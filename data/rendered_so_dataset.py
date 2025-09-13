#!/usr/bin/env python3
# datasets/rendered_so_dataset.py
# ======================================================================
# Teapot / mesh renders over SO(3), S^2 (zero roll), and a *flat torus*
# (S^1 x S^1 via azimuth+roll with fixed elevation) with fast, safe
# multi-GPU generation, anti-aliased downsampling, and geodesic rendering.
# Includes: CLI to generate & cache the dataset; OOM-safe micro-batching.
# ======================================================================

from __future__ import annotations
import argparse
import math
import multiprocessing as mp
import os
import pathlib
import time
from contextlib import suppress
from typing import Sequence, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.multiprocessing as _mp
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    BlendParams,
    PointLights,
    TexturesVertex,
    Materials,
    look_at_view_transform,
)
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate

# Avoid CUDA shared memory issues across processes
_mp.set_sharing_strategy("file_system")

# ───────────────────────────── Shader ──────────────────────────────── #
class MultiLightSoftShader(SoftPhongShader):
    """SoftPhongShader that accepts lights.location shaped (B, L, 3).
    When L>1 we loop lights to reuse PyTorch3D internals.
    """

    def forward(self, fragments, meshes, **kw):
        cameras = super()._get_cameras(**kw)
        lights = kw.get("lights", self.lights)
        materials = kw.get("materials", self.materials)
        blend = kw.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)

        if hasattr(lights, "location") and lights.location.ndim == 3:
            # Batched multi-light: average contributions
            _, L, _ = lights.location.shape
            col = 0.0
            for li in range(L):
                single = PointLights(
                    device=lights.device,
                    location=lights.location[:, li],
                    ambient_color=lights.ambient_color[:, li],
                    diffuse_color=lights.diffuse_color[:, li],
                    specular_color=lights.specular_color[:, li],
                )
                col += phong_shading(
                    meshes=meshes,
                    fragments=fragments,
                    texels=texels,
                    lights=single,
                    cameras=cameras,
                    materials=materials,
                )
            colours = col / L
        else:
            colours = phong_shading(
                meshes=meshes,
                fragments=fragments,
                texels=texels,
                lights=lights,
                cameras=cameras,
                materials=materials,
            )

        znear = kw.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kw.get("zfar", getattr(cameras, "zfar", 100.0))
        return softmax_rgb_blend(colours, fragments, blend, znear=znear, zfar=zfar)

# ───────────────────── Constants & Helpers ─────────────────────────── #
FOV = 25.0
CAM_DIST = 1.05 / math.sin(math.radians(FOV / 2))


def _aa(final_H: int):
    """Anti-aliasing presets per target height.
    - render: super-sample resolution (square)
    - blur: rasterization blur radius
    - faces: faces_per_pixel (bigger → smoother silhouettes; heavier)
    - down_mid: first downsample size (area)
    - down_final: optional second downsample (bicubic)
    """
    if final_H == 128:
        return dict(render=256, blur=1e-4, faces=12, down_mid=128, down_final=None)
    if final_H == 64:
        return dict(render=512, blur=1e-6, faces=30, down_mid=128, down_final=64)
    raise ValueError("image_size must be 64 or 128")


def _to_srgb(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) ** (1 / 2.2)


def _normalise_mesh(mesh):
    v = mesh.verts_packed()
    mesh.offset_verts_(-v.mean(0))
    mesh.scale_verts_((1.0 / v.norm(dim=1).max()).item())
    return mesh


def _base_camera(dev: torch.device):
    return FoVPerspectiveCameras(
        device=dev,
        R=torch.eye(3, device=dev)[None],
        T=torch.tensor([[0.0, 0.0, CAM_DIST]], device=dev),
        fov=FOV,
    )


def _build_renderer(
    res: int,
    blur: float,
    faces: int,
    dev: torch.device,
    *,
    cull_backfaces: bool = False,
    bin_size: Optional[int] = 0,
):
    raster = RasterizationSettings(
        image_size=(res, res),
        blur_radius=blur,
        faces_per_pixel=faces,
        cull_backfaces=cull_backfaces,
        bin_size=bin_size,
    )
    cam = _base_camera(dev)
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))
    shader = MultiLightSoftShader(device=dev, cameras=cam, blend_params=blend, lights=None)
    return MeshRenderer(MeshRasterizer(cam, raster), shader)


def _downsample(lin: torch.Tensor, aa):
    """Downsample with area → (optional) bicubic; accepts (B,C,H,W) or (C,H,W)."""
    is_3d = lin.dim() == 3
    if is_3d:
        lin = lin.unsqueeze(0)
    mid = torch.nn.functional.interpolate(lin, size=aa["down_mid"], mode="area")
    if aa["down_final"]:
        mid = torch.nn.functional.interpolate(mid, size=aa["down_final"], mode="bicubic", align_corners=False)
    if is_3d:
        mid = mid.squeeze(0)
    return mid

# ───────────────────── SO(3) Utilities ─────────────────────────────── #

def _so3_log_batch(R: torch.Tensor) -> torch.Tensor:
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    theta2 = theta * theta
    coef = torch.where(theta.abs() < 1e-5, 0.5 - theta2 / 12 + theta2 * theta2 / 720, theta / (2 * torch.sin(theta)))
    return coef.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2, -1))


def _so3_exp_batch(A: torch.Tensor) -> torch.Tensor:
    w = torch.stack((A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]), -1)
    theta = torch.linalg.vector_norm(w, dim=-1)
    theta2 = theta * theta
    s = torch.where(theta.abs() < 1e-5, 1 - theta2 / 6 + theta2 * theta2 / 120, torch.sin(theta) / theta)
    c = torch.where(theta.abs() < 1e-5, 0.5 - theta2 / 12 + theta2 * theta2 / 720, (1 - torch.cos(theta)) / theta2)
    s, c = s.unsqueeze(-1).unsqueeze(-1), c.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, device=A.device).expand_as(A)
    return I + s * A + c * (A @ A)


def _build_lights_batch(R_view: torch.Tensor, dev: torch.device, dist=4.0, rim=True) -> PointLights:
    """Create per-sample multi-light in world coords from camera-to-world rotation."""
    dirs_cam = torch.tensor([[0.35, 0.35, 1.0], [-0.35, 0.20, 1.0], [0.0, 0.0, -1.0]], device=dev)
    if rim:
        dirs_cam = torch.cat([dirs_cam, torch.tensor([[0.0, 0.0, -1.0]], device=dev)], 0)
    # transform to world
    dirs_w = (R_view @ dirs_cam.t()).transpose(1, 2)
    dirs_w = dirs_w / dirs_w.norm(dim=-1, keepdim=True)
    locs = dirs_w * dist
    L = locs.size(1)
    amb = locs.new_full((locs.size(0), L, 3), 0.10)
    diff = locs.new_full((locs.size(0), L, 3), 0.65)
    spec = locs.new_full((locs.size(0), L, 3), 0.25)
    if rim:
        diff[:, -1], amb[:, -1], spec[:, -1] = torch.tensor([0.35, 0.35, 0.35], device=dev), 0.0, 0.0
    return PointLights(device=dev, location=locs, ambient_color=amb, diffuse_color=diff, specular_color=spec)

# ───────────────────── Torus helpers ────────────────────────────────── #

def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi

def _Rz_cam(deg: float, device) -> torch.Tensor:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device)

# ───────────────────── MP Worker Renderer (OOM-safe) ───────────────── #

def _worker_render(
    rank: int,
    device: str,
    chunk: Sequence,
    mesh_path: str,
    aa,
    channels: int,
    grid_mode: bool,
    seed: int,
    queue,
    *,
    flush: int = 1024,
    render_batch: int = 32,
    cull_backfaces: bool = False,
    bin_size: Optional[int] = 0,
):
    torch.manual_seed(seed)
    dev = torch.device(device)

    # Prepare base mesh & renderer on this device
    mesh0 = load_objs_as_meshes([mesh_path], device=dev)
    mesh0 = _normalise_mesh(mesh0)
    if mesh0.textures is None:
        mesh0.textures = TexturesVertex(
            verts_features=torch.ones_like(mesh0.verts_padded(), dtype=torch.float32)[None] * 0.7
        )

    renderer = _build_renderer(aa["render"], aa["blur"], aa["faces"], dev, cull_backfaces=cull_backfaces, bin_size=bin_size)
    renderer.shader.materials = Materials(device=dev, specular_color=((0.9, 0.9, 0.9),), shininess=100.0)

    V0 = mesh0.verts_padded()  # (1,V,3)
    F0 = mesh0.faces_padded()  # (1,F,3)
    C0 = mesh0.textures.verts_features_padded()  # (1,V,3)

    def to_R(item):
        if not grid_mode:
            return item.to(dev)
        az, el, rl = item
        R_cam, _ = look_at_view_transform(CAM_DIST, el, az, degrees=True, device=dev)
        R_cam = R_cam.squeeze(0)
        if abs(float(rl)) > 0.0:
            R_cam = _Rz_cam(float(rl), dev) @ R_cam
        return R_cam.transpose(0, 1)

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    imgs, rots = [], []
    pack_size = int(render_batch)

    for pack in batched(list(chunk), pack_size):
        Rs = torch.stack([to_R(x) for x in pack], 0)  # (B,3,3)

        start = 0
        while start < len(pack):
            cur_B = len(pack) - start
            cur_B = min(pack_size, cur_B)
            sub_Rs = Rs[start : start + cur_B]

            try:
                RV = Rotate(sub_Rs, device=dev).transform_points(V0.expand(cur_B, -1, -1))  # (cur_B,V,3)
                verts_list = [RV[i] for i in range(cur_B)]
                faces_list = [F0[0] for _ in range(cur_B)]
                color_list = [C0[0] for _ in range(cur_B)]
                mesh_batch = Meshes(verts=verts_list, faces=faces_list, textures=TexturesVertex(verts_features=color_list))

                # Lights expect camera-to-world rotations; R_view = R_obj^T
                renderer.shader.lights = _build_lights_batch(sub_Rs.transpose(1, 2).contiguous(), dev)

                hi = renderer(mesh_batch)[..., :3].permute(0, 3, 1, 2)  # (cur_B,3,Hhi,Whi)
                lo = _downsample(hi, aa)  # (cur_B,C,H,W)
                if channels == 1:
                    lo = lo.mean(1, keepdim=True)

                imgs.append((_to_srgb(lo) * 255).byte().cpu())
                rots.append(sub_Rs.cpu())

                start += cur_B

                if sum(x.size(0) for x in imgs) >= flush:
                    queue.put((torch.cat(imgs, 0), torch.cat(rots, 0)))
                    imgs, rots = [], []

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_B == 1:
                    raise
                pack_size = max(1, cur_B // 2)
                continue

    if imgs:
        queue.put((torch.cat(imgs, 0), torch.cat(rots, 0)))

# ───────────────────── Dataset Class ───────────────────────────────── #

class RenderedSO3Dataset(Dataset):
    """Rendered mesh dataset over SO(3), S^2 (zero roll), or Torus(az,roll) views.

    Supported modes:
      - manifold_dim = 3: Haar-uniform SO(3) samples
      - manifold_dim = 2 & submanifold == "s2_zeroroll": uniform S^2 views with zero roll
      - manifold_dim = 2 & submanifold == "torus_az_roll": S^1×S^1 with fixed elevation (flat torus)

    Features:
      - Multi-process, multi-GPU rendering with OOM-safe micro-batching
      - Anti-aliased supersampling and downsampling
      - Optional ambient projection to a higher-dimensional embedding (ambient_dim)
      - Geodesic rendering for SO(3), S^2 (zero-roll), and Torus(az,roll)
    """

    def __init__(self, args, *, seed: int = 0):
        g = torch.Generator().manual_seed(seed)

        # Args (mirror CLI)
        self.mesh_path: str = args.mesh_path
        self.cache_path: Optional[str] = getattr(args, "dataset_path", None)
        self.image_size: int = int(args.image_size)
        self.channels: int = int(args.channels)
        self.device: str = str(getattr(args, "device", "cuda")).lower()
        self.azim_step: Optional[float] = getattr(args, "azim_step", None)
        self.elev_step: Optional[float] = getattr(args, "elev_step", None)
        self.roll_step: Optional[float] = getattr(args, "roll_step", None)
        self.N_random: Optional[int] = getattr(args, "data_samples", None)
        self.overwrite: bool = bool(getattr(args, "overwrite_cache", False))
        self.ambient_dim: Optional[int] = getattr(args, "ambient_dim", None)
        self.manifold_dim: int = int(getattr(args, "manifold_dim", 3))
        self.submanifold: str = str(getattr(args, "submanifold", "s2_zeroroll"))
        self.torus_elev_deg: float = float(getattr(args, "torus_elev_deg", 25.0))
        self.n_workers: int = int(getattr(args, "n_workers", 0) or self._default_workers())
        self.render_batch: int = int(getattr(args, "render_batch", 32))
        self.cull_backfaces: bool = bool(getattr(args, "cull_backfaces", False))
        self.bin_size: Optional[int] = getattr(args, "bin_size", 0)

        self.H = self.W = self.image_size
        self._aa = _aa(self.image_size)

        # Only support S^2 zero-roll or torus for 2D manifold
        if self.manifold_dim == 2 and self.submanifold.lower() not in ("s2_zeroroll", "torus_az_roll", "s1xs1", "flat_torus"):
            raise ValueError("For manifold_dim=2 use submanifold in {'s2_zeroroll','torus_az_roll'}.")

        # Load cache or generate
        if self.cache_path and os.path.isfile(self.cache_path) and not self.overwrite:
            blob = torch.load(self.cache_path, map_location="cpu")
            self.data, self.rotmats, self.P = blob["data"], blob["rot"], blob["proj"]
            print(f"[RenderedSO3] loaded {len(self)} cached samples from {self.cache_path}")
        else:
            self._generate_dataset(g)
            if self.cache_path:
                pathlib.Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"data": self.data, "rot": self.rotmats, "proj": self.P}, self.cache_path)
                print(f"[RenderedSO3] cached → {self.cache_path}")
        self._materialise_images()

    # ─────────────── helpers ─────────────── #
    @staticmethod
    def _visible_cuda() -> List[str]:
        env = os.getenv("CUDA_VISIBLE_DEVICES")
        if env and env.strip():
            return [f"cuda:{i}" for i in range(len(env.split(",")))]
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    def _default_workers(self) -> int:
        return max(1, len(self._visible_cuda())) if self.device.startswith("cuda") else mp.cpu_count()

    def _materialise_images(self):
        self.images = None
        if self.P.shape[0] == self.P.shape[1] and torch.allclose(self.P, torch.eye(self.P.shape[0])):
            self.images = self.data.view(-1, self.channels, self.H, self.W).contiguous()

    # ─────────────── dataset generation ─────────────── #
    def _generate_dataset(self, g: torch.Generator):
        grid_mode = self.azim_step is not None and self.elev_step is not None and self.roll_step is not None

        if grid_mode:
            sub = self.submanifold.lower()
            if self.manifold_dim == 2 and sub == "s2_zeroroll":
                if float(self.roll_step) != 0.0:
                    raise ValueError("For manifold_dim=2 (S^2 zero-roll), set roll_step=0 in grid mode.")
                az = torch.arange(-180.0, 180.0, float(self.azim_step))
                el = torch.arange(-90.0, 90.0 + 1e-9, float(self.elev_step))
                rl = torch.tensor([0.0])
                tasks = [(a.item(), e.item(), r.item()) for r in rl for e in el for a in az]
                print(f"[RenderedSO3] Euler grid (S^2 zero-roll) → {len(tasks)} rotations")
                worker_grid_mode = True
            elif self.manifold_dim == 2 and sub in ("torus_az_roll", "s1xs1", "flat_torus"):
                az = torch.arange(-180.0, 180.0, float(self.azim_step))
                rl = torch.arange(-180.0, 180.0, float(self.roll_step))
                el0 = float(self.torus_elev_deg)
                tasks = [(a.item(), el0, r.item()) for r in rl for a in az]
                print(f"[RenderedSO3] Torus grid → {len(tasks)} rotations at elev={el0}°")
                worker_grid_mode = True
            else:
                # SO(3) grid via Euler (rarely used; keep consistent)
                az = torch.arange(-180.0, 180.0, float(self.azim_step))
                el = torch.arange(-90.0, 90.0 + 1e-9, float(self.elev_step))
                rl = torch.arange(-180.0, 180.0, float(self.roll_step))
                tasks = [(a.item(), e.item(), r.item()) for r in rl for e in el for a in az]
                print(f"[RenderedSO3] Euler grid (SO(3)) → {len(tasks)} rotations")
                worker_grid_mode = True

        else:
            if self.manifold_dim == 3:
                N = int(self.N_random or 100_000)
                G = torch.randn(N, 3, 3, generator=g)
                Q, _ = torch.linalg.qr(G)
                det = torch.linalg.det(Q)
                Q[det < 0, :, 0] *= -1
                tasks = [Q[i] for i in range(N)]
                print(f"[RenderedSO3] Haar-uniform sample N = {N}")
                worker_grid_mode = False
            elif self.manifold_dim == 2:
                sub = self.submanifold.lower()
                if sub == "s2_zeroroll":
                    # Uniform S^2 with zero roll
                    N = int(self.N_random or 100_000)
                    az = torch.rand(N, generator=g) * 360.0 - 180.0  # degrees
                    u = torch.rand(N, generator=g) * 2.0 - 1.0
                    el = torch.asin(u) * (180.0 / math.pi)
                    R_cam, _ = look_at_view_transform(CAM_DIST, el, az, degrees=True, device="cpu")
                    R_obj_all = R_cam.transpose(1, 2).contiguous()
                    tasks = [R_obj_all[i] for i in range(N)]
                    print(f"[RenderedSO3] 2-D submanifold=S2 (zero roll), N={N} (precomputed rotations)")
                    worker_grid_mode = False
                elif sub in ("torus_az_roll", "s1xs1", "flat_torus"):
                    # Torus: random azimuth & roll; fixed elevation
                    N = int(self.N_random or 100_000)
                    az = torch.rand(N, generator=g) * 360.0 - 180.0
                    rol = torch.rand(N, generator=g) * 360.0 - 180.0
                    el0 = float(self.torus_elev_deg)
                    tasks = [(float(az[i]), el0, float(rol[i])) for i in range(N)]
                    print(f"[RenderedSO3] 2-D submanifold=Torus(az,roll) @ elev={el0}°, N={N}")
                    worker_grid_mode = True
                else:
                    raise ValueError(f"Unknown 2-D submanifold '{self.submanifold}'.")
            else:
                raise ValueError(f"manifold_dim must be 2 or 3, got {self.manifold_dim}")

        if self.n_workers <= 1:
            self._gen_serial(tasks, worker_grid_mode)
        else:
            self._gen_mp(tasks, worker_grid_mode)

        # Embed (optional) into ambient_dim >= D_img via random isometry
        flat = self.data.view(self.data.size(0), -1)
        d_img = flat.size(1)
        d_emb = int(self.ambient_dim or d_img)
        if d_emb > d_img:
            A, _ = torch.linalg.qr(torch.randn(d_emb, d_img, generator=g))
            self.P = A.float()
            self.data = (A @ flat.T).T.float()
        else:
            self.P = torch.eye(d_img).float()
            self.data = flat.float()

        # Preview contact sheet (best-effort)
        with suppress(Exception):
            preview = (os.path.splitext(self.cache_path)[0] if self.cache_path else "rendered") + "_preview.png"
            pathlib.Path(preview).parent.mkdir(parents=True, exist_ok=True)
            save_image(self.data.view(-1, self.channels, self.H, self.W)[:64], preview, nrow=8, normalize=True)
            print("[RenderedSO3] preview saved →", preview)

    def _gen_mp(self, tasks, grid_mode):
        devs = self._visible_cuda() if self.device == "cuda" or self.device.startswith("cuda:") else [self.device]
        n_proc = min(self.n_workers, len(devs))
        chunk = math.ceil(len(tasks) / n_proc)
        chunks = [tasks[i : i + chunk] for i in range(0, len(tasks), chunk)]
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(maxsize=2 * n_proc)
        procs = []
        for rk, ch in enumerate(chunks):
            p = ctx.Process(
                target=_worker_render,
                kwargs=dict(
                    rank=rk,
                    device=devs[rk % len(devs)],
                    chunk=ch,
                    mesh_path=self.mesh_path,
                    aa=self._aa,
                    channels=self.channels,
                    grid_mode=grid_mode,
                    seed=17 + rk,
                    queue=queue,
                    flush=1024,
                    render_batch=self.render_batch,
                    cull_backfaces=self.cull_backfaces,
                    bin_size=self.bin_size,
                ),
            )
            p.start()
            procs.append(p)
        imgs, rots, rec = [], [], 0
        with tqdm(total=len(tasks), desc="Rendering SO(3) (mp)") as pbar:
            while rec < len(tasks):
                i, r = queue.get()
                imgs.append(i)
                rots.append(r)
                rec += i.size(0)
                pbar.update(i.size(0))
        for p in procs:
            p.join()
        self.data = torch.cat(imgs).float() / 255.0
        self.rotmats = torch.cat(rots).float()

    def _gen_serial(self, tasks, grid_mode):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        _worker_render(
            0,
            self.device,
            tasks,
            self.mesh_path,
            self._aa,
            self.channels,
            grid_mode,
            42,
            queue,
            flush=len(tasks),
            render_batch=self.render_batch,
            cull_backfaces=self.cull_backfaces,
            bin_size=self.bin_size,
        )
        imgs, rots = queue.get()
        self.data = imgs.float() / 255.0
        self.rotmats = rots.float()

    # ─────────────── Dataset Interface ─────────────── #
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        if self.images is not None:
            return self.images[idx], self.rotmats[idx]
        return self.data[idx].view(self.channels, self.H, self.W), self.rotmats[idx]

    # ─────────────── Geodesics (SO(3), S^2 zero-roll, Torus) ────────── #
    def _get_geo_ctx(self, dev: torch.device):
        """Cache (mesh, renderer) on a device. Keep float32 throughout."""
        if not hasattr(self, "_geo_ctx"):
            self._geo_ctx = {}
        if dev not in self._geo_ctx:
            mesh = load_objs_as_meshes([self.mesh_path], device=dev)
            if mesh.textures is None:
                mesh.textures = TexturesVertex(
                    verts_features=torch.ones_like(mesh.verts_padded(), dtype=torch.float32)[None] * 0.7
                )
            mesh = _normalise_mesh(mesh)
            renderer = _build_renderer(
                self._aa["render"], self._aa["blur"], self._aa["faces"], dev,
                cull_backfaces=self.cull_backfaces, bin_size=self.bin_size,
            )
            renderer.shader.materials = Materials(device=dev, specular_color=((0.9, 0.9, 0.9),), shininess=100.0)
            self._geo_ctx[dev] = dict(mesh=mesh, renderer=renderer)
        return self._geo_ctx[dev]["mesh"], self._geo_ctx[dev]["renderer"]

    @staticmethod
    def _camera_view_from_direction(n: torch.Tensor) -> torch.Tensor:
        """Return camera-to-world rotation with ZERO roll for a world direction n (B,3)."""
        B = n.size(0)
        f = F.normalize(-n, dim=-1)
        up0 = n.new_tensor([0.0, 1.0, 0.0]).expand(B, 3)
        r = torch.cross(up0, f, dim=-1)
        bad = r.norm(dim=-1, keepdim=True) < 1e-8
        up1 = n.new_tensor([0.0, 0.0, 1.0]).expand(B, 3)
        up = torch.where(bad, up1, up0)
        r = F.normalize(torch.cross(up, f, dim=-1), dim=-1)
        u = F.normalize(torch.cross(f, r, dim=-1), dim=-1)
        return torch.stack([r, u, f], dim=-1)

    @staticmethod
    def _s2_geodesic_directions(n0: torch.Tensor, n1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, T = n0.size(0), t.numel()
        n0 = F.normalize(n0, dim=-1)
        n1 = F.normalize(n1, dim=-1)
        k_raw = torch.cross(n0, n1, dim=-1)
        k_norm = k_raw.norm(dim=-1, keepdim=True)
        dot = (n0 * n1).sum(-1)
        theta = torch.atan2(k_norm.squeeze(-1), dot.clamp(-1.0, 1.0))
        eps = 1e-8
        k = torch.where(k_norm > eps, k_raw / (k_norm + 1e-12), k_raw)
        nearpi = (k_norm.squeeze(-1) < 1e-8) & (dot < 0)
        if nearpi.any():
            idx = nearpi.nonzero(as_tuple=False).squeeze(-1)
            a = torch.tensor([1.0, 0.0, 0.0], device=n0.device).expand(idx.numel(), 3)
            alt = torch.tensor([0.0, 1.0, 0.0], device=n0.device).expand(idx.numel(), 3)
            v = torch.cross(n0[idx], a, dim=-1)
            bad = v.norm(dim=-1, keepdim=True) < 1e-6
            v = torch.where(bad, torch.cross(n0[idx], alt, dim=-1), v)
            k[idx] = F.normalize(torch.cross(n0[idx], v, dim=-1), dim=-1)
            theta[idx] = math.pi
        kBT = k.unsqueeze(1).expand(B, T, 3)
        n0BT = n0.unsqueeze(1).expand(B, T, 3)
        ang = (theta.unsqueeze(1) * t.view(1, T)).unsqueeze(-1)
        ca = torch.cos(ang)
        sa = torch.sin(ang)
        k_dot_n0 = (kBT * n0BT).sum(-1, keepdim=True)
        n = n0BT * ca + torch.cross(kBT, n0BT, dim=-1) * sa + kBT * k_dot_n0 * (1.0 - ca)
        return F.normalize(n, dim=-1)

    def _compute_geodesic_so3(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dev = P.device
        B, T = P.size(0), t.numel()
        N = B * T
        A = _so3_log_batch(P.transpose(-2, -1) @ Q)
        Rts = P.unsqueeze(1) @ _so3_exp_batch(A.unsqueeze(1) * t.view(1, T, 1, 1))
        Rts_flat = Rts.reshape(N, 3, 3).contiguous()
        mesh0, renderer = self._get_geo_ctx(dev)
        hi_frames: List[torch.Tensor] = []
        for k in range(N):
            R_obj = Rts_flat[k]
            mesh_inst = mesh0.clone().update_padded(Rotate(R_obj[None], device=dev).transform_points(mesh0.verts_padded()))
            renderer.shader.lights = _build_lights_batch(R_obj.transpose(0, 1).unsqueeze(0).contiguous(), dev)
            hi = renderer(mesh_inst)[0, :, :, :3].permute(2, 0, 1)
            hi_frames.append(hi)
        hi_stack = torch.stack(hi_frames, dim=0)
        lin = _downsample(hi_stack, self._aa)
        if self.channels == 1:
            lin = lin.mean(1, keepdim=True)
        return _to_srgb(lin).view(B, T, self.channels, self.H, self.W)

    def _compute_geodesic_s2_zeroroll(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dev = P.device
        B, T = P.size(0), t.numel()
        mesh0, renderer = self._get_geo_ctx(dev)
        Rv0 = P.transpose(-2, -1)
        Rv1 = Q.transpose(-2, -1)
        z_cam = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(B, 3)
        n0 = (Rv0 @ z_cam.unsqueeze(-1)).squeeze(-1)
        n1 = (Rv1 @ z_cam.unsqueeze(-1)).squeeze(-1)
        k_raw = torch.cross(n0, n1, dim=-1)
        k_norm = k_raw.norm(dim=-1, keepdim=True)
        dot = (n0 * n1).sum(-1)
        theta = torch.atan2(k_norm.squeeze(-1), dot.clamp(-1.0, 1.0))
        eps = 1e-8
        k = torch.where(k_norm > eps, k_raw / (k_norm + 1e-12), k_raw)
        nearpi = (k_norm.squeeze(-1) < 1e-8) & (dot < 0)
        if nearpi.any():
            idx = nearpi.nonzero(as_tuple=False).squeeze(-1)
            a = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(idx.numel(), 3)
            alt = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(idx.numel(), 3)
            v = torch.cross(n0[idx], a, dim=-1)
            bad = v.norm(dim=-1, keepdim=True) < 1e-6
            v = torch.where(bad, torch.cross(n0[idx], alt, dim=-1), v)
            k[idx] = F.normalize(torch.cross(n0[idx], v, dim=-1), dim=-1)
            theta[idx] = math.pi
        Rv_start = self._camera_view_from_direction(n0)

        frames = []
        for b in range(B):
            for ti in range(T):
                ang = float(theta[b]) * float(t[ti])
                ax = k[b]
                K = torch.tensor(
                    [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]], device=dev
                )
                I = torch.eye(3, device=dev)
                s, c = math.sin(ang), math.cos(ang)
                Rax = I + s * K + (1.0 - c) * (K @ K)
                Rv = Rax @ Rv_start[b]
                Ro = Rv.transpose(0, 1).contiguous()
                mesh_inst = mesh0.clone().update_padded(Rotate(Ro[None], device=dev).transform_points(mesh0.verts_padded()))
                renderer.shader.lights = _build_lights_batch(Rv.unsqueeze(0).contiguous(), dev)
                hi = renderer(mesh_inst)[0, :, :, :3].permute(2, 0, 1)
                frames.append(hi)
        hi_stack = torch.stack(frames, dim=0)
        lin = _downsample(hi_stack, self._aa)
        if self.channels == 1:
            lin = lin.mean(1, keepdim=True)
        return _to_srgb(lin).view(B, T, self.channels, self.H, self.W)

    @staticmethod
    def _torus_angles_from_Rview(Rv: torch.Tensor, elev_deg: float) -> Tuple[float, float]:
        """
        Decompose camera-to-world rotation as Rv ≈ Rz(roll) @ Rview0(az),
        where Rview0(az) = look_at_view_transform(CAM_DIST, elev_deg, az, ...).
        Returns (az_deg, roll_deg) in degrees.
        Assumes elevation not at poles (avoid ±90°).
        """
        device = Rv.device
        # forward (camera z in world)
        f = Rv[:, 2]  # (3,)
        # azimuth from forward x,z; negate to match look_at forward
        az = math.degrees(math.atan2(float(-f[2]), float(-f[0])))

        R0, _ = look_at_view_transform(CAM_DIST, elev_deg, az, degrees=True, device=device)
        R0 = R0.squeeze(0)  # camera-to-world, zero roll at this azimuth

        # r0,u0 are right/up of zero-roll frame; r is right of actual frame
        r0 = R0[:, 0]; u0 = R0[:, 1]
        r  = Rv[:, 0]
        a = float(torch.dot(r, r0))
        b = float(torch.dot(r, u0))
        roll = math.degrees(math.atan2(b, a))
        return az, roll

    def _compute_geodesic_torus(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Geodesic on S^1×S^1 with flat metric: straight line in (az, roll) with wrap.
        Inputs:
          P, Q: (B,3,3) object-to-world rotations used in dataset (your convention)
          t:    (T,) linspace in [0,1]
        Output: (B,T,C,H,W) sRGB frames
        """
        dev = P.device
        B, T = P.size(0), t.numel()
        elev = float(self.torus_elev_deg)

        # convert to camera-to-world
        Rv0 = P.transpose(1, 2).contiguous()  # (B,3,3)
        Rv1 = Q.transpose(1, 2).contiguous()

        # recover (az, roll) for endpoints
        az0, rl0, az1, rl1 = [], [], [], []
        for b in range(B):
            a0, r0 = self._torus_angles_from_Rview(Rv0[b], elev)
            a1, r1 = self._torus_angles_from_Rview(Rv1[b], elev)
            az0.append(a0); rl0.append(r0)
            az1.append(a1); rl1.append(r1)
        az0 = torch.tensor(az0, device=dev) * (math.pi/180)
        rl0 = torch.tensor(rl0, device=dev) * (math.pi/180)
        az1 = torch.tensor(az1, device=dev) * (math.pi/180)
        rl1 = torch.tensor(rl1, device=dev) * (math.pi/180)

        # shortest modular deltas on S^1
        d_az = _wrap_to_pi(az1 - az0)
        d_rl = _wrap_to_pi(rl1 - rl0)

        mesh0, renderer = self._get_geo_ctx(dev)
        frames: List[torch.Tensor] = []
        for b in range(B):
            for ti in range(T):
                a = float(az0[b] + d_az[b] * t[ti])
                r = float(rl0[b] + d_rl[b] * t[ti])
                # back to degrees for look_at
                a_deg = math.degrees((a + 2*math.pi) % (2*math.pi))
                r_deg = math.degrees((r + 2*math.pi) % (2*math.pi))
                Rcam, _ = look_at_view_transform(CAM_DIST, elev, a_deg, degrees=True, device=dev)
                Rcam = Rcam.squeeze(0)
                Rcam = _Rz_cam(r_deg, dev) @ Rcam
                Robj = Rcam.transpose(0, 1).contiguous()

                mesh_inst = mesh0.clone().update_padded(Rotate(Robj[None], device=dev).transform_points(mesh0.verts_padded()))
                renderer.shader.lights = _build_lights_batch(Rcam.unsqueeze(0).contiguous(), dev)
                hi = renderer(mesh_inst)[0, :, :, :3].permute(2, 0, 1)
                frames.append(hi)

        hi_stack = torch.stack(frames, dim=0)  # (B*T,3,Hhi,Whi)
        lin = _downsample(hi_stack, self._aa)
        if self.channels == 1:
            lin = lin.mean(1, keepdim=True)
        return _to_srgb(lin).view(B, T, self.channels, self.H, self.W)

    def compute_geodesic(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.manifold_dim == 3:
            return self._compute_geodesic_so3(P, Q, t)
        sub = self.submanifold.lower()
        if self.manifold_dim == 2 and sub == "s2_zeroroll":
            return self._compute_geodesic_s2_zeroroll(P, Q, t)
        if self.manifold_dim == 2 and sub in ("torus_az_roll", "s1xs1", "flat_torus"):
            return self._compute_geodesic_torus(P, Q, t)
        raise NotImplementedError("Geodesics for the selected submanifold are not implemented.")

# ───────────────── Debug & CLI ──────────────────────────────────────── #

def _geo_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> float:
    cos = ((Ra.T @ Rb).trace() - 1).clamp(-1.0, 1.0) / 2
    return torch.acos(cos).item()


def _debug_geodesics(ds: "RenderedSO3Dataset", pairs=5, frames=10, out_root="datasets/geo_debug"):
    os.makedirs(out_root, exist_ok=True)
    chosen: List[Tuple[int, int]] = []
    cand = torch.randperm(len(ds))[: min(500, len(ds))]

    for _ in range(pairs):
        if not chosen:
            i = cand[0].item()
        else:
            used = torch.tensor([u for ij in chosen for u in ij])
            remain = cand[~cand.unsqueeze(1).eq(used).any(1)]
            scores = [
                (min(_geo_angle(ds.rotmats[k], ds.rotmats[u]) for u in used), k.item()) for k in remain
            ]
            i = max(scores)[1] if scores else remain[0].item()
        j = max(((
            _geo_angle(ds.rotmats[i], ds.rotmats[j]), j.item()) for j in cand if j != i
        ), key=lambda x: x[0])[1]
        chosen.append((i, j))

    t = torch.linspace(0, 1, frames)
    dev = torch.device(ds.device if ("cuda" in ds.device and torch.cuda.is_available()) else "cpu")
    p_rots = ds.rotmats[torch.tensor([c[0] for c in chosen])].to(dev).contiguous()
    q_rots = ds.rotmats[torch.tensor([c[1] for c in chosen])].to(dev).contiguous()
    t = t.to(dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    sheet = ds.compute_geodesic(p_rots, q_rots, t).flatten(0, 1)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    B, T = len(chosen), frames
    print(f"[profile] geodesic render B={B} T={T} → {dt*1000:.1f} ms  ({dt/(B*T)*1000:.2f} ms/frame) on {dev}")

    save_image(sheet, os.path.join(out_root, "geodesics_grid.png"), nrow=frames, normalize=True)
    print("Saved contact sheet →", f"{out_root}/geodesics_grid.png")


if __name__ == "__main__":
    P = argparse.ArgumentParser("Build rendered SO(3)/S^2/Torus dataset")
    P.add_argument("--mesh_path", required=True)
    P.add_argument("--dataset_path", required=True)
    P.add_argument("--image_size", type=int, choices=[64, 128], default=128)
    P.add_argument("--channels", type=int, choices=[1, 3], default=1)

    # sampling/grid
    P.add_argument("--azim_step", type=float)
    P.add_argument("--elev_step", type=float)
    P.add_argument("--roll_step", type=float)
    P.add_argument("--data_samples", type=int)

    # geometry
    P.add_argument("--manifold_dim", type=int, default=3)
    P.add_argument("--submanifold", type=str, default="s2_zeroroll",
                   help="one of: s2_zeroroll, torus_az_roll")
    P.add_argument("--torus_elev_deg", type=float, default=25.0,
                   help="Fixed elevation (degrees) for torus_az_roll (avoid poles).")

    # embed / cache
    P.add_argument("--ambient_dim", type=int)
    P.add_argument("--overwrite_cache", action="store_true")

    # perf / devices
    P.add_argument("--device", default="cuda")
    P.add_argument("--n_workers", type=int, default=0)
    P.add_argument("--render_batch", type=int, default=32)
    P.add_argument("--cull_backfaces", action="store_true")
    P.add_argument("--bin_size", type=int, default=0)

    # debug
    P.add_argument("--debug_pairs", type=int, default=0, help="if >0, render debug geodesic sheet with this many pairs")
    P.add_argument("--debug_frames", type=int, default=20)

    args = P.parse_args()

    ds = RenderedSO3Dataset(args)
    print(f"Dataset built: {len(ds)} samples → {args.dataset_path}")

    if args.debug_pairs > 0:
        _debug_geodesics(ds, pairs=args.debug_pairs, frames=args.debug_frames)
