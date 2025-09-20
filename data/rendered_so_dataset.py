#!/usr/bin/env python3
# datasets/rendered_so_dataset.py
# ======================================================================
# Anti-aliased teapot renders with batched geodesic renderer.
# FULLY FUNCTIONAL: Restores dataset generation and keeps fast geodesics.
# ======================================================================

from __future__ import annotations
import argparse, math, multiprocessing as mp, os, pathlib, time
from contextlib import suppress
from typing import Sequence, List, Tuple
import math, torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import Materials
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.transforms import Rotate
from pytorch3d.renderer import look_at_view_transform

import torch, torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRasterizer, MeshRenderer, RasterizationSettings,
    BlendParams, PointLights, TexturesVertex, Materials, look_at_view_transform,
)
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.transforms import Rotate
from pytorch3d.structures import Meshes

# NEW: used for S^2 geodesics / frame construction
import torch.nn.functional as F

# ───────────────────────────── Shader (Unchanged) ────────────────────────── #
class MultiLightSoftShader(SoftPhongShader):
    """SoftPhongShader that accepts lights.location shaped (B,L,3)."""
    def forward(self, fragments, meshes, **kw):
        cameras   = super()._get_cameras(**kw)
        lights    = kw.get("lights",    self.lights)
        materials = kw.get("materials", self.materials)
        blend     = kw.get("blend_params", self.blend_params)
        texels    = meshes.sample_textures(fragments)

        if lights.location.ndim == 3:  # batched multi-light
            _, L, _ = lights.location.shape
            col = 0.0
            for li in range(L):
                single = PointLights(device         = lights.device,
                                     location       = lights.location[:, li],
                                     ambient_color  = lights.ambient_color[:, li],
                                     diffuse_color  = lights.diffuse_color[:, li],
                                     specular_color = lights.specular_color[:, li])
                col += phong_shading(meshes=meshes, fragments=fragments, texels=texels,
                                     lights=single, cameras=cameras, materials=materials)
            colours = col / L
        else:                           # single-light
            colours = phong_shading(meshes=meshes, fragments=fragments, texels=texels,
                                    lights=lights, cameras=cameras, materials=materials)

        znear = kw.get("znear", getattr(cameras, "znear", 1.0))
        zfar  = kw.get("zfar",  getattr(cameras, "zfar", 100.0))
        return softmax_rgb_blend(colours, fragments, blend, znear=znear, zfar=zfar)

# ───────────────────── Constants & Helpers (Unchanged) ──────────────────── #
FOV      = 25.0
CAM_DIST = 1.05 / math.sin(math.radians(FOV / 2))

def _aa(final_H: int):
    if final_H == 128: return dict(render=256, blur=1e-4, faces=12, down_mid=128, down_final=None)
    if final_H == 64: return dict(render=512, blur=1e-6, faces=30, down_mid=128, down_final=64)
    raise ValueError("image_size must be 64 or 128")

def _to_srgb(x: torch.Tensor): return x.clamp(0, 1) ** (1/2.2)

def _normalise_mesh(mesh):
    v = mesh.verts_packed()
    mesh.offset_verts_(-v.mean(0))
    mesh.scale_verts_((1.0 / v.norm(dim=1).max()).item())
    return mesh

def _base_camera(dev):
    return FoVPerspectiveCameras(
        device=dev,
        R=torch.eye(3, device=dev)[None],
        T=torch.tensor([[0., 0., CAM_DIST]], device=dev),
        fov=FOV
    )

def _build_renderer(res: int, blur: float, faces: int, dev):
    raster = RasterizationSettings(
        image_size=(res, res),
        blur_radius=blur,
        faces_per_pixel=faces,
        cull_backfaces=False,
        bin_size=None
    )
    cam = _base_camera(dev)
    blend = BlendParams(background_color=(1., 1., 1.))
    shader = MultiLightSoftShader(device=dev, cameras=cam, blend_params=blend, lights=None)
    return MeshRenderer(MeshRasterizer(cam, raster), shader)

def _downsample(lin: torch.Tensor, aa):
    # Note: lin must be 4D tensor for interpolate
    is_3d = lin.dim() == 3
    if is_3d:
        lin = lin.unsqueeze(0)
    mid = torch.nn.functional.interpolate(lin, size=aa['down_mid'], mode="area")
    if aa['down_final']:
        mid = torch.nn.functional.interpolate(mid, size=aa['down_final'], mode="bicubic", align_corners=False)
    if is_3d:
        mid = mid.squeeze(0)
    return mid

# ───────────────────── Batched SO(3) & Lighting Utilities (Unchanged) ─────── #
def _so3_log_batch(R: torch.Tensor) -> torch.Tensor:
    tr  = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) * 0.5).clamp(-1., 1.)
    θ   = torch.acos(cos)
    θ2  = θ * θ
    coef= torch.where(θ.abs() < 1e-5, 0.5 - θ2/12 + θ2*θ2/720, θ / (2 * torch.sin(θ)))
    return coef.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2, -1))

def _so3_exp_batch(A: torch.Tensor) -> torch.Tensor:
    w  = torch.stack((A[...,2,1], A[...,0,2], A[...,1,0]), -1)
    θ  = torch.linalg.vector_norm(w, dim=-1)
    θ2 = θ * θ
    s  = torch.where(θ.abs() < 1e-5, 1 - θ2/6 + θ2*θ2/120, torch.sin(θ) / θ)
    c  = torch.where(θ.abs() < 1e-5, 0.5 - θ2/12 + θ2*θ2/720, (1 - torch.cos(θ)) / θ2)
    s, c = s.unsqueeze(-1).unsqueeze(-1), c.unsqueeze(-1).unsqueeze(-1)
    return (torch.eye(3, device=A.device).expand_as(A) + s * A + c * (A @ A))

def _build_lights_batch(R_view: torch.Tensor, dev: torch.device, dist=4.0, rim=True) -> PointLights:
    dirs_cam = torch.tensor([[0.35, 0.35, 1.0], [-0.35, 0.20, 1.0], [0.0, 0.0, -1.0]], device=dev)
    if rim:
        dirs_cam = torch.cat([dirs_cam, torch.tensor([[0., 0., -1.]], device=dev)], 0)

    dirs_w = (R_view @ dirs_cam.t()).transpose(1, 2)
    dirs_w = dirs_w / dirs_w.norm(dim=-1, keepdim=True)
    locs   = dirs_w * dist
    L = locs.size(1)

    amb  = locs.new_full((locs.size(0), L, 3), 0.10)
    diff = locs.new_full((locs.size(0), L, 3), 0.65)
    spec = locs.new_full((locs.size(0), L, 3), 0.25)
    if rim:
        diff[:, -1], amb[:, -1], spec[:, -1] = torch.tensor([0.35, 0.35, 0.35], device=dev), 0., 0.

    return PointLights(device=dev, location=locs, ambient_color=amb, diffuse_color=diff, specular_color=spec)

def _worker_render(
    rank:int, device:str, chunk, mesh_path:str, aa, channels:int,
    grid_mode: bool, seed:int, queue,
    flush:int=1024, render_batch:int=32,  # start conservative; we’ll scale up if you want
):
    torch.manual_seed(seed)
    dev = torch.device(device)

    mesh0 = load_objs_as_meshes([mesh_path], device=dev)
    mesh0 = _normalise_mesh(mesh0)
    if mesh0.textures is None:
        mesh0.textures = TexturesVertex(
            verts_features=torch.ones_like(mesh0.verts_packed(), dtype=torch.float32)[None] * 0.7
        )

    renderer = _build_renderer(aa['render'], aa['blur'], aa['faces'], dev)
    renderer.shader.materials = Materials(device=dev, specular_color=((0.9,0.9,0.9),), shininess=100.0)

    V0 = mesh0.verts_padded()               # (1,V,3)
    F0 = mesh0.faces_padded()               # (1,F,3)
    C0 = mesh0.textures.verts_features_padded()  # (1,V,3)

    def to_R(item):
        if not grid_mode:
            return item.to(dev)
        az, el, rl = item
        R_cam, _ = look_at_view_transform(CAM_DIST, el, az, degrees=True, device=dev)
        R_cam = R_cam.squeeze(0)
        if abs(float(rl)) > 0.0:
            th = math.radians(float(rl)); c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], device=dev)
            R_cam = Rz @ R_cam
        return R_cam.transpose(0, 1)

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    imgs, rots = [], []
    pack_size = int(render_batch)

    for pack in batched(list(chunk), pack_size):
        # Convert to rotation matrices
        Rs = torch.stack([to_R(x) for x in pack], 0)  # (B,3,3)

        # We may need to split this pack if we OOM
        start = 0
        while start < len(pack):
            cur_B = min(pack_size, len(pack) - start)
            sub_Rs = Rs[start:start+cur_B]  # (cur_B,3,3)

            # Build a proper batch of independent meshes
            verts_list, faces_list, color_list = [], [], []
            RV = Rotate(sub_Rs, device=dev).transform_points(V0.expand(cur_B, -1, -1))  # (cur_B,V,3)
            for i in range(cur_B):
                verts_list.append(RV[i])
                faces_list.append(F0[0])
                color_list.append(C0[0])

            mesh_batch = Meshes(
                verts=verts_list,
                faces=faces_list,
                textures=TexturesVertex(verts_features=color_list),
            )

            try:
                # batched lights (B,L,3)
                renderer.shader.lights = _build_lights_batch(sub_Rs.transpose(1,2).contiguous(), dev)
                hi = renderer(mesh_batch)[..., :3].permute(0,3,1,2)   # (cur_B,3,Hhi,Whi)
                lo = _downsample(hi, aa)                              # (cur_B,C,H,W)
                if channels == 1:
                    lo = lo.mean(1, keepdim=True)

                imgs.append((_to_srgb(lo) * 255).byte().cpu())
                rots.append(sub_Rs.cpu())
                start += cur_B

                # flush to parent to reduce queue backpressure
                if sum(x.size(0) for x in imgs) >= flush:
                    queue.put((torch.cat(imgs,0), torch.cat(rots,0)))
                    imgs, rots = [], []

            except torch.cuda.OutOfMemoryError:
                # Back off: halve the current chunk and retry
                torch.cuda.empty_cache()
                if cur_B == 1:
                    raise  # even B=1 fails -> give up
                pack_size = max(1, cur_B // 2)
                # retry with a smaller sub-pack (don’t advance `start`)
                continue

    if imgs:
        queue.put((torch.cat(imgs,0), torch.cat(rots,0)))

# ───────────────────── Dataset Class (MODIFIED) ─────────────────── #
class RenderedSO3Dataset(Dataset):
    # ---------- Initialization & Dataset Generation (MODIFIED) --------------
    def __init__(self, args, *, seed:int=0):
        g = torch.Generator().manual_seed(seed)

        self.manifold_dim = getattr(args, "manifold_dim", 3)
        self.submanifold  = getattr(args, "submanifold", "s2_zeroroll" if self.manifold_dim == 2 else "so3")
        self.mesh_path   = args.mesh_path
        self.channels    = args.channels
        self.H = self.W  = int(args.image_size)
        self.device      = args.device.lower()
        self.azim_step   = args.azim_step
        self.elev_step   = args.elev_step
        self.roll_step   = args.roll_step
        self.N_random    = args.data_samples
        self.cache_path  = args.dataset_path
        self.overwrite   = args.overwrite_cache
        self.ambient_dim = args.ambient_dim
        self.n_workers   = args.n_workers or self._default_workers()
        self._aa = _aa(self.H)
        self.render_batch = getattr(args, "render_batch", 1)

        # Only support submanifolds with analytic geodesics (S^2 zero-roll)
        if self.manifold_dim == 2 and self.submanifold.lower() != "s2_zeroroll":
            raise ValueError(
                f"Submanifold {self.submanifold!r} not supported with ground-truth geodesics. "
                "Use 's2_zeroroll'."
            )

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

    @staticmethod
    def _visible_cuda() -> List[str]:
        env = os.getenv("CUDA_VISIBLE_DEVICES")
        return [f"cuda:{i}" for i in range(len(env.split(",")))] if env and env.strip() else [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    def _default_workers(self) -> int:
        return max(1, len(self._visible_cuda())) if self.device.startswith("cuda") else mp.cpu_count()

    def _materialise_images(self):
        self.images = None
        if self.P.shape[0] == self.P.shape[1] and torch.allclose(self.P, torch.eye(self.P.shape[0])):
            self.images = self.data.view(-1, self.channels, self.H, self.W).contiguous()

    def _generate_dataset(self, g):
        grid_mode = (self.azim_step is not None and self.elev_step is not None and self.roll_step is not None)

        if grid_mode:
            # If user chooses a 2D manifold, require roll_step == 0 to match S^2 (zero roll).
            if self.manifold_dim == 2 and (self.roll_step is not None) and (float(self.roll_step) != 0.0):
                raise ValueError("For manifold_dim=2 (S^2 zero-roll), set roll_step=0 in grid mode.")
            az = torch.arange(-180., 180., self.azim_step)
            el = torch.arange(-90.,  90. + 1e-9, self.elev_step)
            rl = torch.tensor([0.]) if self.roll_step == 0 else torch.arange(-180., 180., self.roll_step)
            tasks = [(a.item(), e.item(), r.item()) for r in rl for e in el for a in az]
            print(f"[RenderedSO3] Euler grid → {len(tasks)} rotations (roll_step={self.roll_step})")
            worker_grid_mode = True
        else:
            N = int(self.N_random or 100_000)
            if self.manifold_dim == 3:
                G = torch.randn(N, 3, 3, generator=g); Q, _ = torch.linalg.qr(G)
                det = torch.linalg.det(Q); Q[det < 0, :, 0] *= -1
                tasks = [Q[i] for i in range(N)]
                print(f"[RenderedSO3] Haar-uniform sample N = {N}")
                worker_grid_mode = False
            elif self.manifold_dim == 2:
                mode = str(self.submanifold).lower()
                if mode == "s2_zeroroll":
                    # Uniform S^2: az ~ U(-180,180), u ~ U(-1,1), elev = asin(u), roll=0
                    az = torch.rand(N, generator=g) * 360.0 - 180.0   # deg
                    u  = torch.rand(N, generator=g) * 2.0   - 1.0
                    el = torch.asin(u) * (180.0 / math.pi)           # deg

                    # Vectorized camera rotations on CPU (single call, no Python loop)
                    R_cam, _ = look_at_view_transform(CAM_DIST, el, az, degrees=True, device="cpu")  # (N,3,3)
                    # zero roll ⇒ nothing else to apply

                    R_obj_all = R_cam.transpose(1, 2).contiguous()   # (N,3,3)  object rotation to render
                    tasks = [R_obj_all[i] for i in range(N)]         # match SO(3) path: list of 3×3 tensors
                    print(f"[RenderedSO3] 2-D submanifold=S2 (zero roll), N={N} (precomputed rotations)")
                    worker_grid_mode = False                          # use the fast path in _worker_render
                else:
                    raise ValueError(f"Unknown 2-D submanifold: {self.submanifold!r}")
            else:
                raise ValueError(f"manifold_dim must be 2 or 3, got {self.manifold_dim}")

        if self.n_workers <= 1:
            self._gen_serial(tasks, worker_grid_mode)
        else:
            self._gen_mp(tasks, worker_grid_mode)

        flat = self.data.view(self.data.size(0), -1)
        d_img = flat.size(1); d_emb = self.ambient_dim or d_img
        if d_emb > d_img:
            A, _ = torch.linalg.qr(torch.randn(d_emb, d_img, generator=g))
            self.P = A.float(); self.data = (A @ flat.T).T.float()
        else:
            self.P = torch.eye(d_img).float(); self.data = flat.float()

        with suppress(Exception):
            preview = (os.path.splitext(self.cache_path)[0] if self.cache_path else "rendered") + "_preview.png"
            pathlib.Path(preview).parent.mkdir(parents=True, exist_ok=True)
            save_image(self.data.view(-1, self.channels, self.H, self.W)[:64], preview, nrow=8, normalize=True)
            print("[RenderedSO3] preview saved →", preview)

    def _gen_mp(self, tasks, grid_mode):
        devs = self._visible_cuda() if self.device == "cuda" or self.device.startswith("cuda:") else [self.device]
        n_proc = min(self.n_workers, len(devs))
        chunk = math.ceil(len(tasks) / n_proc)
        chunks = [tasks[i:i+chunk] for i in range(0, len(tasks), chunk)]
        ctx = mp.get_context("spawn"); queue = ctx.Queue(maxsize=2*n_proc); procs = []
        for rk, ch in enumerate(chunks):
            p = ctx.Process(target=_worker_render, kwargs=dict(
                rank=rk, device=devs[rk % len(devs)], chunk=ch, mesh_path=self.mesh_path,
                aa=self._aa, channels=self.channels, grid_mode=grid_mode, seed=17+rk,
                queue=queue, flush=1024, render_batch=self.render_batch))
            p.start(); procs.append(p)
        imgs, rots, rec = [], [], 0
        with tqdm(total=len(tasks), desc="Rendering SO(3) (mp)") as pbar:
            while rec < len(tasks):
                i, r = queue.get()
                imgs.append(i); rots.append(r); rec += i.size(0); pbar.update(i.size(0))
        for p in procs: p.join()
        self.data = torch.cat(imgs).float() / 255.; self.rotmats = torch.cat(rots).float()

    def _gen_serial(self, tasks, grid_mode):
        # This serial generation can be kept as a fallback, reusing the _worker_render logic.
        ctx = mp.get_context("spawn"); queue = ctx.Queue()
        _worker_render(0, self.device, tasks, self.mesh_path, self._aa, self.channels,
                       grid_mode, 42, queue, flush=len(tasks), render_batch=self.render_batch)
        imgs, rots = queue.get()
        self.data = imgs.float() / 255.; self.rotmats = rots.float()

    # ---------- Dataset Interface (Unchanged) ---------------------------------
    def __len__(self): return self.data.size(0)
    def __getitem__(self, idx):
        if self.images is not None: return self.images[idx], self.rotmats[idx]
        return self.data[idx].view(self.channels, self.H, self.W), self.rotmats[idx]

    # ───────────────── Fast Batched Geodesic Renderer (Unchanged) ───────────────────
    def _get_geo_ctx(self, dev):
        """
        Lazily create and cache (mesh, renderer) on the requested device.
        Avoids invalid Meshes.to(dtype=...) usage; keeps everything float32.
        """
        if not hasattr(self, "_geo_ctx") or self._geo_ctx["mesh"].device != dev:
            mesh = load_objs_as_meshes([self.mesh_path], device=dev)
            if mesh.textures is None:
                mesh.textures = TexturesVertex(
                    verts_features=torch.ones_like(mesh.verts_packed(), dtype=torch.float32)[None] * 0.7
                )
            mesh = _normalise_mesh(mesh)

            renderer = _build_renderer(self._aa['render'], self._aa['blur'], self._aa['faces'], dev)
            renderer.shader.materials = Materials(
                device=dev, specular_color=((0.9, 0.9, 0.9),), shininess=100.0
            )

            self._geo_ctx = dict(mesh=mesh, renderer=renderer)
        return self._geo_ctx["mesh"], self._geo_ctx["renderer"]

    # Build camera-to-world rotation with ZERO roll from direction
    @staticmethod
    def _camera_view_from_direction(n: torch.Tensor) -> torch.Tensor:
        """
        n: (B*[T], 3) unit view direction in WORLD coords (camera looks along -n).
        Returns R_view: (B*[T], 3, 3) camera-to-world rotation with ZERO ROLL wrt world up.
        """
        B = n.size(0)
        f = F.normalize(-n, dim=-1)                     # camera forward axis in world
        up0 = n.new_tensor([0.0, 1.0, 0.0]).expand(B, 3)
        r = torch.cross(up0, f, dim=-1)
        bad = (r.norm(dim=-1, keepdim=True) < 1e-8)
        up1 = n.new_tensor([0.0, 0.0, 1.0]).expand(B, 3)
        up = torch.where(bad, up1, up0)
        r = F.normalize(torch.cross(up, f, dim=-1), dim=-1)
        u = F.normalize(torch.cross(f, r, dim=-1), dim=-1)
        R_view = torch.stack([r, u, f], dim=-1)        # columns are camera axes in world coords
        return R_view

    # S^2 great-circle (slerp) between unit vectors
    @staticmethod
    def _s2_geodesic_directions(n0: torch.Tensor, n1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Shortest-arc S^2 geodesic using axis-angle with atan2 for numerical robustness.
        n0, n1: (B,3), unit vectors (we normalize anyway)
        t: (T,) in [0,1]
        returns n(t): (B,T,3)
        """
        B, T = n0.size(0), t.numel()
        n0 = F.normalize(n0, dim=-1)
        n1 = F.normalize(n1, dim=-1)

        # Rotation axis and angle for the shortest arc
        k_raw = torch.cross(n0, n1, dim=-1)                         # (B,3)
        k_norm = k_raw.norm(dim=-1, keepdim=True)                   # (B,1)
        dot = (n0 * n1).sum(-1)                                     # (B,)
        # θ in [0, π], robust near 0/π
        theta = torch.atan2(k_norm.squeeze(-1), dot.clamp(-1.0, 1.0))  # (B,)

        # Handle degenerate cases
        eps = 1e-8
        # Near identical: lerp+normalize
        near0 = theta < 1e-6
        # Near antipodal: axis undefined -> pick any axis ⟂ n0, set θ=π
        nearpi = (k_norm.squeeze(-1) < 1e-8) & (dot < 0)

        # Choose an axis for generic case
        k = torch.where(k_norm > eps, k_raw / (k_norm + 1e-12), k_raw)  # placeholder; will overwrite in branches

        if nearpi.any():
            idx = nearpi.nonzero(as_tuple=False).squeeze(-1)
            # pick axis ⟂ n0
            a = torch.tensor([1.0, 0.0, 0.0], device=n0.device).expand(idx.numel(), 3)
            alt = torch.tensor([0.0, 1.0, 0.0], device=n0.device).expand(idx.numel(), 3)
            v = torch.cross(n0[idx], a, dim=-1)
            bad = (v.norm(dim=-1, keepdim=True) < 1e-6)
            v = torch.where(bad, torch.cross(n0[idx], alt, dim=-1), v)
            k[idx] = F.normalize(torch.cross(n0[idx], v, dim=-1), dim=-1)
            theta[idx] = math.pi

        # Batch Rodrigues for directions: n(t) = R(k, θ t) n0
        # Expand to (B,T,*) shapes
        kBT = k.unsqueeze(1).expand(B, T, 3)
        n0BT = n0.unsqueeze(1).expand(B, T, 3)
        ang = (theta.unsqueeze(1) * t.view(1, T)).unsqueeze(-1)     # (B,T,1)
        ca = torch.cos(ang)                                         # (B,T,1)
        sa = torch.sin(ang)                                         # (B,T,1)
        k_dot_n0 = (kBT * n0BT).sum(-1, keepdim=True)               # (B,T,1)

        n = n0BT * ca + torch.cross(kBT, n0BT, dim=-1) * sa + kBT * k_dot_n0 * (1.0 - ca)

        if near0.any():
            idx = near0.nonzero(as_tuple=False).squeeze(-1)
            tt = t.view(1, T, 1).expand(idx.numel(), T, 1)
            n_lin = (1 - tt) * n0[idx].unsqueeze(1) + tt * n1[idx].unsqueeze(1)
            n[idx] = n_lin

        return F.normalize(n, dim=-1)

    # SO(3) geodesic
    def _compute_geodesic_so3(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Safe CUDA path: render frames in a micro-batched loop using a single-mesh clone
        per frame (mirrors the worker path). Returns (B, T, C, H, W) in sRGB.
        """
        dev = P.device
        B, T = P.size(0), t.numel()
        N = B * T

        # Geodesic rotations: R(t) = P * exp(t * log(P^T Q))
        A = _so3_log_batch(P.transpose(-2, -1) @ Q)  # (B,3,3)
        Rts = P.unsqueeze(1) @ _so3_exp_batch(A.unsqueeze(1) * t.view(1, T, 1, 1))  # (B,T,3,3)
        Rts_flat = Rts.reshape(N, 3, 3).contiguous()

        mesh0, renderer = self._get_geo_ctx(dev)

        hi_frames: List[torch.Tensor] = []
        for k in range(N):
            R_obj = Rts_flat[k]  # (3,3)

            # Clone single base mesh and update its verts (safe path)
            mesh_inst = mesh0.clone().update_padded(
                Rotate(R_obj[None], device=dev).transform_points(mesh0.verts_padded())
            )

            # Lights expect camera-to-world rotations; R_view = R_obj^T (shape (1,3,3))
            renderer.shader.lights = _build_lights_batch(R_obj.transpose(0, 1).unsqueeze(0).contiguous(), dev)

            # Render a single frame; collect (3, H_hi, W_hi)
            hi = renderer(mesh_inst)[0, :, :, :3].permute(2, 0, 1)
            hi_frames.append(hi)

        # Stack to (N, 3, H_hi, W_hi) and downsample
        hi_stack = torch.stack(hi_frames, dim=0)
        lin = _downsample(hi_stack, self._aa)  # (N, C, H, W)
        if self.channels == 1:
            lin = lin.mean(1, keepdim=True)
        return _to_srgb(lin).view(B, T, self.channels, self.H, self.W)


    # S^2 (zero roll) submanifold geodesic
    def _compute_geodesic_s2_zeroroll(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        FINAL VERSION: Fast, memory-efficient geodesic rendering that downsamples
        inside the micro-batch loop to prevent internal PyTorch errors with giant tensors.
        """
        dev = P.device
        B, T = P.size(0), t.numel()
        N = B * T
        mesh0, renderer = self._get_geo_ctx(dev)

        # --- 1. Calculate all geodesic rotations (memory-light) ---
        Rv0 = P.transpose(-2, -1)
        Rv1 = Q.transpose(-2, -1)
        z_cam = torch.tensor([0.0, 0.0, -1.0], device=dev)
        n0 = F.normalize((Rv0 @ z_cam.unsqueeze(-1)).squeeze(-1), dim=-1)
        n1 = F.normalize((Rv1 @ z_cam.unsqueeze(-1)).squeeze(-1), dim=-1)
        n_t = self._s2_geodesic_directions(n0, n1, t)
        n_t_flat = n_t.reshape(N, 3)
        R_view_flat = self._camera_view_from_direction(n_t_flat)
        R_obj_flat = R_view_flat.transpose(1, 2).contiguous()

        # --- 2. Render and Downsample in micro-batches ---
        micro_batch_size = self.render_batch
        V0, F0, C0 = mesh0.verts_padded(), mesh0.faces_padded(), mesh0.textures.verts_features_padded()
        
        # We will collect the final, low-resolution frames here
        all_low_res_frames = []
        
        for i in tqdm(range(0, N, micro_batch_size), desc=f"Rendering in batches of {micro_batch_size}"):
            start, end = i, min(i + micro_batch_size, N)
            chunk_size = end - start
            
            # Select the chunk of rotations
            R_obj_chunk, R_view_chunk = R_obj_flat[start:end], R_view_flat[start:end]
            
            # Build the mesh batch
            verts_rotated_chunk = Rotate(R_obj_chunk, device=dev).transform_points(V0.expand(chunk_size, -1, -1))
            mesh_chunk = Meshes(
                verts=[v for v in verts_rotated_chunk],
                faces=[F0[0]] * chunk_size,
                textures=TexturesVertex(verts_features=[C0[0]] * chunk_size)
            )
            
            # Render the current micro-batch (high-res)
            renderer.shader.lights = _build_lights_batch(R_view_chunk, dev)
            hi_chunk = renderer(mesh_chunk)[..., :3].permute(0, 3, 1, 2)
            
            # THE FIX: Immediately downsample the high-res chunk
            lin_chunk = _downsample(hi_chunk, self._aa)
            
            # Append the small, downsampled chunk to our list
            all_low_res_frames.append(lin_chunk)

        # --- 3. Combine the already-downsampled results ---
        # This tensor is now much smaller and will not crash
        lin_stack = torch.cat(all_low_res_frames, dim=0)

        if self.channels == 1:
            lin_stack = lin_stack.mean(1, keepdim=True)
            
        return _to_srgb(lin_stack).view(B, T, self.channels, self.H, self.W)


    # Dispatch to correct geodesic depending on manifold/submanifold
    def compute_geodesic(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.manifold_dim == 3:
            return self._compute_geodesic_so3(P, Q, t)
        if self.manifold_dim == 2 and self.submanifold.lower() == "s2_zeroroll":
            return self._compute_geodesic_s2_zeroroll(P, Q, t)
        raise NotImplementedError("Geodesics for the selected submanifold are not implemented.")

# ───────────────── Debug & CLI (MODIFIED) ──────────────────────
def _geo_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> float:
    cos = ((Ra.T @ Rb).trace() - 1).clamp(-1.0, 1.0) / 2
    return torch.acos(cos).item()

def _debug_geodesics(ds: "RenderedSO3Dataset", pairs=5, frames=10, out_root="datasets/geo_debug"):
    os.makedirs(out_root, exist_ok=True)
    chosen: List[Tuple[int,int]] = []
    cand  = torch.randperm(len(ds))[:min(500, len(ds))]

    for _ in range(pairs):
        if not chosen:
            i = cand[0].item()
        else:
            used = torch.tensor([u for ij in chosen for u in ij])
            remain = cand[~cand.unsqueeze(1).eq(used).any(1)]
            scores = [(min(_geo_angle(ds.rotmats[k], ds.rotmats[u]) for u in used), k.item()) for k in remain]
            i = max(scores)[1] if scores else remain[0].item()
        j = max(((_geo_angle(ds.rotmats[i], ds.rotmats[j]), j.item()) for j in cand if j != i), key=lambda x: x[0])[1]
        chosen.append((i, j))

    t = torch.linspace(0, 1, frames)

    # Render device & tensor placement
    dev = torch.device(ds.device if ("cuda" in ds.device and torch.cuda.is_available()) else "cpu")
    p_rots = ds.rotmats[torch.tensor([c[0] for c in chosen])].to(dev).contiguous()
    q_rots = ds.rotmats[torch.tensor([c[1] for c in chosen])].to(dev).contiguous()
    t = t.to(dev)

    # Timing (sync for accurate GPU time)
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
    P = argparse.ArgumentParser("Build rendered SO(3) dataset")
    P.add_argument("--mesh_path", required=True); P.add_argument("--dataset_path", required=True)
    P.add_argument("--image_size", type=int, choices=[64,128], default=128)
    P.add_argument("--channels", type=int, choices=[1,3], default=1)
    P.add_argument("--azim_step", type=float); P.add_argument("--elev_step", type=float); P.add_argument("--roll_step", type=float)
    P.add_argument("--data_samples", type=int); P.add_argument("--ambient_dim", type=int)
    P.add_argument("--manifold_dim", type=int, default=3)
    P.add_argument("--submanifold", type=str, default="s2_zeroroll")
    P.add_argument("--overwrite_cache", action="store_true"); P.add_argument("--n_workers", type=int, default=0)
    P.add_argument("--device", default="cuda")
    P.add_argument("--render_batch", type=int, default=64)
    args = P.parse_args()

    ds = RenderedSO3Dataset(args)
    print(f"Dataset built: {len(ds)} samples → {args.dataset_path}")
    _debug_geodesics(ds, pairs=10, frames=20)