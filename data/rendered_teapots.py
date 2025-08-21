#!/usr/bin/env python3
# datasets/rendered_so_dataset.py
# Anti-aliased teapot renders with unified renderer and fast geodesics.
# Parameters match your command: 64x64, grayscale, directional lights,
# ambient=0.03, key=1.0, fill=0.5, back=0.5, rim=0.9, specular=0.9, shininess=50,
# NO backface culling, NO outline.

from __future__ import annotations
import argparse, math, multiprocessing as mp, os, pathlib
from contextlib import suppress
from typing import Sequence, List, Tuple

import torch
import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRasterizer, MeshRenderer, RasterizationSettings,
    BlendParams, DirectionalLights, PointLights, TexturesVertex, Materials,
    look_at_view_transform,
)
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.transforms import Rotate


# ───────────────────────────── Multi-light shader ───────────────────────────── #

class MultiLightSoftShader(SoftPhongShader):
    """Supports PointLights or DirectionalLights with (B,3) or (B,L,3)."""
    def forward(self, fragments, meshes, **kw):
        cameras   = super()._get_cameras(**kw)
        lights    = kw.get("lights",    self.lights)
        materials = kw.get("materials", self.materials)
        blend     = kw.get("blend_params", self.blend_params)
        texels    = meshes.sample_textures(fragments)

        def shade(single_lights):
            return phong_shading(meshes=meshes, fragments=fragments, texels=texels,
                                 lights=single_lights, cameras=cameras, materials=materials)

        # Multi point
        if hasattr(lights, "location") and isinstance(lights.location, torch.Tensor) and lights.location.ndim == 3:
            _, L, _ = lights.location.shape
            acc = 0.0
            for li in range(L):
                single = PointLights(
                    device=lights.device,
                    location=lights.location[:, li],
                    ambient_color=lights.ambient_color[:, li],
                    diffuse_color=lights.diffuse_color[:, li],
                    specular_color=lights.specular_color[:, li],
                )
                acc = acc + shade(single)
            colours = acc / L

        # Multi directional
        elif hasattr(lights, "direction") and isinstance(lights.direction, torch.Tensor) and lights.direction.ndim == 3:
            _, L, _ = lights.direction.shape
            acc = 0.0
            for li in range(L):
                single = DirectionalLights(
                    device=lights.device,
                    direction=lights.direction[:, li],
                    ambient_color=lights.ambient_color[:, li],
                    diffuse_color=lights.diffuse_color[:, li],
                    specular_color=lights.specular_color[:, li],
                )
                acc = acc + shade(single)
            colours = acc / L

        else:
            colours = shade(lights)

        znear = kw.get("znear", getattr(cameras, "znear", 1.0))
        zfar  = kw.get("zfar",  getattr(cameras, "zfar", 100.0))
        return softmax_rgb_blend(colours, fragments, blend, znear=znear, zfar=zfar)


# ───────────────────────────── constants & helpers ─────────────────────────── #

FOV      = 25.0
CAM_DIST = 1.05 / math.sin(math.radians(FOV / 2))

def _aa(final_H: int):
    # unified AA scheme: render high-res → area to 128 → bicubic to final
    if final_H == 128: return dict(render=256, blur=1e-4, faces=12, down_mid=128, down_final=None)
    if final_H == 64:  return dict(render=512, blur=1e-6, faces=30, down_mid=128, down_final=64)
    raise ValueError("image_size must be 64 or 128")

def _to_srgb(x: torch.Tensor): return x.clamp(0, 1) ** (1/2.2)

def _normalise_mesh(mesh):
    v = mesh.verts_packed()
    mesh.offset_verts_(-v.mean(0))
    mesh.scale_verts_((1.0 / v.norm(dim=1).max()).item())
    return mesh

def _base_camera(dev):
    return FoVPerspectiveCameras(device=dev, R=torch.eye(3, device=dev)[None],
                                 T=torch.tensor([[0., 0., CAM_DIST]], device=dev), fov=FOV)

def _build_renderer(res: int, blur: float, faces: int, dev):
    # cull_backfaces=False (i.e., --no-cull)
    raster = RasterizationSettings(image_size=(res, res), blur_radius=blur,
                                   faces_per_pixel=faces, cull_backfaces=False, bin_size=None)
    cam = _base_camera(dev)
    blend = BlendParams(background_color=(1., 1., 1.))
    shader = MultiLightSoftShader(device=dev, cameras=cam, blend_params=blend, lights=None)
    return MeshRenderer(MeshRasterizer(cam, raster), shader)

def _downsample(lin: torch.Tensor, aa):
    # accepts (B,C,H,W) or (C,H,W)
    is_3d = (lin.dim() == 3)
    if is_3d: lin = lin.unsqueeze(0)
    mid = torch.nn.functional.interpolate(lin, size=aa['down_mid'], mode="area")
    if aa['down_final']:
        mid = torch.nn.functional.interpolate(mid, size=aa['down_final'], mode="bicubic", align_corners=False)
    return mid.squeeze(0) if is_3d else mid


# ───────────────────── batched SO(3) & lighting utilities ──────────────────── #

def _so3_log_batch(R: torch.Tensor) -> torch.Tensor:
    tr  = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    th  = torch.acos(cos)
    th2 = th * th
    coef = torch.where(th.abs() < 1e-5, 0.5 - th2/12 + th2*th2/720, th / (2 * torch.sin(th)))
    return coef.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2, -1))

def _so3_exp_batch(A: torch.Tensor) -> torch.Tensor:
    w  = torch.stack((A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]), -1)
    th = torch.linalg.vector_norm(w, dim=-1)
    th2= th * th
    s  = torch.where(th.abs() < 1e-5, 1 - th2/6 + th2*th2/120, torch.sin(th)/th)
    c  = torch.where(th.abs() < 1e-5, 0.5 - th2/12 + th2*th2/720, (1 - torch.cos(th)) / (th2 + 1e-12))
    s, c = s.unsqueeze(-1).unsqueeze(-1), c.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, device=A.device).expand_as(A)
    return I + s * A + c * (A @ A)

def _build_dir_lights_batch(
    R_view: torch.Tensor, dev: torch.device,
    ambient=0.03, key=1.0, fill=0.5, back=0.5, rim=0.9
) -> DirectionalLights:
    """
    Directional 3-point (+rim) rig in camera space, rotated to world by R_view (B,3,3).
    +Z points toward the camera in camera space.
    """
    dirs_cam = torch.tensor(
        [[ 0.35,  0.35,  1.0],   # key
         [-0.35,  0.20,  1.0],   # fill
         [ 0.00,  0.15, -1.0],   # back
         [ 0.00,  0.00, -1.0]],  # rim
        device=dev, dtype=torch.float32
    )
    B, L = R_view.shape[0], dirs_cam.shape[0]
    dirs_w = (R_view @ dirs_cam.t()).transpose(1, 2)        # (B,L,3)
    dirs_w = dirs_w / dirs_w.norm(dim=-1, keepdim=True)

    amb  = dirs_w.new_full((B, L, 3), ambient)
    diff = torch.stack([
        torch.full((B, 3), key,  device=dev),
        torch.full((B, 3), fill, device=dev),
        torch.full((B, 3), back, device=dev),
        torch.full((B, 3), rim,  device=dev),
    ], dim=1)
    spec = diff * 0.6
    return DirectionalLights(
        device=dev, direction=-dirs_w,   # toward origin
        ambient_color=amb, diffuse_color=diff, specular_color=spec
    )


# ───────────────────── MP worker (unified renderer) ────────────────────────── #

def _worker_render(rank:int, device:str, chunk: Sequence, mesh_path:str, aa, channels:int,
                   grid_mode: bool, seed:int, queue, flush:int=256):
    torch.manual_seed(seed)
    dev = torch.device(device)

    mesh = load_objs_as_meshes([mesh_path], device=dev)
    mesh = _normalise_mesh(mesh)
    if mesh.textures is None:
        mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed())[None] * 0.7)

    renderer = _build_renderer(aa['render'], aa['blur'], aa['faces'], dev)
    renderer.shader.materials = Materials(
        device=dev, specular_color=((0.9, 0.9, 0.9),), shininess=50.0
    )

    imgs, rots = [], []
    for item in chunk:
        if grid_mode:
            az, el, rl = item
            R_cam, _ = look_at_view_transform(CAM_DIST, el, az, degrees=True, device=dev)
            if rl:
                θ = math.radians(rl); c, s = math.cos(θ), math.sin(θ)
                Rz = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], device=dev)
                R_cam = Rz @ R_cam
            R_obj = R_cam.t()
        else:
            R_obj = item.to(dev)

        # Lights for this view (directional + rim), NO culling
        renderer.shader.lights = _build_dir_lights_batch(R_obj.t().unsqueeze(0), dev)

        mesh_inst = mesh.clone().update_padded(
            Rotate(R_obj[None], device=dev).transform_points(mesh.verts_padded())
        )
        hi  = renderer(mesh_inst)[0, ..., :3].permute(2, 0, 1)   # (3,H,W) linear
        lin = _downsample(hi, aa)
        if channels == 1:
            lin = lin.mean(0, keepdim=True)

        imgs.append((_to_srgb(lin).clamp(0,1) * 255).byte().cpu())
        rots.append(R_obj.cpu())

        if len(imgs) >= flush:
            queue.put((torch.stack(imgs), torch.stack(rots)))
            imgs, rots = [], []

    if imgs:
        queue.put((torch.stack(imgs), torch.stack(rots)))


# ───────────────────────────── main Dataset class ──────────────────────────── #

class RenderedTeapots(Dataset):
    """
    Cached SO(3) teapot renders.
    __getitem__ returns (image[C,H,W], R[3,3]).
    Geodesics can be rendered via compute_geodesic(P, Q, t) → (B,T,C,H,W).
    """
    def __init__(self, args, *, seed:int=0):
        g = torch.Generator().manual_seed(seed)

        self.mesh_path   = args.mesh_path
        self.channels    = int(args.channels)
        self.H = self.W  = int(args.image_size)
        self.device      = args.device
        self.azim_step   = args.azim_step
        self.elev_step   = args.elev_step
        self.roll_step   = args.roll_step
        self.N_random    = args.data_samples
        self.cache_path  = args.dataset_path
        self.overwrite   = args.overwrite_cache
        self.ambient_dim = args.ambient_dim
        self.n_workers   = args.n_workers or self._default_workers()
        self._aa         = _aa(self.H)

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
        if env and env.strip():
            return [f"cuda:{i}" for i in range(len(env.split(",")))]
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]

    def _default_workers(self) -> int:
        return max(1, len(self._visible_cuda())) if str(self.device).startswith("cuda") else max(1, mp.cpu_count() // 2)

    def _materialise_images(self):
        self.images = None
        if self.ambient_dim == self.channels * self.H * self.W:
            self.images = self.data.view(-1, self.channels, self.H, self.W).contiguous()

    def _generate_dataset(self, g: torch.Generator):
        grid_mode = (self.azim_step is not None and self.elev_step is not None and self.roll_step is not None)
        if grid_mode:
            az = torch.arange(-180., 180., self.azim_step)
            el = torch.arange(-90.,  90.+1e-9, self.elev_step)
            rl = torch.tensor([0.]) if (self.roll_step == 0 or self.roll_step is None) \
                 else torch.arange(-180., 180., self.roll_step)
            tasks = [(a.item(), e.item(), r.item()) for r in rl for e in el for a in az]
            print(f"[RenderedSO3] Euler grid → {len(tasks)} rotations")
        else:
            N = int(self.N_random or 200_000)
            G = torch.randn(N, 3, 3, generator=g)
            Q, _ = torch.linalg.qr(G)
            det = torch.linalg.det(Q)
            Q[det < 0, :, 0] *= -1
            tasks = [Q[i] for i in range(N)]
            print(f"[RenderedSO3] Haar-uniform sample N = {N}")

        if self._default_workers() <= 1:
            self._gen_serial(tasks, grid_mode)
        else:
            self._gen_mp(tasks, grid_mode)

        # flatten to ambient_dim (identity projection by default)
        flat = self.data.view(self.data.size(0), -1)
        d_img = flat.size(1)
        d_emb = self.ambient_dim or d_img
        if d_emb == d_img:
            self.P = torch.eye(d_img, dtype=torch.float32)
            self.data = flat.float()
        else:
            A, _ = torch.linalg.qr(torch.randn(d_emb, d_img, generator=g))
            self.P = A.float()
            self.data = (A @ flat.T).T.float()

        # quick preview
        with suppress(Exception):
            preview = (os.path.splitext(self.cache_path)[0] if self.cache_path else "rendered") + "_preview.png"
            pathlib.Path(preview).parent.mkdir(parents=True, exist_ok=True)
            save_image(self.data.view(-1, self.channels, self.H, self.W)[:25], preview, nrow=5, normalize=True)
            print("[RenderedSO3] preview saved →", preview)

    def _gen_mp(self, tasks, grid_mode):
        devs = self._visible_cuda() if str(self.device).startswith("cuda") else [self.device]
        n_proc = min(self._default_workers(), len(devs))
        chunk = math.ceil(len(tasks) / n_proc)
        chunks = [tasks[i:i + chunk] for i in range(0, len(tasks), chunk)]
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(maxsize=2 * n_proc)
        procs = []
        for rk, ch in enumerate(chunks):
            p = ctx.Process(target=_worker_render, kwargs=dict(
                rank=rk, device=devs[rk % len(devs)], chunk=ch,
                mesh_path=self.mesh_path, aa=self._aa, channels=self.channels,
                grid_mode=grid_mode, seed=17 + rk, queue=queue))
            p.start(); procs.append(p)

        imgs, rots, rec = [], [], 0
        with tqdm(total=len(tasks), desc="Rendering SO(3) (mp)") as pbar:
            while rec < len(tasks):
                i, r = queue.get()
                imgs.append(i); rots.append(r); rec += i.size(0); pbar.update(i.size(0))
        for p in procs: p.join()

        self.data = torch.cat(imgs).float() / 255.0
        self.rotmats = torch.cat(rots).float()

    def _gen_serial(self, tasks, grid_mode):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        _worker_render(0, self.device, tasks, self.mesh_path, self._aa, self.channels,
                       grid_mode, 42, queue, flush=len(tasks))
        imgs, rots = queue.get()
        self.data = imgs.float() / 255.0
        self.rotmats = rots.float()

    # ---------- standard dataset API ----------
    def __len__(self): return self.data.size(0)
    def __getitem__(self, idx):
        if self.images is not None:
            return self.images[idx], self.rotmats[idx]
        return self.data[idx].view(self.channels, self.H, self.W), self.rotmats[idx]

    # ───────────────── fast batched geodesic renderer (unified) ─────────────── #
    def _get_geo_ctx(self, dev):
        if not hasattr(self, "_geo_ctx") or self._geo_ctx["mesh"].device != dev:
            mesh = load_objs_as_meshes([self.mesh_path], device=dev)
            if mesh.textures is None:
                mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed())[None] * 0.7)
            mesh = _normalise_mesh(mesh)
            renderer = _build_renderer(self._aa['render'], self._aa['blur'], self._aa['faces'], dev)
            renderer.shader.materials = Materials(device=dev, specular_color=((0.9, 0.9, 0.9),), shininess=50.0)
            self._geo_ctx = dict(mesh=mesh, renderer=renderer)
        return self._geo_ctx["mesh"], self._geo_ctx["renderer"]

    def compute_geodesic(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        P,Q: (B,3,3) start/end rotations; t: (T,) in [0,1]
        Returns: (B, T, C, H, W) sRGB in [0,1]
        """
        dev, B, T = P.device, P.size(0), t.numel()
        A    = _so3_log_batch(P.transpose(-2, -1) @ Q)
        Rts  = P.unsqueeze(1) @ _so3_exp_batch(A.unsqueeze(1) * t.view(1, T, 1, 1))  # (B,T,3,3)
        R_bt = Rts.reshape(B * T, 3, 3)

        mesh0, renderer = self._get_geo_ctx(dev)
        mesh_bt = mesh0.extend(B * T)
        verts   = Rotate(R_bt, device=dev).transform_points(mesh_bt.verts_padded())
        mesh_bt = mesh_bt.update_padded(verts)

        # same lighting as dataset renders
        renderer.shader.lights = _build_dir_lights_batch(R_bt.transpose(1, 2), dev)

        hi  = renderer(mesh_bt)[..., :3].permute(0, 3, 1, 2)  # (B*T,3,H',W')
        lin = _downsample(hi, self._aa)
        if self.channels == 1:
            lin = lin.mean(1, keepdim=True)
        out = _to_srgb(lin).clamp(0, 1).view(B, T, self.channels, self.H, self.W)
        return out


# ──────────────────────────────── CLI helper ───────────────────────────────── #

def _geo_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> float:
    cos = ((Ra.T @ Rb).trace() - 1).clamp(-1.0, 1.0) / 2
    return torch.acos(cos).item()

def _debug_geodesics(ds: "RenderedTeapots", pairs=5, frames=11, out_root="datasets/geo_debug"):
    os.makedirs(out_root, exist_ok=True)
    idx = torch.randperm(len(ds))[:2 * pairs]
    P = ds.rotmats[idx[:pairs]]
    Q = ds.rotmats[idx[pairs:2 * pairs]]
    t = torch.linspace(0, 1, frames)
    sheet = ds.compute_geodesic(P, Q, t).flatten(0, 1)
    save_image(sheet, os.path.join(out_root, f"geodesics_{ds.H}.png"),
               nrow=frames, normalize=True)
    print("Saved contact sheet →", f"{out_root}/geodesics_{ds.H}.png")


if __name__ == "__main__":
    P = argparse.ArgumentParser("Build rendered SO(3) dataset")
    P.add_argument("--mesh_path",   required=True)
    P.add_argument("--dataset_path", required=True)
    P.add_argument("--image_size", type=int, choices=[64, 128], default=64)
    P.add_argument("--channels",   type=int, choices=[1, 3], default=1)
    P.add_argument("--azim_step",  type=float)
    P.add_argument("--elev_step",  type=float)
    P.add_argument("--roll_step",  type=float)
    P.add_argument("--data_samples", type=int)  # used if steps are None ⇒ Haar sample
    P.add_argument("--ambient_dim",  type=int)
    P.add_argument("--overwrite_cache", action="store_true")
    P.add_argument("--n_workers", type=int, default=0)
    P.add_argument("--device", default="cuda:0")
    args = P.parse_args()

    ds = RenderedTeapots(args)
    print(f"Dataset built: {len(ds)} samples → {args.dataset_path}")
    _debug_geodesics(ds, pairs=6, frames=11)
