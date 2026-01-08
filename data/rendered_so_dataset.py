#!/usr/bin/env python3
# datasets/rendered_so_dataset.py
# ======================================================================
# Teapot renders with batched geodesic renderer.
# Adds an S^2 submanifold in SO(3): R(n) = Exp(alpha * [n]_x)  (homeomorphic to S^2)
# Supports grayscale and 64x64 images.
# Proper progress bars during dataset generation (MP + serial) and geodesics.
# ======================================================================

from __future__ import annotations
import argparse, math, multiprocessing as mp, os, pathlib, time
from contextlib import suppress
from typing import Sequence, List, Tuple
from queue import Empty as QueueEmpty  # robust queue timeout handling

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRasterizer, MeshRenderer, RasterizationSettings,
    BlendParams, PointLights, TexturesVertex, Materials, look_at_view_transform,
)
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.transforms import Rotate

# ───────────────────────────── Shader ────────────────────────── #
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
        else:
            colours = phong_shading(meshes=meshes, fragments=fragments, texels=texels,
                                    lights=lights, cameras=cameras, materials=materials)

        znear = kw.get("znear", getattr(cameras, "znear", 1.0))
        zfar  = kw.get("zfar",  getattr(cameras, "zfar", 100.0))
        return softmax_rgb_blend(colours, fragments, blend, znear=znear, zfar=zfar)

# ───────────────────── Constants & Helpers ──────────────────── #
FOV      = 25.0
CAM_DIST = 1.05 / math.sin(math.radians(FOV / 2))

def _aa(final_H: int):
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
    return FoVPerspectiveCameras(
        device=dev,
        R=torch.eye(3, device=dev)[None],
        T=torch.tensor([[0., 0., CAM_DIST]], device=dev),
        fov=FOV
    )

def _build_renderer(res: int, blur: float, faces: int, dev):
    raster = RasterizationSettings(
        image_size=(res, res), blur_radius=blur, faces_per_pixel=faces,
        cull_backfaces=False, bin_size=None
    )
    cam = _base_camera(dev)
    blend = BlendParams(background_color=(1., 1., 1.))
    shader = MultiLightSoftShader(device=dev, cameras=cam, blend_params=blend, lights=None)
    return MeshRenderer(MeshRasterizer(cam, raster), shader)

def _downsample(lin: torch.Tensor, aa):
    is_3d = lin.dim() == 3
    if is_3d: lin = lin.unsqueeze(0)
    mid = torch.nn.functional.interpolate(lin, size=aa['down_mid'], mode="area")
    if aa['down_final']:
        mid = torch.nn.functional.interpolate(mid, size=aa['down_final'], mode="bicubic", align_corners=False)
    if is_3d: mid = mid.squeeze(0)
    return mid

# ───────────────────── SO(3) math utils ─────────────────────── #
def _skew(n: torch.Tensor) -> torch.Tensor:
    """n: (...,3) -> (...,3,3) skew-symmetric"""
    nx, ny, nz = n.unbind(-1)
    O = torch.zeros_like(nx)
    return torch.stack([
        torch.stack([ O,   -nz,  ny], -1),
        torch.stack([ nz,   O,  -nx], -1),
        torch.stack([-ny,   nx,   O], -1),
    ], -2)

def _exp_axis_angle(n: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' formula, batched.
    n: (B,3) unit axes, alpha: (B,) or scalar
    returns R: (B,3,3)
    """
    n = F.normalize(n, dim=-1)
    K = _skew(n)
    if alpha.ndim == 0: alpha = alpha.expand(n.size(0))
    a = alpha.view(-1, 1, 1)
    I = torch.eye(3, device=n.device).view(1,3,3).expand_as(K)
    s, c = torch.sin(a), torch.cos(a)
    return I + s*K + (1 - c) * (K @ K)

def _build_lights_batch(
    R_view: torch.Tensor,
    dev: torch.device,
    dist: float = 4.0,
    rim: bool = True,
    ambient: float = 0.18,
    diffuse: float = 0.85,
    specular: float = 0.35,
) -> PointLights:
    """
    Build a batched multi-light rig aligned per-frame to the camera view.

    Args:
        R_view: (B, 3, 3) camera-to-world rotation for each frame.
        dev:    torch.device.
        dist:   distance of the lights from the origin (scene scale ~1).
        rim:    if True, adds an extra back rim light along -Z_cam.
        ambient/diffuse/specular: per-light color intensities.

    Returns:
        PointLights with location/ambient/diffuse/specular shaped (B, L, 3).
    """
    # Key, fill, back rim, and a soft "headlight" (facing camera) to avoid dark frames.
    dirs_cam = torch.tensor(
        [
            [ 0.35,  0.35,  1.00],  # key
            [-0.35,  0.20,  1.00],  # fill
            [ 0.00,  0.00, -1.00],  # back (rim)
            [ 0.00,  0.00,  1.00],  # headlight
        ],
        device=dev,
        dtype=torch.float32,
    )
    if rim:
        # Extra strong rim from behind to pop the silhouette
        dirs_cam = torch.cat([dirs_cam, torch.tensor([[0.0, 0.0, -1.0]], device=dev)], dim=0)

    # Rotate light directions from camera space into world space for each frame
    # (R_view maps camera→world, so dirs_world = R_view @ dirs_cam^T).
    dirs_w = (R_view @ dirs_cam.t()).transpose(1, 2)             # (B, L, 3)
    dirs_w = dirs_w / (dirs_w.norm(dim=-1, keepdim=True) + 1e-9)  # normalize
    locs   = dirs_w * dist                                        # place on a sphere of radius 'dist'

    B, L, _ = locs.shape
    amb  = locs.new_full((B, L, 3), ambient)
    diff = locs.new_full((B, L, 3), diffuse)
    spec = locs.new_full((B, L, 3), specular)

    # Slightly dim the extra rim (last light) to keep it from blowing out edges
    if rim:
        diff[:, -1] *= 0.5
        spec[:, -1] *= 0.5

    return PointLights(
        device=dev,
        location=locs,
        ambient_color=amb,
        diffuse_color=diff,
        specular_color=spec,
    )


# ───────────────────── Worker renderer ───────────────────────── #
def _worker_render(rank:int, device:str, chunk, mesh_path:str, aa, channels:int,
                   queue, flush:int=1024, render_batch:int=32, show_pbar:bool=False):
    torch.manual_seed(17 + rank)
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

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    imgs, rots = [], []
    pack_size = int(render_batch)

    # Optional per-worker progress (disabled by default for MP noise)
    pbar = tqdm(total=len(chunk), desc=f"Worker {rank}", position=rank+1, leave=False,
                dynamic_ncols=True, disable=not show_pbar)

    for pack in batched(list(chunk), pack_size):
        Rs = torch.stack([x.to(dev) for x in pack], 0)  # (B,3,3)

        start = 0
        while start < len(pack):
            cur_B = min(pack_size, len(pack) - start)
            sub_Rs = Rs[start:start+cur_B]  # (cur_B,3,3)

            verts_list = []
            RV = Rotate(sub_Rs, device=dev).transform_points(V0.expand(cur_B, -1, -1))  # (cur_B,V,3)
            for i in range(cur_B):
                verts_list.append(RV[i])

            mesh_batch = Meshes(
                verts=verts_list,
                faces=[F0[0]] * cur_B,
                textures=TexturesVertex(verts_features=[C0[0]] * cur_B),
            )

            try:
                # camera-to-world rotations for lighting
                R_view = sub_Rs.transpose(1,2).contiguous()
                renderer.shader.lights = _build_lights_batch(R_view, dev)

                hi = renderer(mesh_batch)[..., :3].permute(0,3,1,2)   # (cur_B,3,Hhi,Whi)
                lo = _downsample(hi, aa)                              # (cur_B,C,H,W)
                if channels == 1:
                    lo = lo.mean(1, keepdim=True)

                imgs.append((_to_srgb(lo) * 255).byte().cpu())
                rots.append(sub_Rs.cpu())
                start += cur_B
                if show_pbar: pbar.update(cur_B)

                if sum(x.size(0) for x in imgs) >= flush:
                    queue.put((torch.cat(imgs,0), torch.cat(rots,0)))
                    imgs, rots = [], []

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_B == 1: raise
                pack_size = max(1, cur_B // 2)
                continue

    if show_pbar: pbar.close()
    if imgs:
        queue.put((torch.cat(imgs,0), torch.cat(rots,0)))

# ───────────────────── Worker renderer (indexed, preserves order) ─────────── #
def _worker_render_indexed(rank:int, device:str, chunk_indices, chunk_rots, mesh_path:str, aa, channels:int,
                           queue, flush:int=1024, render_batch:int=32, show_pbar:bool=False):
    """
    Same rendering as _worker_render, but returns (idx_batch, img_batch) so caller
    can place frames into the correct positions. This preserves global ordering.
    """
    torch.manual_seed(17 + rank)
    dev = torch.device(device)

    mesh0 = load_objs_as_meshes([mesh_path], device=dev)
    mesh0 = _normalise_mesh(mesh0)
    if mesh0.textures is None:
        mesh0.textures = TexturesVertex(
            verts_features=torch.ones_like(mesh0.verts_packed(), dtype=torch.float32)[None] * 0.7
        )

    renderer = _build_renderer(aa['render'], aa['blur'], aa['faces'], dev)
    renderer.shader.materials = Materials(device=dev, specular_color=((0.9,0.9,0.9),), shininess=100.0)

    V0 = mesh0.verts_padded()
    F0 = mesh0.faces_padded()
    C0 = mesh0.textures.verts_features_padded()

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    imgs, ids = [], []
    pack_size = int(render_batch)

    pbar = tqdm(total=len(chunk_rots), desc=f"Worker {rank}", position=rank+1, leave=False,
                dynamic_ncols=True, disable=not show_pbar)

    for idx_pack, R_pack in zip(batched(list(chunk_indices), pack_size), batched(list(chunk_rots), pack_size)):
        Rs = torch.stack([x.to(dev) for x in R_pack], 0)  # (B,3,3)
        start = 0
        while start < Rs.size(0):
            cur_B = min(pack_size, Rs.size(0) - start)
            sub_Rs = Rs[start:start+cur_B]

            try:
                verts_list = []
                RV = Rotate(sub_Rs, device=dev).transform_points(V0.expand(cur_B, -1, -1))
                for i in range(cur_B):
                    verts_list.append(RV[i])

                mesh_batch = Meshes(
                    verts=verts_list,
                    faces=[F0[0]] * cur_B,
                    textures=TexturesVertex(verts_features=[C0[0]] * cur_B),
                )

                # camera-to-world rotations for lighting
                R_view = sub_Rs.transpose(1,2).contiguous()
                renderer.shader.lights = _build_lights_batch(R_view, dev)

                hi = renderer(mesh_batch)[..., :3].permute(0,3,1,2)   # (cur_B,3,Hhi,Whi)
                lo = _downsample(hi, aa)                              # (cur_B,C,H,W)
                if channels == 1:
                    lo = lo.mean(1, keepdim=True)

                imgs.append((_to_srgb(lo) * 255).byte().cpu())
                ids.append(torch.tensor(idx_pack[start:start+cur_B], dtype=torch.long))

                start += cur_B
                if show_pbar: pbar.update(cur_B)

                # periodic flush
                if sum(x.size(0) for x in imgs) >= flush:
                    queue.put((torch.cat(ids, 0), torch.cat(imgs, 0)))
                    imgs, ids = [], []

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_B == 1:
                    raise
                pack_size = max(1, cur_B // 2)
                continue

    if show_pbar: pbar.close()
    if imgs:
        queue.put((torch.cat(ids, 0), torch.cat(imgs, 0)))


# ───────────────────── Dataset Class ────────────────────────── #
class RenderedSO3Dataset(Dataset):
    """
    Submanifolds:
      - 'so3'         : full SO(3) (Haar approx)
      - 's2_zeroroll' : camera pointing directions, zero roll (has a seam)
      - 's2_axisangle': NEW — R(n) = Exp(alpha [n]_x), alpha∈(0,π), homeomorphic to S^2
    """
    def __init__(self, args, *, seed:int=0):
        g = torch.Generator().manual_seed(seed)

        self.mesh_path   = args.mesh_path
        self.channels    = int(args.channels)
        self.H = self.W  = int(args.image_size)
        self.device      = args.device.lower()
        self.manifold_dim= int(args.manifold_dim)
        self.submanifold = str(args.submanifold).lower()
        self.alpha_deg   = float(getattr(args, "axis_angle_deg", 90.0))  # only for s2_axisangle
        self.alpha_rad   = max(1e-3, min(math.pi - 1e-3, math.radians(self.alpha_deg)))

        self.azim_step   = args.azim_step
        self.elev_step   = args.elev_step
        self.roll_step   = args.roll_step
        self.N_random    = args.data_samples

        self.cache_path  = args.dataset_path
        self.overwrite   = args.overwrite_cache
        self.ambient_dim = args.ambient_dim
        self.n_workers   = args.n_workers or self._default_workers()
        self._aa         = _aa(self.H)
        self.render_batch= int(getattr(args, "render_batch", 64))

        # Geodesic-specific rendering parameters
        self.geo_render       = int(getattr(args, "geo_render", 384))
        self.geo_faces        = int(getattr(args, "geo_faces", 12))
        self.geo_render_batch = int(getattr(args, "geo_render_batch", 16))
        self.geo_amp          = bool(getattr(args, "geo_amp", False))

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
            self.images = self.data.reshape(-1, self.channels, self.H, self.W).contiguous()

    # ---------- generation ----------
    def _generate_dataset(self, g):
        sub = self.submanifold
        if sub not in {"so3","s2_zeroroll","s2_axisangle"}:
            raise ValueError(f"Unknown submanifold: {sub}")

        tasks: List[torch.Tensor] = []
        grid_mode = (self.azim_step is not None and self.elev_step is not None and (sub!="s2_zeroroll" or self.roll_step is not None))

        if sub == "so3":
            if grid_mode:
                raise ValueError("Grid mode not supported for full SO(3). Use random sampling.")
            N = int(self.N_random or 100_000)
            G = torch.randn(N, 3, 3, generator=g); Q, _ = torch.linalg.qr(G)
            det = torch.linalg.det(Q); Q[det < 0, :, 0] *= -1
            tasks = [Q[i] for i in range(N)]
            print(f"[RenderedSO3] Haar-uniform sample N = {N}")

        elif sub == "s2_zeroroll":
            # (kept for completeness; note: topologically S^2 with a seam)
            if grid_mode:
                az = torch.arange(-180., 180., self.azim_step)
                el = torch.arange(-90.,  90. + 1e-9, self.elev_step)
                if self.roll_step is None or float(self.roll_step) != 0.0:
                    raise ValueError("For s2_zeroroll grid: set --roll_step 0")
                AZ, EL = torch.meshgrid(az, el, indexing="ij")
                R_cam, _ = look_at_view_transform(CAM_DIST, EL.reshape(-1), AZ.reshape(-1), degrees=True, device="cpu")
                R_obj = R_cam.transpose(1,2).contiguous()
                tasks = [R_obj[i] for i in range(R_obj.size(0))]
                print(f"[RenderedSO3] s2_zeroroll grid -> {len(tasks)}")
            else:
                N = int(self.N_random or 100_000)
                az = torch.rand(N, generator=g) * 360.0 - 180.0
                u  = torch.rand(N, generator=g) * 2.0 - 1.0
                el = torch.asin(u) * (180.0 / math.pi)
                R_cam, _ = look_at_view_transform(CAM_DIST, el, az, degrees=True, device="cpu")
                R_obj = R_cam.transpose(1, 2).contiguous()
                tasks = [R_obj[i] for i in range(N)]
                print(f"[RenderedSO3] s2_zeroroll random N={N}")

        elif sub == "s2_axisangle":
            # NEW: true S^2 submanifold via fixed-angle rotations about axis n
            alpha = torch.tensor(self.alpha_rad)
            if grid_mode:
                az = torch.arange(-180., 180., self.azim_step)
                el = torch.arange(-90.,  90. + 1e-9, self.elev_step)
                AZ, EL = torch.meshgrid(az, el, indexing="ij")
                azr, elr = torch.deg2rad(AZ.reshape(-1)), torch.deg2rad(EL.reshape(-1))
                n = torch.stack([
                    torch.cos(elr) * torch.cos(azr),
                    torch.sin(elr),
                    torch.cos(elr) * torch.sin(azr),
                ], -1)  # (M,3)
                R = _exp_axis_angle(n, alpha)
                tasks = [R[i].cpu() for i in range(R.size(0))]
                print(f"[RenderedSO3] s2_axisangle grid (alpha={self.alpha_deg:.1f}°) -> {len(tasks)}")
            else:
                N = int(self.N_random or 100_000)
                v = torch.randn(N,3, generator=g); n = F.normalize(v, dim=-1)
                R = _exp_axis_angle(n, alpha)
                tasks = [R[i].cpu() for i in range(N)]
                print(f"[RenderedSO3] s2_axisangle random N={N}, alpha={self.alpha_deg:.1f}°")

        # multiprocessing render
        if self._default_workers() <= 1:
            self._gen_serial(tasks)
        else:
            self._gen_mp(tasks)

        flat = self.data.reshape(self.data.size(0), -1)
        d_img = flat.size(1); d_emb = self.ambient_dim or d_img
        if d_emb > d_img:
            A, _ = torch.linalg.qr(torch.randn(d_emb, d_img, generator=g))
            self.P = A.float(); self.data = (A @ flat.T).T.float()
        else:
            self.P = torch.eye(d_img).float(); self.data = flat.float()

        with suppress(Exception):
            preview = (os.path.splitext(self.cache_path)[0] if self.cache_path else "rendered") + "_preview.png"
            pathlib.Path(preview).parent.mkdir(parents=True, exist_ok=True)
            save_image(self.data.reshape(-1, self.channels, self.H, self.W)[:64], preview, nrow=8, normalize=True)
            print("[RenderedSO3] preview saved →", preview)

    def _gen_mp(self, tasks):
        devs = self._visible_cuda() if ("cuda" in self.device) else [self.device]
        n_proc = min(self.n_workers, len(devs))
        chunk = math.ceil(len(tasks) / n_proc)
        chunks = [tasks[i:i+chunk] for i in range(0, len(tasks), chunk)]
        ctx = mp.get_context("spawn"); queue = ctx.Queue(maxsize=2*n_proc); procs = []

        print(f"Spawning {n_proc} worker processes...")
        for rk, ch in enumerate(chunks):
            p = ctx.Process(target=_worker_render, kwargs=dict(
                rank=rk, device=devs[rk % len(devs)], chunk=ch, mesh_path=self.mesh_path,
                aa=self._aa, channels=self.channels, queue=queue,
                flush=1024, render_batch=self.render_batch,
                show_pbar=bool(int(os.getenv("WORKER_TQDM", "0"))),  # off by default
            ))
            p.start(); procs.append(p)

        imgs, rots, rec = [], [], 0
        with tqdm(total=len(tasks), desc="Rendering (all workers)", dynamic_ncols=True) as pbar:
            while rec < len(tasks):
                try:
                    i, r = queue.get(timeout=10)  # prevent hangs
                    imgs.append(i); rots.append(r)
                    n_now = i.size(0)
                    rec += n_now
                    pbar.update(n_now)
                except QueueEmpty:
                    # if all workers are done and queue empty -> exit
                    if all(not p.is_alive() for p in procs):
                        print("\n[warn] Queue empty and all workers finished.")
                        break

        for p in procs: p.join()
        if imgs:
            self.data = torch.cat(imgs).float() / 255.; self.rotmats = torch.cat(rots).float()
        else:
            raise RuntimeError("No frames were produced by workers.")

    def _gen_serial(self, tasks):
        ctx = mp.get_context("spawn"); queue = ctx.Queue()
        _worker_render(0, self.device, tasks, self.mesh_path, self._aa, self.channels,
                       queue, flush=len(tasks), render_batch=self.render_batch, show_pbar=True)
        imgs, rots = queue.get()
        self.data = imgs.float() / 255.; self.rotmats = rots.float()
    
    # ---------- dataset-style MP renderer for arbitrary rotations ----------
    def _render_rotations_dataset_style(self, R_obj_flat: torch.Tensor, show_pbar: bool = False) -> torch.Tensor:
        """
        Render rotations using the SAME pipeline as dataset generation:
        - Multi-process across devices from _visible_cuda()
        - MultiLight SoftPhong with per-view lights
        - AA settings from self._aa, faces_per_pixel=self._aa['faces']
        Returns float tensor in [0,1], shape (N,C,H,W), order preserved.
        """
        N = R_obj_flat.size(0)
        if N == 0:
            return torch.empty(0, self.channels, self.H, self.W)

        # Single-process serial path (keeps exact worker code)
        if self._default_workers() <= 1:
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            _worker_render(
                0, self.device, list(R_obj_flat), self.mesh_path, self._aa, self.channels,
                queue, flush=N, render_batch=self.render_batch, show_pbar=True
            )
            imgs, _ = queue.get()
            return imgs.float() / 255.

        # Multi-process path with order preservation
        devs = self._visible_cuda() if ("cuda" in self.device) else [self.device]
        n_proc = min(self.n_workers, len(devs))
        chunk = math.ceil(N / n_proc)
        idx_chunks = [list(range(i, min(i+chunk, N))) for i in range(0, N, chunk)]
        rot_chunks = [ [R_obj_flat[j] for j in ich] for ich in idx_chunks ]

        ctx = mp.get_context("spawn")
        queue = ctx.Queue(maxsize=2*n_proc)
        procs = []

        for rk, (ich, rch) in enumerate(zip(idx_chunks, rot_chunks)):
            p = ctx.Process(target=_worker_render_indexed, kwargs=dict(
                rank=rk, device=devs[rk % len(devs)], chunk_indices=ich, chunk_rots=rch,
                mesh_path=self.mesh_path, aa=self._aa, channels=self.channels, queue=queue,
                flush=1024, render_batch=self.render_batch, show_pbar=False
            ))
            p.start(); procs.append(p)

        frames_u8 = torch.empty((N, self.channels, self.H, self.W), dtype=torch.uint8)
        rec = 0
        with tqdm(total=N, desc="Rendering (dataset-style MP)", dynamic_ncols=True, disable=not show_pbar) as pbar:
            while rec < N:
                try:
                    idx_batch, img_batch = queue.get(timeout=10)
                    frames_u8[idx_batch] = img_batch
                    rec += img_batch.size(0)
                    if show_pbar: pbar.update(img_batch.size(0))
                except QueueEmpty:
                    if all(not p.is_alive() for p in procs):
                        break

        for p in procs: p.join()
        if rec < N:
            missing = N - rec
            raise RuntimeError(f"Geodesic render incomplete: missing {missing} frames")

        return frames_u8.float() / 255.

    # ---------- Dataset Interface ----------
    def __len__(self): return self.data.size(0)
    def __getitem__(self, idx):
        if self.images is not None: return self.images[idx], self.rotmats[idx]
        return self.data[idx].reshape(self.channels, self.H, self.W), self.rotmats[idx]

    # ---------- Geodesic rendering in parameter space ----------
    @torch.no_grad()
    def compute_geodesic(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        If submanifold is s2_axisangle: parameters are axes n0,n1 extracted from P,Q, path n(t)=slerp, R(t)=Exp(alpha[n(t)]_x).
        If so3: uses true SO(3) geodesic. If s2_zeroroll: uses zero-roll S^2 camera path.
        Returns (B,T,C,H,W) in sRGB.
        """
        dev = P.device
        if self.submanifold == "so3":
            return self._compute_geodesic_so3(P, Q, t)
        if self.submanifold == "s2_zeroroll":
            return self._compute_geodesic_s2_zeroroll(P, Q, t)
        if self.submanifold == "s2_axisangle":
            return self._compute_geodesic_s2_axisangle(P, Q, t.to(dev))
        raise NotImplementedError

    def _get_geo_ctx(self, dev):
        """
        Geodesic renderer context (separate AA and faces from the dataset builder).
        """
        want = (dev, self.geo_render, self.geo_faces)
        if not hasattr(self, "_geo_ctx") or self._geo_ctx.get("key") != want:
            from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, MeshRenderer, BlendParams
            mesh = load_objs_as_meshes([self.mesh_path], device=dev)
            if mesh.textures is None:
                mesh.textures = TexturesVertex(
                    verts_features=torch.ones_like(mesh.verts_packed(), dtype=torch.float32)[None] * 0.7
                )
            mesh = _normalise_mesh(mesh)

            raster = RasterizationSettings(
                image_size=(self.geo_render, self.geo_render),
                blur_radius=self._aa['blur'],           # reuse blur
                faces_per_pixel=self.geo_faces,
                cull_backfaces=False,
                bin_size=None,
            )
            cam = _base_camera(dev)
            blend = BlendParams(background_color=(1., 1., 1.))
            shader = MultiLightSoftShader(device=dev, cameras=cam, blend_params=blend, lights=None)
            renderer = MeshRenderer(MeshRasterizer(cam, raster), shader)
            renderer.shader.materials = Materials(device=dev, specular_color=((0.9, 0.9, 0.9),), shininess=100.0)

            self._geo_ctx = dict(mesh=mesh, renderer=renderer, key=want)
        return self._geo_ctx["mesh"], self._geo_ctx["renderer"]

    # Build camera-to-world rotation with ZERO roll from direction (used by s2_zeroroll)
    @staticmethod
    def _camera_view_from_direction(n: torch.Tensor) -> torch.Tensor:
        """
        n: (B,3) unit view direction in WORLD coords (camera looks along -n).
        Returns R_view: (B,3,3) camera-to-world rotation with ZERO roll wrt world up.
        """
        B = n.size(0)
        f = F.normalize(-n, dim=-1)                       # camera forward
        up0 = n.new_tensor([0.0, 1.0, 0.0]).expand(B, 3)  # world up
        r = torch.cross(up0, f, dim=-1)
        bad = (r.norm(dim=-1, keepdim=True) < 1e-8)
        up1 = n.new_tensor([0.0, 0.0, 1.0]).expand(B, 3)  # fallback
        up = torch.where(bad, up1, up0)
        r = F.normalize(torch.cross(up, f, dim=-1), dim=-1)
        u = F.normalize(torch.cross(f, r, dim=-1), dim=-1)
        return torch.stack([r, u, f], dim=-1)            # columns are axes

    # True SO(3) geodesic between P,Q  → dataset-style renderer
    def _compute_geodesic_so3(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, T = P.size(0), t.numel()
        # R(t) = P * exp(t * log(P^T Q))
        A = self._so3_log_batch(P.transpose(-2, -1) @ Q)  # (B,3,3)
        Rts = P.unsqueeze(1) @ self._so3_exp_batch(A.unsqueeze(1) * t.view(1, T, 1, 1))
        R_obj_flat = Rts.reshape(B * T, 3, 3).contiguous().cpu()
        frames = self._render_rotations_dataset_style(R_obj_flat, show_pbar=True)
        return frames.view(B, T, self.channels, self.H, self.W)

    # S^2 zero-roll geodesic → dataset-style renderer
    def _compute_geodesic_s2_zeroroll(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, T = P.size(0), t.numel()
        Rv0 = P.transpose(-2, -1)
        Rv1 = Q.transpose(-2, -1)
        z_cam = torch.tensor([0.0, 0.0, -1.0], device=P.device)
        n0 = F.normalize((Rv0 @ z_cam.unsqueeze(-1)).squeeze(-1), dim=-1)
        n1 = F.normalize((Rv1 @ z_cam.unsqueeze(-1)).squeeze(-1), dim=-1)
        n_t = self._s2_geodesic_directions(n0, n1, t).reshape(B*T,3)

        # build camera-to-world with ZERO roll, then object rotation = R_view^T
        R_view_flat = self._camera_view_from_direction(n_t)
        R_obj_flat = R_view_flat.transpose(1,2).contiguous().cpu()
        frames = self._render_rotations_dataset_style(R_obj_flat, show_pbar=True)
        return frames.view(B, T, self.channels, self.H, self.W)

    # NEW: S^2 axis-angle geodesic (parameter geodesic on S^2) → dataset-style renderer
    def _compute_geodesic_s2_axisangle(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dev = P.device
        B, T = P.size(0), t.numel()

        def _axis_from_R(R: torch.Tensor) -> torch.Tensor:
            S = R - R.transpose(-2, -1)
            nx = (S[...,2,1] - S[...,1,2]) / 2
            ny = (S[...,0,2] - S[...,2,0]) / 2
            nz = (S[...,1,0] - S[...,0,1]) / 2
            n = torch.stack([nx, ny, nz], -1)
            return F.normalize(n, dim=-1)

        n0 = _axis_from_R(P)  # (B,3)
        n1 = _axis_from_R(Q)  # (B,3)
        n_t = self._s2_geodesic_directions(n0, n1, t.to(dev)).reshape(B*T,3)

        alpha = torch.tensor(self.alpha_rad, device=dev)
        R_obj_flat = _exp_axis_angle(n_t, alpha).cpu()
        frames = self._render_rotations_dataset_style(R_obj_flat, show_pbar=True)
        return frames.view(B, T, self.channels, self.H, self.W)

    # Shared micro-batch renderer for a list of rotations (with progress)
    def _render_rotations(self, R_obj_flat: torch.Tensor, dev: torch.device) -> torch.Tensor:
        mesh0, renderer = self._get_geo_ctx(dev)
        V0, F0, C0 = mesh0.verts_padded(), mesh0.faces_padded(), mesh0.textures.verts_features_padded()
        N = R_obj_flat.size(0)

        mb = int(self.geo_render_batch) if self.geo_render_batch > 0 else 1
        frames_cpu = []

        i = 0
        with tqdm(total=N, desc=f"GeoRender x{mb}", dynamic_ncols=True) as pbar:
            while i < N:
                cur = min(mb, N - i)
                R_chunk = R_obj_flat[i:i+cur]

                try:
                    # Rotate verts (GPU)
                    verts_rot_chunk = Rotate(R_chunk, device=dev).transform_points(V0.expand(cur, -1, -1))
                    mesh_chunk = Meshes(
                        verts=[v for v in verts_rot_chunk],
                        faces=[F0[0]] * cur,
                        textures=TexturesVertex(verts_features=[C0[0]] * cur)
                    )

                    # Lights per frame
                    R_view = R_chunk.transpose(1, 2).contiguous()
                    renderer.shader.lights = _build_lights_batch(R_view, dev)

                    # Render (optionally under AMP)
                    if self.geo_amp:
                        import torch.cuda.amp as amp
                        with amp.autocast():
                            hi = renderer(mesh_chunk)[..., :3].permute(0, 3, 1, 2)
                    else:
                        hi = renderer(mesh_chunk)[..., :3].permute(0, 3, 1, 2)

                    # Downsample to final image size; grayscale if needed
                    lin = _downsample(hi, {'down_mid': self.H, 'down_final': None,
                                           'render': self.geo_render, 'blur': self._aa['blur'], 'faces': self.geo_faces})
                    if self.channels == 1:
                        lin = lin.mean(1, keepdim=True)

                    # sRGB + move to CPU immediately to free GPU RAM
                    frames_cpu.append(_to_srgb(lin).cpu())

                    i += cur
                    pbar.update(cur)

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if cur == 1:
                        if self.geo_faces > 4:
                            self.geo_faces = max(4, self.geo_faces // 2)
                            mesh0, renderer = self._get_geo_ctx(dev)  # rebuild with fewer faces/px
                            pbar.set_description(f"GeoRender x{mb} (faces_per_pixel={self.geo_faces})")
                            continue
                        raise
                    mb = max(1, mb // 2)  # back off micro-batch and retry
                    pbar.set_description(f"GeoRender x{mb} (OOM retry)")

        return torch.cat(frames_cpu, 0)

    # small SO(3) log/exp helpers used in so3 geodesic
    @staticmethod
    def _so3_log_batch(R: torch.Tensor) -> torch.Tensor:
        tr  = R.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos = ((tr - 1) * 0.5).clamp(-1., 1.)
        theta   = torch.acos(cos)
        theta2  = theta * theta
        coef= torch.where(theta.abs() < 1e-5, 0.5 - theta2/12 + theta2*theta2/720, theta / (2 * torch.sin(theta)))
        return coef.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2, -1))

    @staticmethod
    def _so3_exp_batch(A: torch.Tensor) -> torch.Tensor:
        w  = torch.stack((A[...,2,1], A[...,0,2], A[...,1,0]), -1)
        theta  = torch.linalg.vector_norm(w, dim=-1)
        theta2 = theta * theta
        s  = torch.where(theta.abs() < 1e-5, 1 - theta2/6 + theta2*theta2/120, torch.sin(theta) / theta)
        c  = torch.where(theta.abs() < 1e-5, 0.5 - theta2/12 + theta2*theta2/720, (1 - torch.cos(theta)) / theta2)
        s, c = s.unsqueeze(-1).unsqueeze(-1), c.unsqueeze(-1).unsqueeze(-1)
        I = torch.eye(3, device=A.device).expand_as(A)
        return I + s * A + c * (A @ A)

    # S^2 great-circle interpolation (directions)
    @staticmethod
    def _s2_geodesic_directions(n0: torch.Tensor, n1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n0 = F.normalize(n0, dim=-1); n1 = F.normalize(n1, dim=-1)
        B, T = n0.size(0), t.numel()
        k_raw = torch.cross(n0, n1, dim=-1); k_norm = k_raw.norm(dim=-1, keepdim=True)
        dot = (n0 * n1).sum(-1).clamp(-1.0, 1.0)
        theta = torch.atan2(k_norm.squeeze(-1), dot)  # (B,)
        eps = 1e-8
        k = torch.where(k_norm > eps, k_raw / (k_norm + 1e-12), k_raw)
        # antipodal fix
        nearpi = (k_norm.squeeze(-1) < 1e-8) & (dot < 0)
        if nearpi.any():
            idx = nearpi.nonzero(as_tuple=False).squeeze(-1)
            a = torch.tensor([1.,0.,0.], device=n0.device).expand(idx.numel(),3)
            v = torch.cross(n0[idx], a, dim=-1)
            bad = (v.norm(dim=-1, keepdim=True) < 1e-6)
            v = torch.where(bad, torch.cross(n0[idx], torch.tensor([0.,1.,0.], device=n0.device), dim=-1), v)
            k[idx] = F.normalize(torch.cross(n0[idx], v, dim=-1), dim=-1)
            theta[idx] = math.pi
        kBT = k.unsqueeze(1).expand(B, T, 3)
        n0BT = n0.unsqueeze(1).expand(B, T, 3)
        ang = (theta.unsqueeze(1) * t.view(1, T)).unsqueeze(-1)
        ca, sa = torch.cos(ang), torch.sin(ang)
        k_dot_n0 = (kBT * n0BT).sum(-1, keepdim=True)
        n = n0BT * ca + torch.cross(kBT, n0BT, dim=-1) * sa + kBT * k_dot_n0 * (1.0 - ca)
        return F.normalize(n, dim=-1)

# ───────────────── Debug & CLI ──────────────────────
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
    dev = torch.device(ds.device if ("cuda" in ds.device and torch.cuda.is_available()) else "cpu")
    p_rots = ds.rotmats[torch.tensor([c[0] for c in chosen])].to(dev).contiguous()
    q_rots = ds.rotmats[torch.tensor([c[1] for c in chosen])].to(dev).contiguous()
    t = t.to(dev)

    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    sheet = ds.compute_geodesic(p_rots, q_rots, t).flatten(0, 1)
    if dev.type == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    B, T = len(chosen), frames
    print(f"[profile] geodesic render B={B} T={T} → {dt*1000:.1f} ms  ({dt/(B*T)*1000:.2f} ms/frame) on {dev}")

    save_image(
    sheet, os.path.join(out_root, "geodesics_grid.png"),
    nrow=frames, padding=0, normalize=True
    )
    print("Saved contact sheet →", f"{out_root}/geodesics_grid.png")

if __name__ == "__main__":
    P = argparse.ArgumentParser("Build rendered SO(3) dataset (with S^2 submanifold)")
    P.add_argument("--mesh_path", required=True); P.add_argument("--dataset_path", required=True)
    P.add_argument("--image_size", type=int, choices=[64,128], default=64)  # grayscale 64x64 ready
    P.add_argument("--channels", type=int, choices=[1,3], default=1)        # grayscale ready
    P.add_argument("--azim_step", type=float); P.add_argument("--elev_step", type=float); P.add_argument("--roll_step", type=float)
    P.add_argument("--data_samples", type=int); P.add_argument("--ambient_dim", type=int)
    P.add_argument("--manifold_dim", type=int, default=2)  # 2 for S^2 submanifolds by default
    P.add_argument("--submanifold", type=str, default="s2_axisangle",
                   help="one of: so3, s2_zeroroll, s2_axisangle")
    P.add_argument("--axis_angle_deg", type=float, default=90.0,
                   help="Fixed axis-angle (degrees) for s2_axisangle; in (0,180)")
    P.add_argument("--overwrite_cache", action="store_true"); P.add_argument("--n_workers", type=int, default=0)
    P.add_argument("--device", default="cuda")
    P.add_argument("--render_batch", type=int, default=64)
    # Geodesic debug knobs
    P.add_argument("--geo_render", type=int, default=384,
                   help="AA render size used only during geodesic debug (e.g., 384 lowers mem vs 512)")
    P.add_argument("--geo_faces", type=int, default=12,
                   help="faces_per_pixel used only during geodesic debug (e.g., 12 lowers mem vs 30)")
    P.add_argument("--geo_render_batch", type=int, default=16,
                   help="micro-batch size used only during geodesic debug")
    P.add_argument("--geo_amp", action="store_true",
                   help="use AMP (float16) in geodesic debug to further cut memory")
    args = P.parse_args()

    ds = RenderedSO3Dataset(args)
    print(f"Dataset built: {len(ds)} samples → {args.dataset_path}")
    _debug_geodesics(ds, pairs=8, frames=12)
