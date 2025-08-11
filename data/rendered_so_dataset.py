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
    v = mesh.verts_packed(); mesh.offset_verts_(-v.mean(0)); mesh.scale_verts_((1.0 / v.norm(dim=1).max()).item()); return mesh

def _base_camera(dev):
    return FoVPerspectiveCameras(device=dev, R=torch.eye(3, device=dev)[None], T=torch.tensor([[0., 0., CAM_DIST]], device=dev), fov=FOV)

def _build_renderer(res: int, blur: float, faces: int, dev):
    raster = RasterizationSettings(image_size=(res,res), blur_radius=blur, faces_per_pixel=faces, cull_backfaces=True, bin_size=None)
    cam = _base_camera(dev)
    blend = BlendParams(background_color=(1.,1.,1.))
    shader = MultiLightSoftShader(device=dev, cameras=cam, blend_params=blend, lights=None)
    return MeshRenderer(MeshRasterizer(cam, raster), shader)

def _downsample(lin: torch.Tensor, aa):
    # Note: lin must be 4D tensor for interpolate
    is_3d = lin.dim() == 3
    if is_3d: lin = lin.unsqueeze(0)
    
    mid = torch.nn.functional.interpolate(lin, size=aa['down_mid'], mode="area")
    if aa['down_final']:
        mid = torch.nn.functional.interpolate(mid, size=aa['down_final'], mode="bicubic", align_corners=False)
    
    if is_3d: mid = mid.squeeze(0)
    return mid

# ───────────────────── Batched SO(3) & Lighting Utilities (NEW) ──────────────── #
def _so3_log_batch(R: torch.Tensor) -> torch.Tensor:
    tr  = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1)*0.5).clamp(-1.,1.)
    θ   = torch.acos(cos)
    θ2  = θ*θ
    coef= torch.where(θ.abs()<1e-5, 0.5 - θ2/12 + θ2*θ2/720, θ / (2*torch.sin(θ)))
    return coef.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2,-1))

def _so3_exp_batch(A: torch.Tensor) -> torch.Tensor:
    w  = torch.stack((A[...,2,1], A[...,0,2], A[...,1,0]), -1)
    θ  = torch.linalg.vector_norm(w, dim=-1)
    θ2 = θ*θ
    s  = torch.where(θ.abs()<1e-5, 1 - θ2/6 + θ2*θ2/120, torch.sin(θ)/θ)
    c  = torch.where(θ.abs()<1e-5, 0.5 - θ2/12 + θ2*θ2/720, (1-torch.cos(θ))/θ2)
    s, c = s.unsqueeze(-1).unsqueeze(-1), c.unsqueeze(-1).unsqueeze(-1)
    return (torch.eye(3, device=A.device).expand_as(A) + s*A + c*(A@A))

def _build_lights_batch(R_view: torch.Tensor, dev: torch.device, dist=4.0, rim=True) -> PointLights:
    dirs_cam = torch.tensor([[0.35, 0.35, 1.0], [-0.35, 0.20, 1.0], [0.0, 0.0, -1.0]], device=dev)
    if rim: dirs_cam = torch.cat([dirs_cam, torch.tensor([[0.,0.,-1.]], device=dev)], 0)
    
    dirs_w = (R_view @ dirs_cam.t()).transpose(1,2)
    dirs_w = dirs_w / dirs_w.norm(dim=-1, keepdim=True)
    locs   = dirs_w * dist
    L = locs.size(1)
    
    amb  = locs.new_full((locs.size(0),L,3), 0.10)
    diff = locs.new_full((locs.size(0),L,3), 0.65)
    spec = locs.new_full((locs.size(0),L,3), 0.25)
    if rim:
        diff[:,-1], amb[:,-1], spec[:,-1] = torch.tensor([0.35,0.35,0.35], device=dev), 0., 0.
        
    return PointLights(device=dev, location=locs, ambient_color=amb, diffuse_color=diff, specular_color=spec)

# ───────────────────── MP Worker Renderer (RESTORED) ────────────────────────── #
def _worker_render(rank:int, device:str, chunk: Sequence, mesh_path:str, aa, channels:int, grid_mode: bool, seed:int, queue, flush:int=256):
    torch.manual_seed(seed)
    dev = torch.device(device)

    mesh = load_objs_as_meshes([mesh_path], device=dev); mesh = _normalise_mesh(mesh)
    if mesh.textures is None:
        mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed())[None]*0.7)

    renderer = _build_renderer(aa['render'], aa['blur'], aa['faces'], dev)
    renderer.shader.materials = Materials(device=dev, specular_color=((0.9,0.9,0.9),), shininess=100.0)

    imgs, rots = [], []
    for item in chunk:
        if grid_mode:
            az, el, rl = item
            R_cam, _ = look_at_view_transform(CAM_DIST, el, az, degrees=True, device=dev)
            if rl:
                θ = math.radians(rl); c, s = math.cos(θ), math.sin(θ)
                Rz = torch.tensor([[c,-s,0], [s,c,0], [0,0,1]], device=dev)
                R_cam = Rz @ R_cam
            R_obj = R_cam.t()
        else:
            R_obj = item.to(dev)

        lights = _build_lights_batch(R_obj.t().unsqueeze(0), dev) # Use new batched lights helper
        renderer.shader.lights = lights
        
        mesh_inst = mesh.clone().update_padded(Rotate(R_obj[None], device=dev).transform_points(mesh.verts_padded()))
        lin_hi = renderer(mesh_inst)[0,:,:,:3].permute(2,0,1)
        lin    = _downsample(lin_hi, aa)
        if channels == 1:
            lin = lin.mean(0, keepdim=True)

        imgs.append((_to_srgb(lin)*255).byte().cpu())
        rots.append(R_obj.cpu())

        if len(imgs) >= flush:
            queue.put((torch.stack(imgs), torch.stack(rots))); imgs, rots = [], []
    if imgs:
        queue.put((torch.stack(imgs), torch.stack(rots)))

# ───────────────────── Dataset Class (RESTORED & ENHANCED) ─────────────────── #
class RenderedSO3Dataset(Dataset):
    # ---------- Initialization & Dataset Generation (RESTORED) --------------
    def __init__(self, args, *, seed:int=0):
        g = torch.Generator().manual_seed(seed)

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
            az = torch.arange(-180., 180., self.azim_step); el = torch.arange(-90., 90.+1e-9, self.elev_step)
            rl = torch.tensor([0.]) if self.roll_step == 0 else torch.arange(-180., 180., self.roll_step)
            tasks = [(a.item(), e.item(), r.item()) for r in rl for e in el for a in az]
            print(f"[RenderedSO3] Euler grid → {len(tasks)} rotations")
        else:
            N = int(self.N_random or 100_000)
            G = torch.randn(N,3,3, generator=g); Q,_ = torch.linalg.qr(G); det=torch.linalg.det(Q); Q[det<0,:,0]*=-1
            tasks = [Q[i] for i in range(N)]
            print(f"[RenderedSO3] Haar-uniform sample N = {N}")

        if self.n_workers <= 1: self._gen_serial(tasks, grid_mode)
        else: self._gen_mp(tasks, grid_mode)
        
        flat = self.data.view(self.data.size(0), -1); d_img = flat.size(1); d_emb = self.ambient_dim or d_img
        if d_emb > d_img:
            A,_ = torch.linalg.qr(torch.randn(d_emb, d_img, generator=g)); self.P = A.float(); self.data = (A @ flat.T).T.float()
        else:
            self.P = torch.eye(d_img).float(); self.data = flat.float()

        with suppress(Exception):
            preview = (os.path.splitext(self.cache_path)[0] if self.cache_path else "rendered")+"_preview.png"
            pathlib.Path(preview).parent.mkdir(parents=True, exist_ok=True)
            save_image(self.data.view(-1,self.channels,self.H,self.W)[:25], preview, nrow=5, normalize=True)
            print("[RenderedSO3] preview saved →", preview)

    def _gen_mp(self, tasks, grid_mode):
        devs = self._visible_cuda() if self.device=="cuda" else [self.device]; n_proc= min(self.n_workers, len(devs))
        chunk = math.ceil(len(tasks) / n_proc); chunks = [tasks[i:i+chunk] for i in range(0,len(tasks),chunk)]
        ctx = mp.get_context("spawn"); queue = ctx.Queue(maxsize=2*n_proc); procs = []
        for rk,ch in enumerate(chunks):
            p = ctx.Process(target=_worker_render, kwargs=dict(rank=rk, device=devs[rk % len(devs)], chunk=ch, mesh_path=self.mesh_path, aa=self._aa, channels=self.channels, grid_mode=grid_mode, seed=17+rk, queue=queue))
            p.start(); procs.append(p)
        imgs, rots, rec = [], [], 0
        with tqdm(total=len(tasks), desc="Rendering SO(3) (mp)") as pbar:
            while rec < len(tasks):
                i,r = queue.get(); imgs.append(i); rots.append(r); rec += i.size(0); pbar.update(i.size(0))
        for p in procs: p.join()
        self.data = torch.cat(imgs).float() / 255.; self.rotmats = torch.cat(rots).float()

    def _gen_serial(self, tasks, grid_mode):
        # This serial generation can be kept as a fallback, reusing the _worker_render logic.
        ctx = mp.get_context("spawn"); queue = ctx.Queue()
        _worker_render(0, self.device, tasks, self.mesh_path, self._aa, self.channels, grid_mode, 42, queue, flush=len(tasks))
        imgs, rots = queue.get()
        self.data = imgs.float() / 255.; self.rotmats = rots.float()

    # ---------- Dataset Interface (Unchanged) ---------------------------------
    def __len__(self): return self.data.size(0)
    def __getitem__(self, idx):
        if self.images is not None: return self.images[idx], self.rotmats[idx]
        return self.data[idx].view(self.channels, self.H, self.W), self.rotmats[idx]

    # ───────────────── Fast Batched Geodesic Renderer (NEW) ───────────────────
    def _get_geo_ctx(self, dev):
        if not hasattr(self, "_geo_ctx") or self._geo_ctx["mesh"].device != dev:
            mesh = load_objs_as_meshes([self.mesh_path], device=dev)
            if mesh.textures is None: mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed())[None]*0.7)
            mesh = _normalise_mesh(mesh)
            renderer = _build_renderer(self._aa['render'], self._aa['blur'], self._aa['faces'], dev)
            renderer.shader.materials = Materials(device=dev, specular_color=((0.9,0.9,0.9),), shininess=100.0)
            self._geo_ctx = dict(mesh=mesh, renderer=renderer)
        return self._geo_ctx["mesh"], self._geo_ctx["renderer"]

    def compute_geodesic(self, P: torch.Tensor, Q: torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        dev, B, T = P.device, P.size(0), t.numel()
        A = _so3_log_batch(P.transpose(-2,-1) @ Q)
        Rts = P.unsqueeze(1) @ _so3_exp_batch(A.unsqueeze(1) * t.view(1,T,1,1))
        Rts_flat = Rts.reshape(B*T, 3,3)
        mesh0, renderer = self._get_geo_ctx(dev)
        mesh_bt = mesh0.extend(B*T)
        verts = Rotate(Rts_flat, device=dev).transform_points(mesh_bt.verts_padded())
        mesh_bt = mesh_bt.update_padded(verts)
        renderer.shader.lights = _build_lights_batch(Rts_flat.transpose(1,2), dev)
        hi = renderer(mesh_bt)[...,:3].permute(0,3,1,2)
        lin = _downsample(hi, self._aa)
        if self.channels == 1: lin = lin.mean(1, keepdim=True)
        return _to_srgb(lin).view(B, T, self.channels, self.H, self.W)

# ───────────────── Debug & CLI (RESTORED & Corrected) ──────────────────────
def _geo_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> float:
    cos = ((Ra.T @ Rb).trace() - 1).clamp(-1.0,1.0)/2; return torch.acos(cos).item()

def _debug_geodesics(ds: "RenderedSO3Dataset", pairs=5, frames=10, out_root="datasets/geo_debug"):
    os.makedirs(out_root, exist_ok=True)
    chosen: List[Tuple[int,int]] = []
    cand  = torch.randperm(len(ds))[:min(500, len(ds))]
    for _ in range(pairs):
        if not chosen: i = cand[0].item()
        else:
            used = torch.tensor([u for ij in chosen for u in ij])
            remain = cand[~cand.unsqueeze(1).eq(used).any(1)]
            scores = [(min(_geo_angle(ds.rotmats[k], ds.rotmats[u]) for u in used), k.item()) for k in remain]
            i = max(scores)[1] if scores else remain[0].item()
        j = max(((_geo_angle(ds.rotmats[i], ds.rotmats[j]), j.item()) for j in cand if j != i), key=lambda x: x[0])[1]
        chosen.append((i,j))
    
    t = torch.linspace(0,1,frames)
    p_rots = ds.rotmats[torch.tensor([c[0] for c in chosen])]
    q_rots = ds.rotmats[torch.tensor([c[1] for c in chosen])]

    print(f"Rendering {pairs} geodesic paths with {frames} frames each...")
    sheet = ds.compute_geodesic(p_rots, q_rots, t).flatten(0,1)
    
    save_image(sheet, os.path.join(out_root,"geodesics_grid.png"), nrow=frames, normalize=True)
    print("Saved contact sheet →", f"{out_root}/geodesics_grid.png")


if __name__ == "__main__":
    P = argparse.ArgumentParser("Build rendered SO(3) dataset")
    P.add_argument("--mesh_path", required=True); P.add_argument("--dataset_path", required=True)
    P.add_argument("--image_size", type=int, choices=[64,128], default=128)
    P.add_argument("--channels", type=int, choices=[1,3], default=1)
    P.add_argument("--azim_step", type=float); P.add_argument("--elev_step", type=float); P.add_argument("--roll_step", type=float)
    P.add_argument("--data_samples", type=int); P.add_argument("--ambient_dim", type=int)
    P.add_argument("--overwrite_cache", action="store_true"); P.add_argument("--n_workers", type=int, default=0)
    P.add_argument("--device", default="cuda")
    args = P.parse_args()

    ds = RenderedSO3Dataset(args)
    print(f"Dataset built: {len(ds)} samples → {args.dataset_path}")
    _debug_geodesics(ds, pairs=5, frames=10)