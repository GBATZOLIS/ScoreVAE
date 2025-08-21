#!/usr/bin/env python3
# tools/test_rendering_so3.py
# Unified rendering: samples and geodesics use the exact same renderer,
# lighting, materials, and silhouette outline. Geodesics are chunked to avoid OOM.

from __future__ import annotations
import os, math, argparse
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRenderer, MeshRasterizer, RasterizationSettings,
    BlendParams, PointLights, DirectionalLights, Materials, TexturesVertex
)
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer.mesh.shader import SoftPhongShader, SoftSilhouetteShader
from pytorch3d.transforms import random_rotations, Rotate


# ──────────────────────────────── shader ─────────────────────────────── #

class MultiLightSoftShader(SoftPhongShader):
    """Supports PointLights or DirectionalLights with (B,3) or (B,L,3)."""
    def forward(self, fragments, meshes, **kwargs):
        cameras   = super()._get_cameras(**kwargs)
        lights    = kwargs.get("lights",    self.lights)
        materials = kwargs.get("materials", self.materials)
        blend     = kwargs.get("blend_params", self.blend_params)
        texels    = meshes.sample_textures(fragments)

        def shade_with_single(single_lights):
            return phong_shading(
                meshes    = meshes,
                fragments = fragments,
                texels    = texels,
                lights    = single_lights,
                cameras   = cameras,
                materials = materials,
            )

        # Multi point
        if hasattr(lights, "location") and isinstance(lights.location, torch.Tensor) and lights.location.ndim == 3:
            B, L, _ = lights.location.shape
            acc = 0.0
            for li in range(L):
                single = PointLights(
                    device         = lights.device,
                    location       = lights.location[:, li],
                    ambient_color  = lights.ambient_color[:, li],
                    diffuse_color  = lights.diffuse_color[:, li],
                    specular_color = lights.specular_color[:, li],
                )
                acc = acc + shade_with_single(single)
            colours = acc / L

        # Multi directional
        elif hasattr(lights, "direction") and isinstance(lights.direction, torch.Tensor) and lights.direction.ndim == 3:
            B, L, _ = lights.direction.shape
            acc = 0.0
            for li in range(L):
                single = DirectionalLights(
                    device         = lights.device,
                    direction      = lights.direction[:, li],
                    ambient_color  = lights.ambient_color[:, li],
                    diffuse_color  = lights.diffuse_color[:, li],
                    specular_color = lights.specular_color[:, li],
                )
                acc = acc + shade_with_single(single)
            colours = acc / L

        else:
            colours = shade_with_single(lights)

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar  = kwargs.get("zfar",  getattr(cameras, "zfar", 100.0))
        return softmax_rgb_blend(colours, fragments, blend, znear=znear, zfar=zfar)


# ──────────────────────── small helpers (batch-safe) ─────────────────────── #

def to_srgb(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1) ** (1/2.2)

def _resize(x: torch.Tensor, size: int, mode: str, *, align_corners=None) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
        out = (F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
               if align_corners is not None else
               F.interpolate(x, size=size, mode=mode))
        return out[0]
    if x.dim() == 4:
        return (F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
                if align_corners is not None else
                F.interpolate(x, size=size, mode=mode))
    raise NotImplementedError(f"Expected 3D/4D tensor, got {x.dim()}D.")

def area(x: torch.Tensor, size: int) -> torch.Tensor:
    return _resize(x, size, "area")

def bicubic(x: torch.Tensor, size: int) -> torch.Tensor:
    return _resize(x, size, "bicubic", align_corners=False)

def min_pool2d(x: torch.Tensor, kernel_size: int, stride: int = 1, padding: int = 0):
    # emulate min-pooling via max-pooling
    return -F.max_pool2d(-x, kernel_size, stride, padding)


# ────────────────────────────── renderer bits ────────────────────────────── #

def make_renderer(image_size: int, blur: float, faces_per_pixel: int,
                  fov: float, cam_dist: float, dev: torch.device,
                  cull_backfaces: bool = True) -> MeshRenderer:
    R = torch.eye(3, device=dev)[None]
    T = torch.tensor([[0., 0., cam_dist]], device=dev)
    cams  = FoVPerspectiveCameras(device=dev, R=R, T=T, fov=fov)
    rast  = RasterizationSettings(
        image_size=image_size, blur_radius=blur, faces_per_pixel=faces_per_pixel,
        cull_backfaces=cull_backfaces, bin_size=None
    )
    blend = BlendParams(background_color=(1., 1., 1.))
    shader = MultiLightSoftShader(device=dev, cameras=cams, blend_params=blend, lights=None)
    return MeshRenderer(MeshRasterizer(cams, rast), shader)

def make_silhouette_renderer(image_size: int, fov: float, cam_dist: float,
                             dev: torch.device, cull_backfaces: bool) -> MeshRenderer:
    R = torch.eye(3, device=dev)[None]
    T = torch.tensor([[0., 0., cam_dist]], device=dev)
    cams = FoVPerspectiveCameras(device=dev, R=R, T=T, fov=fov)
    rast = RasterizationSettings(
        image_size=image_size, blur_radius=1e-6, faces_per_pixel=1,
        cull_backfaces=cull_backfaces, bin_size=None
    )
    return MeshRenderer(MeshRasterizer(cams, rast), SoftSilhouetteShader())

def build_lights_batch(
    R_view: torch.Tensor,
    dev: torch.device,
    kind: str = "directional",
    dist: float = 4.0,
    ambient: float = 0.03,
    key: float = 1.00,
    fill: float = 0.50,
    back: float = 0.50,
    rim: float = 0.90
):
    # camera-space directions (+Z = toward camera)
    dirs_cam = torch.tensor(
        [[ 0.35,  0.35,  1.0],   # key
         [-0.35,  0.20,  1.0],   # fill
         [ 0.00,  0.15, -1.0],   # back
         [ 0.00,  0.00, -1.0]],  # rim
        device=dev, dtype=torch.float32
    )
    B = R_view.shape[0]; L = dirs_cam.shape[0]
    dirs_w = (R_view @ dirs_cam.t()).transpose(1, 2)      # (B,L,3)
    dirs_w = dirs_w / dirs_w.norm(dim=-1, keepdim=True)

    amb  = dirs_w.new_full((B, L, 3), ambient)
    diff = torch.stack([
        torch.full((B, 3), key,  device=dev),
        torch.full((B, 3), fill, device=dev),
        torch.full((B, 3), back, device=dev),
        torch.full((B, 3), rim,  device=dev),
    ], dim=1)
    spec = diff * 0.6

    if kind == "directional":
        return DirectionalLights(
            device=dev,
            direction=-dirs_w,  # lights point toward origin
            ambient_color=amb, diffuse_color=diff, specular_color=spec
        )
    else:
        locs = dirs_w * dist
        return PointLights(
            device=dev,
            location=locs, ambient_color=amb, diffuse_color=diff, specular_color=spec
        )

def overlay_outline(rgb_bchw: torch.Tensor, mesh, sil_renderer: MeshRenderer,
                    k: int, strength: float) -> torch.Tensor:
    """
    Thin edge darkening from a silhouette pass (identical res/fpp as main renderer).
    rgb_bchw: (B,3,H,W) linear; returns same shape.
    """
    sil = sil_renderer(mesh)[..., 3].unsqueeze(1)         # (B,1,H,W)
    pad = k // 2
    maxf = F.max_pool2d(sil, k, 1, pad)
    minf = min_pool2d(sil, k, 1, pad)
    edge = (maxf - minf).clamp(0, 1)
    return rgb_bchw * (1 - strength * edge)               # gently darken edges


# ───────────────────────────── SO(3) utilities ───────────────────────────── #

def so3_log_batch(R: torch.Tensor) -> torch.Tensor:
    tr  = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    th2   = theta * theta
    coef  = torch.where(
        theta.abs() < 1e-5,
        0.5 - th2/12 + th2*th2/720,
        theta / (2 * torch.sin(theta))
    )
    return coef.unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2, -1))

def so3_exp_batch(A: torch.Tensor) -> torch.Tensor:
    w  = torch.stack((A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]), -1)
    th = torch.linalg.vector_norm(w, dim=-1)
    th2= th * th
    s  = torch.where(th.abs() < 1e-5, 1 - th2/6 + th2*th2/120, torch.sin(th)/th)
    c  = torch.where(th.abs() < 1e-5, 0.5 - th2/12 + th2*th2/720, (1 - torch.cos(th))/(th2 + 1e-12))
    s, c = s.unsqueeze(-1).unsqueeze(-1), c.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, device=A.device).expand_as(A)
    return I + s * A + c * (A @ A)


# ─────────────────────────────────── main ─────────────────────────────────── #

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Preview SO(3) teapot renders + geodesics")
    ap.add_argument("--mesh", default="datasets/meshes/teapot.obj")
    ap.add_argument("--out",  default="datasets/debug_out")
    ap.add_argument("--size", type=int, choices=[64, 128], default=64)
    ap.add_argument("--channels", type=int, choices=[1, 3], default=1)
    ap.add_argument("--num-random", type=int, default=16)
    ap.add_argument("--num-pairs",  type=int, default=6)
    ap.add_argument("--frames",     type=int, default=11)
    ap.add_argument("--device", default="cuda:0")

    # ONE set of render params for both samples and geodesics
    ap.add_argument("--render-res", type=int, default=512,
                    help="Internal render resolution before downsampling.")
    ap.add_argument("--faces-per-pixel", type=int, default=30,
                    help="Faces per pixel for rasterizer.")

    # Geodesic memory knob (same renderer; just chunk the batch)
    ap.add_argument("--geo-chunk", type=int, default=4,
                    help="How many geodesic frames to render per chunk.")

    # Lighting / material knobs
    ap.add_argument("--no-cull", action="store_true", help="Disable backface culling.")
    ap.add_argument("--light-type", choices=["point", "directional"], default="directional")
    ap.add_argument("--ambient",  type=float, default=0.03)
    ap.add_argument("--key",      type=float, default=1.00)
    ap.add_argument("--fill",     type=float, default=0.50)
    ap.add_argument("--back",     type=float, default=0.50)
    ap.add_argument("--rim",      type=float, default=0.90)
    ap.add_argument("--specular", type=float, default=0.90)
    ap.add_argument("--shininess", type=float, default=50.0)

    # Outline controls
    ap.add_argument("--no-outline", action="store_true", help="Disable silhouette outline.")
    ap.add_argument("--outline-k", type=int, default=3, help="Outline kernel size (odd).")
    ap.add_argument("--outline-strength", type=float, default=0.65, help="Edge darkening strength [0,1].")

    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    # Anti-alias presets → down to target size
    if a.size == 128:
        blur = 1e-4
        def down(x):  # identity to 128
            return x
        down_to_aa = 128
    else:  # 64×64 target
        blur = 1e-6
        def down(x):
            return bicubic(x, a.size)
        down_to_aa = 128

    fov = 25.0
    cam_dist = 1.05 / math.sin(math.radians(fov / 2))

    # Load & normalise mesh
    mesh = load_objs_as_meshes([a.mesh], device=dev)
    if mesh.textures is None or (
        hasattr(mesh.textures, "_maps_padded") and mesh.textures._maps_padded.shape[1] == 0
    ):
        v = mesh.verts_packed()
        mesh.textures = TexturesVertex(verts_features=torch.ones_like(v)[None] * 0.7)
    v = mesh.verts_packed()
    mesh.offset_verts_(-v.mean(0))
    mesh.scale_verts_((1.0 / v.norm(dim=1).max()).item())

    # Single renderer for everything
    renderer = make_renderer(
        image_size=a.render_res, blur=blur, faces_per_pixel=a.faces_per_pixel,
        fov=fov, cam_dist=cam_dist, dev=dev, cull_backfaces=not a.no_cull
    )
    renderer.shader.materials = Materials(
        device=dev,
        specular_color=((a.specular, a.specular, a.specular),),
        shininess=a.shininess
    )

    # Matching silhouette renderer (same res & culling)
    sil_renderer = make_silhouette_renderer(
        image_size=a.render_res, fov=fov, cam_dist=cam_dist,
        dev=dev, cull_backfaces=not a.no_cull
    )

    # ── random samples
    R = random_rotations(a.num_random, device=dev)
    tiles = []
    for i in range(a.num_random):
        R_obj  = R[i]
        view_R = R_obj.t()
        renderer.shader.lights = build_lights_batch(
            view_R[None], dev, a.light_type,
            ambient=a.ambient, key=a.key, fill=a.fill, back=a.back, rim=a.rim
        )

        m = mesh.clone().update_padded(
            Rotate(R_obj[None], device=dev).transform_points(mesh.verts_padded())
        )
        hi = renderer(m)[0, ..., :3].permute(2, 0, 1).unsqueeze(0)  # (1,3,H',W')
        if not a.no_outline:
            hi = overlay_outline(hi, m, sil_renderer, k=a.outline-k if hasattr(a,'outline-k') else a.outline_k, strength=a.outline_strength)
        hi  = hi[0]
        mid = area(hi, down_to_aa)
        img = down(mid)
        if a.channels == 1:
            img = img.mean(0, keepdim=True)
        tiles.append(to_srgb(img).cpu())

    grid_cols = int(a.num_random ** 0.5 + 0.5)
    samples   = torch.stack(tiles)
    samples3  = samples.repeat(1, 3, 1, 1) if a.channels == 1 else samples
    grid      = make_grid(samples3, nrow=grid_cols)
    out_samples = os.path.join(a.out, f"samples_{a.size}.png")
    save_image(grid, out_samples)
    print("✅ wrote samples grid →", out_samples)

    # ── geodesics (closed-form on SO(3)) — SAME renderer, chunked
    K, T = a.num_pairs, a.frames
    idx  = torch.randperm(a.num_random)
    P    = R[idx[:K]]
    Q    = R[idx[K:2*K]]
    t    = torch.linspace(0, 1, T, device=dev)

    A    = so3_log_batch(P.transpose(1, 2) @ Q)
    Rts  = P.unsqueeze(1) @ so3_exp_batch(A.unsqueeze(1) * t.view(1, T, 1, 1))  # (K,T,3,3)
    R_bt = Rts.reshape(K * T, 3, 3)

    imgs_cpu = []
    chunk = max(1, a.geo_chunk)
    for s in range(0, K * T, chunk):
        e = min(s + chunk, K * T)
        R_chunk = R_bt[s:e]

        mesh_chunk = mesh.extend(R_chunk.shape[0])
        verts      = Rotate(R_chunk, device=dev).transform_points(mesh_chunk.verts_padded())
        mesh_chunk = mesh_chunk.update_padded(verts)

        renderer.shader.lights = build_lights_batch(
            R_chunk.transpose(1, 2), dev, a.light_type,
            ambient=a.ambient, key=a.key, fill=a.fill, back=a.back, rim=a.rim
        )

        hi = renderer(mesh_chunk)[..., :3].permute(0, 3, 1, 2)  # (B,3,H',W')
        if not a.no_outline:
            hi = overlay_outline(hi, mesh_chunk, sil_renderer, k=a.outline_k, strength=a.outline_strength)
        mid = area(hi, down_to_aa)
        lo  = bicubic(mid, a.size) if a.size != 128 else mid
        if a.channels == 1:
            lo = lo.mean(1, keepdim=True)

        imgs_cpu.append(to_srgb(lo).cpu())

        del mesh_chunk, verts, hi, mid, lo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    geos = torch.cat(imgs_cpu, dim=0)
    geos3 = geos.repeat(1, 3, 1, 1) if a.channels == 1 else geos
    out_geos = os.path.join(a.out, f"geodesics_{a.size}.png")
    save_image(geos3, out_geos, nrow=T)
    print("✅ wrote geodesics grid →", out_geos)


if __name__ == "__main__":
    main()
