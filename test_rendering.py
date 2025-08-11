#!/usr/bin/env python3
"""
random_so3_teapot_grid_lit.py  —  dataset-ready version
=======================================================

• Samples N random elements of SO(3) (intrinsic-dim = 3).  
• Puts a 3-point (+rim) light rig **in the camera frame**, so the visible
  side is always well lit, independent of object orientation.  
• Uses a custom MultiLightSoftShader that supports any number of point
  lights without ballooning VRAM.

Ground-truth geodesics on SO(3) are still computable from the stored
rotation matrices.
"""

import os, math, torch, torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    BlendParams, PointLights, TexturesVertex, Materials,
)
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.transforms import random_rotations, Rotate

# ────────────────────────── custom multi-light shader ────────────────── #

class MultiLightSoftShader(SoftPhongShader):
    """SoftPhongShader that accepts lights.location shaped (B, L, 3)."""

    def forward(self, fragments, meshes, **kwargs):
        cameras   = super()._get_cameras(**kwargs)
        lights    = kwargs.get("lights",    self.lights)
        materials = kwargs.get("materials", self.materials)
        blend     = kwargs.get("blend_params", self.blend_params)
        texels    = meshes.sample_textures(fragments)

        # ── multi-light branch ───────────────────────────────────────────
        if lights.location.ndim == 3:                 # (B, L, 3)
            B, L, _ = lights.location.shape
            colour_accum = 0.0
            for li in range(L):
                single = PointLights(
                    device         = lights.device,
                    location       = lights.location[:, li],
                    ambient_color  = lights.ambient_color[:, li],
                    diffuse_color  = lights.diffuse_color[:, li],
                    specular_color = lights.specular_color[:, li],
                )
                colour_accum += phong_shading(
                    meshes    = meshes,
                    fragments = fragments,
                    texels    = texels,
                    lights    = single,
                    cameras   = cameras,
                    materials = materials,
                )
            colours = colour_accum / L               # average contribution
        # ── single-light (stock) branch ──────────────────────────────────
        else:
            colours = phong_shading(
                meshes    = meshes,
                fragments = fragments,
                texels    = texels,
                lights    = lights,
                cameras   = cameras,
                materials = materials,
            )

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar  = kwargs.get("zfar",  getattr(cameras, "zfar", 100.0))
        return softmax_rgb_blend(colours, fragments, blend, znear=znear, zfar=zfar)

# ───────────────────── user-tweakable knobs ───────────────────── #

BASE          = 128         # 64 or 128  — final tile size
NUM_RANDOM    = 16          # N random rotations ⇒ grid √N × √N
ADD_RIM_LIGHT = True        # extra halo from behind camera
ADD_OUTLINE   = False       # black outline pass (excludes rim light)
OUTLINE_SCALE = 1.03        # scale for outline mesh

if BASE == 128:             # —— anti-alias presets
    RENDER_SIZE     = 256   # 2× supersample
    BLUR_RADIUS     = 1.0e-4
    FACES_PER_PIXEL = 12

    def post_downsample(img):       # already 128×128 after area filter
        return img

elif BASE == 64:
    RENDER_SIZE     = 512   # 8× supersample
    BLUR_RADIUS     = 1.0e-6
    FACES_PER_PIXEL = 30

    def post_downsample(img):
        return F.interpolate(img[None], size=64, mode="bicubic",
                             align_corners=False)[0]
else:
    raise ValueError("BASE must be 64 or 128")

FOV       = 25.0
CAM_DIST  = 1.05 / math.sin(math.radians(FOV / 2))
GRID_COLS = int(NUM_RANDOM ** 0.5 + 0.5)
ROOT      = "datasets"
OBJ_PATH  = f"{ROOT}/meshes/teapot.obj"
OUT_PATH  = f"{ROOT}/teapot_debug_grid.png"

print(f"BASE={BASE}, render={RENDER_SIZE}×{RENDER_SIZE}, "
      f"{NUM_RANDOM} random views → grid "
      f"{GRID_COLS}×{math.ceil(NUM_RANDOM/GRID_COLS)}")

# ───────────────────── helper: lights in camera space ─────────────────── #

def build_lights(view_R: torch.Tensor, dev: torch.device,
                 add_rim=ADD_RIM_LIGHT, outline=ADD_OUTLINE, dist=4.0) -> PointLights:
    """
    Create a 3-point (+rim) rig defined **in camera coordinates** and rotate
    it to world space with `view_R` (camera-to-world rotation matrix).
    """
    dirs_cam = torch.tensor([
        [ 0.35,  0.35,  1.0],   # key
        [-0.35,  0.20,  1.0],   # fill
        [ 0.00,  0.00, -1.0],   # back / kicker
    ], dtype=torch.float32, device=dev)
    if add_rim and not outline:
        dirs_cam = torch.cat([dirs_cam,
                              torch.tensor([[0., 0., -1.0]], device=dev)], dim=0)

    # Rotate directions into world space (normalise afterwards).
    dirs_world = (view_R @ dirs_cam.t()).t()
    dirs_world = dirs_world / dirs_world.norm(dim=1, keepdim=True)
    locs = dirs_world * dist                                   # positions

    amb  = torch.full_like(locs, 0.10)
    diff = torch.full_like(locs, 0.65)
    spec = torch.full_like(locs, 0.25)
    if add_rim and not outline:
        diff[-1] = torch.tensor([0.35, 0.35, 0.35], device=dev)
        amb [-1] = 0.0
        spec[-1] = 0.0

    loc_t, amb_t, diff_t, spec_t = [x.unsqueeze(0) for x in (locs, amb, diff, spec)]
    return PointLights(device=dev,
                       location       = loc_t,
                       ambient_color  = amb_t,
                       diffuse_color  = diff_t,
                       specular_color = spec_t)

def to_srgb(img_lin: torch.Tensor) -> torch.Tensor:
    """Linear-RGB → sRGB in-place clamp."""
    return img_lin.clamp(0, 1) ** (1 / 2.2)

# ───────────────────── renderer factory ────────────────────── #

def make_renderer(camera, dev):
    raster = RasterizationSettings(
        image_size      = RENDER_SIZE,
        blur_radius     = BLUR_RADIUS,
        faces_per_pixel = FACES_PER_PIXEL,
        bin_size        = None,
    )
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))
    return MeshRenderer(
        rasterizer = MeshRasterizer(camera, raster),
        shader     = MultiLightSoftShader(device=dev, cameras=camera,
                                          blend_params=blend, lights=None),
    )

# ─────────────────────────── main procedure ─────────────────────────── #

@torch.no_grad()
def main() -> None:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    # —— load and normalise mesh ————————————————
    mesh = load_objs_as_meshes([OBJ_PATH], device=dev)
    if mesh.textures is None or mesh.textures._maps_padded.shape[1] == 0:
        verts = mesh.verts_packed()
        mesh.textures = TexturesVertex(
            verts_features=torch.ones_like(verts)[None] * 0.7)

    verts = mesh.verts_packed()
    mesh.offset_verts_(-verts.mean(0))
    mesh.scale_verts_((1.0 / verts.norm(dim=1).max()).item())

    # —— camera & renderer ————————————————————————
    R_cam, T_cam = look_at_view_transform(CAM_DIST, 0, 0, device=dev)
    camera   = FoVPerspectiveCameras(device=dev, R=R_cam, T=T_cam, fov=FOV)
    renderer = make_renderer(camera, dev)

    renderer.shader.materials = Materials(
        device=dev,
        specular_color=((0.9, 0.9, 0.9),),
        shininess=100.0)

    # —— random SO(3) rotations ————————————————————
    R_rand = random_rotations(NUM_RANDOM, device=dev)
    torch.save(R_rand.cpu(), f"{ROOT}/teapot_rotations.pt")   # ground-truth

    # —— render loop ————————————————————————————————
    tiles = []
    for i in range(NUM_RANDOM):
        R_obj = R_rand[i]                    # rotate the mesh
        view_R = R_obj.t()                   # camera-to-world rotation
        renderer.shader.lights = build_lights(view_R, dev)

        m = mesh.clone().update_padded(
            Rotate(R_obj[None], device=dev).transform_points(mesh.verts_padded()))

        # forward pass
        hi  = renderer(m)[0, ..., :3].permute(2, 0, 1)      # (3,H,W)
        mid = F.interpolate(hi[None], size=128, mode="area")[0]
        main_pass = post_downsample(mid)

        if ADD_OUTLINE:
            outline_mesh = mesh.clone()
            outline_mesh.scale_verts_(OUTLINE_SCALE)
            outline_mesh.textures = TexturesVertex(
                verts_features=torch.zeros_like(outline_mesh.verts_packed())[None])

            orig_cull = renderer.rasterizer.raster_settings.cull_backfaces
            renderer.rasterizer.raster_settings.cull_backfaces = False
            ol = renderer(outline_mesh)[0, ..., :3].permute(2, 0, 1)
            renderer.rasterizer.raster_settings.cull_backfaces = orig_cull

            ol_mid   = F.interpolate(ol[None], size=128, mode="area")[0]
            ol_final = post_downsample(ol_mid)
            mask     = ol_final.sum(0, keepdim=True) > 0.01
            final    = torch.where(mask, ol_final, main_pass)
        else:
            final = main_pass

        tiles.append(to_srgb(final).mean(0, keepdim=True).cpu())
        torch.cuda.empty_cache()

    # —— make & save grid ————————————————————————
    grid = make_grid(torch.stack(tiles).repeat(1, 3, 1, 1), nrow=GRID_COLS)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    save_image(grid, OUT_PATH)
    print("✅ wrote", OUT_PATH)

if __name__ == "__main__":
    main()

