#!/usr/bin/env python3
# datasets/analytic_manifold_dataset.py
# ============================================================================
# Analytic image manifolds (no 3D rendering) with exact geodesics:
#   - S^2  : "Bump-Shaded Spherical Glyphs"
#   - T^2  : "Neural Tapestry Torus"
#
# Mirrors the style of rendered_so_dataset.py:
#   * CLI: generate & cache dataset, preview sheet, random or grid sampling
#   * Optional ambient projection via random isometry (P)
#   * Ground-truth geodesic image sequences
# ============================================================================

from __future__ import annotations
import argparse
import math
import os
import pathlib
from contextlib import suppress
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image

# tqdm (nice CLI progress bars) — safe fallback if not installed
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


# ───────────────────────────── Utilities ────────────────────────────── #

def _grid_xy(H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij",
    )
    return x, y  # in [-1,1]

def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi

def _slerp(n0: torch.Tensor, n1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Great-circle interpolation on S^2.
    n0, n1: (B,3) unit vectors
    t: (T,) in [0,1]
    returns (B,T,3)
    """
    n0 = F.normalize(n0, dim=-1)
    n1 = F.normalize(n1, dim=-1)

    dot = (n0 * n1).sum(-1, keepdim=True).clamp(-1.0, 1.0)  # (B,1)
    theta = torch.acos(dot).squeeze(-1)                      # (B,)
    eps = 1e-8
    sin_th = torch.sin(theta).clamp_min(eps)                 # (B,)

    # reshape to broadcast with tt = (1,T,1)
    theta  = theta.view(-1, 1, 1)                            # (B,1,1)
    sin_th = sin_th.view(-1, 1, 1)                           # (B,1,1)
    tt = t.view(1, -1, 1)                                    # (1,T,1)

    A  = torch.sin((1 - tt) * theta) / sin_th                # (B,T,1)
    Bc = torch.sin(tt * theta) / sin_th                      # (B,T,1)

    return A * n0.unsqueeze(1) + Bc * n1.unsqueeze(1)        # (B,T,3)


# ─────────────────────── Image Generators ───────────────────────────── #

@torch.no_grad()
def s2_bump_shaded(n: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    S^2 → image (B,3,H,W). Smooth, asymmetric; glossy + tail marker.
    n: (B,3) unit vectors
    """
    device = n.device
    B = n.size(0)
    x, y = _grid_xy(H, W, device)
    x = x.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    y = y.unsqueeze(0).expand(B, -1, -1)

    nx, ny, nz = n[:, 0].view(B, 1, 1), n[:, 1].view(B, 1, 1), n[:, 2].view(B, 1, 1)

    # Angle-conditioned multi-frequency heightfield
    f1 = torch.sin(6 * (x * nx + y * ny) + 1.1 * nz)
    f2 = torch.sin(9 * (x * ny - y * nx) + 0.7 * nx + 1.3 * ny)
    f3 = torch.sin(13 * (0.7 * x + 0.2 * y) + 0.9 * nz - 0.4 * nx)
    h = 0.4 * f1 + 0.3 * f2 + 0.25 * f3

    # Pseudo normal map from gradients
    dx = h[:, :, 2:] - h[:, :, :-2]
    dy = h[:, 2:, :] - h[:, :-2, :]
    dx = F.pad(dx, (1, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 1, 1))
    nmap = torch.stack([-dx, -dy, torch.ones_like(dx) * 0.6], dim=1)  # (B,3,H,W)
    nmap = F.normalize(nmap, dim=1)

    # Lights vary smoothly with n (no roll)
    L1 = torch.stack([0.6 * nx + 0.3, 0.4 * ny + 0.2, 0.8 * nz + 0.1], dim=1).squeeze(-1).squeeze(-1)  # (B,3)
    L2 = torch.stack([-0.7 * nx + 0.2, 0.5 * ny, 0.5 * nz + 0.3], dim=1).squeeze(-1).squeeze(-1)        # (B,3)
    L1 = F.normalize(L1, dim=-1)
    L2 = F.normalize(L2, dim=-1)

    def lambert(L: torch.Tensor) -> torch.Tensor:
        L = L[:, :, None, None]  # (B,3,1,1)
        return (nmap * L).sum(1).clamp_min(0.0)  # (B,H,W)

    diff = 0.8 * lambert(L1) + 0.6 * lambert(L2)  # (B,H,W)
    spec = (nmap[:, 2].clamp_min(0.0) ** 12) * 0.6

    # Asymmetric tail along d = normalize(n × a)
    a = torch.tensor([0.57, -0.31, 0.77], device=device).expand(B, 3)
    dvec = F.normalize(torch.cross(n, a, dim=1), dim=-1)  # (B,3)
    cx = 0.35 * nx
    cy = 0.35 * ny
    tx = cx + 0.15 * dvec[:, 0].view(B, 1, 1)
    ty = cy + 0.15 * dvec[:, 1].view(B, 1, 1)
    dist2 = (x - tx) ** 2 + (y - ty) ** 2
    tail = torch.exp(-dist2 / (2 * (0.09 ** 2)))

    # Palette (asymmetric)
    Rw = 0.55 + 0.45 * torch.sin(1.3 * nx + 0.9 * ny - 0.4 * nz)
    Gw = 0.50 + 0.50 * torch.sigmoid(2.1 * nz - 1.0 * nx + 0.6 * ny)
    Bw = 0.45 + 0.55 * torch.sin(0.8 * ny + 1.7 * nz + 0.3)

    base = (0.55 + 0.45 * torch.tanh(1.1 * h)) * (0.45 + 0.55 * diff)
    R = base * Rw + 0.6 * spec + 0.5 * tail
    G = base * Gw + 0.5 * spec + 0.35 * tail
    Bc = base * Bw + 0.4 * spec + 0.25 * tail
    img = torch.stack([R, G, Bc], 1).clamp(0, 1)
    return img


@torch.no_grad()
def torus_neural_tapestry(alpha: torch.Tensor, beta: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    T^2 → image (B,3,H,W). Rich periodic texture with warp + occluder.
    alpha, beta: (B,) angles in radians
    """
    device = alpha.device
    B = alpha.numel()
    x, y = _grid_xy(H, W, device)
    x = x.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    y = y.unsqueeze(0).expand(B, -1, -1)

    a = alpha.view(B, 1, 1)
    b = beta.view(B, 1, 1)

    # Periodic, angle-conditioned warp
    wx = 0.18 * torch.sin(3 * x + a) + 0.10 * torch.sin(2 * y + 0.7 * b)
    wy = 0.18 * torch.cos(2 * y - b) + 0.08 * torch.sin(3 * x + 1.1 * a - 0.4 * b)
    xp, yp = x + wx, y + wy

    # Multiscale oriented Gabor weave: orientation=beta, phase=alpha
    def gabor(_xp, _yp, k, gain):
        u = _xp * torch.cos(b) + _yp * torch.sin(b)
        v = -_xp * torch.sin(b) + _yp * torch.cos(b)
        return gain * torch.sin(k * u + a) * torch.exp(-((v ** 2) * (k * 0.12) ** 2))

    base = gabor(xp, yp, 7.0, 0.7) + gabor(xp, yp, 11.0, 0.5) + gabor(xp, yp, 15.0, 0.35)

    # Occluder + soft shadow (centers are periodic in (a,b))
    u = 0.35 * torch.sin(a) + 0.25 * torch.sin(b + 0.7 * a)
    v = 0.30 * torch.cos(b) + 0.22 * torch.sin(a - 0.4 * b)
    dist2 = (xp - u) ** 2 + (yp - v) ** 2
    occl = torch.exp(-dist2 / (2 * (0.12 ** 2)))
    shadow = torch.exp(-torch.sqrt(dist2 + 1e-6) / 0.35)

    # Angle-conditioned palette (breaks π symmetry)
    r = 0.55 + 0.45 * torch.sin(1.1 * a + 0.7 * b)
    g = 0.50 + 0.50 * torch.sigmoid(2.2 * a - 1.3 * b)
    bl = 0.45 + 0.55 * torch.sin(0.6 * a + 1.9 * b + 0.7)

    I = 0.55 + 0.45 * torch.tanh(1.2 * base)
    R = I * r + 0.6 * occl + 0.25 * shadow
    G = I * g + 0.45 * occl + 0.20 * shadow
    Bc = I * bl + 0.35 * occl + 0.30 * shadow
    img = torch.stack([R, G, Bc], 1).clamp(0, 1)
    return img


# ─────────────────────── Dataset Class ──────────────────────────────── #

class AnalyticManifoldDataset(Dataset):
    """
    Analytic image manifolds with exact geodesics.

    Modes (manifold):
      - 's2'    : unit sphere samples (parameter = n ∈ S^2) → slerp geodesics
      - 'torus' : flat torus S^1×S^1 (parameter = (alpha,beta)) → shortest wrap geodesics

    Features:
      - Random or grid sampling
      - Optional ambient projection to ambient_dim via random isometry
      - Cached tensor {data, params, proj} at dataset_path
      - Preview contact sheet
    """

    def __init__(self, args, *, seed: int = 0):
        self.cache_path: Optional[str] = getattr(args, "dataset_path", None)
        self.image_size: int = int(args.image_size)
        self.channels: int = int(args.channels)
        self.manifold: str = str(getattr(args, "manifold", "s2")).lower()
        assert self.manifold in ("s2", "torus")
        self.N_random: Optional[int] = getattr(args, "data_samples", None)
        self.overwrite: bool = bool(getattr(args, "overwrite_cache", False))
        self.ambient_dim: Optional[int] = getattr(args, "ambient_dim", None)
        self.gen_batch: int = int(getattr(args, "gen_batch", 8192))

        # grid sampling (optional)
        self.azim_step: Optional[float] = getattr(args, "azim_step", None)      # for s2: longitude step (deg)
        self.elev_step: Optional[float] = getattr(args, "elev_step", None)      # for s2: latitude step (deg)
        self.alpha_step: Optional[float] = getattr(args, "alpha_step", None)    # for torus: α step (deg)
        self.beta_step: Optional[float] = getattr(args, "beta_step", None)      # for torus: β step (deg)

        self.device: str = str(getattr(args, "device", "cpu")).lower()
        self.H = self.W = self.image_size

        g = torch.Generator(device="cpu").manual_seed(seed)

        if self.cache_path and os.path.isfile(self.cache_path) and not self.overwrite:
            blob = torch.load(self.cache_path, map_location="cpu")
            self.data, self.params, self.P = blob["data"], blob["params"], blob["proj"]
            print(f"[AnalyticManifold] loaded {len(self)} cached samples from {self.cache_path}")
        else:
            self._generate_dataset(g)
            if self.cache_path:
                pathlib.Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"data": self.data, "params": self.params, "proj": self.P}, self.cache_path)
                print(f"[AnalyticManifold] cached → {self.cache_path}")
        self._materialise_images()

    # ───────── helpers ───────── #

    def _materialise_images(self):
        self.images = None
        # Raw data is already image vectors if P is identity of size d_img
        flat = self.data
        d = self.channels * self.H * self.W
        if flat.size(1) == d and torch.allclose(self.P, torch.eye(d)):
            self.images = flat.view(-1, self.channels, self.H, self.W).contiguous()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        if self.images is not None:
            return self.images[idx], self.params[idx]
        # Fallback: matmul P^{-1}? not needed—always store image-space then embed.
        return self.data[idx].view(self.channels, self.H, self.W), self.params[idx]

    # ───────── dataset generation ───────── #

    def _generate_dataset(self, g: torch.Generator):
        dev = torch.device(self.device if (self.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

        if self._use_grid():
            if self.manifold == "s2":
                az = torch.arange(-180.0, 180.0, float(self.azim_step))
                el = torch.arange(-90.0, 90.0 + 1e-9, float(self.elev_step))
                # spherical to Cartesian (longitude=az, latitude=el) in radians
                AZ, EL = torch.meshgrid(az, el, indexing="ij")
                AZr = torch.deg2rad(AZ.reshape(-1))
                ELr = torch.deg2rad(EL.reshape(-1))
                nx = torch.cos(ELr) * torch.cos(AZr)
                ny = torch.sin(ELr)
                nz = torch.cos(ELr) * torch.sin(AZr)
                n = torch.stack([nx, ny, nz], dim=-1).to(dev)
                params = F.normalize(n, dim=-1)  # (N,3)
                print(f"[AnalyticManifold] S^2 grid → {params.size(0)} points")
                # Chunked generation with tqdm
                imgs_list = []
                N = params.size(0)
                for i in tqdm(range(0, N, self.gen_batch), desc="Generating S² images", dynamic_ncols=True):
                    chunk = params[i:i + self.gen_batch].to(dev, non_blocking=True)
                    imgs_list.append(s2_bump_shaded(chunk, self.H, self.W).cpu())
                imgs = torch.cat(imgs_list, dim=0)
            else:
                al = torch.arange(-180.0, 180.0, float(self.alpha_step))
                be = torch.arange(-180.0, 180.0, float(self.beta_step))
                AL, BE = torch.meshgrid(al, be, indexing="ij")
                alpha = torch.deg2rad(AL.reshape(-1)).to(dev)
                beta = torch.deg2rad(BE.reshape(-1)).to(dev)
                params = torch.stack([alpha, beta], dim=-1)  # (N,2)
                print(f"[AnalyticManifold] T^2 grid → {params.size(0)} points")
                # Chunked generation with tqdm
                imgs_list = []
                N = alpha.size(0)
                for i in tqdm(range(0, N, self.gen_batch), desc="Generating T² images", dynamic_ncols=True):
                    a = alpha[i:i + self.gen_batch].to(dev, non_blocking=True)
                    b = beta[i:i + self.gen_batch].to(dev, non_blocking=True)
                    imgs_list.append(torus_neural_tapestry(a, b, self.H, self.W).cpu())
                imgs = torch.cat(imgs_list, dim=0)
        else:
            N = int(self.N_random or 100_000)
            if self.manifold == "s2":
                v = torch.randn(N, 3, generator=g).to(dev)
                params = F.normalize(v, dim=-1)  # (N,3)
                print(f"[AnalyticManifold] S^2 random N={N}")
                # Chunked generation with tqdm
                imgs_list = []
                for i in tqdm(range(0, N, self.gen_batch), desc="Generating S² images", dynamic_ncols=True):
                    chunk = params[i:i + self.gen_batch].to(dev, non_blocking=True)
                    imgs_list.append(s2_bump_shaded(chunk, self.H, self.W).cpu())
                imgs = torch.cat(imgs_list, dim=0)
            else:
                alpha = (torch.rand(N, generator=g) * 2 * math.pi - math.pi).to(dev)
                beta = (torch.rand(N, generator=g) * 2 * math.pi - math.pi).to(dev)
                params = torch.stack([alpha, beta], dim=-1)  # (N,2)
                print(f"[AnalyticManifold] T^2 random N={N}")
                # Chunked generation with tqdm
                imgs_list = []
                for i in tqdm(range(0, N, self.gen_batch), desc="Generating T² images", dynamic_ncols=True):
                    a = alpha[i:i + self.gen_batch].to(dev, non_blocking=True)
                    b = beta[i:i + self.gen_batch].to(dev, non_blocking=True)
                    imgs_list.append(torus_neural_tapestry(a, b, self.H, self.W).cpu())
                imgs = torch.cat(imgs_list, dim=0)

        if self.channels == 1:
            imgs = imgs.mean(1, keepdim=True)

        # Optional ambient projection
        flat = imgs.view(imgs.size(0), -1)  # (N, d_img)
        d_img = flat.size(1)
        d_emb = int(self.ambient_dim or d_img)
        if d_emb > d_img:
            A, _ = torch.linalg.qr(torch.randn(d_emb, d_img, generator=g))
            P = A.float().to(flat.device)
            data = (P @ flat.T).T.float()
        else:
            P = torch.eye(d_img, device=flat.device).float()
            data = flat.float()

        # Save
        self.data = data.cpu()
        self.params = params.cpu()
        self.P = P.cpu()

        # Preview contact sheet
        with suppress(Exception):
            preview = (os.path.splitext(self.cache_path)[0] if self.cache_path else "analytic") + "_preview.png"
            pathlib.Path(preview).parent.mkdir(parents=True, exist_ok=True)
            save_image(imgs[:64].cpu(), preview, nrow=8, normalize=True)
            print("[AnalyticManifold] preview saved →", preview)

    def _use_grid(self) -> bool:
        if self.manifold == "s2":
            return self.azim_step is not None and self.elev_step is not None
        else:
            return self.alpha_step is not None and self.beta_step is not None

    # ─────────────── Ground-truth geodesics ─────────────── #

    @torch.no_grad()
    def compute_geodesic(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return (B,T,C,H,W) images along the geodesic between parameters P and Q.
        - For S^2  : P,Q are (B,3) unit vectors, geodesic is slerp on S^2.
        - For T^2  : P,Q are (B,2) angles (alpha,beta) in radians; geodesic is
                     shortest straight line with wrap on each circle.
        """
        dev = P.device
        B, T = P.size(0), t.numel()
        H, W, C = self.H, self.W, self.channels

        frames: List[torch.Tensor] = []

        if self.manifold == "s2":
            n_path = _slerp(P, Q, t.to(dev))  # (B,T,3)
            for b in range(B):
                imgs = s2_bump_shaded(n_path[b], H, W)  # (T,3,H,W)
                if C == 1:
                    imgs = imgs.mean(1, keepdim=True)
                frames.append(imgs)
        else:  # torus
            d = torch.stack([
                _wrap_to_pi(Q[:, 0] - P[:, 0]),
                _wrap_to_pi(Q[:, 1] - P[:, 1]),
            ], dim=-1).to(dev)                                                 # (B,2)
            tt = t.to(dev).view(1, T, 1)                                       # (1,T,1)
            path = P.to(dev).unsqueeze(1) + tt * d.unsqueeze(1)                # (B,T,2)
            for b in range(B):
                a = path[b, :, 0]
                bb = path[b, :, 1]
                imgs = torus_neural_tapestry(a, bb, H, W)                      # (T,3,H,W)
                if C == 1:
                    imgs = imgs.mean(1, keepdim=True)
                frames.append(imgs)

        return torch.stack(frames, dim=0)  # (B,T,C,H,W)


# ───────────────── Debug & CLI ──────────────────────────────────────── #

def _random_pairs(n: int, k: int, g=None) -> List[Tuple[int, int]]:
    g = g or torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=g)
    pairs = []
    for i in range(min(k, n // 2)):
        pairs.append((idx[2 * i].item(), idx[2 * i + 1].item()))
    return pairs

def _debug_geodesics(ds: "AnalyticManifoldDataset", pairs=5, frames=16, out_root="datasets/geo_debug"):
    os.makedirs(out_root, exist_ok=True)
    pr = _random_pairs(len(ds), pairs)
    t = torch.linspace(0, 1, frames)

    if ds.manifold == "s2":
        P = ds.params[torch.tensor([i for i, _ in pr])]
        Q = ds.params[torch.tensor([j for _, j in pr])]
    else:
        P = ds.params[torch.tensor([i for i, _ in pr])]  # (B,2)
        Q = ds.params[torch.tensor([j for _, j in pr])]

    sheet = ds.compute_geodesic(P, Q, t).flatten(0, 1)  # (B*T,C,H,W)
    save_image(sheet, os.path.join(out_root, f"geo_{ds.manifold}_grid.png"), nrow=frames, normalize=True)
    print("Saved geodesic sheet →", f"{out_root}/geo_{ds.manifold}_grid.png")


if __name__ == "__main__":
    P = argparse.ArgumentParser("Build analytic S^2 / T^2 dataset")
    P.add_argument("--dataset_path", required=True)
    P.add_argument("--manifold", type=str, choices=["s2", "torus"], default="s2")
    P.add_argument("--image_size", type=int, choices=[32, 64, 128], default=32)
    P.add_argument("--channels", type=int, choices=[1, 3], default=3)

    # sampling/grid
    P.add_argument("--data_samples", type=int, help="N random samples if grid not specified")
    # S^2 grid (degrees)
    P.add_argument("--azim_step", type=float, help="longitude step (deg) for S^2 grid")
    P.add_argument("--elev_step", type=float, help="latitude step (deg) for S^2 grid")
    # T^2 grid (degrees)
    P.add_argument("--alpha_step", type=float, help="alpha step (deg) for T^2 grid")
    P.add_argument("--beta_step", type=float, help="beta step (deg) for T^2 grid")

    # embed / cache
    P.add_argument("--ambient_dim", type=int)
    P.add_argument("--overwrite_cache", action="store_true")

    # perf / device
    P.add_argument("--device", default="cpu")
    P.add_argument("--gen_batch", type=int, default=8192,
                   help="Chunk size for on-the-fly generation with tqdm.")

    # debug
    P.add_argument("--debug_pairs", type=int, default=0, help="if >0, render debug geodesic sheet with this many pairs")
    P.add_argument("--debug_frames", type=int, default=16)

    args = P.parse_args()

    ds = AnalyticManifoldDataset(args)
    print(f"Dataset built: {len(ds)} samples → {args.dataset_path}")

    if args.debug_pairs > 0:
        _debug_geodesics(ds, pairs=args.debug_pairs, frames=args.debug_frames)
