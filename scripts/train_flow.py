#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-2: Train a latent→latent flow f so that the score-induced geometry on u = f(z)
is smooth and easy for geodesic ODEs. Uses ONLY the score function (no image decoder).

Everything (paths, device, hparams, weights) is read from the flow config, which only
points to the AE config and the latent diffusion config. Those configs contain their own
model hyperparams and checkpoints.

CRITICAL: Training is performed in the NORMALIZED latent space.
We therefore:
  1) Load AE RUNNING checkpoint to fetch normalization buffers.
  2) Load AE EMA checkpoint for weights.
  3) Copy buffers from RUNNING → EMA model before encoding/normalizing.
"""

from __future__ import annotations
import os, math, argparse
from typing import List, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.func import jacrev, jvp, vmap, grad

# ── your repo bits
from configs import load_config
from data.data_utils_ddp import get_dataloaders
from models import get_model
from utils.train_utils import EMA, load_model
from utils.ae_utils import add_isometry_compact_layout, perturb_latents_at_time
from sde import configure_sde
from loss.isometry import decoder_isometry_regularisation  # reusing for flow isometry

# tqdm (optional, nice progress bars)
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

import math
import matplotlib.pyplot as plt
from utils.sampling_utils import Algorithm1


@torch.no_grad()
def _sample_normalized_latents(lat_model, lat_sde, d_lat, device, *, steps=250, count=6000):
    """Sample normalized latents z_hat ~ diffusion prior (shape: (count, d_lat))."""
    score_fn = lat_model.get_score_fn(lat_sde)
    z_hat = Algorithm1(lat_sde, int(steps), score_fn, (int(count), int(d_lat)), device, y=None)
    return z_hat  # already normalized space

def _scatter2d(ax, Z, title):
    ax.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.6)
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.set_title(title); ax.grid(True, ls="--", alpha=0.3)

def _scatter3d(ax, Z, title):
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=4, alpha=0.6, depthshade=True)
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]"); ax.set_zlabel("z[2]")
    ax.set_title(title)

def _log_latent_scatter(writer, tag, Z, step):
    """
    Z: (N, d) on CPU. Supports d==2 or d==3. Else silently skip.
    """
    Z = Z.detach().float().cpu()
    N, d = Z.shape
    if d not in (2, 3) or N < 2:
        print(f"[{tag}] skip: latent dim={d} (need 2 or 3) or insufficient points.")
        return

    # subsample for speed/clarity
    max_points = 6000
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        Z = Z[idx]

    if d == 2:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        _scatter2d(ax, Z, title=tag)
        writer.add_figure(tag, fig, global_step=step)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(6.2, 6.2))
        ax = fig.add_subplot(111, projection='3d')
        _scatter3d(ax, Z, title=tag)
        writer.add_figure(tag, fig, global_step=step)
        plt.close(fig)

@torch.no_grad()
def _log_overlay_scatter(writer, tag, Z_a, Z_b, labels=("A","B"), step=0):
    """
    Overlay two clouds (supports d=2 or 3).
    """
    Za = Z_a.detach().float().cpu()
    Zb = Z_b.detach().float().cpu()
    if Za.numel() == 0 or Zb.numel() == 0: return
    assert Za.shape[1] == Zb.shape[1]
    d = Za.shape[1]
    if d not in (2,3): 
        print(f"[{tag}] skip overlay: d={d} (need 2 or 3)")
        return

    # subsample equally
    m = min(Za.size(0), Zb.size(0), 6000)
    if Za.size(0) > m: Za = Za[torch.randperm(Za.size(0))[:m]]
    if Zb.size(0) > m: Zb = Zb[torch.randperm(Zb.size(0))[:m]]

    import matplotlib.pyplot as plt
    if d == 2:
        fig, ax = plt.subplots(figsize=(5.8,5.8))
        ax.scatter(Za[:,0], Za[:,1], s=4, alpha=0.45, label=labels[0])
        ax.scatter(Zb[:,0], Zb[:,1], s=4, alpha=0.45, label=labels[1])
        ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
        ax.grid(True, ls="--", alpha=0.3); ax.legend()
        writer.add_figure(tag, fig, global_step=step); plt.close(fig)
    else:
        fig = plt.figure(figsize=(6.6,6.6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Za[:,0], Za[:,1], Za[:,2], s=3, alpha=0.40, label=labels[0])
        ax.scatter(Zb[:,0], Zb[:,1], Zb[:,2], s=3, alpha=0.40, label=labels[1])
        ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]"); ax.set_zlabel("z[2]")
        ax.legend()
        writer.add_figure(tag, fig, global_step=step); plt.close(fig)

@torch.no_grad()
def _log_displacement_stats(writer, tag_prefix, Z0, Zt, step):
    """
    Log L2 displacement and per-dim stats between Zt and Z0.
    """
    Z0 = Z0.detach().float().cpu()
    Zt = Zt.detach().float().cpu()
    D = (Zt - Z0)
    l2 = D.norm(dim=1)  # (N,)

    # scalars
    writer.add_scalar(f"{tag_prefix}/disp_l2_mean", float(l2.mean()), step)
    writer.add_scalar(f"{tag_prefix}/disp_l2_p50",  float(l2.quantile(0.50)), step)
    writer.add_scalar(f"{tag_prefix}/disp_l2_p90",  float(l2.quantile(0.90)), step)
    writer.add_scalar(f"{tag_prefix}/disp_l2_p95",  float(l2.quantile(0.95)), step)

    # hist
    writer.add_histogram(f"{tag_prefix}/disp_l2_hist", l2.numpy(), global_step=step)

    # per-dim stats
    mean = D.mean(0); std = D.std(0, unbiased=False)
    for i in range(D.shape[1]):
        writer.add_scalar(f"{tag_prefix}/per_dim/delta_mean_{i}", float(mean[i]), step)
        writer.add_scalar(f"{tag_prefix}/per_dim/delta_std_{i}",  float(std[i]), step)

@torch.no_grad()
def _print_latent_stats(tag, Z):
    """Console stats per dimension for quick sanity."""
    Zc = Z.detach().float().cpu()
    N, d = Zc.shape
    mean = Zc.mean(0); std = Zc.std(0, unbiased=False)
    zmin = Zc.min(0).values; zmax = Zc.max(0).values
    print(f"[{tag}] stats over N={N}, d={d}")
    print("dim\tmean\t\tstd\t\tmin\t\tmax")
    for i in range(d):
        print(f"{i:3d}\t{mean[i]:+.6e}\t{std[i]:.6e}\t{zmin[i]:+.6e}\t{zmax[i]:+.6e}")


def inv_softplus(y: float) -> float:
    # returns x such that softplus(x) = y  (stable for y>0)
    import math
    # softplus(x) = log(1 + exp(x))
    # For y not too small: x = log(exp(y) - 1)
    return math.log(math.expm1(y))

# -----------------------------------------------------------------------------
# Model summaries / pretty printing
# -----------------------------------------------------------------------------
def _human_count(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def print_flow_summary(flow: nn.Module, latent_dim: int, cfg) -> None:
    total_params = sum(p.numel() for p in flow.parameters())
    train_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in flow.buffers(recurse=True))
    try:
        transform = str(cfg.flow.transform)
        n_blocks  = int(cfg.flow.n_blocks)
        hidden    = int(cfg.flow.hidden)
        layers    = int(cfg.flow.layers)
        bins      = getattr(cfg.flow, "bins", None)
    except Exception:
        transform, n_blocks, hidden, layers, bins = "?", "?", "?", "?", None

    arch_bits = f"{transform} coupling | blocks={n_blocks} | hidden={hidden} | layers={layers}"
    if transform == "rq" and bins is not None:
        arch_bits += f" | bins={bins}"

    print("[Flow] ----------------------------------------------")
    print(f"[Flow] latent_dim={latent_dim}")
    print(f"[Flow] arch: {arch_bits}")
    print(f"[Flow] params: {_human_count(train_params)} trainable / {_human_count(total_params)} total")
    if total_buffers:
        print(f"[Flow] buffers: {_human_count(total_buffers)}")
    print("[Flow] ----------------------------------------------")


def _finite_or_nan(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()

def _safe_cholesky(G: torch.Tensor, *, abs_jitter: float, rel_jitter: float, max_tries: int = 5):
    """
    Try cholesky with progressively higher jitter. Returns (L, used_abs_jitter, used_rel_jitter).
    """
    B, m, _ = G.shape
    I = torch.eye(m, device=G.device, dtype=G.dtype).expand(B, m, m)
    # per-sample relative scale from diagonal mean
    diag_mean = torch.diagonal(G, dim1=1, dim2=2).mean(dim=1).view(B, 1, 1)

    a = float(abs_jitter)
    r = float(rel_jitter)
    for k in range(max_tries):
        G_ = G + a * I + r * diag_mean * I
        try:
            L = torch.linalg.cholesky(G_)
            if _finite_or_nan(L):
                return L, a, r
        except RuntimeError:
            pass
        # backoff: increase jitter x10 each attempt
        a *= 10.0
        r *= 10.0
    # final attempt with heavy jitter
    G_ = G + a * I + r * diag_mean * I
    L = torch.linalg.cholesky(G_)
    return L, a, r


# =============================================================================
# Diagnostic visualization callback
# =============================================================================
def visualize_z_and_u_spaces(
    val_latents_clean: torch.Tensor,
    flow: nn.Module,
    sde,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
    t_scalar: float,
    max_points: int = 2000,
):
    """Creates scatter plots for the source (z_t) and target (u_t) latent spaces."""
    if not is_primary():
        return

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    z_sample = val_latents_clean[:max_points].to(device)
    with torch.no_grad():
        z_t_sample = perturb_latents_at_time(z_sample, sde, t_scalar)
        u_t_sample = flow(z_t_sample)

    z_t_np = z_t_sample.cpu().numpy()
    u_t_np = u_t_sample.cpu().numpy()
    d = z_t_np.shape[1]

    def _plot_scatter(data, space_name):
        # 2D plot of first two dimensions
        if d >= 2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=5)
            ax.set_title(f'{space_name} Space (Epoch {epoch}) - Dims 0,1')
            ax.set_xlabel('Dim 0'); ax.set_ylabel('Dim 1')
            ax.grid(True, linestyle='--', alpha=0.6)
            writer.add_figure(f"FLOW2/Viz/{space_name}_2D", fig, global_step=epoch)
            plt.close(fig)

        # 3D plot of first three dimensions
        if d >= 3:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.5, s=5, depthshade=True)
            ax.set_title(f'{space_name} Space (Epoch {epoch}) - Dims 0,1,2')
            ax.set_xlabel('Dim 0'); ax.set_ylabel('Dim 1'); ax.set_zlabel('Dim 2')
            writer.add_figure(f"FLOW2/Viz/{space_name}_3D", fig, global_step=epoch)
            plt.close(fig)

    _plot_scatter(z_t_np, "Source_Z_t")
    _plot_scatter(u_t_np, "Target_U_t")


# =============================================================================
# small helpers
# =============================================================================
def is_primary() -> bool:
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

@torch.no_grad()
def encode_full_split(
    ae: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    normalize: bool = True,
    desc: Optional[str] = None,
    use_tqdm: bool = True
) -> torch.Tensor:
    """Encode an entire split to latents (optionally normalized) with an optional tqdm bar."""
    Zs = []
    pbar = None
    if desc is None:
        desc = "Encode latents"
    if use_tqdm and is_primary():
        try:
            from tqdm.auto import tqdm as _tqdm
            pbar = _tqdm(total=len(loader), desc=desc, leave=False)
        except Exception:
            pbar = None  # tqdm not available; just run without a bar

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        z = ae.encode(x)
        if normalize and hasattr(ae, "normalize_latent"):
            z = ae.normalize_latent(z)
        Zs.append(z.to(torch.float32).cpu())
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return torch.cat(Zs, 0)


def take_first(ds: TensorDataset, n: int) -> Subset:
    idx = list(range(min(n, len(ds))))
    return Subset(ds, idx)

def _resolve_ae_pair_paths(ae_ckpt_path: str) -> Tuple[str, str]:
    """
    Given an AE checkpoint path (either running or EMA), derive both:
      returns (running_ckpt, ema_ckpt).
    """
    if not ae_ckpt_path.endswith(".pth"):
        ae_ckpt_path += ".pth"
    if ae_ckpt_path.endswith("_EMA.pth"):
        ema_p = ae_ckpt_path
        run_p = ae_ckpt_path.replace("_EMA.pth", ".pth")
    else:
        run_p = ae_ckpt_path
        ema_p = ae_ckpt_path.replace(".pth", "_EMA.pth")
    return run_p, ema_p

def _copy_named_buffers(src: nn.Module, dst: nn.Module):
    """
    Copy buffers (e.g., latent_norm_mean/std) from src → dst by matching names.
    Leaves parameters untouched (EMA will govern parameters).
    """
    src_bufs = dict(src.named_buffers())
    for name, buf in dst.named_buffers():
        if name in src_bufs and src_bufs[name].shape == buf.shape:
            buf.copy_(src_bufs[name])


# =============================================================================
# Rational-Quadratic spline flow (masked coupling), with closed-form logdet
# =============================================================================

def _identity_spline_params(B, M, K, rmin, rmax, device, dtype):
    # Equal bins on [-R, R]
    edges = torch.linspace(rmin, rmax, K + 1, device=device, dtype=dtype)  # (K+1,)
    xk = edges.view(1, 1, K + 1).expand(B, M, K + 1)  # (B,M,K+1)
    yk = xk.clone()                                   # identity map
    dk = torch.ones(B, M, K + 1, device=device, dtype=dtype)  # derivative=1 at knots
    return xk, yk, dk

def _params_to_spline(w_raw, h_raw, d_raw, K, rmin=-3.0, rmax=3.0, min_bin=1e-3, min_deriv=1e-3):
    R = rmax - rmin
    w = F.softmax(w_raw, dim=-1) * (R - K * min_bin) + min_bin
    h = F.softmax(h_raw, dim=-1) * (R - K * min_bin) + min_bin
    xk = F.pad(torch.cumsum(w, dim=-1), (1, 0), value=0.0) + rmin  # (B,M,K+1)
    yk = F.pad(torch.cumsum(h, dim=-1), (1, 0), value=0.0) + rmin
    d  = F.softplus(d_raw) + min_deriv
    return xk, yk, d

def _search_bins(x, knots):  # x:(B,M), knots:(B,M,K+1)
    B, M = x.shape
    K1 = knots.size(-1)
    flat_x = x.contiguous().view(B * M)
    flat_k = knots.contiguous().view(B * M, K1)
    idx = torch.searchsorted(flat_k, flat_x.unsqueeze(-1)).squeeze(-1) - 1
    return idx.clamp_(0, K1 - 2).view(B, M)

# vmap-safe, branch-free forward/inverse with gather rank fix
def _rq_forward_and_logdet(x, xk, yk, dk, eps=1e-12):
    B, M = x.shape
    left, right = xk[..., 0], xk[..., -1]
    bottom, top = yk[..., 0], yk[..., -1]
    inside = (x >= left) & (x <= right)
    y_out = torch.where(x < left, x - left + bottom,
                        torch.where(x > right, x - right + top, torch.zeros_like(x)))
    logabsdx = torch.zeros_like(x)
    x_clamped = torch.max(torch.min(x, right), left)

    def gather(a, i):  # a: (B,M,K+1), i: (B,M)
        return a.gather(-1, i.unsqueeze(-1)).squeeze(-1)

    idx = _search_bins(x_clamped, xk)  # (B,M)

    x0 = gather(xk, idx);  x1 = gather(xk, idx + 1)
    y0 = gather(yk, idx);  y1 = gather(yk, idx + 1)
    d0 = gather(dk, idx);  d1 = gather(dk, idx + 1)

    w = (x1 - x0); h = (y1 - y0)
    s = h / (w + eps)
    t = (x_clamped - x0) / (w + eps)

    A  = s - d0
    Bc = d0
    N  = A * t * t + Bc * t

    Q = (d0 + d1 - 2 * s)
    P = (2 * s - d0 - d1)
    D = s + Q * t + P * t * t

    y_inside = y0 + h * (N / (D + eps))

    Np = 2 * A * t + Bc
    Dp = Q + 2 * P * t
    slope = (h / (w + eps)) * ((Np * D - N * Dp) / ((D + eps) * (D + eps)))
    slope = slope.clamp_min(1e-12)
    y_inside = torch.nan_to_num(y_inside, nan=0.0, posinf=0.0, neginf=0.0)
    log_inside = torch.log(slope)

    y = torch.where(inside, y_inside, y_out)
    logabsdx = torch.where(inside, log_inside, logabsdx)
    return y, logabsdx

def _rq_inverse(y, xk, yk, dk, eps=1e-12):
    B, M = y.shape
    bottom, top = yk[..., 0], yk[..., -1]
    left, right = xk[..., 0], xk[..., -1]
    inside = (y >= bottom) & (y <= top)
    x_out = torch.where(y < bottom, y - bottom + left,
                        torch.where(y > top, y - top + right, torch.zeros_like(y)))
    y_clamped = torch.max(torch.min(y, top), bottom)

    def gather(a, i):
        return a.gather(-1, i.unsqueeze(-1)).squeeze(-1)

    idx = _search_bins(y_clamped, yk)  # (B,M)

    y0 = gather(yk, idx);  y1 = gather(yk, idx + 1)
    x0 = gather(xk, idx);  x1 = gather(xk, idx + 1)
    d0 = gather(dk, idx);  d1 = gather(dk, idx + 1)

    h = (y1 - y0); w = (x1 - x0)
    s = h / (w + eps)
    z = (y_clamped - y0) / (h + eps)

    c  = d0 + d1 - 2 * s
    A  = (-z * c) - s + d0
    Bq = (z * c) - d0
    C  = z * s

    disc = (Bq * Bq - 4 * A * C).clamp_min(0.0)
    sqrt_disc = torch.sqrt(disc + 1e-12)
    t1 = (-Bq + sqrt_disc) / (2 * A + 1e-12)
    t2 = (-Bq - sqrt_disc) / (2 * A + 1e-12)
    t = torch.where((t1 >= 0.0) & (t1 <= 1.0), t1, t2)
    t = torch.clamp(torch.nan_to_num(t, nan=0.5, posinf=0.0, neginf=0.0), 0.0, 1.0)

    x_inside = x0 + w * t
    x = torch.where(inside, x_inside, x_out)
    return x

# =============================================================================
# Flow blocks
# =============================================================================
class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, n_layers=2, act=nn.SiLU):
        super().__init__()
        layers, dim = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(dim, d_hid), act()]
            dim = d_hid
        layers += [nn.Linear(dim, d_out)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

class CouplingRQ(nn.Module):
    def __init__(self, d, mask: torch.Tensor, bins=8, hidden=128, layers=2, rng=3.0):
        super().__init__()
        self.register_buffer("mask", mask)
        self.K, self.range = int(bins), float(rng)
        m = int(mask.sum().item()); u = d - m
        out_per_dim = 2*self.K + (self.K + 1)
        self.cond = MLP(m, hidden, u * out_per_dim, n_layers=layers)

        # alpha blends identity (alpha=0) → learned params (alpha=1)
        self.register_buffer("alpha", torch.tensor(0.0))

    def set_alpha(self, a: float):
        # clamp to [0,1]
        self.alpha.fill_(float(max(0.0, min(1.0, a))))

    def _cond_params(self, x_masked):
        B, m = x_masked.shape
        u = int((~self.mask.bool()).sum().item())
        out = self.cond(x_masked).view(B, u, 2*self.K + (self.K + 1))

        w_raw = out[..., :self.K]
        h_raw = out[..., self.K:2*self.K]
        d_raw = out[..., 2*self.K:]

        # learned params
        xk_l, yk_l, dk_l = _params_to_spline(
            w_raw, h_raw, d_raw, self.K,
            rmin=-self.range, rmax=self.range
        )

        # identity params
        xk_i, yk_i, dk_i = _identity_spline_params(
            B, u, self.K, -self.range, self.range,
            device=x_masked.device, dtype=x_masked.dtype
        )

        # blend: params = (1-alpha)*identity + alpha*learned
        a = self.alpha
        xk = (1 - a) * xk_i + a * xk_l
        yk = (1 - a) * yk_i + a * yk_l
        dk = (1 - a) * dk_i + a * dk_l
        return xk, yk, dk  # (B,u,K+1)


    def forward_and_logdet(self, z):
        m = self.mask.bool()
        zm = z[:, m]                 # (B,m)
        zu = z[:, ~m]                # (B,u)
        xk, yk, dk = self._cond_params(zm)
        yu, logabs = _rq_forward_and_logdet(zu, xk, yk, dk)
        out = z.clone()
        out[:, m]  = zm
        out[:, ~m] = yu
        logdet = logabs.sum(dim=-1)  # (B,)
        return out, logdet

    def inverse(self, y):
        m = self.mask.bool()
        ym = y[:, m]
        yu = y[:, ~m]
        xk, yk, dk = self._cond_params(ym)
        xu = _rq_inverse(yu, xk, yk, dk)
        out = y.clone()
        out[:, m]  = ym
        out[:, ~m] = xu
        return out

class CouplingAffine(nn.Module):
    def __init__(self, d, mask, hidden=128, layers=2, scale_clip=5.0):
        super().__init__()
        self.register_buffer("mask", mask); self.scale_clip=float(scale_clip)
        m = int(mask.sum().item()); u = d - m
        self.nn = MLP(m, hidden, 2*u, n_layers=layers)

    def forward_and_logdet(self, z):
        m = self.mask.bool()
        zm = z[:, m]
        params = self.nn(zm)
        B, u2 = params.shape; u = u2 // 2
        s, t = params[:, :u], params[:, u:]
        s = torch.tanh(s).clamp(-self.scale_clip, self.scale_clip)
        zu = z[:, ~m]
        yu = zu * torch.exp(s) + t
        out = z.clone(); out[:, m] = zm; out[:, ~m] = yu
        logdet = s.sum(dim=-1)
        return out, logdet

    def inverse(self, y):
        m = self.mask.bool()
        ym = y[:, m]
        params = self.nn(ym)
        B, u2 = params.shape; u = u2 // 2
        s, t = params[:, :u], params[:, u:]
        s = torch.tanh(s).clamp(-self.scale_clip, self.scale_clip)
        yu = y[:, ~m]
        xu = (yu - t) * torch.exp(-s)
        out = y.clone(); out[:, m]  = ym; out[:, ~m] = xu
        return out

class Flow(nn.Module):
    def __init__(self, d, n_blocks=6, hidden=128, layers=2, transform="rq", bins=8, range_=3.0):
        super().__init__()
        masks = []
        for i in range(n_blocks):
            mask = torch.zeros(d)
            mask[i % d::2] = 1.0
            masks.append(mask)
        blocks: List[nn.Module] = []
        for mask in masks:
            if transform == "rq":
                blocks.append(CouplingRQ(d, mask, bins=bins, hidden=hidden, layers=layers, rng=range_))
            else:
                blocks.append(CouplingAffine(d, mask, hidden=hidden, layers=layers))
        self.blocks = nn.ModuleList(blocks)
        print("[Flow] initialized to identity (per-block coupling nets)")

    def set_alpha(self, a: float):
        for b in self.blocks:
            if hasattr(b, "set_alpha"):
                b.set_alpha(a)

    def forward(self, z):
        y = z
        for b in self.blocks:
            y, _ = b.forward_and_logdet(y)
        return y

    def forward_with_logdet(self, z):
        y, acc = z, torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        for b in self.blocks:
            y, ld = b.forward_and_logdet(y)
            acc = acc + ld
        return y, acc

    def inverse(self, u):
        x = u
        for b in reversed(self.blocks):
            x = b.inverse(x)
        return x


# =============================================================================
# Latent diffusion loader (EMA-aware) and score interface @ time t
# =============================================================================

def resolve_latent_checkpoint(
    *,
    latent_config_path: str,
    ae_cfg,                      # the AE config object you already loaded
    ae_cfg_path: str,            # path to the AE config file (string)
    explicit_ckpt: Optional[str] = None,
    extra_candidates: Optional[List[str]] = None,
) -> str:
    """
    Resolve the latent diffusion checkpoint path robustly.
    """
    from configs import load_config  # local import to avoid cycles at import time

    def _norm_pth(name: str) -> str:
        return name if name.endswith(".pth") else (name + ".pth")

    def _pair_names(basename: str) -> List[str]:
        names = [basename]
        if basename.endswith("_EMA.pth"):
            names.append(basename.replace("_EMA.pth", ".pth"))
        elif basename.endswith(".pth"):
            names.append(basename.replace(".pth", "_EMA.pth"))
        else:
            names.append(basename + "_EMA.pth")
        return list(dict.fromkeys(names))  # dedupe

    if explicit_ckpt:
        if os.path.isabs(explicit_ckpt) and os.path.exists(explicit_ckpt):
            return explicit_ckpt
        rel_try = os.path.abspath(explicit_ckpt)
        if os.path.exists(rel_try):
            return rel_try

    lat_cfg = load_config(latent_config_path)
    lat_ckpt_name = _norm_pth(getattr(lat_cfg.model, "checkpoint", "LatentDiff_last.pth"))

    if os.path.isabs(lat_ckpt_name):
        if os.path.exists(lat_ckpt_name):
            return lat_ckpt_name
        raise FileNotFoundError(f"Latent checkpoint not found at absolute path: {lat_ckpt_name}")

    candidates_dirs: List[str] = []
    if getattr(lat_cfg, "checkpoint_dir", None):
        candidates_dirs.append(lat_cfg.checkpoint_dir)
    if getattr(ae_cfg, "checkpoint_dir", None):
        candidates_dirs.append(ae_cfg.checkpoint_dir)
    candidates_dirs.append(os.path.dirname(os.path.abspath(ae_cfg_path)))
    candidates_dirs.append(os.path.dirname(os.path.abspath(latent_config_path)))
    candidates_dirs.append(os.getcwd())
    if extra_candidates:
        candidates_dirs.extend(extra_candidates)

    tried = []
    for d in candidates_dirs:
        for fname in _pair_names(os.path.basename(lat_ckpt_name)):
            p = os.path.abspath(os.path.join(d, fname))
            tried.append(p)
            if os.path.exists(p):
                return p

    tried_str = "\n  - ".join(tried)
    raise FileNotFoundError(
        "Could not resolve latent diffusion checkpoint.\n"
        f"Looked for '{lat_ckpt_name}' and its EMA/non-EMA pair in:\n  - {tried_str}"
    )

def load_latent_diffusion_and_sde(
    *,
    ae_latent_dim: int,
    device: torch.device,
    latent_config: str,
    latent_ckpt: Optional[str],
) -> tuple[nn.Module, object]:
    lat_cfg = load_config(latent_config)
    lat_cfg.training.device = str(device)

    # Align dims (safety in case of mismatch)
    if getattr(lat_cfg.data, "latent_dim", None) != ae_latent_dim:
        lat_cfg.data.latent_dim = ae_latent_dim
        lat_cfg.data.shape = [ae_latent_dim]
        lat_cfg.model.state_size = ae_latent_dim

    model = get_model(lat_cfg.model).to(device)
    sde = configure_sde(lat_cfg)

    ckpt = latent_ckpt or getattr(lat_cfg.model, "checkpoint", None) or "LatentDiff_last.pth"
    if not ckpt.endswith(".pth"):
        ckpt += ".pth"
    if not os.path.isabs(ckpt) and getattr(lat_cfg, "checkpoint_dir", None):
        ckpt = os.path.join(lat_cfg.checkpoint_dir, os.path.basename(ckpt))

    ema = EMA(model, decay=float(getattr(lat_cfg.model, "ema_decay", 0.999)))
    if os.path.exists(ckpt):
        is_ema = ckpt.endswith("_EMA.pth")
        load_model(model, ema, ckpt, "LatentDiff", device=device, is_ema=is_ema)
        if is_ema:
            ema.apply_shadow()
        print(f"[LatentDiff] loaded '{ckpt}' (EMA={is_ema})")
    else:
        raise FileNotFoundError(f"latent diffusion checkpoint not found: {ckpt}")

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, sde


def make_score_fn(model: nn.Module, sde, t_scalar: float):
    base = model.get_score_fn(sde)  # expects (x, y, t)
    t_val = float(t_scalar)
    def s_fixed(x: torch.Tensor) -> torch.Tensor:
        t = x.new_full((x.size(0),), t_val)
        return base(x, None, t)
    return s_fixed


# =============================================================================
# Geometry bits — score-only losses (MSM & CONN)
# =============================================================================
def isometry_forward_loss(phi: nn.Module, z: torch.Tensor, num_v: int = 8, device="cpu"):
    return decoder_isometry_regularisation(phi, z, num_v=num_v, train=True, device=device)

def _as_single_map(f_batched):
    def f_single(u1d: torch.Tensor) -> torch.Tensor:
        y = f_batched(u1d.unsqueeze(0))
        return y.reshape(1, -1).squeeze(0)
    return f_single

def _chol_solve_batch(L: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    add_dim = (b.dim() == 2)
    if add_dim: b = b.unsqueeze(-1)
    x = torch.cholesky_solve(b, L)
    return x.squeeze(-1) if add_dim else x

def _solve_tri_lower(L: torch.Tensor, B_: torch.Tensor) -> torch.Tensor:
    add_dim = (B_.dim() == 2)
    if add_dim: B_ = B_.unsqueeze(-1)
    out = torch.linalg.solve_triangular(L, B_, upper=False)
    return out.squeeze(-1) if add_dim else out

def _solve_tri_upper(U: torch.Tensor, B_: torch.Tensor) -> torch.Tensor:
    add_dim = (B_.dim() == 2)
    if add_dim: B_ = B_.unsqueeze(-1)
    out = torch.linalg.solve_triangular(U, B_, upper=True)
    return out.squeeze(-1) if add_dim else out

def _rademacher(shape, device, dtype):
    return (torch.randint(0, 2, shape, device=device) * 2 - 1).to(dtype)

def _build_JGL_batched(s_fn, u: torch.Tensor, *, chol_abs: float = 1e-6, chol_rel: float = 1e-3):
    f_single = _as_single_map(s_fn)
    jac = jacrev(f_single)
    J = vmap(jac)(u)                                # (B,m,m)
    # Guard: if J has NaNs, bail early
    if not _finite_or_nan(J):
        raise RuntimeError("NaNs in Jacobian")

    G = torch.einsum('bim,bjm->bij', J, J)          # (B,m,m)
    # Very small negative eigenvalues from roundoff can happen; force symmetry
    G = 0.5 * (G + G.transpose(1, 2))

    L, used_abs, used_rel = _safe_cholesky(G, abs_jitter=chol_abs, rel_jitter=chol_rel)
    return J, G, L


def metric_smoothness_score_only(
    *, s_fn, u: torch.Tensor, K_w: int = 1,
    use_rademacher: bool = True, use_exact_hessian: bool = True,
    fd_eps: float = 1e-3, normalize_by_dim: bool = True, chol_reg_abs=1e-6, chol_reg_rel=1e-3
) -> torch.Tensor:
    """
    MSM(u) = E_w || L^{-1} (∂_w G) L^{-T} ||_F^2 ,  G = J^T J
    Returns (B,)
    """
    B, m = u.shape
    f_single = _as_single_map(s_fn)
    with torch.cuda.amp.autocast(enabled=False):
        u = u.float()
        J, G, L = _build_JGL_batched(s_fn, u, chol_abs=chol_reg_abs, chol_rel=chol_reg_rel)

        if use_rademacher:
            W = _rademacher((K_w, B, m), u.device, torch.float32)
            S = _rademacher((K_w, B, m), u.device, torch.float32)
        else:
            W = torch.randn(K_w, B, m, device=u.device, dtype=torch.float32)
            S = torch.randn(K_w, B, m, device=u.device, dtype=torch.float32)

        def H_mixed(uu, y, w):
            if use_exact_hessian:
                g = lambda z: jvp(f_single, (z,), (y,))[1]
                return jvp(g, (uu,), (w,))[1]  # (m,)
            else:
                eps = fd_eps
                return (jvp(f_single, (uu + eps*w,), (y,))[1] - jvp(f_single, (uu,), (y,))[1]) / eps

        def one_pair(w_row, s_row):
            y = _solve_tri_upper(L.transpose(1, 2), s_row)          # (B,m) = L^{-T}s
            a = torch.einsum('bij,bj->bi', J, y)                    # (B,m) = J y

            def c_single(ui, ai, wi):
                def g_local(z): return (ai * f_single(z)).sum()
                grad_g = grad(g_local)
                return jvp(grad_g, (ui,), (wi,))[1]
            c = vmap(c_single)(u, a, w_row)                         # (B,m)

            b  = vmap(H_mixed, in_dims=(0,0,0))(u, y, w_row)        # (B,m) = D^2 s[u][y,w]
            JT_b = torch.einsum('bij,bj->bi', J.transpose(1,2), b)  # (B,m)

            dGy = c + JT_b                                          # (B,m)
            uvec = _solve_tri_lower(L, dGy)                         # L^{-1} dG[w] y
            return (uvec * uvec).sum(dim=-1)                        # (B,)

        vals = vmap(one_pair, in_dims=(0,0))(W, S)                  # (K_w,B)
        out = vals.mean(dim=0)                                      # (B,)
        if normalize_by_dim: out = out / (m * m)
    return out

def _dG_times_vec_single(f_single, ui, Ji, v, w):
    """d(G v)[w] = (dJ[w])^T (Jv) + J^T (dJ[w] v)"""
    a = Ji @ v
    def g_local(z): return (a * f_single(z)).sum()
    grad_g = grad(g_local)
    term1 = jvp(grad_g, (ui,), (w,))[1]
    def jv_local(z): return jvp(f_single, (z,), (v,))[1]
    dv = jvp(jv_local, (ui,), (w,))[1]
    term2 = Ji.transpose(0,1) @ dv
    return term1 + term2

def connection_norm_score_only(
    *, s_fn, u: torch.Tensor, K_pairs: int = 1,
    use_rademacher: bool = True, normalize_by_dim: bool = True,
    chol_reg_abs=1e-6, chol_reg_rel=1e-3
) -> torch.Tensor:
    """
    CONN(u) ≈ E_{a,b} || Γ(a,b) ||_G^2, surrogate with fp32 linalg.
    Returns (B,)
    """
    B, m = u.shape
    with torch.cuda.amp.autocast(enabled=False):
        u = u.float()
        f_single = _as_single_map(s_fn)
        J, G, L = _build_JGL_batched(s_fn, u, chol_abs=chol_reg_abs, chol_rel=chol_reg_rel)

        if use_rademacher:
            A  = _rademacher((K_pairs, B, m), u.device, torch.float32)
            Bv = _rademacher((K_pairs, B, m), u.device, torch.float32)
        else:
            A  = torch.randn(K_pairs, B, m, device=u.device, dtype=torch.float32)
            Bv = torch.randn(K_pairs, B, m, device=u.device, dtype=torch.float32)

        def one_pair(a_row, b_row):
            dGb_a = vmap(_dG_times_vec_single, in_dims=(None,0,0,0,0))(f_single, u, J, b_row, a_row)
            dGa_b = vmap(_dG_times_vec_single, in_dims=(None,0,0,0,0))(f_single, u, J, a_row, b_row)
            rhs = 0.5 * (dGb_a + dGa_b)
            v = _chol_solve_batch(L, rhs)     # G^{-1} rhs
            Lv = _solve_tri_lower(L, v)       # L v
            return (Lv * Lv).sum(dim=-1)

        vals = vmap(one_pair, in_dims=(0,0))(A, Bv)  # (K_pairs,B)
        out = vals.mean(dim=0)
        if normalize_by_dim: out = out / m
    return out


# =============================================================================
# Training loop
# =============================================================================
def train_flow_step2(cfg):
    # device
    device = torch.device(cfg.training.device if (cfg.training.device != "cuda" or torch.cuda.is_available()) else "cpu")
    logdir = getattr(cfg, "tensorboard_dir", "runs/flow_step2_scoregeom")
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir) if is_primary() else None
    if writer:
        add_isometry_compact_layout(writer)

    # 1) Load AE config
    ae_cfg_path = cfg.paths.ae_config
    ae_cfg = load_config(ae_cfg_path)

    # Build AE models
    ae = get_model(ae_cfg.model).to(device)
    ae_run = get_model(ae_cfg.model).to(device)

    # Resolve AE checkpoint path
    ae_ckpt_in = ae_cfg.model.checkpoint
    if not os.path.isabs(ae_ckpt_in) and getattr(ae_cfg, "checkpoint_dir", None):
        ae_ckpt_in = os.path.join(ae_cfg.checkpoint_dir, os.path.basename(ae_ckpt_in))
    run_ckpt, ema_ckpt = _resolve_ae_pair_paths(ae_ckpt_in)

    if not os.path.exists(run_ckpt):
        raise FileNotFoundError(f"[AE] Running checkpoint not found: {run_ckpt}")
    if not os.path.exists(ema_ckpt):
        raise FileNotFoundError(f"[AE] EMA checkpoint not found: {ema_ckpt}")

    if is_primary():
        print(f"[AE] config: {ae_cfg_path}")
        print(f"[AE] running ckpt: {run_ckpt}")
        print(f"[AE] EMA ckpt:     {ema_ckpt}")

    # Load buffers and EMA weights
    ema_dummy = EMA(ae_run, decay=float(getattr(ae_cfg.model, "ema_decay", 0.999)))
    load_model(ae_run, ema_dummy, run_ckpt, "AE", device=device, is_ema=False)

    ema_main = EMA(ae, decay=float(getattr(ae_cfg.model, "ema_decay", 0.999)))
    load_model(ae, ema_main, ema_ckpt, "AE", device=device, is_ema=True)
    ema_main.apply_shadow()

    _copy_named_buffers(ae_run, ae)

    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    # Build loaders from AE's data config
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        ae_cfg.data, seed=getattr(ae_cfg, "random_seed", cfg.random_seed),
        distributed=False, rank=0, world_size=1, return_samplers=True
    )

    # 2) Materialize NORMALIZED latents (ẑ) with progress bars
    if is_primary():
        print("[Latents] Encoding normalized latents for train/val/test ...")
        
    # 2) Materialize NORMALIZED latents (ẑ)
    with torch.no_grad():
        z_train = encode_full_split(ae, train_loader, device, normalize=True, desc="Latents/train")
        z_val   = encode_full_split(ae, val_loader,   device, normalize=True, desc="Latents/val")
        z_test  = encode_full_split(ae, test_loader,  device, normalize=True, desc="Latents/test") \
                if test_loader is not None else torch.empty(0, z_train.size(1))

    if is_primary():
        msg = f"[Latents] shapes — train: {tuple(z_train.shape)}, val: {tuple(z_val.shape)}"
        if z_test.numel() > 0:
            msg += f", test: {tuple(z_test.shape)}"
        print(msg)

    d = int(z_train.size(1))
    ds_train = TensorDataset(z_train)
    ds_val   = TensorDataset(z_val)

    # 3) Flow acts on normalized space
    flow = Flow(
        d=d,
        n_blocks=cfg.flow.n_blocks,
        hidden=cfg.flow.hidden,
        layers=cfg.flow.layers,
        transform=str(cfg.flow.transform),
        bins=cfg.flow.bins,
        range_=cfg.flow.range,
    ).to(device).float()  # keep params fp32

    if is_primary():
        print_flow_summary(flow, latent_dim=int(z_train.size(1)), cfg=cfg)

    opt = torch.optim.Adam(
        flow.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        weight_decay=cfg.optim.weight_decay,
        eps=getattr(cfg.optim, "eps", 1e-8),
    )
    train_lat_loader = DataLoader(ds_train, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
    val_lat_loader   = DataLoader(ds_val,   batch_size=cfg.training.batch_size, shuffle=False, drop_last=False)

    # 4) Load latent diffusion (config provides its own checkpoint)
    lat_cfg_path = cfg.paths.latent_config
    resolved_lat_ckpt = resolve_latent_checkpoint(
        latent_config_path=lat_cfg_path,
        ae_cfg=ae_cfg,
        ae_cfg_path=ae_cfg_path,
        explicit_ckpt=None,
        extra_candidates=None,
    )
    lat_model, lat_sde = load_latent_diffusion_and_sde(
        ae_latent_dim=d,
        device=device,
        latent_config=lat_cfg_path,
        latent_ckpt=resolved_lat_ckpt,
    )
    ae.float(); lat_model.float()

    if is_primary():
        _tmp = load_config(lat_cfg_path)
        lat_ckpt = _tmp.model.checkpoint
        if not os.path.isabs(lat_ckpt) and getattr(_tmp, "checkpoint_dir", None):
            lat_ckpt = os.path.join(_tmp.checkpoint_dir, os.path.basename(lat_ckpt))
        print(f"[Latent] config: {lat_cfg_path}")
        print(f"[Latent] checkpoint: {lat_ckpt}")

    # ── PRE-TRAIN: visualize perturbation at score_time (only if d=2 or 3)
    if writer is not None:
        d_lat = int(z_val.size(1))
        if d_lat in (2, 3):
            try:
                t_scalar = float(cfg.latent.score_time)

                # Use a decent-sized sample for a clear picture
                n_vis = min(6000, z_val.size(0))
                z_vis = z_val[:n_vis].to(device)

                # Perturb in normalized space at the chosen time
                with torch.no_grad():
                    z_t_vis = perturb_latents_at_time(z_vis, lat_sde, t_scalar)

                # 1) Plot perturbed cloud alone
                _log_latent_scatter(writer, "Sanity/Perturb/Latents_t", z_t_vis, step=0)

                # 2) Overlay original vs perturbed to see the displacement
                _log_overlay_scatter(
                    writer,
                    tag="Sanity/Perturb/Overlay_Real_vs_t",
                    Z_a=z_vis, Z_b=z_t_vis,
                    labels=("real_norm", f"t={t_scalar:g}"),
                    step=0
                )

                # 3) Displacement stats
                _log_displacement_stats(writer, "Sanity/Perturb", z_vis, z_t_vis, step=0)

                if is_primary():
                    print(f"[Sanity] Logged perturbation at score_time={t_scalar}.")
            except Exception as e:
                print(f"[Sanity] perturbation plotting failed (non-fatal): {e}")
        else:
            if is_primary():
                print(f"[Sanity] Skipping perturb plots: latent dim={d_lat} (only 2D/3D supported).")



    s_z_t = make_score_fn(lat_model, lat_sde, t_scalar=float(cfg.latent.score_time))

    # ===================== Precision setup =====================
    # Preferred: cfg.training.precision in {"fp32","fp16","bf16"}
    prec = str(getattr(cfg.training, "precision", None) or ("bf16" if getattr(cfg.training, "use_bf16", False) else "fp32")).lower()
    if prec not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unknown precision: {prec}")
    use_amp = (prec != "fp32") and torch.cuda.is_available()
    amp_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[prec]
    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if is_primary():
        print(f"[Precision] mode={prec} (AMP={use_amp}, dtype={amp_dtype}) — geometry losses run in fp32")
    # ===========================================================

    global_step = 0
    epochs_total = int(cfg.training.epochs)

    for epoch in range(1, epochs_total + 1):
        flow.train()

        # -------- train progress bar --------
        train_iter = train_lat_loader
        pbar = None
        if is_primary() and _tqdm is not None:
            try:
                pbar = _tqdm(train_lat_loader, total=len(train_lat_loader),
                            desc=f"[FLOW] Epoch {epoch}/{epochs_total}", leave=False)
                train_iter = pbar
            except Exception:
                pbar = None  # fallback silently

        # running averages for pretty postfix
        run_loss = run_iso = run_msm = run_conn = 0.0

        for i, (z_clean_hat,) in enumerate(train_iter):
            alpha_ramp = getattr(cfg.training, "init_alpha", 0.0) + \
             (getattr(cfg.training, "final_alpha", 1.0) - getattr(cfg.training, "init_alpha", 0.0)) * \
             min(1.0, global_step / float(getattr(cfg.training, "alpha_warmup_steps", 2000)))
            flow.set_alpha(alpha_ramp)

            z_clean_hat = z_clean_hat.to(device, non_blocking=True)

            with torch.no_grad():
                z_t_hat = perturb_latents_at_time(z_clean_hat, lat_sde, float(cfg.latent.score_time))

            with autocast_ctx:
                # map to target in normalized space
                u_t = flow(z_t_hat)

                # local isometry at z_t
                loss_iso = isometry_forward_loss(flow, z_t_hat, num_v=min(int(cfg.loss.iso_num_v), d), device=device)

                # ---- GENERAL score in u-space via change-of-variables (functorch-safe) ----
                def score_u(u_in: torch.Tensor) -> torch.Tensor:
                    def inv_only(u):
                        return flow.inverse(u)
                    z_pre, vjp_inv = torch.func.vjp(inv_only, u_in)

                    def f_logdet(z):
                        _, logdet = flow.forward_with_logdet(z)
                        return logdet.sum()
                    grad_g = torch.func.grad(f_logdet)(z_pre)
                    grad_g = torch.nan_to_num(grad_g, nan=0.0, posinf=0.0, neginf=0.0)

                    s_src = s_z_t(z_pre)
                    v = s_src - grad_g
                    return vjp_inv(v)[0]

                # ---- score-only geometry losses (internally upcast to fp32) ----
                msm_vals = metric_smoothness_score_only(
                    s_fn=s_z_t, u=z_t_hat,
                    K_w=max(1, int(cfg.loss.msm.K_w)),
                    use_rademacher=True,
                    use_exact_hessian=bool(cfg.loss.msm.use_exact_hessian),
                    fd_eps=float(cfg.loss.msm.fd_eps),
                    normalize_by_dim=bool(cfg.loss.msm.normalize_by_dim),
                    chol_reg_abs=float(getattr(cfg.loss.msm,  "chol_reg_abs", 1e-6)),
                    chol_reg_rel=float(getattr(cfg.loss.msm,  "chol_reg_rel", 1e-3)),
                )
                conn_vals = connection_norm_score_only(
                    s_fn=s_z_t, u=z_t_hat,
                    K_pairs=max(1, int(cfg.loss.conn.K_pairs)),
                    use_rademacher=bool(cfg.loss.conn.use_rademacher),
                    normalize_by_dim=bool(cfg.loss.conn.normalize_by_dim),
                    chol_reg_abs=float(getattr(cfg.loss.conn, "chol_reg_abs", 1e-6)),
                    chol_reg_rel=float(getattr(cfg.loss.conn, "chol_reg_rel", 1e-3)),
                )
                loss_msm  = msm_vals.mean()
                loss_conn = conn_vals.mean()

                warm  = min(1.0, global_step / int(getattr(cfg.training, "geom_warmup_steps", 1000)))
                w_iso  = float(cfg.loss.w_iso)
                w_msm  = float(cfg.loss.w_msm)  * warm
                w_conn = float(cfg.loss.w_conn) * warm
                loss = w_iso*loss_iso + w_msm*loss_msm + w_conn*loss_conn

            opt.zero_grad(set_to_none=True)
            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
            if getattr(cfg.training, "grad_clip", 0.0) and cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), float(cfg.training.grad_clip))
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            # --- TB logging ---
            if writer and (global_step % int(cfg.training.log_every) == 0):
                writer.add_scalar("FLOW2/train/loss_total", float(loss.item()), global_step)
                writer.add_scalar("FLOW2/train/loss_iso",   float(loss_iso.item()), global_step)
                writer.add_scalar("FLOW2/train/loss_msm",   float(loss_msm.item()), global_step)
                writer.add_scalar("FLOW2/train/loss_conn",  float(loss_conn.item()), global_step)
                writer.add_scalar("FLOW2/train/score_time", float(cfg.latent.score_time), global_step)
                writer.add_scalar("FLOW2/train/alpha", float(alpha_ramp), global_step)

            # --- tqdm postfix (running averages) ---
            if pbar is not None:
                run_loss += float(loss.item())
                run_iso  += float(loss_iso.item())
                run_msm  += float(loss_msm.item())
                run_conn += float(loss_conn.item())
                denom = i + 1
                try:
                    lr = opt.param_groups[0]["lr"]
                except Exception:
                    lr = float("nan")
                pbar.set_postfix({
                    "step":   global_step,
                    "loss":   f"{run_loss/denom:.4f}",
                    "iso":    f"{run_iso/denom:.4f}",
                    "msm":    f"{run_msm/denom:.4f}",
                    "conn":   f"{run_conn/denom:.4f}",
                    "lr":     f"{lr:.2e}",
                })

            global_step += 1

        if pbar is not None:
            pbar.close()

        # ========================================================================
        # ─── VALIDATION LOOP ─────────────────────────────────────────────────────
        # ========================================================================
        flow.eval()
        val_loss_iso, val_loss_msm, val_loss_conn = [], [], []

        val_iter = val_lat_loader
        vpbar = None
        if is_primary() and _tqdm is not None:
            try:
                vpbar = _tqdm(val_lat_loader, total=len(val_lat_loader),
                            desc=f"[FLOW][Val] Epoch {epoch}/{epochs_total}", leave=False)
                val_iter = vpbar
            except Exception:
                vpbar = None

        # freeze a single alpha for the whole validation pass
        alpha_val = (
            getattr(cfg.training, "init_alpha", 0.0) +
            (getattr(cfg.training, "final_alpha", 1.0) - getattr(cfg.training, "init_alpha", 0.0)) *
            min(1.0, global_step / float(getattr(cfg.training, "alpha_warmup_steps", 2000)))
        )
        flow.set_alpha(alpha_val)

        with torch.no_grad():
            for i, (z_clean_hat_val,) in enumerate(val_iter):
                z_clean_hat_val = z_clean_hat_val.to(device, non_blocking=True)
                z_t_hat_val = perturb_latents_at_time(z_clean_hat_val, lat_sde, float(cfg.latent.score_time))

                with autocast_ctx:
                    u_t_val = flow(z_t_hat_val)

                    # 1) isometry
                    loss_iso_val = isometry_forward_loss(flow, z_t_hat_val, num_v=min(int(cfg.loss.iso_num_v), d), device=device)
                    val_loss_iso.append(loss_iso_val.item())

                    # 2) MSM & CONN
                    def score_u_val(u_in: torch.Tensor) -> torch.Tensor:
                        def inv_only(u):
                            return flow.inverse(u)
                        z_pre, vjp_inv = torch.func.vjp(inv_only, u_in)
                        def f_logdet(z):
                            _, logdet = flow.forward_with_logdet(z)
                            return logdet.sum()
                        grad_g = torch.func.grad(f_logdet)(z_pre)
                        grad_g = torch.nan_to_num(grad_g, nan=0.0, posinf=0.0, neginf=0.0)
                        s_src = s_z_t(z_pre)
                        v = s_src - grad_g
                        return vjp_inv(v)[0]

                    msm_vals_val = metric_smoothness_score_only(
                        s_fn=score_u_val, u=u_t_val, K_w=max(1, int(cfg.loss.msm.K_w)),
                        chol_reg_abs=float(getattr(cfg.loss.msm, "chol_reg_abs", 1e-6)),
                        chol_reg_rel=float(getattr(cfg.loss.msm, "chol_reg_rel", 1e-3)),
                    )
                    conn_vals_val = connection_norm_score_only(
                        s_fn=score_u_val, u=u_t_val, K_pairs=max(1, int(cfg.loss.conn.K_pairs)),
                        chol_reg_abs=float(getattr(cfg.loss.conn, "chol_reg_abs", 1e-6)),
                        chol_reg_rel=float(getattr(cfg.loss.conn, "chol_reg_rel", 1e-3)),
                    )
                    val_loss_msm.append(msm_vals_val.mean().item())
                    val_loss_conn.append(conn_vals_val.mean().item())

                if vpbar is not None:
                    denom = i + 1
                    vpbar.set_postfix({
                        "iso":  f"{(sum(val_loss_iso)/denom):.4f}",
                        "msm":  f"{(sum(val_loss_msm)/denom):.4f}",
                        "conn": f"{(sum(val_loss_conn)/denom):.4f}",
                    })

        if vpbar is not None:
            vpbar.close()

        # --- TensorBoard epoch aggregates ---
        if writer:
            avg_iso = sum(val_loss_iso) / max(1, len(val_loss_iso))
            avg_msm = sum(val_loss_msm) / max(1, len(val_loss_msm))
            avg_conn = sum(val_loss_conn) / max(1, len(val_loss_conn))
            avg_total = (float(cfg.loss.w_iso) * avg_iso +
                        float(cfg.loss.w_msm) * avg_msm +
                        float(cfg.loss.w_conn) * avg_conn)

            writer.add_scalar("FLOW2/val/loss_total", avg_total, epoch)
            writer.add_scalar("FLOW2/val/loss_iso",   avg_iso, epoch)
            writer.add_scalar("FLOW2/val/loss_msm",   avg_msm, epoch)
            writer.add_scalar("FLOW2/val/loss_conn",  avg_conn, epoch)

        # --- Visualization Call ---
        if writer and epoch % int(cfg.training.viz_every) == 0:
            visualize_z_and_u_spaces(
                ds_val.tensors[0], flow, lat_sde, writer, device,
                epoch=epoch, t_scalar=float(cfg.latent.score_time)
            )

        # checkpoint
        if (epoch % int(cfg.training.ckpt_every) == 0) or (epoch == int(cfg.training.epochs)):
            ckpt = {"flow": flow.state_dict(), "epoch": epoch}
            ckpt_dir = getattr(cfg, "checkpoint_dir", logdir)
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"flow_step2_e{epoch:03d}.pt")
            torch.save(ckpt, path)
            if is_primary():
                print(f"[ckpt] saved → {path}")

    if writer:
        writer.close()



# =============================================================================
# CLI (minimal)
# =============================================================================
def main():
    P = argparse.ArgumentParser("Step-2: latent→latent flow regularized by score-only geometry (MSM + CONN)")
    P.add_argument("--config", required=True, help="path to ml_collections .py flow config")
    args = P.parse_args()

    cfg = load_config(args.config)

    # Allow TF32 and bf16 as requested in cfg
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # If the config uses legacy flags, map to precision (default fp32)
    if not hasattr(cfg.training, "precision"):
        if getattr(cfg.training, "use_bf16", False):
            cfg.training.precision = "bf16"
        else:
            cfg.training.precision = "fp32"

    train_flow_step2(cfg)

if __name__ == "__main__":
    main()
