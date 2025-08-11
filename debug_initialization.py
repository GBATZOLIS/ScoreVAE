#!/usr/bin/env python
"""
DDIM-based path initialization debugger (VPSDE).

Exports (for reuse):
- flatten, unflatten
- get_score_fn
- save_path_grid
- solve_forward_ddim_to
- solve_reverse_ddim_to
- generate_ode_initialized_path

CLI (optional):
- debug_init_paths(...) and unconditional_ddim_samples(...)
"""
from __future__ import annotations
import os, math, argparse
from typing import Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Optional project imports used only by the CLI `main()` below.
from configs import load_config
from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import EMA, load_model, prepare_training_dirs

Tensor = torch.Tensor
Model = torch.nn.Module
SDE = Any

# ---------------------------------------------------------------------
# Shapes & simple utils
# ---------------------------------------------------------------------
def flatten(x: Tensor) -> Tensor:
    return x.view(x.size(0), -1)

def unflatten(x: Tensor, shp: Tuple[int, ...]) -> Tensor:
    return x.view(x.size(0), *shp)

def _as_time(t: Tensor, B: int) -> Tensor:
    return t.expand(B) if t.dim() == 0 else t

def _to_uint8_img(x: Tensor) -> np.ndarray:
    """x: (C,H,W) in [-1,1] or [0,1] -> uint8 image."""
    x = x.detach().cpu()
    if x.dim() == 2:
        arr = x
    elif x.size(0) == 1:
        arr = x[0]
    else:
        arr = x.permute(1, 2, 0)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    arr = arr.clamp(0, 1).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)

def save_path_grid(path_nodes: List[Tensor], orig_shape: Tuple[int, ...], save_path: str, max_rows: int | None = None):
    """Path nodes: list of K tensors, each (B_pairs, D). Save as a BxK image grid."""
    K = len(path_nodes)
    B_pairs = path_nodes[0].size(0)
    if max_rows is not None:
        B_pairs = min(B_pairs, max_rows)
    imgs_per_node = [unflatten(p[:B_pairs], orig_shape).cpu() for p in path_nodes]  # K x (B,C,H,W)
    rows = []
    for b in range(B_pairs):
        row_imgs = [imgs_per_node[k][b] for k in range(K)]  # list of (C,H,W)
        rows.append(torch.cat(row_imgs, dim=-1))
    grid = torch.cat(rows, dim=-2)

    arr = _to_uint8_img(grid)
    plt.figure(figsize=(max(3, K) * 1.2, max(2, B_pairs) * 1.2))
    if arr.ndim == 2:
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(arr)
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved path grid to {save_path}")

# ---------------------------------------------------------------------
# Score & DDIM machinery (VPSDE)
# ---------------------------------------------------------------------
def get_score_fn(sde: SDE, model: Model, shape: Tuple[int, ...]):
    model.eval()
    base_score = model.get_score_fn(sde)  # (x_unflat, y, t) -> score

    def score_fn(x_flat: Tensor, t: Tensor) -> Tensor:
        B = x_flat.size(0)
        t_batch = _as_time(t, B)
        x_unflat = unflatten(x_flat, shape)
        with torch.no_grad():
            s_unflat = base_score(x_unflat, None, t_batch)
        return flatten(s_unflat)
    return score_fn

def _score_unflat(sde: SDE, model: Model, x_unflat: Tensor, t_batch: Tensor) -> Tensor:
    with torch.no_grad():
        eps = model(x_unflat, None, t_batch)
    B = x_unflat.size(0)
    _, std = sde.marginal_prob(x_unflat, t_batch)  # (B,)
    std = std.view(B, *([1] * (x_unflat.dim() - 1)))
    return -eps / std

def _slerp(z0: Tensor, z1: Tensor, lam: Tensor, eps: float = 1e-8) -> Tensor:
    """Batched SLERP in R^D. z0,z1: (B_pairs,D); lam: (K,) -> (B_pairs,K,D)"""
    z0n = z0 / (z0.norm(dim=1, keepdim=True).clamp_min(eps))
    z1n = z1 / (z1.norm(dim=1, keepdim=True).clamp_min(eps))
    dot = (z0n * z1n).sum(dim=1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dot)
    sin_th = torch.sin(theta).clamp_min(eps)

    lam = lam.view(1, -1, 1).to(z0)
    theta = theta.unsqueeze(1)
    sin_th = sin_th.unsqueeze(1)

    s0 = torch.sin((1 - lam) * theta) / sin_th
    s1 = torch.sin(lam * theta) / sin_th
    return s0 * z0.unsqueeze(1) + s1 * z1.unsqueeze(1)

def _lift_to_t_sampled(x0_flat: Tensor, orig_shape: Tuple[int, ...], sde: SDE, t_scalar: float) -> Tuple[Tensor, Tensor]:
    x0_un = unflatten(x0_flat, orig_shape)
    B = x0_un.size(0)
    t_batch = torch.full((B,), float(t_scalar), device=x0_un.device, dtype=x0_un.dtype)
    mean, std = sde.marginal_prob(x0_un, t_batch)
    std = std.view(B, *([1] * (x0_un.ndim - 1)))
    z = torch.randn_like(x0_un)
    x_t_un = mean + std * z
    return flatten(x_t_un), z

def _ddim_step_unflat(sde: SDE, model: Model, x_unflat: Tensor, t_scalar: Tensor, s_scalar: Tensor) -> Tensor:
    """Deterministic DDIM update x_t -> x_s using ε-prediction with VPSDE α,σ."""
    t_scalar = torch.as_tensor(t_scalar, device=x_unflat.device, dtype=x_unflat.dtype)
    s_scalar = torch.as_tensor(s_scalar, device=x_unflat.device, dtype=x_unflat.dtype)

    a_t, sig_t = sde.perturbation_coefficients(t_scalar)
    a_s, sig_s = sde.perturbation_coefficients(s_scalar)

    B = x_unflat.size(0)
    t_batch = torch.full((B,), float(t_scalar.item()), device=x_unflat.device, dtype=x_unflat.dtype)

    score = _score_unflat(sde, model, x_unflat, t_batch)
    x0_hat = (sig_t ** 2) * score + x_unflat
    x0_hat = x0_hat / a_t

    x_s = (sig_s / sig_t) * x_unflat + (a_s - (sig_s / sig_t) * a_t) * x0_hat
    return x_s

def _ddim_evolve_flat(x_flat: Tensor, orig_shape: Tuple[int, ...], sde: SDE, model: Model, t_grid: Tensor) -> Tensor:
    x = x_flat
    with torch.no_grad():
        for i in tqdm(range(len(t_grid) - 1), desc="[DDIM] evolve", leave=False):
            t_i, t_ip1 = t_grid[i], t_grid[i + 1]
            x_un = unflatten(x, orig_shape)
            x_un_next = _ddim_step_unflat(sde, model, x_un, t_i, t_ip1)
            x = flatten(x_un_next)
    return x

def solve_forward_ddim_to(x0_flat: Tensor, orig_shape: Tuple[int, ...],
                          sde: SDE, model: Model, t_target: float, steps: int, t0: float) -> Tensor:
    """Encode: 0 -> t_target with a sampled x_{t0} start, then DDIM to t_target."""
    device, dtype = x0_flat.device, x0_flat.dtype
    x_t0_flat, _ = _lift_to_t_sampled(x0_flat, orig_shape, sde, t0)
    t_grid = torch.linspace(torch.as_tensor(t0, device=device, dtype=dtype),
                            torch.as_tensor(t_target, device=device, dtype=dtype),
                            steps + 1, device=device, dtype=dtype)
    return _ddim_evolve_flat(x_t0_flat, orig_shape, sde, model, t_grid)

def solve_reverse_ddim_to(x_flat: Tensor, orig_shape: Tuple[int, ...],
                          sde: SDE, model: Model, t_target: float, steps: int) -> Tensor:
    """Decode: from T -> t_target deterministically with DDIM."""
    device, dtype = x_flat.device, x_flat.dtype
    t_grid = torch.linspace(sde.T, t_target, steps + 1, device=device, dtype=dtype)
    return _ddim_evolve_flat(x_flat, orig_shape, sde, model, t_grid)

def generate_ode_initialized_path(
    p: Tensor, q: Tensor, sde: SDE, model: Model,
    orig_shape: tuple, n_segments: int, end_time: float,
    num_solver_steps: int, t0: float, device: torch.device
) -> List[Tensor]:
    """Encode p,q → T with DDIM (starting at t0), SLERP, decode nodes to end_time with DDIM."""
    print("[Initializer/DDIM] Encoding endpoints to latents at T...")
    p_latent = solve_forward_ddim_to(p, orig_shape, sde, model, t_target=sde.T, steps=num_solver_steps, t0=t0)
    q_latent = solve_forward_ddim_to(q, orig_shape, sde, model, t_target=sde.T, steps=num_solver_steps, t0=t0)
    with torch.no_grad():
        print(f"[Debug] MSE(p_latent, q_latent) = {torch.mean((p_latent - q_latent)**2).item():.4e}")

    B = p.size(0)
    K = n_segments + 1
    lam = torch.linspace(0, 1, K, device=device, dtype=p.dtype)
    lat_path = _slerp(p_latent, q_latent, lam)         # (B,K,D)
    x_all = lat_path.reshape(B * K, -1)                # (B*K, D)

    x = solve_reverse_ddim_to(x_all, orig_shape, sde, model, t_target=end_time, steps=num_solver_steps)
    final = x.view(B, K, -1).permute(1, 0, 2)
    path = [final[k].detach().clone().requires_grad_(k not in {0, K-1}) for k in range(K)]

    with torch.no_grad():
        p_to_t = solve_forward_ddim_to(p, orig_shape, sde, model, t_target=end_time, steps=num_solver_steps, t0=t0)
        q_to_t = solve_forward_ddim_to(q, orig_shape, sde, model, t_target=end_time, steps=num_solver_steps, t0=t0)
        e_p = torch.mean((path[0]  - p_to_t)**2).item()
        e_q = torch.mean((path[-1] - q_to_t)**2).item()
        print(f"[Debug/DDIM] endpoint consistency vs encode→t={end_time:g}: "
              f"start MSE={e_p:.4e}, end MSE={e_q:.4e}")

    print("[Initializer/DDIM] Deterministic path generated.")
    return path

# ---------------------------------------------------------------------
# Debug-only pipeline/CLI
# ---------------------------------------------------------------------
def debug_init_paths(model: Model, sde: SDE, images: Tensor, *,
                     segments: int, steps: int, t0: float, t_end: float, save_dir: str):
    device = images.device
    B_imgs, C, H, W = images.shape
    orig_shape = (C, H, W)

    if B_imgs % 2 == 1:
        images = images[:B_imgs - 1]
        B_imgs -= 1
    p_img, q_img = images.chunk(2)
    B_pairs = p_img.size(0)
    p_flat, q_flat = flatten(p_img), flatten(q_img)

    print(f"[Debug] images={B_imgs} -> pairs(B)={B_pairs}, K={segments+1}")
    print("[Debug] MSE(p0, q0):", torch.mean((p_flat - q_flat) ** 2).item())

    print("[Init/DDIM] Encoding endpoints to T…")
    p_lat = solve_forward_ddim_to(p_flat, orig_shape, sde, model, t_target=sde.T, steps=steps, t0=t0)
    q_lat = solve_forward_ddim_to(q_flat, orig_shape, sde, model, t_target=sde.T, steps=steps, t0=t0)

    K = segments + 1
    lam = torch.linspace(0, 1, K, device=device, dtype=images.dtype)
    lat_path = _slerp(p_lat, q_lat, lam)
    x_decoded = solve_reverse_ddim_to(lat_path.reshape(B_pairs * K, -1), orig_shape, sde, model, t_target=t_end, steps=steps)
    final = x_decoded.view(B_pairs, K, -1)

    with torch.no_grad():
        p_to_t = solve_forward_ddim_to(p_flat, orig_shape, sde, model, t_target=t_end, steps=steps, t0=t0)
        q_to_t = solve_forward_ddim_to(q_flat, orig_shape, sde, model, t_target=t_end, steps=steps, t0=t0)
        e_p = torch.mean((final[:, 0, :] - p_to_t) ** 2).item()
        e_q = torch.mean((final[:, -1, :] - q_to_t) ** 2).item()
        print(f"[Debug] endpoint MSE vs direct encode→t_end: start={e_p:.4e}, end={e_q:.4e}")

    path_nodes = [final[:, k, :] for k in range(K)]
    os.makedirs(save_dir, exist_ok=True)
    save_path_grid(path_nodes, orig_shape, os.path.join(save_dir, "initialized_paths.png"))
    return path_nodes, orig_shape

def unconditional_ddim_samples(model: Model, sde: SDE, shape: Tuple[int, int, int, int], *,
                               n: int, steps: int, t_end: float, save_path: str):
    C, H, W = shape[1:]
    xT = torch.randn(n, C, H, W, device=next(model.parameters()).device)
    x_flat = flatten(xT)
    out_flat = solve_reverse_ddim_to(x_flat, (C, H, W), sde, model, t_target=t_end, steps=steps)
    imgs = unflatten(out_flat, (C, H, W))

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pads = rows * cols - n
    if pads > 0:
        imgs = torch.cat([imgs, torch.zeros(pads, C, H, W, device=imgs.device)], dim=0)
    rows_list = []
    for r in range(rows):
        rows_list.append(torch.cat([imgs[r * cols + c] for c in range(cols)], dim=-1))
    grid = torch.cat(rows_list, dim=-2)

    arr = _to_uint8_img(grid)
    plt.figure(figsize=(cols * 1.5, rows * 1.5))
    if arr.ndim == 2:
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(arr)
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved unconditional DDIM grid to {save_path}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("DDIM path-initialization debugger (VPSDE)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-pairs", type=int, default=2)
    ap.add_argument("--segments", type=int, default=20)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--t0", type=float, default=1e-3)
    ap.add_argument("--t-end", type=float, default=1e-3)
    ap.add_argument("--save-dir", type=str, default="./results/debug_init")
    ap.add_argument("--uncond", type=int, default=0)
    a = ap.parse_args()

    diff_cfg = load_config(a.config)
    device = torch.device(diff_cfg.training.device)
    _, _, _ = prepare_training_dirs(diff_cfg)

    loader, _, _ = get_dataloaders(diff_cfg.data, seed=42)
    dset = loader.dataset

    model = get_model(diff_cfg.model).to(device)
    ema = EMA(model, diff_cfg.model.ema_decay)

    from collections import defaultdict
    import torch.serialization as ts
    try:
        ts.add_safe_globals([defaultdict, dict])
        ckpt_path = os.path.join(diff_cfg.checkpoint_dir, diff_cfg.model.checkpoint)
        load_model(model, ema, ckpt_path, "Model", device=device, is_ema=True)
    except Exception as e:
        print(f"[warn] Safe load failed ({e}); falling back to full torch.load.")
        load_model(model, ema, ckpt_path, "Model", device=device, is_ema=True)

    ema.apply_shadow(); model.eval()
    sde = configure_sde(diff_cfg)

    Btot = min(2 * a.num_pairs, len(dset))
    idx = torch.randperm(len(dset))[:Btot]
    items = [dset[i] for i in idx]
    if isinstance(items[0], (tuple, list)):
        imgs = torch.stack([it[0] for it in items]).to(device)
    else:
        imgs = torch.stack(items).to(device)

    os.makedirs(a.save_dir, exist_ok=True)
    path_nodes, orig_shape = debug_init_paths(
        model, sde, imgs,
        segments=a.segments, steps=a.steps,
        t0=a.t0, t_end=a.t_end, save_dir=a.save_dir
    )

    if a.uncond > 0:
        C, H, W = orig_shape
        unconditional_ddim_samples(
            model, sde, (a.uncond, C, H, W),
            n=a.uncond, steps=a.steps, t_end=0.0,
            save_path=os.path.join(a.save_dir, "uncond_ddim.png")
        )

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
