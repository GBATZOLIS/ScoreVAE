#!/usr/bin/env python
"""
Batch geodesic computation on a **score‑based data manifold**.
--------------------------------------------------------------------------------
CLI
--------------------------------------------------------------------------------
```bash
python geodesic_batch.py \
    --config      path/to/diffusion_config.py \
    --geo-config  path/to/geodesic_config.py
```

--------------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import time
import random
import numpy as np
import pickle
import torch
import torch.multiprocessing as mp
from argparse import ArgumentParser
from typing import Tuple, Dict, Any

# -----------------------------------------------------------------------------
# Diffusion‑model imports (mirror of optimisation script)
# -----------------------------------------------------------------------------
from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import prepare_training_dirs, EMA, load_model
from configs import load_config

# -----------------------------------------------------------------------------
# Geodesic routine & visualisation
# -----------------------------------------------------------------------------
from data_geometry.geodesics.fast_geodesic_computation import compute_geodesic

from data_geometry.utils.visualization import (
    visualize_riemannian_optimization_selector,
)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    """Flatten all non‑batch dims → (B, d)."""
    return x.view(x.size(0), -1)


def unflatten_tensor(x: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    return x.view(x.size(0), *orig_shape)


def ensure_time_tensor(t: torch.Tensor, B: int):
    if t.dim() == 0:
        return t.expand(B)
    if t.dim() == 1 and t.size(0) == B:
        return t
    return t.expand(B)

# -------------------------- score / denoiser wrappers ------------------------

def get_score_fn(sde, model, t: torch.Tensor, orig_shape):
    sigma_fn = sde.get_sigma_fn()

    def score_fn(x_t: torch.Tensor):
        B = x_t.size(0)
        t_corr = ensure_time_tensor(t, B)
        sigma_t = sigma_fn(t_corr).view(B, 1)
        noise_pred = model(unflatten_tensor(x_t, orig_shape), None, t_corr)
        return -flatten_tensor(noise_pred) / sigma_t

    return score_fn


def get_denoiser_fn(sde, model, t: torch.Tensor, orig_shape):
    alpha_fn, sigma_fn = sde.get_alpha_fn(), sde.get_sigma_fn()

    def denoise_fn(x_t: torch.Tensor):
        B = x_t.size(0)
        t_corr = ensure_time_tensor(t, B)
        alpha_t = alpha_fn(t_corr).view(B, 1)
        sigma_t = sigma_fn(t_corr).view(B, 1)
        noise_pred = model(unflatten_tensor(x_t, orig_shape), None, t_corr)
        return (x_t - sigma_t * flatten_tensor(noise_pred)) / alpha_t

    return denoise_fn

# -----------------------------------------------------------------------------
# Config loader (plain Python file with CONFIG dict)
# -----------------------------------------------------------------------------

def load_geo_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    scope: Dict[str, Any] = {}
    with open(path, "r") as f:
        exec(compile(f.read(), path, "exec"), scope)
    if "CONFIG" not in scope:
        raise ValueError("Geo config must define a CONFIG dict")
    return scope["CONFIG"]

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def geodesic_batch(diff_cfg, geo_cfg):
    device = torch.device(diff_cfg.training.device)
    print(f"[Geodesic‑Batch] using device: {device}")

    # 1. Logging dirs ---------------------------------------------------------
    _, _, eval_dir = prepare_training_dirs(diff_cfg)

    # 2. Dataset -------------------------------------------------------------
    train_loader, _, _ = get_dataloaders(diff_cfg.data, seed=geo_cfg.get("random_seed", 42))
    dataset = train_loader.dataset

    num_pairs = geo_cfg.get("num_pairs", 24)
    if len(dataset) < 2 * num_pairs:
        raise RuntimeError("Dataset too small for requested num_pairs")

    samples = torch.stack([
        dataset[i][0] if isinstance(dataset[i], tuple) else dataset[i]
        for i in range(2 * num_pairs)
    ])
    p_img, q_img = samples[:num_pairs], samples[num_pairs:]
    p_img, q_img = p_img.to(device), q_img.to(device)
    orig_shape = p_img.shape[1:]
    p_flat, q_flat = flatten_tensor(p_img), flatten_tensor(q_img)

    # 3. Model + SDE ---------------------------------------------------------
    model = get_model(diff_cfg.model).to(device)
    ema = EMA(model=model, decay=diff_cfg.model.ema_decay)

    ckpt = diff_cfg.model.checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(diff_cfg.checkpoint_dir, ckpt)
    if not ckpt.endswith(".pth"):
        ckpt += ".pth"
    load_model(model, ema, ckpt, "Model", device=device, is_ema=True)
    ema.apply_shadow()
    model.eval()

    sde = configure_sde(diff_cfg)

    # 4. Diffusion coefficients ---------------------------------------------
    t_pert = torch.tensor(geo_cfg.get("time_for_perturbation", 0.01), device=device)
    alpha_t, sigma_t = sde.get_alpha_fn()(t_pert).item(), sde.get_sigma_fn()(t_pert).item()

    score_fn   = get_score_fn(sde, model, t_pert, orig_shape)
    denoise_fn = get_denoiser_fn(sde, model, t_pert, orig_shape)

    # 5. Jacobian‑CG kwargs (optional) --------------------------------------
    cg_kwargs = None
    if geo_cfg.get("metric_type", "stein").lower() == "jacobian":
        cg_kwargs = dict(
            preconditioner       = geo_cfg.get("cg_preconditioner", "diagonal"),
            precond_diag_samples = geo_cfg.get("cg_precond_diag_samples", 8),
            tol                  = geo_cfg.get("cg_tol", 5e-5),
            max_iter             = geo_cfg.get("cg_max_iter", 4),
        )

    start = time.time()
    path, loss_info = compute_geodesic(
        p_flat,
        q_flat,
        score_fn,
        metric_type   = geo_cfg.get("metric_type", "stein"),
        lam_metric    = geo_cfg.get("lam_metric", 1.0),
        n_segments    = geo_cfg.get("n_segments", 16),
        alpha_t       = alpha_t,
        sigma_t       = sigma_t,
        lam_smooth    = geo_cfg.get("lam_smooth", 1e-3),
        lam_mono      = geo_cfg.get("lam_mono", 0.0),
        adam_lr       = geo_cfg.get("adam_lr", 1e-2),
        max_iters     = geo_cfg.get("max_iters", 500),
        tol           = geo_cfg.get("tol", 1e-6),
        patience      = geo_cfg.get("patience", 30),
        verbose       = True,
        denoise_fn    = denoise_fn,
        jacobian_cg_kwargs = cg_kwargs,
    )

    elapsed = time.time() - start
    print(f"compute_geodesic finished in {elapsed:.2f} s")

    print((
        "┌─ Loss breakdown (averaged over batch) ───────────────────────────────┐\n"
        f"│  total            : {loss_info['total']:.4e}\n"
        f"│  geodesic energy  : {loss_info['geodesic_energy']:.4e}\n"
        f"│  smoothness pen.  : {loss_info['smoothness']:.4e}\n"
        f"│  monotonicity pen.: {loss_info['monotonicity']:.4e}\n"
        f"│  best iteration    : {loss_info['iter']}\n"
        "└───────────────────────────────────────────────────────────────────────┘"
    ))

    # 7. Visualisation background -------------------------------------------
    with torch.no_grad():
        bg_cap = min(1000, len(dataset))
        bg = torch.stack([
            dataset[i][0] if isinstance(dataset[i], tuple) else dataset[i]
            for i in range(bg_cap)
        ]).to(device)
        bg_noised = alpha_t * bg + sigma_t * torch.randn_like(bg)
        bg_flat   = flatten_tensor(bg_noised)

    visualize_riemannian_optimization_selector(
        bg_flat.cpu(),
        lambda x: score_fn(x.to(device)).cpu(),
        t_pert.item(),
        trajectories=[p.cpu() for p in path],
        metrics={},
        min_point=None,
        orig_shape=orig_shape,
        log_dir=eval_dir,
        plot_filename=geo_cfg.get("plot_filename", "geodesics.png"),
    )

    return path

# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    parser = ArgumentParser("Batch geodesic computation for diffusion models")
    parser.add_argument("--config", required=True, help="Diffusion config (.py)")
    parser.add_argument("--geo-config", required=True, help="Geodesic config (.py)")
    args = parser.parse_args()

    diff_cfg = load_config(args.config)
    cfg_dir  = os.path.join(diff_cfg.base_log_dir, diff_cfg.experiment)
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.pkl"), "wb") as f:
        pickle.dump(diff_cfg.to_dict(), f)

    geo_cfg = load_geo_config(args.geo_config)
    
    # ----------------- Seed setting (reproducibility) ------------------
    seed = geo_cfg.get("seed", 42)  # or read from diff_cfg if more appropriate
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    geodesic_batch(diff_cfg, geo_cfg)
