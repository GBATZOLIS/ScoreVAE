#!/usr/bin/env python
"""
Unified Riemannian-optimisation driver (refactored – helper-free)
----------------------------------------------------------------
This version off-loads **all** standalone utilities to
``utils.ro_utils`` while preserving *bit-for-bit* behaviour.
The public CLI is unchanged.
"""
from __future__ import annotations

# ------------------------------------------------------------------
# Standard lib / third-party
# ------------------------------------------------------------------
import os
import time  # noqa: F401 – kept for BC (was imported before)
import pickle  # noqa: F401 – idem
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

import numpy as np  # noqa: F401 – downstream code may still use it
import torch
import torch.multiprocessing as mp

# ------------------------------------------------------------------
# Project-local stacks
# ------------------------------------------------------------------
from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import prepare_training_dirs, EMA, load_model
from configs import load_config

from data_geometry.optim_function import get_optim_function
from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from data_geometry.riemannian_optimization.optimizers import (
    RiemannianGD,
    GenericRiemannianGD,
)
from data_geometry.metrics import create_metric
from data_geometry.utils.visualization import (
    visualize_riemannian_optimization_selector,
)

# ------------------------------------------------------------------
# Refactored helpers – single import line keeps driver tidy
# ------------------------------------------------------------------
from utils.ro_utils import (
    set_global_seeds,
    flatten,
    unflatten,  # not used directly but kept for parity if downstream hooks import it
    ensure_B,  # same as above
    get_score_fn,
    get_denoiser_fn,
    load_py_config,
    sample_indices_excluding,
    build_projector,
    evaluate_trajectory_and_best_points,
    enrich_riem_cfg_with_manifold_info
)

# ------------------------------------------------------------------
# Main optimisation routine
# ------------------------------------------------------------------

def riemannian_optimization(diff_cfg, riem_cfg):
    """Entry-point orchestrating data, model & optimisation (unchanged signatures)."""

    # ───────────────── reproducibility ────────────────────────────
    SEED = int(riem_cfg.get("random_seed", diff_cfg.random_seed))
    set_global_seeds(SEED)

    device = torch.device(diff_cfg.training.device)
    print(f"[driver] using device: {device} | seed={SEED}")

    # ───────────────── logging dirs ───────────────────────────────
    _, _, eval_dir = prepare_training_dirs(diff_cfg)

    # ───────────────── dataset snapshot (all points) ──────────────
    train_loader, _, _ = get_dataloaders(diff_cfg.data)
    dataset = train_loader.dataset
    data_tensor = torch.stack([
        d[0] if isinstance(d, (tuple, list)) else d for d in dataset
    ])

    # ─ Riemanian config enrichment with manifold/dataset specific details ─
    # Information such as the embedding matrix, etc. useful for the construction of the optim function.
    riem_cfg = enrich_riem_cfg_with_manifold_info(riem_cfg, diff_cfg, dataset)

    # total number of available data points
    N = len(data_tensor)

    # ───────────────── minima selection (soft wells) ──────────────
    if not riem_cfg.get("min_point"):
        k_minima = riem_cfg.get("n_minima", 6)
        minima_idx = sample_indices_excluding(N, k_minima, SEED)
        minima = [data_tensor[i].flatten().tolist() for i in minima_idx]
        riem_cfg["min_point"] = minima
        print(f"[driver] ➊ sampled {k_minima} minima deterministically:")
        for j, m in enumerate(minima, 1):
            print(f"        m{j}: {m}")
    else:
        minima_idx = torch.tensor([], dtype=torch.long)
        print(f"[driver] ➊ using {len(riem_cfg['min_point'])} minima from config.")

    # ───────────────── initial optimisation points ────────────────
    n_points = riem_cfg.get("num_points", 32)
    x0_idx = sample_indices_excluding(N, n_points, SEED + 1, exclude_idx=minima_idx)
    x0 = data_tensor[x0_idx].to(device)
    orig_sh = x0.shape[1:]
    x0_flat = flatten(x0)
    riem_cfg["initial_point"] = x0_flat  # required by classic optimiser

    # ───────────────── diffusion model stack ──────────────────────
    model = get_model(diff_cfg.model).to(device)
    sde = configure_sde(diff_cfg)
    ema = EMA(model, decay=diff_cfg.model.ema_decay)

    ckpt_path = diff_cfg.model.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(diff_cfg.checkpoint_dir, ckpt_path)
    if not ckpt_path.endswith(".pth"):
        ckpt_path += ".pth"
    load_model(model, ema, ckpt_path, "Model", device=device, is_ema=True)
    ema.apply_shadow()
    model.eval()

    # – perturbation time in the SDE –––––––––––––––––––––––––––––––
    t_val = float(riem_cfg["time_for_perturbation"])
    t_tensor = torch.tensor(t_val, dtype=torch.float32, device=device)

    snr_value = sde.snr(t_tensor).item()
    print(f"[driver] 💡 Optimization at t={t_val:.4f} corresponds to an SNR of {snr_value:.4f}")

    score_fn = get_score_fn(sde, model, t_tensor, orig_sh)
    denoiser_fn = get_denoiser_fn(sde, model, t_tensor, orig_sh)

    # ───────────────── objective function  f(x)  ──────────────────
    opt_fn = get_optim_function(riem_cfg)

    # ───────────────── choose optimiser implementation ────────────
    impl = riem_cfg.get("gradient_impl", "classic").lower()
    if impl == "generic":
        metric = create_metric(
            metric_type=riem_cfg.get("metric_type", "jacobian"),
            score_fn=score_fn,
            lam_metric=riem_cfg.get("lam_metric", riem_cfg.get("reg_lambda", 1.0)),
            denoiser_fn=denoiser_fn,
            cg_kwargs=dict(
                reg_lambda=riem_cfg.get("reg_lambda", 1e-6),
                max_iter=riem_cfg.get("cg_max_iter", 50),
                tol=riem_cfg.get("cg_tol", 1e-6),
                preconditioner=riem_cfg.get("cg_preconditioner", "diagonal"),
                precond_diag_samples=riem_cfg.get("cg_precond_diag_samples", 10),
            ),
        )
        optimiser = GenericRiemannianGD(metric, opt_fn, riem_cfg)
        trajectory, metrics = optimiser.run(x0_flat)

    elif impl == "classic":
        retraction_fn = create_retraction_fn(
            retraction_type=riem_cfg.get("retraction_operator", "identity"),
            denoiser_fn=denoiser_fn,
        )
        optimiser = RiemannianGD(score_fn, opt_fn, riem_cfg, retraction_fn)
        trajectory, metrics = optimiser.run()

    else:
        raise ValueError("gradient_impl must be 'classic' or 'generic', got '{impl}'")

    print(f"[driver] optimisation finished – {len(trajectory) - 1} iterations")

    # ------------------------------------------------------------------
    # Evaluate manifold closeness – trajectory + best points
    # ------------------------------------------------------------------
    try:
        expected_radius = float(riem_cfg.get("manifold_radius", 1.0))
        Q = riem_cfg.get("embedding_matrix")
        P = build_projector(Q.to(device)) if Q is not None else None

        evaluate_trajectory_and_best_points(
            trajectory=trajectory,
            opt_fn=opt_fn,
            expected_radius=expected_radius,
            projector=P,
            device=device,
            metrics=metrics,
        )
    except Exception as e:
        print(f"[driver] manifold-error computation skipped ({e})")

    # ------------------------------------------------------------------
    # Optional visualisation
    # ------------------------------------------------------------------
    if riem_cfg.get("visualize", True):
        alpha_fn, sigma_fn = sde.get_alpha_fn(), sde.get_sigma_fn()
        alpha_t, sigma_t = alpha_fn(t_tensor), sigma_fn(t_tensor)
        base_pts = data_tensor[:2500].to(device)
        noise = torch.randn_like(base_pts)
        perturbed = alpha_t * base_pts + sigma_t * noise

        visualize_riemannian_optimization_selector(
            perturbed,
            score_fn,
            t_val,
            [p.cpu() for p in trajectory],
            metrics,
            torch.tensor(riem_cfg["min_point"]).cpu().numpy(),
            orig_sh,
            log_dir=eval_dir,
            plot_filename=riem_cfg["plot_filename"],
        )


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ro-config", required=True)
    cli = parser.parse_args()

    diff_cfg = load_config(cli.config)
    riem_cfg = load_py_config(cli.ro_config)

    riemannian_optimization(diff_cfg, riem_cfg)
