#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ambient_geodesic_onepair.py
Run image/ambient-space geodesic *for exactly one pair* chosen previously by
save_latent_pairs.py. Produces estimated-vs-GT comparison and metrics.

Usage:
  python ambient_geodesic_onepair.py \
      --config path/to/diffusion_or_ae_cfg.py \
      --geo-config path/to/ambient_geo_config.py \
      --pair-from-file /path/to/eval_logs/latent_pair_selection.pt \
      --pair-id 10 \
      --visualize

Notes:
- --pair-id is 0-based (so 10 = 11th pair).
- Uses the same split that was saved in the pair file, unless you override with --split.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from configs                 import load_config
from data.data_utils_fast    import get_dataloaders
from utils.train_utils       import prepare_training_dirs, EMA, load_model
from models                  import get_model
from sde                     import configure_sde
from data_geometry.geodesics.fast_geodesic_computation import compute_geodesic
from utils.ae_utils          import _save_geodesic_comparison_grid

# Optional viz helpers (safe no-ops if you skip --visualize)
from data_geometry.utils.visualization import visualize_riemannian_optimization_selector

# ─── tiny helpers ────────────────────────────────────────────────────
def _unwrap_base_dataset(eval_dataset):
    subset = eval_dataset
    while hasattr(subset, "dataset"):
        subset = subset.dataset
    base_ds = subset
    if hasattr(eval_dataset, "indices"):
        idx_pool = list(eval_dataset.indices)
    else:
        idx_pool = list(range(len(eval_dataset)))
    return base_ds, idx_pool

def _load_geo_cfg(path: str) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    with open(path, "r") as f:
        exec(compile(f.read(), path, "exec"), scope)
    if "CONFIG" not in scope:
        raise ValueError(f"Config file '{path}' must define a CONFIG dict.")
    return scope["CONFIG"]

def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)

def unflatten(x: torch.Tensor, c_h_w: Tuple[int, int, int]) -> torch.Tensor:
    C, H, W = c_h_w
    return x.view(x.size(0), C, H, W)

def _split_data(items: List[Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(items[0], (tuple, list)):
        imgs = torch.stack([it[0] for it in items])
        rots = torch.stack([it[1] for it in items])
        return imgs, rots
    return torch.stack(items), None

# ─── core one-pair routine ───────────────────────────────────────────
def run_one_pair(geo: Dict[str, Any],
                 model: torch.nn.Module,
                 sde: Any,
                 dset,
                 device: torch.device,
                 eval_dir: str,
                 pair_from_file: str,
                 pair_id: int,
                 split_override: Optional[str],
                 visualize: bool):
    """Compute ambient geodesic for exactly ONE pair from a saved selection."""

    # choose split (align to saved unless overridden)
    saved = torch.load(pair_from_file, map_location="cpu")
    saved_split = saved.get("split", "val")
    split_name = (split_override or saved_split).lower()
    print(f"[Driver] Using split: {split_name} (saved was: {saved_split})")

    # unwrap current dataset (we assume caller passed the correct split loader.dataset)
    base_ds, idx_pool_current = _unwrap_base_dataset(dset)

    B_saved       = int(saved["B"])
    perm_local    = list(map(int, saved["perm_local"]))
    idx_pool_saved= list(map(int, saved["idx_pool"]))

    if not (0 <= pair_id < B_saved):
        raise ValueError(f"pair_id {pair_id} out of range for saved B={B_saved}")

    # Resolve *global* base-dataset indices selected by the latent run
    p_global = idx_pool_saved[ perm_local[pair_id] ]
    q_global = idx_pool_saved[ perm_local[pair_id + B_saved] ]

    # Fetch those exact items from *this* base dataset
    p_item = base_ds[p_global]
    q_item = base_ds[q_global]

    imgs, rots = _split_data([p_item, q_item])  # (2,...) tensors
    C, H, W = imgs.shape[1:]
    p_img_full, q_img_full = imgs[:1], imgs[1:]  # B=1
    p_rot_full, q_rot_full = (rots[:1], rots[1:]) if rots is not None else (None, None)

    # Flatten endpoints
    p_flat, q_flat = flatten(p_img_full.to(device)), flatten(q_img_full.to(device))

    # Metric params (CG only used for "jacobian"/"jsm")
    mt = str(geo.get("metric_type", "stein")).lower()
    if mt in {"jacobian", "jsm"}:
        cg_kwargs = dict(
            reg_lambda=float(geo.get("lam_metric", geo.get("reg_lambda", 1e-5))),
            max_iter=int(geo.get("cg_max_iter", 10)),
            tol=float(geo.get("cg_tol", 1e-6)),
            preconditioner=str(geo.get("cg_preconditioner", "diagonal")),
            precond_diag_samples=int(geo.get("cg_precond_diag_samples", 10)),
        )
    else:
        cg_kwargs = {}

    # Optimizer/LS policy
    optimizer   = str(geo.get("optimizer", "adam")).lower()
    line_search = str(geo.get("line_search", "armijo")).lower()
    if optimizer == "adam" and line_search == "strong_wolfe":
        print("[Driver] Adam + strong_wolfe is unsupported; using Armijo.")
        line_search = "armijo"

    # Run geodesic
    t0 = time.time()
    path, info, _ = compute_geodesic(
        p=p_flat, q=q_flat,
        initial_path=None,                                 # rely on endpoint_mode below
        model=model, sde=sde, orig_shape=(C, H, W),

        # discretisation & schedule
        n_segments=int(geo.get("n_segments", 16)),
        time_schedule=geo.get("time_schedule", [0.03]),

        # regularisers
        lam_smooth=float(geo.get("lam_smooth", 2.0)),
        lam_mono=float(geo.get("lam_mono", 2.0)),

        # budget
        max_iters=int(geo.get("max_iters", 800)),
        tol=float(geo.get("tol", 1e-6)),
        patience=int(geo.get("patience", 50)),

        # optimizer
        optimizer=optimizer,
        adam_lr=float(geo.get("adam_lr", 5e-4)),
        betas=tuple(geo.get("betas", (0.9, 0.999))),
        line_search=line_search,
        use_retraction_update=bool(geo.get("use_retraction_update", True)),

        # Armijo
        armijo_rho=float(geo.get("armijo_rho", 5e-4)),
        armijo_beta=float(geo.get("armijo_beta", 0.5)),
        armijo_max_iter=int(geo.get("armijo_max_iter", 15)),

        # Strong-Wolfe params are ignored for Adam

        # metric
        metric_type=str(geo.get("metric_type", "jacobian")),
        lam_metric=float(geo.get("lam_metric", 1e-2)),
        cg_kwargs=cg_kwargs,

        devices=geo.get("devices", None),
        post_denoise_fn=None,                              # single final denoise = None
        endpoint_mode=str(geo.get("endpoint_mode", "noisy")),
        fixed_noise=None,

        verbose=True,
        vis_kwargs=geo.get("visualization", {}),
    )
    runtime = time.time() - t0
    print(f"[Done] Optimization runtime: {runtime:.2f}s")

    # Realized T (frames)
    T_realized = len(path)
    tv = torch.linspace(0.0, 1.0, T_realized)

    # Predicted stack (B=1, T, C, H, W)
    # ---- Build (B, T, C, H, W) from path: list[T] of (B, D) ----
    B = path[0].shape[0]
    pred_per_b = []
    for b in range(B):
        frames_b = []
        for k in range(T_realized):
            # take the b-th trajectory vector at step k: (D,)
            x_bk = path[k][b:b+1].detach().cpu()                 # (1, D)
            img_bk = unflatten(x_bk, (C, H, W)).squeeze(0)       # (C, H, W)
            frames_b.append(img_bk.clamp(0.0, 1.0))
        pred_per_b.append(torch.stack(frames_b, dim=0))          # (T, C, H, W)
    pred_stack = torch.stack(pred_per_b, dim=0)                  # (B, T, C, H, W)
    
    # GT stack — uses rotations if dataset provides them
    if hasattr(base_ds, "compute_geodesic"):
        with torch.no_grad():
            if p_rot_full is not None:
                gt_stack = base_ds.compute_geodesic(p_rot_full, q_rot_full, tv).detach().cpu()
            else:
                gt_stack = base_ds.compute_geodesic(flatten(p_img_full).cpu(),
                                                    flatten(q_img_full).cpu(), tv).detach().cpu()
        # RMSE
        mse = ((pred_stack - gt_stack) ** 2).mean().item()
        rmse = float(np.sqrt(mse))
        print(f"[GT] RMSE vs GT (B=1, T={T_realized}): {rmse:.6f}")

        # Comparison image (top: estimated, bottom: GT)
        comp_path = os.path.join(eval_dir, "geodesic_onepair_vs_gt.png")
        _save_geodesic_comparison_grid(pred_stack, gt_stack, comp_path, padding=2, row_gap=2, pair_gap=12)
        print(f"[GT] Saved comparison → {comp_path}")

        with open(os.path.join(eval_dir, "geodesic_onepair_eval.txt"), "w") as f:
            f.write(f"pair_id={pair_id} (0-based)\n")
            f.write(f"T={T_realized}\n")
            f.write(f"RMSE_vs_GT={rmse:.6f}\n")

    # Optional vector-field viz overlays
    if visualize:
        t_pert = torch.tensor(float(geo.get("time_schedule", [0.03])[0]), device=device)
        # Background points: a small random subset of dataset to draw the field (safe on CPU)
        bg_n = min(2000, len(dset))
        bg_imgs, _ = _split_data([dset[i] for i in range(bg_n)])
        with torch.no_grad():
            noise_bg = sde.perturb(bg_imgs.to(device), t_pert)
        from .debug_initialization import get_score_fn, save_path_grid  # if these are in your tree
        vis_score_fn = get_score_fn(sde, model, bg_imgs.shape[1:])
        vis_kwargs = dict(
            perturbed_points=flatten(noise_bg).cpu(),
            score_fn=lambda x: vis_score_fn(x.to(device), t_pert.to(device)).cpu(),
            t_val=t_pert.item(),
            metrics={}, min_point=None,
            orig_shape=bg_imgs.shape[1:],
            log_dir=eval_dir
        )
        fname = "geodesic_onepair.png"
        print(f"[Viz] Plotting estimated geodesic field overlay → {os.path.join(eval_dir, fname)}")
        visualize_riemannian_optimization_selector(
            **vis_kwargs, trajectories=[torch.stack(path, 0).cpu().permute(1, 0, 2)], plot_filename=fname
        )
        save_path_grid([p.detach().cpu() for p in path], (C, H, W),
                       os.path.join(eval_dir, "geodesic_onepair_estimated_raw.png"))

def main():
    import argparse
    P = argparse.ArgumentParser("Ambient geodesic for exactly one saved pair")
    P.add_argument("--config", required=True, help="Main model/training config.")
    P.add_argument("--geo-config", required=True, help="Ambient geodesic CONFIG.")
    P.add_argument("--pair-from-file", required=True, help="latent_pair_selection.pt from the latent run.")
    P.add_argument("--pair-id", required=True, type=int, help="0-based pair id (e.g., 10 for the 11th pair).")
    P.add_argument("--split", type=str, default=None, choices=["train","val","test"],
                   help="Override split. If omitted, use the one saved in pair file.")
    P.add_argument("--visualize", action="store_true", help="Enable vector-field overlays (optional).")
    a = P.parse_args()

    diff_cfg = load_config(a.config)
    geo_cfg = _load_geo_cfg(a.geo_config)

    # Output dir
    _, _, eval_dir = prepare_training_dirs(diff_cfg)
    os.makedirs(eval_dir, exist_ok=True)

    # Build loaders with deterministic split seed (match latent’s)
    seed_for_split = int(geo_cfg.get("random_seed", 42))
    train_loader, val_loader, test_loader = get_dataloaders(diff_cfg.data, seed=seed_for_split)

    saved = torch.load(a.pair_from_file, map_location="cpu")
    split_name = (a.split or saved.get("split", "val")).lower()
    if split_name == "train":
        loader = train_loader
    elif split_name == "test":
        loader = test_loader
    else:
        loader = val_loader

    dset = loader.dataset

    # Model + SDE
    device = torch.device(diff_cfg.training.device)
    model = get_model(diff_cfg.model).to(device)
    ema = EMA(model, diff_cfg.model.ema_decay)
    ckpt_path = os.path.join(diff_cfg.checkpoint_dir, diff_cfg.model.checkpoint)
    if not ckpt_path.endswith(".pth"):
        ckpt_path += ".pth"
    load_model(model, ema, ckpt_path, "Model", device=device, is_ema=True)
    ema.apply_shadow()
    model.eval()
    sde = configure_sde(diff_cfg)

    # Float precision nicety
    torch.set_float32_matmul_precision("high")

    # Seed reproducibly but we don't resample anything heavy here
    seed = int(geo_cfg.get("random_seed", saved.get("seed_used", 42)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_one_pair(
        geo=geo_cfg, model=model, sde=sde, dset=dset, device=device, eval_dir=eval_dir,
        pair_from_file=a.pair_from_file, pair_id=int(a.pair_id),
        split_override=a.split, visualize=a.visualize
    )

if __name__ == "__main__":
    main()
