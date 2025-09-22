#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ambient-space geodesic evaluation that mirrors the latent-space pipeline,
now with:
  • Micro-batched geodesic optimization (e.g., 5 pairs at a time) and
    global aggregation so outputs look like B_total were run at once.
  • Entropy profile computation & saving for the ambient diffusion model.

Artifacts:
    ambient_geodesics_grid.png
    ambient_geodesics_vs_gt.png
    geodesic_eval.txt
    ambient_geodesics_info.pkl
    ambient_geodesics_path_flat.pt
    entropy_profile.png / entropy_profile.npz
"""

import os
import argparse
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from configs import load_config
from data.data_utils_ddp import get_dataloaders
from models import get_model
from utils.train_utils import prepare_training_dirs, EMA, load_model
from sde import configure_sde

# Geodesic driver (same as latent)
from data_geometry.geodesics.fast_geodesic_computation import compute_geodesic

# Entropy profile (ambient version uses the image loader)
from utils.entropy_profile import compute_entropy_profile  # you pasted this file above


# ────────────────────────── small utils ──────────────────────────

Tensor = torch.Tensor


def _load_geo_cfg(path: str) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    with open(path, "r") as f:
        code = compile(f.read(), path, "exec")
        exec(code, scope)
    if "CONFIG" not in scope:
        raise ValueError(f"Geodesic config '{path}' must define a CONFIG dict.")
    return scope["CONFIG"]  # type: ignore[return-value]


def _build_cg_kwargs(geo: Dict[str, Any]) -> Dict[str, Any]:
    """Match the latent script: only build CG kwargs for Jacobian/JSM metrics."""
    mt = str(geo.get("metric_type", "stein")).lower()
    if mt in {"jacobian", "jsm"}:
        return dict(
            reg_lambda=float(geo.get("lam_metric", geo.get("reg_lambda", 1e-5))),
            max_iter=int(geo.get("cg_max_iter", 8)),
            tol=float(geo.get("cg_tol", 5e-5)),
            preconditioner=str(geo.get("cg_preconditioner", "diagonal")),
            precond_diag_samples=int(geo.get("cg_precond_diag_samples", 8)),
        )
    return {}


def _unwrap_base_dataset(eval_dataset):
    """Unwrap Subset chains to the base dataset & return indices for current split."""
    subset = eval_dataset
    while hasattr(subset, "dataset"):
        subset = subset.dataset
    base_ds = subset
    if hasattr(eval_dataset, "indices"):
        idx_pool = list(eval_dataset.indices)
    else:
        idx_pool = list(range(len(eval_dataset)))
    return base_ds, idx_pool


def _flatten(x: Tensor) -> Tensor:
    return x.flatten(1)


def _unflatten(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return x.view(x.shape[0], *shape)


# ── save a comparison grid (per pair: top row estimated, bottom row GT)
def _save_geodesic_comparison_grid(
    pred_stack: Tensor,   # (B, T, C, H, W) in [0,1]
    gt_stack: Tensor,     # (B, T, C, H, W) in [0,1]
    out_path: str,
    *,
    padding: int = 2,
    row_gap: int = 2,
    pair_gap: int = 12,
):
    pred_stack = pred_stack.detach().cpu()
    gt_stack   = gt_stack.detach().cpu()

    B, T, C, H, W = pred_stack.shape
    blocks = []
    for i in range(B):
        pred_grid = vutils.make_grid(pred_stack[i], nrow=T, padding=padding)
        gt_grid   = vutils.make_grid(gt_stack[i],   nrow=T, padding=padding)

        Cg, Hr, Wr = pred_grid.shape
        row_gap_img  = torch.zeros(Cg, row_gap, Wr)
        pair_gap_img = torch.zeros(Cg, pair_gap, Wr)

        pair_block = torch.cat([pred_grid, row_gap_img, gt_grid], dim=1)
        blocks.append(pair_block)
        if i < B - 1:
            blocks.append(pair_gap_img)

    big_img = torch.cat(blocks, dim=1)  # (C, H_total, W)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vutils.save_image(big_img, out_path)
    return out_path


def _save_path_grid(
    path_list: List[Tensor],  # list length T, each (B, D_flat)
    img_shape: Tuple[int, int, int],  # (C,H,W)
    out_path: str,
    *,
    padding: int = 2,
    pair_gap: int = 12,
):
    """
    Save a grid of the estimated geodesic paths only.
    Each pair (row) shows T frames left→right.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    T = len(path_list)
    B = path_list[0].shape[0]

    # (B, T, C, H, W) in [0,1]
    frames = []
    for t in range(T):
        imgs_t = _unflatten(path_list[t], img_shape).detach().cpu().clamp(0.0, 1.0)
        frames.append(imgs_t)
    stack = torch.stack(frames, dim=1)  # (B, T, C, H, W)

    # build rows per pair
    rows = []
    for b in range(B):
        row = vutils.make_grid(stack[b], nrow=T, padding=padding)
        rows.append(row)
        if b < B - 1:
            rows.append(torch.zeros_like(row)[:, :pair_gap, :])  # horizontal spacer

    big = torch.cat(rows, dim=1)
    vutils.save_image(big, out_path)
    return out_path


# ────────────────────────── main driver ──────────────────────────

def eval_ambient_geodesics(
    diff_cfg,
    *,
    geo_config: str,
):
    """
    Ambient-space geodesic evaluation mirroring the latent-space script's behavior/outputs,
    with micro-batching and entropy profiles for the ambient model.
    """
    device = torch.device(diff_cfg.training.device)
    _, _, eval_dir = prepare_training_dirs(diff_cfg)
    writer = SummaryWriter(log_dir=eval_dir)

    # 1) Data
    loader, _, _ = get_dataloaders(diff_cfg.data, seed=diff_cfg.get("random_seed", 42))
    dset = loader.dataset
    base_ds, idx_pool = _unwrap_base_dataset(dset)

    # 2) Model + SDE
    model = get_model(diff_cfg.model).to(device)
    ema = EMA(model=model, decay=diff_cfg.model.ema_decay)
    ck_in = diff_cfg.model.checkpoint
    if ck_in is None:
        raise ValueError("Set diffusion config.model.checkpoint to an EMA .pth (e.g., 'Model_last_EMA.pth').")
    if not ck_in.endswith(".pth"):
        ck_in += ".pth"
    ckpt_path = os.path.join(diff_cfg.checkpoint_dir, ck_in)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")

    load_model(model, ema, ckpt_path, "Model", device=device, is_ema=True)
    ema.apply_shadow()
    model.eval()
    sde = configure_sde(diff_cfg)

    # 3) Geodesic CONFIG (shared with latent script)
    geo = _load_geo_cfg(geo_config) if geo_config is not None else {}
    existing_schedule = list(geo.get("time_schedule", [0.1]))
    time_schedule = existing_schedule[:]  # do NOT change via entropy in this script
    B_total = int(geo.get("num_pairs", 10))
    need = 2 * B_total

    # 4) Sample endpoints (images + optional pose/rotation params)
    g_idx = torch.Generator()
    g_idx.manual_seed(int(geo.get("random_seed", diff_cfg.get("random_seed", 42))))
    perm_local = torch.randperm(len(idx_pool), generator=g_idx).tolist()[:need]

    imgs: List[Tensor] = []
    rots: List[Tensor] = []
    has_pose = False
    for j in perm_local:
        item = base_ds[idx_pool[j]]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x_j, r_j = item[0], item[1]
            has_pose = True
            imgs.append(x_j)
            rots.append(r_j)
        else:
            imgs.append(item)

    X = torch.stack(imgs, dim=0)        # keep on CPU for now; move per batch
    if has_pose:
        Rsel = torch.stack(rots, dim=0)  # (need, ...)

    # Shapes
    C, H, W = int(X.shape[1]), int(X.shape[2]), int(X.shape[3])
    orig_shape = (C, H, W)
    D_flat = C * H * W

    if X.size(0) < need:
        B_total = X.size(0) // 2
        if B_total == 0:
            print("[Geodesics] Not enough samples to form a single pair — aborting.")
            writer.close()
            return
        else:
            print(f"[Geodesics] Reducing num_pairs to {B_total} due to dataset size.")
            need = 2 * B_total

    # ---- Entropy profile (ambient) ---------------------------------
    entropic_profile_enabled = bool(geo.get("entropic_profile", True))
    if entropic_profile_enabled:
        try:
            ent_num_t       = int(geo.get("ent_num_t", 96))
            ent_max_batches = int(geo.get("ent_max_batches", 12))
            ent_t_min       = geo.get("ent_t_min", None)
            ent_t_max       = float(geo.get("ent_t_max", 1.0))

            # Use the existing image loader; compute_entropy_profile only looks at x batches.
            ent = compute_entropy_profile(
                model=model,
                sde=sde,
                loader=loader,
                orig_shape=orig_shape,
                device=device,
                t_min=ent_t_min,
                t_max=ent_t_max,
                num_t=ent_num_t,
                max_batches=ent_max_batches,
                save_dir=eval_dir,
                filename_prefix="entropy_profile",
                progress=True,
            )
            print(f"[Entropy] Saved entropy profile → {os.path.join(eval_dir, 'entropy_profile.png')}")
        except Exception as e:
            print(f"[Entropy] Skipping due to error: {e}")

    # 5) Prep pairs
    p_img_all = X[:B_total]
    q_img_all = X[B_total:2 * B_total]

    # 6) Build CG kwargs (Jacobian metric only), optimizer/line-search
    cg_kwargs = _build_cg_kwargs(geo)

    optimizer   = str(geo.get("optimizer", "adam")).lower()
    line_search = str(geo.get("line_search", "armijo")).lower()
    if optimizer == "adam" and line_search == "strong_wolfe":
        print("[Geodesics] Adam + strong_wolfe is unsupported; using 'armijo' to match latent behavior.")
        line_search = "armijo"

    # 7) Micro-batched geodesic solves
    seed = int(geo.get("random_seed", diff_cfg.get("random_seed", 42)))
    torch.manual_seed(seed)

    batch_pairs = int(geo.get("optimizer_batch_size", 5))  # ← your chosen micro-batch size

    path_flat_all: List[Tensor] = []   # will become length T, each (B_total, D)
    info_last: Dict[str, Any] = {}
    T_frames: Optional[int] = None

    # Iterate over chunks of pairs
    for start in range(0, B_total, batch_pairs):
        end = min(B_total, start + batch_pairs)
        mb = end - start

        p_img = p_img_all[start:end].to(device, non_blocking=True)
        q_img = q_img_all[start:end].to(device, non_blocking=True)
        p_flat = _flatten(p_img)  # (mb, D)
        q_flat = _flatten(q_img)

        # One micro-batch solve
        path_flat_mb, info_mb, _ = compute_geodesic(
            p=p_flat, q=q_flat,
            model=model, sde=sde, orig_shape=orig_shape,
            n_segments=int(geo.get("n_segments", 16)),
            time_schedule=time_schedule,

            lam_smooth=float(geo.get("lam_smooth", 1e-3)),
            lam_mono=float(geo.get("lam_mono", 0.0)),

            max_iters=int(geo.get("max_iters", 500)),
            tol=float(geo.get("tol", 1e-6)),
            patience=int(geo.get("patience", 30)),

            optimizer=optimizer,
            adam_lr=float(geo.get("adam_lr", 1e-2)),
            betas=tuple(geo.get("betas", (0.9, 0.999))),
            line_search=line_search,
            use_retraction_update=bool(geo.get("use_retraction_update", True)),

            armijo_rho=float(geo.get("armijo_rho", 1e-4)),
            armijo_beta=float(geo.get("armijo_beta", 0.5)),
            armijo_max_iter=int(geo.get("armijo_max_iter", 20)),

            wolfe_c1=float(geo.get("wolfe_c1", 1e-4)),
            wolfe_c2=float(geo.get("wolfe_c2", 0.9)),
            wolfe_max_bracket=int(geo.get("wolfe_max_bracket", 10)),
            wolfe_max_zoom=int(geo.get("wolfe_max_zoom", 10)),
            wolfe_max_alpha=float(geo.get("wolfe_max_alpha", 50.0)),

            verbose=True,
            metric_type=str(geo.get("metric_type", "stein")),
            lam_metric=float(geo.get("lam_metric", 1.0)),
            cg_kwargs=cg_kwargs,
            devices=geo.get("devices", None),
            post_denoise_fn=None,
            endpoint_mode=str(geo.get("endpoint_mode", "clean")),  # or "noisy"
            fixed_noise=None,
            # If you patched compute_geodesic with init controls:
            init_method = str(geo.get("init_method", "tproject")),
            init_nseeds = int(geo.get("init_nseeds", 20)),
            init_add_noise  =bool(geo.get("init_add_noise", True)),
        )

        info_last = info_mb  # keep last for logging/diagnostics
        T_this = len(path_flat_mb)
        if T_frames is None:
            T_frames = T_this
            # Initialize global list with empty CPU tensors
            path_flat_all = [torch.empty(0, D_flat) for _ in range(T_frames)]
        else:
            assert T_this == T_frames, "All micro-batches must have same number of frames."

        # Append this micro-batch to global CPU store
        for t in range(T_frames):
            path_flat_all[t] = torch.cat([path_flat_all[t],
                                          path_flat_mb[t].detach().cpu()], dim=0)

        # free micro-batch GPU tensors
        del p_img, q_img, p_flat, q_flat, path_flat_mb
        torch.cuda.empty_cache()

    assert T_frames is not None
    print(f"[Geodesics] Total pairs={B_total}, micro-batch={batch_pairs}, realized frames T={T_frames}")

    # 8) Predicted image stack: (B_total, T, C, H, W) in [0,1]
    frames_cpu = []
    for t in range(T_frames):
        imgs_t = _unflatten(path_flat_all[t], orig_shape).clamp(0.0, 1.0)  # CPU
        frames_cpu.append(imgs_t)
    pred_stack = torch.stack(frames_cpu, dim=1)  # (B_total, T, C, H, W)

    # 9) Save predicted grid
    grid_path = os.path.join(eval_dir, "ambient_geodesics_grid.png")
    _save_path_grid(path_list=path_flat_all, img_shape=orig_shape, out_path=grid_path)
    print(f"[Geodesics] Estimated ambient geodesic grid saved → {grid_path}")

    # 10) Ground-truth comparison + RMSE
    avg_rmse = None
    try:
        if hasattr(base_ds, "compute_geodesic"):
            tv = torch.linspace(0.0, 1.0, T_frames)

            if has_pose:
                P = Rsel[:B_total].detach().cpu()
                Q = Rsel[B_total:2 * B_total].detach().cpu()
                gt_stack = base_ds.compute_geodesic(P, Q, tv).detach().cpu()  # (B, T, C, H, W)
            else:
                gt_stack = base_ds.compute_geodesic(
                    _flatten(p_img_all).cpu(),
                    _flatten(q_img_all).cpu(),
                    tv
                ).detach().cpu()

            # checks
            assert gt_stack.shape[0] == pred_stack.shape[0], "B mismatch"
            assert gt_stack.shape[1] == pred_stack.shape[1], "T mismatch"
            assert gt_stack.shape[2:] == pred_stack.shape[2:], "CHW mismatch"

            # RMSE per pair, then average
            mse_per_pair = ((pred_stack - gt_stack) ** 2).mean(dim=(1, 2, 3, 4))
            rmse_per_pair = torch.sqrt(mse_per_pair)
            avg_rmse = rmse_per_pair.mean().item()

            writer.add_scalar("AmbientGeodesics/AvgRMSE_vs_GT", float(avg_rmse), 0)
            txt_path = os.path.join(eval_dir, "geodesic_eval.txt")
            with open(txt_path, "w") as f:
                f.write(f"Avg RMSE vs GT (B={B_total}, T={T_frames}): {avg_rmse:.6f}\n")
            print(f"[Geodesics][GT] Avg RMSE vs ground-truth (B={B_total}, T={T_frames}): {avg_rmse:.6f}")
            print(f"[Geodesics][GT] Saved → {txt_path}")

            comp_path = os.path.join(eval_dir, "ambient_geodesics_vs_gt.png")
            _save_geodesic_comparison_grid(
                pred_stack=pred_stack, gt_stack=gt_stack, out_path=comp_path,
                padding=2, row_gap=2, pair_gap=12,
            )
            print(f"[Geodesics][GT] Comparison image saved → {comp_path}")
        else:
            print("[Geodesics][GT] Dataset does not expose 'compute_geodesic' — skipping GT evaluation.")
    except Exception as e:
        print(f"[Geodesics][GT] Failed to evaluate GT deviation: {e}")
        try:
            print("[Geodesics][GT][debug]",
                  "B_total:", B_total,
                  "T_frames:", T_frames,
                  "pred_stack:", tuple(pred_stack.shape))
        except Exception:
            pass

    # 11) Persist path + loss info (match latent) — with explicit confirmations
    os.makedirs(eval_dir, exist_ok=True)

    info_path = os.path.join(eval_dir, "ambient_geodesics_info.pkl")
    with open(info_path, "wb") as f:
        pickle.dump(dict(loss_info=info_last, config=geo), f)
    print(f"[Geodesics] Saved info + geo config → {info_path}")

    pt_path = os.path.join(eval_dir, "ambient_geodesics_path_flat.pt")
    payload = {"path_flat": [pt.detach().cpu() for pt in path_flat_all]}
    # Sanity print before save
    print(f"[Geodesics] About to save {len(payload['path_flat'])} frames; "
        f"each frame tensor shape = {tuple(payload['path_flat'][0].shape)}")
    try:
        torch.save(payload, pt_path)
        if os.path.exists(pt_path):
            size_mb = os.path.getsize(pt_path) / (1024 * 1024)
            print(f"[Geodesics] Saved ambient geodesics path → {pt_path} ({size_mb:.2f} MB)")
        else:
            print(f"[Geodesics][WARN] torch.save returned but file not found: {pt_path}")
    except Exception as e:
        print(f"[Geodesics][ERROR] Failed to save geodesics path: {e}")

    writer.close()


def main():
    p = argparse.ArgumentParser("Ambient-space geodesic evaluation (micro-batched + entropy profiles)")
    p.add_argument("--config", required=True, type=str,
                   help="Path to the diffusion model config (ambient).")
    p.add_argument("--geo_config", required=True, type=str,
                   help="Path to the geodesic CONFIG .py (same one you use for latent).")
    args = p.parse_args()

    diff_cfg = load_config(args.config)
    if diff_cfg.model.checkpoint is None:
        raise ValueError("Please set diff_cfg.model.checkpoint to an EMA .pth in your diffusion config.")

    # Save a copy of the config used for eval
    out_dir = os.path.join(diff_cfg.base_log_dir, diff_cfg.experiment)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_eval_ambient.pkl"), "wb") as f:
        pickle.dump(diff_cfg.to_dict(), f)

    torch.set_float32_matmul_precision("high")
    eval_ambient_geodesics(diff_cfg, geo_config=args.geo_config)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
