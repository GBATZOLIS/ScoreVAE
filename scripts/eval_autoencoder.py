#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import math

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader  # ← add

from configs import load_config
from data.data_utils_ddp import get_dataloaders
from models import get_model
from utils.train_utils import prepare_training_dirs, EMA, load_model
from utils.ae_utils import (
    get_reconstruction_callback,
    get_latent_scatter_callback,
    get_prior_variance_callback,
    get_prior_vs_posterior_var_callback,
    plot_latent_corruptions,
    encode_latents_from_loader,
    gather_images,
    decode_and_save_grid,
    save_interactive_latent_geodesic_html,
    save_latent_geodesic_2d,
    _latent_cloud_at_t_with_marginal,
    save_latent_geodesics_stages_2d,
    decode_latent_path_and_save_grid,
    perturb_latents_at_time,
)
from sde import configure_sde
from utils.sampling_utils import Algorithm1

# data-geometry: latent geodesics
from data_geometry.geodesics.fast_geodesic_computation import compute_geodesic
from utils.entropy_profile import compute_entropy_profile, schedule_from_rescaled_entropic_time
import numpy as np


# ────────────────────────── tiny helpers kept locally ──────────────────────────

def _load_geo_cfg(path: str):
    """Exec-load a Python file that defines CONFIG = {...} and return that dict."""
    scope: dict[str, object] = {}
    with open(path, "r") as f:
        code = compile(f.read(), path, "exec")
        exec(code, scope)
    if "CONFIG" not in scope:
        raise ValueError(f"Geodesic config '{path}' must define a CONFIG dict.")
    return scope["CONFIG"]  # type: ignore[return-value]


def _build_cg_kwargs(geo_cfg: dict):
    """Conditionally build CG kwargs for Jacobian-based metrics."""
    mt = str(geo_cfg.get("metric_type", "stein")).lower()
    if mt in {"jacobian", "jsm"}:
        return dict(
            reg_lambda=float(geo_cfg.get("lam_metric", geo_cfg.get("reg_lambda", 1e-5))),
            max_iter=int(geo_cfg.get("cg_max_iter", 8)),
            tol=float(geo_cfg.get("cg_tol", 5e-5)),
            preconditioner=str(geo_cfg.get("cg_preconditioner", "diagonal")),
            precond_diag_samples=int(geo_cfg.get("cg_precond_diag_samples", 8)),
        )
    return {}

def _resolve_ae_pair_paths(base_dir: str, ckpt_name: str) -> tuple[str, str]:
    """
    Return (running_ckpt, ema_ckpt) absolute paths based on a single name the user provides.
    Accepts either AE_last.pth or AE_last_EMA.pth and derives the other.
    """
    if not ckpt_name.endswith(".pth"):
        ckpt_name += ".pth"
    p = os.path.join(base_dir, ckpt_name)
    # If user passed EMA, derive running
    if p.endswith("_EMA.pth"):
        ema_p = p
        run_p = p.replace("_EMA.pth", ".pth")
    else:
        run_p = p
        ema_p = p.replace(".pth", "_EMA.pth")
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


def _load_ae_exact(model, ema: EMA, ckpt_path: str, device: torch.device):
    """
    Load exactly the AE checkpoint provided by the user. If filename ends
    with *_EMA.pth we treat it as EMA and apply the shadow; otherwise we load
    running weights.
    """
    is_ema = ckpt_path.endswith("_EMA.pth")
    load_model(model, ema, ckpt_path, "AE", device=device, is_ema=is_ema)
    if is_ema:
        ema.apply_shadow()
    print(f"[Eval] AE loaded from '{ckpt_path}' (EMA={is_ema})")


def _maybe_load_latent_diffusion_exact(
    *,
    ae_cfg,
    device: torch.device,
    ae_ckpt_dir: str,
    ae_latent_dim: int,
    latent_cfg_path: str | None = None,
    latent_ckpt_path: str | None = None,
):
    if latent_cfg_path is None:
        if not hasattr(ae_cfg.loss, "geom") or not hasattr(ae_cfg.loss.geom, "latent"):
            return None, None
        lat_block = ae_cfg.loss.geom.latent
        if not getattr(lat_block, "enabled", False):
            return None, None
        latent_cfg_path = lat_block.diffusion_config

    lat_cfg = load_config(latent_cfg_path)
    lat_cfg.training.device = str(device)

    # ensure dims align
    if getattr(lat_cfg.data, "latent_dim", None) != ae_latent_dim:
        lat_cfg.data.latent_dim = ae_latent_dim
        lat_cfg.data.shape = [ae_latent_dim]
        lat_cfg.model.state_size = ae_latent_dim

    # 1) Build model (do NOT freeze yet)
    latent_model = get_model(lat_cfg.model).to(device)
    latent_model.train()  # ok to switch to eval later

    # 2) SDE
    latent_sde = configure_sde(lat_cfg)

    # 3) Resolve checkpoint
    ckpt = latent_ckpt_path or getattr(lat_cfg.model, "checkpoint", None) or os.path.join(ae_ckpt_dir, "LatentDiff_last.pth")
    if not ckpt.endswith(".pth"):
        ckpt += ".pth"
    if not os.path.isabs(ckpt) and latent_ckpt_path is None and getattr(lat_cfg, "checkpoint_dir", None):
        ckpt = os.path.join(lat_cfg.checkpoint_dir, ckpt)
    if not os.path.isabs(ckpt) and not os.path.exists(ckpt):
        ckpt = os.path.join(ae_ckpt_dir, os.path.basename(ckpt))  # final fallback

    if not os.path.exists(ckpt):
        print(f"[Eval] Latent diffusion checkpoint not found: {ckpt}. Skipping latent sampling/geodesics.")
        return None, None

    is_ema = ckpt.endswith("_EMA.pth")

    # 4) Create EMA **before** loading so it can hold a shadow
    ema_lat = EMA(latent_model, decay=float(getattr(lat_cfg.model, "ema_decay", 0.999)))

    # 5) Load (this fills either the model or the EMA shadow depending on is_ema)
    load_model(latent_model, ema_lat, ckpt, "LatentDiff", device=device, is_ema=is_ema)

    # 6) If EMA checkpoint, apply it to the model now
    if is_ema:
        ema_lat.apply_shadow()

    # 7) Only now freeze params and switch to eval
    for p in latent_model.parameters():
        p.requires_grad_(False)
    latent_model.eval()

    print(f"[Eval] Latent diffusion model loaded from '{ckpt}' (EMA={is_ema})")
    return latent_model, latent_sde



# ── unwrap Subset(s) to the base dataset and get indices in the current split
def _unwrap_base_dataset(eval_dataset):
    subset = eval_dataset
    while hasattr(subset, "dataset"):
        subset = subset.dataset
    base_ds = subset
    # indices within current split (val/test)
    if hasattr(eval_dataset, "indices"):
        idx_pool = list(eval_dataset.indices)
    else:
        idx_pool = list(range(len(eval_dataset)))
    return base_ds, idx_pool


# ── save a comparison grid (per pair: top row estimated, bottom row GT)
def _save_geodesic_comparison_grid(
    pred_stack: torch.Tensor,   # (B, T, C, H, W) in [0,1]
    gt_stack: torch.Tensor,     # (B, T, C, H, W) in [0,1]
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


# ────────────────────────── eval orchestrator ──────────────────────────

def eval_autoencoder(
    cfg,
    *,
    use_test: bool,
    num_batches: int,
    max_points: int,
    iso_batches: int | None,
    latent_config: str | None,
    latent_ckpt: str | None,
    do_sample: bool,
    sample_steps: int,
    sample_count: int,
    grid_nrow: int | None,
    geo_config: str | None,
):
    _, checkpoint_dir, eval_dir = prepare_training_dirs(cfg)
    writer = SummaryWriter(log_dir=eval_dir)
    device = torch.device(cfg.training.device)

    _, val_loader, test_loader = get_dataloaders(cfg.data, seed=cfg.random_seed)
    eval_loader = test_loader if use_test else val_loader

    # AE setup
    model = get_model(cfg.model).to(device, memory_format=torch.channels_last)
    ema = EMA(model=model, decay=cfg.model.ema_decay)

    # ---- Load both: running (for buffers) and EMA (for weights) ----
    ck_in = cfg.model.checkpoint
    if ck_in is None:
        raise ValueError("Set config.model.checkpoint or pass --checkpoint (either AE_last.pth or AE_last_EMA.pth).")
    if not os.path.isabs(ck_in):
        ck_run, ck_ema = _resolve_ae_pair_paths(checkpoint_dir, ck_in)
    else:
        base_dir = os.path.dirname(ck_in)
        ck_run, ck_ema = _resolve_ae_pair_paths(base_dir, os.path.basename(ck_in))

    # 1) Load RUNNING weights into a temporary model to grab buffers
    model_run = get_model(cfg.model).to(device, memory_format=torch.channels_last)
    ema_run = EMA(model=model_run, decay=cfg.model.ema_decay)
    if not os.path.exists(ck_run):
        raise FileNotFoundError(f"Running checkpoint not found: {ck_run}")
    load_model(model_run, ema_run, ck_run, "AE", device=device, is_ema=False)
    # No apply_shadow here; we just want its buffers.

    # 2) Load EMA weights into the real model
    if not os.path.exists(ck_ema):
        raise FileNotFoundError(f"EMA checkpoint not found: {ck_ema}")
    load_model(model, ema, ck_ema, "AE", device=device, is_ema=True)
    ema.apply_shadow()  # switch parameters to EMA weights

    # 3) Copy normalization (and any other) buffers from running → EMA model
    _copy_named_buffers(model_run, model)

    model.eval()

    # (Optional) quick sanity print
    with torch.no_grad():
        print("[AE buffers copied from RUNNING] μ[:4]:",
            model.latent_norm_mean[:4].tolist(),
            "σ[:4]:", model.latent_norm_std[:4].tolist())


    # always: latent scatter + recon grid (+ optionally prior variance diagnostics)
    latent_cb = get_latent_scatter_callback(num_batches=num_batches, max_points=max_points)
    recon_cb = get_reconstruction_callback()
    prior_var_cb = get_prior_variance_callback()
    prior_vs_post_cb = get_prior_vs_posterior_var_callback()

    epoch = 0
    latent_cb(eval_loader, writer, model, device, epoch, tag_prefix="AE")
    batch = next(iter(eval_loader))
    recon_cb(batch, writer, model, device, epoch, tag_prefix="AE")
    # Diagnostics (they no-op if unavailable)
    prior_var_cb(writer, model, epoch, tag_prefix="AE")
    prior_vs_post_cb(eval_loader, writer, model, device, epoch, tag_prefix="AE")

    # ── latent diffusion model (for viz, geodesics, sampling) ───────────────
    lat_model, lat_sde = _maybe_load_latent_diffusion_exact(
        ae_cfg=cfg,
        device=device,
        ae_ckpt_dir=checkpoint_dir,
        ae_latent_dim=int(cfg.model.latent_dim),
        latent_cfg_path=latent_config,
        latent_ckpt_path=latent_ckpt,
    )

    # ── load geodesic CONFIG once, define orig_shape once ───────────────────
    geo = _load_geo_cfg(geo_config) if geo_config is not None else {}
    orig_shape = (int(cfg.model.latent_dim),)

    # ── encode a pool of latents ONCE and reuse everywhere ───────────────────
    Z_pool = encode_latents_from_loader(
        model=model, loader=eval_loader, device=device, min_count=6000, max_batches=None
    ).float().cpu()

    # normalized copy for diffusion-side ops
    Z_pool_hat = model.normalize_latent(Z_pool.to(device)).detach().cpu()

    # latent-only loader for entropy profile
    bs_ent = int(getattr(cfg.data, "batch_size", 256))
    loader_latents = DataLoader(TensorDataset(Z_pool_hat), batch_size=bs_ent, shuffle=False)

    # ── entropy-rate profile (optional) ──────────────────────────────────────
    entropic_profile_enabled = bool(geo.get("entropic_profile", True))
    existing_schedule = geo.get("time_schedule", [0.2, 0.15, 0.1, 0.05])
    n_stages = len(existing_schedule)

    if entropic_profile_enabled and lat_model is not None and lat_sde is not None:
        ent_num_t       = int(geo.get("ent_num_t", 96))
        ent_max_batches = int(geo.get("ent_max_batches", 12))
        ent_t_min       = geo.get("ent_t_min", None)
        ent_t_max       = float(geo.get("ent_t_max", 1.0))

        ent = compute_entropy_profile(
            model=lat_model,
            sde=lat_sde,
            loader=loader_latents,      # ← use latents, not images
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

        if bool(geo.get("use_entropic_schedule", False)):
            source = str(geo.get("entropic_source", "mmse")).lower()  # {"mmse","score"}
            if source == "score":
                t_grid = ent["Phi_score_t"]; phiR = ent["Phi_rescaled_score"]
            else:
                t_grid = ent["Phi_mmse_t"];  phiR = ent["Phi_rescaled_mmse"]

            t_lo = float(geo.get("ent_t_min", t_grid[0] if len(t_grid) else 1e-4))
            t_hi = float(geo.get("ent_t_max", t_grid[-1] if len(t_grid) else 1.0))
            entropic_sched = schedule_from_rescaled_entropic_time(
                t=np.asarray(t_grid, dtype=np.float64),
                Phi_rescaled=np.asarray(phiR, dtype=np.float64),
                n_stages=n_stages,
                t_lo=t_lo, t_hi=t_hi,
                descending=True,
            )
            print(f"[Entropy] Replacing time_schedule with uniform-in-Φ̃ ({source}) → {entropic_sched}")
            time_schedule = entropic_sched
        else:
            time_schedule = existing_schedule
    else:
        time_schedule = existing_schedule

    # ── latent geodesics (optional) ─────────────────────────────────────────
    if geo_config is not None and lat_model is not None and lat_sde is not None:
        # latent corruption montage (reuse Z_pool)
        viz_out = plot_latent_corruptions(
            latents=Z_pool_hat,
            latent_sde=lat_sde,
            time_schedule=time_schedule,    # smallest → largest shown left→right
            num_points=4000,
            seed=123,
            out_path=os.path.join(eval_dir, "latent_corruptions.png"),
            figsize_unit=3.0,
            dpi=150,
            point_size=4.0,
            alpha=0.8,
        )
        print(f"[Viz] Saved latent corruption montage → {viz_out['saved_path']}")

        # sample endpoints directly from underlying dataset (keep GT params)
        B = int(geo.get("num_pairs", 10))
        need = 2 * B
        base_ds, idx_pool = _unwrap_base_dataset(eval_loader.dataset)

        g_idx = torch.Generator()
        g_idx.manual_seed(int(geo.get("random_seed", getattr(cfg, "random_seed", 42))))
        perm_local = torch.randperm(len(idx_pool), generator=g_idx).tolist()[:need]

        imgs, rots = [], []
        for j in perm_local:
            x_j, r_j = base_ds[idx_pool[j]]  # expects (image, rotation/pose)
            imgs.append(x_j)
            rots.append(r_j)

        X = torch.stack(imgs, dim=0).to(device)
        Rsel = torch.stack(rots, dim=0)  # (need, 2, 2) for SO(2) case; dataset-specific otherwise

        with torch.no_grad():
            Zpairs = model.encode(X).detach()
        if Zpairs.size(0) < need:
            B = Zpairs.size(0) // 2
            if B == 0:
                print("[Geodesics] Not enough samples to form a single pair — skipping.")
            else:
                print(f"[Geodesics] Reducing num_pairs to {B} due to dataset size.")

        if B > 0:
            # normalize endpoints before solver
            p_z = model.normalize_latent(Zpairs[:B])
            q_z = model.normalize_latent(Zpairs[B:2 * B])

            # CG kwargs if needed by Jacobian metric
            cg_kwargs = _build_cg_kwargs(geo)

            optimizer   = str(geo.get("optimizer", "adam")).lower()
            line_search = str(geo.get("line_search", "armijo")).lower()
            if optimizer == "adam" and line_search == "strong_wolfe":
                print("[Geodesics] Adam + strong_wolfe is unsupported; using 'armijo' to match legacy behavior.")
                line_search = "armijo"

            seed = int(geo.get("random_seed", getattr(cfg, "random_seed", 42)))
            g = torch.Generator(device=p_z.device) if p_z.is_cuda else torch.Generator()
            g.manual_seed(seed)
            xi = torch.randn(p_z.shape, dtype=p_z.dtype, device=p_z.device, generator=g)

            path_z, info, _ = compute_geodesic(
                p=p_z, q=q_z,
                model=lat_model, sde=lat_sde, orig_shape=orig_shape,
                n_segments=int(geo.get("n_segments", 15)),
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
                endpoint_mode=str(geo.get("endpoint_mode", "clean")),
                fixed_noise=None,  # relevant only if endpoint_mode=="noisy"
                init_nseeds=20,
            )

            # ── FIX: realized frames are number of path knots, not batch size
            T_realized = int(len(path_z))
            print(f"[Geodesics] Requested n_segments={geo.get('n_segments', 15)} → realized frames T={T_realized}")

            # denormalize before decoding
            path_z_denorm = [model.denormalize_latent(pt) for pt in path_z]

            # Interactive viz (unchanged)
            try:
                latent_dim = int(cfg.model.latent_dim)
            except Exception:
                latent_dim = p_z.shape[1]

            t_final = float(time_schedule[-1] if isinstance(time_schedule, list) else time_schedule)
            Z_bg_hat = Z_pool_hat.to(device)
            Z_bg_t_hat = perturb_latents_at_time(Z_bg_hat, lat_sde, t_final)
            if latent_dim >= 3:
                save_interactive_latent_geodesic_html(
                    path_z_list=path_z,
                    z_bg_pert=Z_bg_t_hat,
                    out_dir=eval_dir,
                    filename="latent_geodesics_latent3d.html",
                )
            elif latent_dim == 2:
                save_latent_geodesic_2d(
                    path_z_list=path_z,
                    z_bg_pert=Z_bg_t_hat,
                    out_dir=eval_dir,
                    filename="latent_geodesics_latent2d.png",
                )
                stage_raw = info.get("stage_raw_paths", None)
                if stage_raw and latent_dim == 2:
                    z_bg_per_stage = {}
                    Z_pool_hat_dev = Z_pool_hat.to(device)
                    for item in stage_raw:
                        t_stage = float(item['t'])
                        with torch.no_grad():
                            z_bg_t = _latent_cloud_at_t_with_marginal(Z_pool_hat_dev, lat_sde, t_stage, seed=123)
                        z_bg_per_stage[t_stage] = z_bg_t.detach().cpu()

                    out_path_stage = save_latent_geodesics_stages_2d(
                        stage_paths=stage_raw,
                        z_bg_per_stage=z_bg_per_stage,
                        out_dir=eval_dir,
                        filename="latent_geodesics_stages_2d.png",
                        order="asc",
                        figsize_unit=3.0, dpi=150, point_size=4.0, alpha=0.8,
                    )
                    print(f"[Viz] Saved stage-wise latent geodesics → {out_path_stage}")

            else:
                print(f"[Geodesics] Latent dim {latent_dim} < 3; skipping interactive latent 3D viz.")

            # save decoded geodesic trajectories grid (unchanged helper)
            decode_latent_path_and_save_grid(
                path_z_list=path_z_denorm,
                decoder=model.decode,
                eval_dir=eval_dir,
                writer=writer,
                tag="LatentGeodesics/DecodedGrid",
                filename="latent_geodesics_grid.png",
            )
            
            # ── GT comparison and RMSE
            avg_rmse = None
            try:
                if hasattr(base_ds, "compute_geodesic"):
                    # Consistent naming avoids accidental reuse
                    B_pairs = int(B)
                    T_frames = int(len(path_z_denorm))  # realized frames = n_segments + 1

                    # 1) Decode predicted trajectories first  → pred_stack: (B, T, C, H, W)
                    pred_imgs = []
                    for b in range(B_pairs):
                        z_traj_b = torch.stack([path_z_denorm[k][b] for k in range(T_frames)], dim=0).to(device)  # (T, D)
                        with torch.no_grad():
                            dec_b = model.decode(z_traj_b)  # (T, C, H, W)
                        pred_imgs.append(dec_b.detach().cpu().clamp(0.0, 1.0))
                    pred_stack = torch.stack(pred_imgs, dim=0)  # (B, T, C, H, W)

                    # 2) Build the GT timeline with the SAME number of frames
                    t_lin = torch.linspace(0.0, 1.0, T_frames)

                    # Endpoints (dataset parameters) for the same pairs you decoded
                    P = Rsel[:B_pairs].detach().cpu()
                    Q = Rsel[B_pairs:2 * B_pairs].detach().cpu()

                    # 3) Compute ground-truth frames with the SAME B and T
                    gt_stack = base_ds.compute_geodesic(P, Q, t_lin).detach().cpu()  # (B, T, C, H, W)

                    print(f'pred_stack.size():{pred_stack.size()}')
                    print(f'gt_stack.size():{gt_stack.size()}')

                    # 4) Sanity checks before subtraction (fail fast if anything drifts)
                    assert gt_stack.shape[0] == pred_stack.shape[0], \
                        f"B mismatch: pred {pred_stack.shape[0]} vs gt {gt_stack.shape[0]}"
                    assert gt_stack.shape[1] == pred_stack.shape[1], \
                        f"T mismatch: pred {pred_stack.shape[1]} vs gt {gt_stack.shape[1]}"
                    assert gt_stack.shape[2:] == pred_stack.shape[2:], \
                        f"CHW mismatch: pred {pred_stack.shape[2:]} vs gt {gt_stack.shape[2:]}"

                    

                    # 5) RMSE per pair, then average
                    mse_per_pair = ((pred_stack - gt_stack) ** 2).mean(dim=(1, 2, 3, 4))
                    rmse_per_pair = torch.sqrt(mse_per_pair)
                    avg_rmse = rmse_per_pair.mean().item()

                    writer.add_scalar("LatentGeodesics/AvgRMSE_vs_GT", float(avg_rmse), epoch)
                    txt_path = os.path.join(eval_dir, "geodesic_eval.txt")
                    with open(txt_path, "w") as f:
                        f.write(f"Avg RMSE vs GT (B={B_pairs}, T={T_frames}): {avg_rmse:.6f}\n")
                    print(f"[Geodesics][GT] Avg RMSE vs ground-truth (B={B_pairs}, T={T_frames}): {avg_rmse:.6f}")
                    print(f"[Geodesics][GT] Saved → {txt_path}")

                    # side-by-side visual
                    comp_path = os.path.join(eval_dir, "latent_geodesics_vs_gt.png")
                    _save_geodesic_comparison_grid(
                        pred_stack=pred_stack, gt_stack=gt_stack, out_path=comp_path,
                        padding=2, row_gap=2, pair_gap=12,
                    )
                    print(f"[Geodesics][GT] Comparison image saved → {comp_path}")

                    # stash into info (optional)
                    if isinstance(info, dict):
                        info["avg_rmse_vs_gt"] = float(avg_rmse)
                        info["geodesic_T"] = int(T_frames)

                else:
                    print("[Geodesics][GT] Dataset does not expose 'compute_geodesic' — skipping GT evaluation.")
            except Exception as e:
                # helpful dump if anything ever goes wrong again
                print(f"[Geodesics][GT] Failed to evaluate GT deviation: {e}")
                try:
                    print("[Geodesics][GT][debug]",
                        "B_pairs:", B_pairs if 'B_pairs' in locals() else None,
                        "T_frames:", T_frames if 'T_frames' in locals() else None,
                        "pred_stack:", tuple(pred_stack.shape) if 'pred_stack' in locals() else None)
                except Exception:
                    pass



            # persist the latent path & loss info (for later analysis)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "latent_geodesics_info.pkl"), "wb") as f:
                pickle.dump(dict(loss_info=info, config=geo), f)
            torch.save(
                {"path_latent": [pt.detach().cpu() for pt in path_z]},
                os.path.join(eval_dir, "latent_geodesics_path_latent.pt"),
            )
            print(f"[Geodesics] Stored latent geodesics and loss info.")

    # latent sampling → decode → grid (optional)
    if do_sample and (lat_model is not None and lat_sde is not None):
        score_fn = lat_model.get_score_fn(lat_sde)
        z_shape = (int(sample_count), int(cfg.model.latent_dim))
        z_hat = Algorithm1(lat_sde, int(sample_steps), score_fn, z_shape, device, y=None)  # normalized
        z = model.denormalize_latent(z_hat)
        decode_and_save_grid(
            decoder=model.decode,
            z_samples=z,
            eval_dir=eval_dir,
            writer=writer,
            grid_nrow=grid_nrow,
            tag="Latent2Image/Samples",
            epoch=0,
        )

    writer.close()


def main():
    p = argparse.ArgumentParser(
        "AE eval: scatter + recon + variance diagnostics + latent corruption viz + latent geodesics + latent sampling"
    )
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override cfg.model.checkpoint (use exact .pth name; *_EMA.pth for EMA).")
    p.add_argument("--use_test", action="store_true")
    p.add_argument("--num_batches", type=int, default=10)
    p.add_argument("--max_points", type=int, default=3000)

    # optional: how many batches to average for isometry eval (kept for compatibility)
    p.add_argument("--iso_batches", type=int, default=None)

    # geodesics
    p.add_argument("--geo_config", type=str, default=None,
                   help="Path to a geodesic CONFIG .py for latent geodesics (optional).")

    # latent sampling controls
    p.add_argument("--latent_config", type=str, default=None,
                   help="Path to latent diffusion config (optional; else from AE loss.geom.latent).")
    p.add_argument("--latent_ckpt", type=str, default=None,
                   help="Exact latent .pth to load (use *_EMA.pth to evaluate EMA).")
    p.add_argument("--sample_latents", action="store_true")
    p.add_argument("--sample_steps", type=int, default=250)
    p.add_argument("--sample_count", type=int, default=36)
    p.add_argument("--grid_nrow", type=int, default=None,
                   help="Images per row in the sampling grid. Auto-square if not set.")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint is not None:
        cfg.model.checkpoint = args.checkpoint  # honor exact filename given

    out_dir = os.path.join(cfg.base_log_dir, cfg.experiment)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_eval_autoencoder.pkl"), "wb") as f:
        pickle.dump(cfg.to_dict(), f)

    eval_autoencoder(
        cfg,
        use_test=bool(args.use_test),
        num_batches=int(args.num_batches),
        max_points=int(args.max_points),
        iso_batches=args.iso_batches,
        latent_config=args.latent_config,
        latent_ckpt=args.latent_ckpt,
        do_sample=bool(args.sample_latents),
        sample_steps=int(args.sample_steps),
        sample_count=int(args.sample_count),
        grid_nrow=args.grid_nrow,
        geo_config=args.geo_config,
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
