#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AE eval with latent Riemannian interpolation (pairwise) + decoding & viz.

Key points in this version
--------------------------
• Uses your RO config *as-is* — no overrides, no setdefaults, no renames.
• Pairwise objective is idx-aware (Strong-Wolfe works with subset eval).
• Metric + CG built strictly from ro_cfg fields.

CLI
---
python -m scripts.autoencoder_ro_interp \
  --config configs/rotated_mnist/autoencoder.py \
  --ro_config configs/rotated_mnist/jsm_ro.py \
  [--checkpoint ...] [--latent_config ...] [--latent_ckpt ...] [--use_test]
"""

from __future__ import annotations

import os
import argparse
import pickle
import math
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from configs import load_config
from data.data_utils_fast import get_dataloaders
from models import get_model
from utils.train_utils import prepare_training_dirs, EMA, load_model
from utils.ae_utils import get_reconstruction_callback, get_latent_scatter_callback
from sde import configure_sde

# data-geometry helpers
from data_geometry.utils.diffusion_utils import get_denoiser_fn, get_score_fn
from data_geometry.metrics import create_metric
from data_geometry.riemannian_optimization.optimizers import GenericRiemannianGD, GenericRiemannianAdam


# ────────────────────────── helpers ──────────────────────────

def _reshape_std(std: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return std.view(-1, *([1] * (like.dim() - 1)))

def _gather_images(loader, device: torch.device, needed: int) -> torch.Tensor:
    xs, count, it = [], 0, iter(loader)
    while count < needed:
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch[0].to(device, non_blocking=True)
        xs.append(x); count += x.size(0)
    if not xs:
        raise RuntimeError("Eval loader is empty; cannot sample pairs.")
    X = torch.cat(xs, dim=0)
    if X.size(0) < needed:
        print(f"[Eval] Warning: requested {needed} images but only found {X.size(0)}.")
    return X[:needed]

@torch.no_grad()
def _gather_encoded_latents(model, loader, device: torch.device, needed: int) -> torch.Tensor:
    X = _gather_images(loader, device, needed)
    return model.encode(X.to(device, non_blocking=True)).detach()

def _perturb_latents_at_time(z: torch.Tensor, sde, t_val: float) -> torch.Tensor:
    t = torch.tensor(float(t_val), device=z.device)
    with torch.no_grad():
        mean, std = sde.marginal_prob(z, t)
        xi = torch.randn_like(z)
        return mean + _reshape_std(std, z) * xi

def _auto_grid_nrow(n: int) -> int:
    return int(math.ceil(math.sqrt(max(1, n))))

@torch.no_grad()
def _decode_and_save_grid_variable_T(
    decoder,
    path_z_list: List[torch.Tensor],   # list[T] of (B, D)
    eval_dir: str,
    writer: SummaryWriter,
    *,
    tag: str = "RO/DecodedGrid",
    filename: str = "ro_latent_path_grid.png",
    epoch: int = 0,
):
    """B×T grid: each row = one pair's trajectory; columns = path nodes (T can vary)."""
    if isinstance(decoder, nn.Module):
        was_training = decoder.training
        decoder.eval()
        decoded_steps = [decoder(z).detach().cpu() for z in path_z_list]
        if was_training:
            decoder.train()
    else:
        decoded_steps = [decoder(z).detach().cpu() for z in path_z_list]

    T = len(decoded_steps)
    B = decoded_steps[0].size(0)
    tiles = []
    for i in range(B):
        for t in range(T):
            tiles.append(decoded_steps[t][i])

    grid = vutils.make_grid(tiles, nrow=T, normalize=True, scale_each=True)
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, filename)
    vutils.save_image(grid, out_path)
    writer.add_image(tag, grid, epoch)
    print(f"[RO] Saved decoded RO path grid → {out_path}  (rows={B}, cols={T})")

def _save_interactive_latent3d(
    *,
    path_z_list: List[torch.Tensor],
    z_bg_pert: torch.Tensor,
    out_dir: str,
    filename: str = "ro_latent_path_3d.html",
) -> str:
    """Interactive 3D viz of the first 3 dims of the latent trajectory."""
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plotly_plot
    except Exception as e:
        print(f"[RO] Plotly not available ({e}); skipping 3D viz.")
        return ""

    traj = torch.stack([pt.detach().cpu() for pt in path_z_list], dim=0)
    if traj.ndim == 2:
        traj = traj.unsqueeze(1)
    T, B, D = traj.shape
    if D < 3:
        print(f"[RO] Latent dim D={D} < 3; skipping 3D viz.")
        return ""

    z_bg = z_bg_pert.detach().cpu()
    if z_bg.shape[1] < 3:
        print(f"[RO] Background latent dim {z_bg.shape[1]} < 3; skipping 3D viz.")
        return ""

    fig = go.Figure()
    for i in range(B):
        tr = traj[:, i, :3].numpy()
        fig.add_trace(go.Scatter3d(x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode="lines", name=f"pair_{i+1}"))
    cloud = z_bg[:, :3].numpy()
    fig.add_trace(go.Scatter3d(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2], mode="markers",
                               name="latent @ t", marker=dict(size=2, opacity=0.35)))
    fig.update_layout(
        title="RO latent paths (first 3 dims)",
        scene=dict(xaxis_title="z1", yaxis_title="z2", zaxis_title="z3"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plotly_plot(fig, filename=out_path, auto_open=False)
    print(f"[RO] Interactive latent 3D plot → {out_path}")
    return out_path

def _load_py_cfg(path: str) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    with open(path, "r") as f:
        code = compile(f.read(), path, "exec")
        exec(code, scope)
    if "CONFIG" not in scope:
        raise ValueError(f"Config '{path}' must define a CONFIG dict.")
    return scope["CONFIG"]

def _load_ae_exact(model, ema: EMA, ckpt_path: str, device: torch.device):
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
    latent_cfg_path: Optional[str],
    latent_ckpt_path: Optional[str],
):
    # config
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

    latent_model = get_model(lat_cfg.model).to(device)
    for p in latent_model.parameters():
        p.requires_grad_(False)
    latent_model.eval()
    latent_sde = configure_sde(lat_cfg)

    # checkpoint
    ckpt = latent_ckpt_path or getattr(lat_cfg.model, "checkpoint", None) or os.path.join(ae_ckpt_dir, "LatentDiff_last.pth")
    if not ckpt.endswith(".pth"): ckpt += ".pth"
    if not os.path.isabs(ckpt) and latent_ckpt_path is None and getattr(lat_cfg, "checkpoint_dir", None):
        ckpt = os.path.join(lat_cfg.checkpoint_dir, ckpt)
    if not os.path.isabs(ckpt) and not os.path.exists(ckpt):
        ckpt = os.path.join(ae_ckpt_dir, os.path.basename(ckpt))

    if not os.path.exists(ckpt):
        print(f"[Eval] Latent diffusion checkpoint not found: {ckpt}. Skipping RO.")
        return None, None

    # strict: trust filename for EMA vs running
    is_ema = ckpt.endswith("_EMA.pth")
    ema_lat = EMA(latent_model, decay=float(getattr(lat_cfg.model, "ema_decay", 0.999)))
    load_model(latent_model, ema_lat, ckpt, "LatentDiff", device=device, is_ema=is_ema)
    if is_ema:
        ema_lat.apply_shadow()
    print(f"[Eval] Latent diffusion model loaded from '{ckpt}' (EMA={is_ema})")
    return latent_model, latent_sde


# ---------------- Riemannian latent path (pairwise p→q) ----------------

def _compute_latent_path_via_riemopt(
    *,
    cfg,
    ro_cfg: Dict[str, Any],
    ae_model,            # AE (for encode/decode only)
    lat_model,           # latent diffusion model (for metric)
    lat_sde,             # latent SDE (for metric/denoiser)
    eval_loader,
    device: torch.device,
    eval_dir: str,
    writer: SummaryWriter,
):
    # how many pairs?
    B = int(ro_cfg.get("num_pairs", 8))
    needed = 2 * B

    # take 2B images → encode
    X = _gather_images(eval_loader, device, needed)
    with torch.no_grad():
        Z = ae_model.encode(X).detach()
    if Z.size(0) < needed:
        B = Z.size(0) // 2
        if B == 0:
            print("[RO] Not enough samples to form a single pair.")
            return
        print(f"[RO] Reducing num_pairs to {B} due to short dataset batch.")
    p_z, q_z = Z[:B], Z[B:2 * B]     # (B,d)
    d = p_z.shape[1]
    orig_shape = (int(cfg.model.latent_dim),)

    # metric @ t = ro_cfg["riem_t"]
    t_val = float(ro_cfg.get("riem_t", 0.01))
    t_tensor = torch.tensor(t_val, dtype=torch.float32, device=device)
    snr_value = lat_sde.snr(t_tensor).item() if hasattr(lat_sde, "snr") else float("nan")
    print(f"[RiemOpt] 💡 Optimizing at t={t_val:.4f} (SNR={snr_value:.4f})")

    score_fn = get_score_fn(lat_sde, lat_model, t_tensor, orig_shape)
    denoiser_fn = get_denoiser_fn(lat_sde, lat_model, t_tensor, orig_shape)

    # Metric strictly from ro_cfg
    metric = create_metric(
        metric_type=ro_cfg.get("metric_type", "jacobian"),
        score_fn=score_fn,
        lam_metric=ro_cfg.get("lam_metric", ro_cfg.get("reg_lambda", 1e-3)),
        denoiser_fn=denoiser_fn,
        cg_kwargs=dict(
            reg_lambda=ro_cfg.get("reg_lambda", 1e-3),
            max_iter=ro_cfg.get("cg_max_iter", 4),
            tol=ro_cfg.get("cg_tol", 5e-5),
            preconditioner=ro_cfg.get("cg_preconditioner", "diagonal"),
            precond_diag_samples=ro_cfg.get("cg_precond_diag_samples", 6),
        ),
    )

    # ---- objective: pairwise Euclidean f(x)=||x - q||^2, idx-aware for SW ----
    def pairwise_objective_builder(targets: torch.Tensor):
        targets = targets.detach().to(device)   # (B,d)
        def f(x: torch.Tensor, idx=None) -> torch.Tensor:
            if idx is None:
                tgt = targets
            else:
                tgt = targets[idx]
            if x.shape != tgt.shape:
                raise RuntimeError(
                    f"Objective called with shape {tuple(x.shape)} but targets have shape {tuple(tgt.shape)}"
                )
            return (x - tgt).pow(2).sum(dim=1)
        return f

    f_obj = pairwise_objective_builder(q_z)

    # ---- run Generic Riemannian GD using ro_cfg EXACTLY as provided ----
    algo = str(ro_cfg.get("optim_algo", "gd")).lower()
    Optim = GenericRiemannianAdam if algo in ("adam", "radam", "riemannian_adam") else GenericRiemannianGD
    optimiser = Optim(metric, f_obj, ro_cfg)
    traj_cpu, metrics = optimiser.run(x0=p_z.detach())

    # convert to device tensors for viz/decoding
    traj = [pt.to(device) for pt in traj_cpu]

    # latent 3D viz (first 3 dims) using perturbed background at t = riem_t
    latent_dim = int(getattr(cfg.model, "latent_dim", d))
    if latent_dim >= 3:
        Z_bg = _gather_encoded_latents(ae_model, eval_loader, device, 3000)
        Z_bg_t = _perturb_latents_at_time(Z_bg, lat_sde, t_val)
        _save_interactive_latent3d(
            path_z_list=traj,
            z_bg_pert=Z_bg_t,
            out_dir=eval_dir,
            filename="ro_latent_path_3d.html",
        )
    else:
        print(f"[RO] Latent dim {latent_dim} < 3; skipping interactive 3D viz.")

    # decode trajectory and save grid (variable T)
    _decode_and_save_grid_variable_T(
        ae_model.decode, traj, eval_dir, writer,
        tag="RO/DecodedGrid", filename="ro_latent_path_grid.png", epoch=0
    )

    # persist the latent path & metrics for analysis
    os.makedirs(eval_dir, exist_ok=True)
    torch.save(
        {"path_latent": [pt.detach().cpu() for pt in traj]},
        os.path.join(eval_dir, "ro_latent_path.pt"),
    )
    with open(os.path.join(eval_dir, "ro_latent_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    print(f"[RO] Stored RO latent path (T={len(traj)}, B={B}) and metrics.")

# ────────────────────────── eval orchestrator ──────────────────────────

def eval_autoencoder(
    cfg,
    *,
    use_test: bool,
    num_batches: int,
    max_points: int,
    iso_batches: Optional[int],
    latent_config: Optional[str],
    latent_ckpt: Optional[str],
    ro_config: Optional[str],
):
    _, checkpoint_dir, eval_dir = prepare_training_dirs(cfg)
    writer = SummaryWriter(log_dir=eval_dir)
    device = torch.device(cfg.training.device)

    _, val_loader, test_loader = get_dataloaders(cfg.data, seed=cfg.random_seed)
    eval_loader = test_loader if use_test else val_loader

    # AE
    ae = get_model(cfg.model).to(device, memory_format=torch.channels_last)
    ema = EMA(model=ae, decay=cfg.model.ema_decay)

    ckpt = cfg.model.checkpoint
    if ckpt is None:
        raise ValueError("Set config.model.checkpoint or pass --checkpoint.")
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(checkpoint_dir, ckpt)
    if not ckpt.endswith(".pth"):
        ckpt += ".pth"
    _load_ae_exact(ae, ema, ckpt, device=device)
    ae.eval()

    # latent diffusion (needed for metric/denoiser)
    lat_model, lat_sde = _maybe_load_latent_diffusion_exact(
        ae_cfg=cfg,
        device=device,
        ae_ckpt_dir=checkpoint_dir,
        ae_latent_dim=int(cfg.model.latent_dim),
        latent_cfg_path=latent_config,
        latent_ckpt_path=latent_ckpt,
    )
    if lat_model is None or lat_sde is None:
        print("[Eval] Skipping RO latent interpolation — latent model not available.")
        return

    # scatter + recon snapshots (for context)
    epoch = 0
    latent_cb = get_latent_scatter_callback(num_batches=num_batches, max_points=max_points)
    recon_cb = get_reconstruction_callback()
    latent_cb(eval_loader, writer, ae, device, epoch, tag_prefix="AE")
    batch = next(iter(eval_loader))
    recon_cb(batch, writer, ae, device, epoch, tag_prefix="AE")

    # Riemannian latent path
    if ro_config is None:
        raise ValueError("Pass --ro_config with a Riemannian optimisation CONFIG .py file.")
    ro_cfg = _load_py_cfg(ro_config)

    _compute_latent_path_via_riemopt(
        cfg=cfg,
        ro_cfg=ro_cfg,
        ae_model=ae,
        lat_model=lat_model,
        lat_sde=lat_sde,
        eval_loader=eval_loader,
        device=device,
        eval_dir=eval_dir,
        writer=writer,
    )

    writer.close()

# ────────────────────────── CLI ──────────────────────────

def main():
    p = argparse.ArgumentParser("AE eval + latent Riemannian interpolation")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override cfg.model.checkpoint (use exact .pth; *_EMA.pth for EMA).")
    p.add_argument("--use_test", action="store_true")
    p.add_argument("--num_batches", type=int, default=10)
    p.add_argument("--max_points", type=int, default=3000)
    p.add_argument("--iso_batches", type=int, default=None)

    # latent diffusion (metric)
    p.add_argument("--latent_config", type=str, default=None,
                   help="Path to latent diffusion config (optional; else from AE loss.geom.latent).")
    p.add_argument("--latent_ckpt", type=str, default=None,
                   help="Exact latent .pth to load (use *_EMA.pth to evaluate EMA).")

    # Riemannian optimisation config
    p.add_argument("--ro_config", type=str, required=True,
                   help="Path to RO CONFIG .py (separate file).")

    args = p.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint is not None:
        cfg.model.checkpoint = args.checkpoint

    out_dir = os.path.join(cfg.base_log_dir, cfg.experiment)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_eval_ro.pkl"), "wb") as f:
        pickle.dump(cfg.to_dict(), f)

    eval_autoencoder(
        cfg,
        use_test=bool(args.use_test),
        num_batches=int(args.num_batches),
        max_points=int(args.max_points),
        iso_batches=args.iso_batches,
        latent_config=args.latent_config,
        latent_ckpt=args.latent_ckpt,
        ro_config=args.ro_config,
    )

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
