#!/usr/bin/env python
# =====================================================================
# Batch geodesic evaluation on a score-based manifold (image/ambient space)
# (aligned with your latent driver: noisy endpoints + t-project init,
#  single final denoise, Jacobian metric, Adam+Armijo, entropy profiles,
#  GT deviation & estimated-vs-GT comparison grid)
# =====================================================================
from __future__ import annotations
import os, time, random, pickle
from argparse import ArgumentParser
from typing import Dict, Any, Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

# ─── Project-local Imports ───────────────────────────────────────────
from data.data_utils_fast import get_dataloaders
from models                  import get_model
from sde                     import configure_sde
from utils.train_utils       import prepare_training_dirs, EMA, load_model
from configs                 import load_config
from data_geometry.utils.visualization import visualize_riemannian_optimization_selector
from data_geometry.geodesics.fast_geodesic_computation import compute_geodesic

import imageio.v2 as imageio
import shutil

# ─── Import initialization/plot helpers from debugger ────────────────
# We keep these imports; by default we won't use DDIM/ODE init, but "linear" and "ode"
# are supported for backwards-compat.
from .debug_initialization import (
    flatten, unflatten,
    get_score_fn,
    save_path_grid,
    generate_ode_initialized_path,
)

# ─── Entropy profile + GT comparison grid helpers ────────────────────
from utils.entropy_profile import compute_entropy_profile, schedule_from_rescaled_entropic_time
from utils.ae_utils import _save_geodesic_comparison_grid  # identical layout to latent driver


# ─── Misc Helpers ────────────────────────────────────────────────────
def load_geo_cfg(path: str) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    with open(path, "r") as f:
        exec(compile(f.read(), path, "exec"), scope)
    if "CONFIG" not in scope:
        raise ValueError(f"Config file '{path}' must define a 'CONFIG' dictionary.")
    return scope["CONFIG"]

def _split_data(items: List[Any]) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Splits dataset items into image and rotation tensors (if present)."""
    if isinstance(items[0], (tuple, list)):
        imgs = torch.stack([it[0] for it in items])
        rots = torch.stack([it[1] for it in items])
        return imgs, rots
    return torch.stack(items), None

# ─── Loss Plot ───────────────────────────────────────────────────────
def _plot_loss(info: Dict[str, Any], lam_s: float, lam_m: float, path: str):
    stages = info.get('optimization_history', [])
    if not stages: return

    it, tot, en, smo, mon, bd, off = [], [], [], [], [], [], 0
    for st in stages:
        h = st['history']
        g = [i + off for i in h['iter']]
        it.extend(g)
        off += h['iter'][-1]
        bd.append(off)
        tot.extend(h['total_loss'])
        en.extend(h['geodesic_energy'])
        smo.extend([v * lam_s for v in h['smoothness_penalty']])
        mon.extend([v * lam_m for v in h['monotonicity_penalty']])

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(16, 22))
    labels = ["Total Loss", "Geodesic Energy", "λ·Smoothness", "λ·Monotonicity"]
    data_series = [tot, en, smo, mon]

    for axis, data, label in zip(ax, data_series, labels):
        axis.plot(it, data, label=label)
        axis.legend()
        axis.grid(linestyle=':')
        for boundary in bd[:-1]:
            axis.axvline(boundary, color='r', linestyle='--', linewidth=0.8)

    ax[-1].set_xlabel("Optimization Iteration")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def create_opt_animation(path_history: List[List[torch.Tensor]],
                         output_filename: str,
                         vis_kwargs: Dict[str, Any],
                         duration: float = 0.1):
    """
    Build a GIF from the recorded optimization path_history.
    Each entry in path_history is a list[Tensor] representing the whole path at that iteration.
    """
    temp_dir = os.path.join(vis_kwargs['log_dir'], "temp_gif_frames")
    os.makedirs(temp_dir, exist_ok=True)

    vw = dict(vis_kwargs)
    vw['log_dir'] = temp_dir

    frame_paths = []
    for i, path_step in enumerate(path_history):
        basename = f"frame_{i:04d}.png"
        # draw one frame
        visualize_riemannian_optimization_selector(
            **vw,
            trajectories=path_step,
            plot_filename=basename
        )
        # The viz util may prefix filenames (e.g., "trajectory_grid_")
        p1 = os.path.join(temp_dir, basename)
        p2 = os.path.join(temp_dir, f"trajectory_grid_{basename}")
        frame_paths.append(p1 if os.path.exists(p1) else p2)

    with imageio.get_writer(output_filename, mode='I', duration=duration, loop=0) as w:
        for fp in frame_paths:
            if os.path.exists(fp):
                w.append_data(imageio.imread(fp))
    print(f"Saved optimization animation to {output_filename}")

    shutil.rmtree(temp_dir, ignore_errors=True)


# ─── Main Geodesic Computation Routine ───────────────────────────────
def run_geodesic_computation(
        geo: Dict[str, Any], model: torch.nn.Module, sde: Any, dset: Dataset,
        device: torch.device, eval_dir: str, visualize: bool):

    B = int(geo.get("num_pairs", 8))
    opt_settings = geo.get("optimizer_settings", {})
    micro_batch_size = int(opt_settings.get("optimizer_batch_size", B))

    g = torch.Generator().manual_seed(int(geo.get("random_seed", 42)))
    idx = torch.randperm(len(dset), generator=g)[:2 * B]
    imgs, rots = _split_data([dset[i] for i in idx])

    p_img_full, q_img_full = imgs.chunk(2)
    p_rot_full, q_rot_full = (rots.chunk(2) if rots is not None else (None, None))

    all_paths: List[List[torch.Tensor]] = []
    all_infos: List[Dict[str, Any]] = []
    all_path_histories: List[List[List[torch.Tensor]]] = []
    initial_path_to_visualize: Optional[List[torch.Tensor]] = None
    first_batch_shape: Optional[Tuple[int, int, int]] = None

    # schedule / init config
    end_time = geo.get("time_schedule", [0.03])[0]
    init_cfg  = geo.get("initialization", {})

    t0_wall = time.time()

    # ── per micro-batch ───────────────────────────────────────────────
    for i in range(0, B, micro_batch_size):
        print(f"\n── Processing Batch {i//micro_batch_size + 1}/{(B + micro_batch_size - 1)//micro_batch_size} (pairs {i} to {i+micro_batch_size-1}) ──")

        p_img = p_img_full[i:i+micro_batch_size].to(device)
        q_img = q_img_full[i:i+micro_batch_size].to(device)
        shp = p_img.shape[1:]  # (C,H,W)
        first_batch_shape = first_batch_shape or shp
        p_flat, q_flat = flatten(p_img), flatten(q_img)

        with torch.no_grad():
            mse_diff = torch.mean((p_flat - q_flat)**2).item()
            print(f"[Debug] MSE between initial p and q: {mse_diff:.4e}")

        # ---- Initialization policy (aligned with latent driver) ----
        init_method = str(init_cfg.get("method", "none")).lower()  # {"none","linear","ode"}
        init_path_opt: Optional[List[torch.Tensor]] = None

        if init_method == "linear":
            T = int(geo.get("n_segments", 16)) + 1
            lam = torch.linspace(0, 1, T, device=device, dtype=p_flat.dtype).view(T, 1, 1)
            gamma0 = (1 - lam) * p_flat.unsqueeze(0) + lam * q_flat.unsqueeze(0)  # (T, B, D)
            init_path_opt = [gamma0[k].clone().detach() for k in range(T)]
        elif init_method == "ode":
            init_path_opt = generate_ode_initialized_path(
                p_flat, q_flat, sde, model, shp,
                n_segments=int(geo.get("n_segments", 16)),
                end_time=float(end_time),
                num_solver_steps=int(init_cfg.get("ode_solver_steps", 250)),
                t0=float(init_cfg.get("t0", 1e-3)),
                device=device
            )
            if i == 0:
                initial_path_to_visualize = [t.detach().clone() for t in init_path_opt]
        else:
            init_path_opt = None  # preferred; compute_geodesic will do t-project init if endpoint_mode=="noisy"

        # ---- Optimizer / line-search options ----
        optimizer = str(geo.get("optimizer", "adam")).lower()
        line_search = str(geo.get("line_search", "armijo")).lower()
        if optimizer == "adam" and line_search == "strong_wolfe":
            print("[driver] Warning: Adam + strong_wolfe is not supported; switching to 'armijo'.")
            line_search = "armijo"

        # ---- Metric CG kwargs (Jacobian / JSM only) ----
        mt = str(geo.get("metric_type", "stein")).lower()
        if mt in {"jacobian", "jsm"}:
            cg_kwargs = dict(
                reg_lambda=float(geo.get("lam_metric", geo.get("reg_lambda", 1e-5))),
                max_iter=int(geo.get("cg_max_iter", 50)),
                tol=float(geo.get("cg_tol", 1e-6)),
                preconditioner=str(geo.get("cg_preconditioner", "diagonal")),
                precond_diag_samples=int(geo.get("cg_precond_diag_samples", 10)),
            )
        else:
            cg_kwargs = {}

        # Force single final denoise (no DDIM chain)
        post_denoise_fn = None

        path, info, path_history = compute_geodesic(
            p=p_flat, q=q_flat,
            initial_path=init_path_opt,
            model=model, sde=sde, orig_shape=shp,

            # discretisation & schedule
            n_segments=int(geo.get("n_segments", 16)),
            time_schedule=geo.get("time_schedule", [0.03]),

            # regularisers
            lam_smooth=float(geo.get("lam_smooth", 1e-3)),
            lam_mono=float(geo.get("lam_mono", 0.0)),

            # learning / budget
            max_iters=int(geo.get("max_iters", 500)),
            tol=float(geo.get("tol", 1e-6)),
            patience=int(geo.get("patience", 30)),

            # optimizer selection & params
            optimizer=optimizer,
            adam_lr=float(geo.get("adam_lr", 1e-2)),
            betas=tuple(geo.get("betas", (0.9, 0.999))),
            line_search=line_search,
            use_retraction_update=bool(geo.get("use_retraction_update", True)),

            # Armijo
            armijo_rho=float(geo.get("armijo_rho", 1e-4)),
            armijo_beta=float(geo.get("armijo_beta", 0.5)),
            armijo_max_iter=int(geo.get("armijo_max_iter", 20)),

            # Strong-Wolfe (kept for RGD pathway)
            wolfe_c1=float(geo.get("wolfe_c1", 1e-4)),
            wolfe_c2=float(geo.get("wolfe_c2", 0.9)),
            wolfe_max_bracket=int(geo.get("wolfe_max_bracket", 10)),
            wolfe_max_zoom=int(geo.get("wolfe_max_zoom", 10)),
            wolfe_max_alpha=float(geo.get("wolfe_max_alpha", 50.0)),

            # metric
            metric_type=str(geo.get("metric_type", "stein")),
            lam_metric=float(geo.get("lam_metric", 1.0)),
            cg_kwargs=cg_kwargs,

            devices=geo.get("devices", None),
            post_denoise_fn=post_denoise_fn,

            # endpoints policy
            endpoint_mode=str(geo.get("endpoint_mode", "clean")),  # set "noisy" in config for t-project init
            fixed_noise=None,

            verbose=True,
            vis_kwargs=geo.get("visualization", {}),
        )

        all_paths.append(path)
        all_infos.append(info)
        all_path_histories.append(path_history)

    runtime = time.time() - t0_wall

    # ── Concatenate batches along B ───────────────────────────────────
    if B > 0:
        num_segments_total = len(all_paths[0])
        path = [torch.cat([batch[k] for batch in all_paths], dim=0) for k in range(num_segments_total)]
        info = all_infos[0]
        path_history = all_path_histories[0]
    else:
        path, info, path_history = [], {}, []

    # ── Comparison Path Calculation & Error Metrics (RMSE vs GT; same realized T) ──
    T_realized = int(len(path)) if path else (int(geo.get("n_segments", 16)) + 1)
    tv = torch.linspace(0.0, 1.0, T_realized)

    lin = (1 - tv.view(1, -1, 1)) * flatten(p_img_full).cpu().unsqueeze(1) + \
          tv.view(1, -1, 1) * flatten(q_img_full).cpu().unsqueeze(1)          # (B, T, D)

    mean_err = std_err = lin_mean = lin_std = np.nan
    avg_rmse = None

    C, H, W = first_batch_shape if first_batch_shape is not None else p_img_full.shape[1:]
    base = dset.dataset if isinstance(dset, Subset) else dset

    if hasattr(base, "compute_geodesic"):
        try:
            with torch.no_grad():
                # GT stack (B, T, C, H, W)
                p_rot, q_rot = (p_rot_full, q_rot_full) if p_rot_full is not None else (None, None)
                if p_rot is not None:
                    gt_stack = base.compute_geodesic(p_rot, q_rot, tv).detach().cpu()
                else:
                    gt_stack = base.compute_geodesic(
                        flatten(p_img_full).cpu(), flatten(q_img_full).cpu(), tv
                    ).detach().cpu()

                # Predicted stack (B, T, C, H, W), reshaped from optimized path
                pred_imgs = []
                for b in range(p_img_full.shape[0]):
                    traj_b = torch.stack([path[k][b].detach().cpu() for k in range(T_realized)], dim=0)  # (T,D) on CPU
                    traj_b_img = unflatten(traj_b, (C, H, W)).clamp(0.0, 1.0)                             # CPU
                    pred_imgs.append(traj_b_img)
                pred_stack = torch.stack(pred_imgs, dim=0)                                  # (B,T,C,H,W)

                # RMSE per pair → average
                mse_per_pair = ((pred_stack - gt_stack) ** 2).mean(dim=(1, 2, 3, 4))
                rmse_per_pair = torch.sqrt(mse_per_pair)
                avg_rmse = float(rmse_per_pair.mean().item())

                # Keep legacy vector-space MSE too
                est_flat = torch.stack(path, 0).permute(1, 0, 2).cpu()                    # (B,T,D)
                gt_flat  = gt_stack.flatten(2)                                            # (B,T,D)
                err = ((est_flat - gt_flat)**2).mean(dim=(1, 2)).numpy()
                mean_err, std_err = err.mean(), err.std()

                le = ((lin - gt_flat)**2).mean(dim=(1, 2)).numpy()
                lin_mean, lin_std = le.mean(), le.std()

                # Save comparison grid (top: estimated, bottom: GT)
                comp_path = os.path.join(eval_dir, "geodesics_vs_gt.png")
                _save_geodesic_comparison_grid(
                    pred_stack=pred_stack, gt_stack=gt_stack, out_path=comp_path,
                    padding=2, row_gap=2, pair_gap=12,
                )
                print(f"[Geodesics][GT] Comparison image saved → {comp_path}")

                with open(os.path.join(eval_dir, "geodesic_eval.txt"), "w") as f:
                    f.write(f"Avg RMSE vs GT (B={pred_stack.size(0)}, T={T_realized}): {avg_rmse:.6f}\n")
                    f.write(f"Vec-MSE ours:   {mean_err:.6e} ± {std_err:.6e}\n")
                    f.write(f"Vec-MSE linear: {lin_mean:.6e} ± {lin_std:.6e}\n")
                print(f"[Geodesics][GT] Avg RMSE vs ground-truth: {avg_rmse:.6f}")

        except Exception as e:
            print(f"[Geodesics][GT] Failed to evaluate GT deviation: {e}")

    # ── Visualisation ─────────────────────────────────────────
    if visualize:
        vis_cfg = geo.get("visualization", {})
        bg = min(2000, len(dset))
        im, _ = _split_data([dset[i] for i in range(bg)])

        # Use the decode target for vector-field visualizations
        t_pert = torch.tensor(float(end_time), device=device)

        with torch.no_grad():
            noise_bg = sde.perturb(im.to(device), t_pert)

        vis_score_fn = get_score_fn(sde, model, im.shape[1:])

        vis_kwargs = dict(
            perturbed_points=flatten(noise_bg).cpu(),
            score_fn=lambda x: vis_score_fn(x.to(device), t_pert.to(device)).cpu(),
            t_val=t_pert.item(),
            metrics={}, min_point=None,
            orig_shape=imgs.shape[1:],
            log_dir=eval_dir
        )
        base_filename = str(geo.get("plot_filename", "geodesics.png"))

        # Initial path (if ODE init was used)
        if initial_path_to_visualize is not None:
            init_filename = base_filename.replace(".png", "_ode_initialization.png")
            print(f"Plotting ODE-initialized path to {os.path.join(eval_dir, init_filename)}...")
            visualize_riemannian_optimization_selector(
                **vis_kwargs,
                trajectories=[p.cpu() for p in initial_path_to_visualize],
                plot_filename=init_filename
            )
            save_path_grid(
                [p.cpu() for p in initial_path_to_visualize],
                first_batch_shape if first_batch_shape is not None else imgs.shape[1:],
                os.path.join(eval_dir, base_filename.replace(".png", "_ode_initialization_raw.png"))
            )

        # Estimated geodesic – both styles
        est_filename = base_filename.replace(".png", "_estimated.png")
        print(f"Plotting estimated geodesic to {os.path.join(eval_dir, est_filename)}...")
        visualize_riemannian_optimization_selector(
            **vis_kwargs, trajectories=[p.cpu() for p in path], plot_filename=est_filename
        )
        save_path_grid(
            [p.detach().cpu() for p in path],
            imgs.shape[1:],
            os.path.join(eval_dir, base_filename.replace(".png", "_estimated_raw.png"))
        )

        # Linear interpolation – vector-field + raw grid
        lin_filename = base_filename.replace(".png", "_linear.png")
        lin_traj = [t for t in lin.permute(1, 0, 2)]
        print(f"Plotting linear interpolation to {os.path.join(eval_dir, lin_filename)}...")
        visualize_riemannian_optimization_selector(
            **vis_kwargs, trajectories=lin_traj, plot_filename=lin_filename
        )
        save_path_grid(
            [lt.detach().cpu() for lt in lin_traj],
            imgs.shape[1:],
            os.path.join(eval_dir, base_filename.replace(".png", "_linear_raw.png"))
        )

        # Ground truth (if available)
        if hasattr(base, "compute_geodesic"):
            try:
                # reuse gt_stack if computed, else compute from first B pairs
                if 'gt_stack' not in locals():
                    t_lin = torch.linspace(0.0, 1.0, T_realized)
                    if p_rot_full is not None:
                        gt_stack = base.compute_geodesic(p_rot_full, q_rot_full, t_lin).detach().cpu()
                    else:
                        gt_stack = base.compute_geodesic(
                            flatten(p_img_full).cpu(), flatten(q_img_full).cpu(), t_lin
                        ).detach().cpu()

                gt_filename = base_filename.replace(".png", "_ground_truth.png")
                gt_traj = [t for t in gt_stack.flatten(2).permute(1, 0, 2)]  # list[T] of (B,D)
                print(f"Plotting ground truth geodesic to {os.path.join(eval_dir, gt_filename)}...")
                visualize_riemannian_optimization_selector(
                    **vis_kwargs, trajectories=gt_traj, plot_filename=gt_filename
                )
                save_path_grid(
                    [g.detach().cpu() for g in gt_traj],
                    imgs.shape[1:],
                    os.path.join(eval_dir, base_filename.replace(".png", "_ground_truth_raw.png"))
                )
            except Exception as e:
                print(f"[Viz] Failed to render GT geodesic: {e}")

        print("Plotting loss evolution...")
        _plot_loss(info, float(geo.get("lam_smooth", 0.)), float(geo.get("lam_mono", 0.)),
                   os.path.join(eval_dir, "loss_evolution.png"))

        if vis_cfg.get("animate_optimization", False) and path_history:
            anim_path = os.path.join(
                eval_dir,
                base_filename.replace(".png", "_optimization.gif")
            )
            create_opt_animation(
                path_history=path_history,
                output_filename=anim_path,
                vis_kwargs=vis_kwargs,
                duration=vis_cfg.get("animation_duration_ms", 100) / 1000.0
            )

    return dict(
        path=path, loss_info=info,
        mean_error=mean_err, std_error=std_err,
        linear_mean_error=lin_mean, linear_std_error=lin_std,
        avg_rmse_vs_gt=avg_rmse,
        runtime_secs=runtime
    )


# ─── Main Execution Wrapper ──────────────────────────────────────────
def geodesic_batch_runner(diff_cfg: Dict[str, Any], geo_cfg: Dict[str, Any], visualize: bool):
    dev = torch.device(diff_cfg.training.device)
    _, _, eval_dir = prepare_training_dirs(diff_cfg)
    loader, _, _ = get_dataloaders(diff_cfg.data, seed=int(geo_cfg.get("random_seed", 42)))
    dset = loader.dataset

    model = get_model(diff_cfg.model).to(dev)
    ema = EMA(model, diff_cfg.model.ema_decay)
    ckpt_path = os.path.join(diff_cfg.checkpoint_dir, diff_cfg.model.checkpoint)
    load_model(model, ema, ckpt_path, "Model", device=dev, is_ema=True)
    ema.apply_shadow()
    model.eval()
    sde = configure_sde(diff_cfg)

    # ── Entropy profile (optional; mirrors latent driver) ─────────────
    entropic_profile_enabled = bool(geo_cfg.get("entropic_profile", True))
    existing_schedule = geo_cfg.get("time_schedule", [0.05])
    n_stages = len(existing_schedule) if isinstance(existing_schedule, (list, tuple)) else 1

    # infer orig_shape from one sample
    sample0 = dset[0][0] if isinstance(dset[0], (tuple, list)) else dset[0]
    orig_shape = tuple(sample0.shape)  # (C,H,W)

    if entropic_profile_enabled:
        ent_num_t       = int(geo_cfg.get("ent_num_t", 96))
        ent_max_batches = int(geo_cfg.get("ent_max_batches", 12))
        ent_t_min       = geo_cfg.get("ent_t_min", None)
        ent_t_max       = float(geo_cfg.get("ent_t_max", 1.0))

        ent = compute_entropy_profile(
            model=model,
            sde=sde,
            loader=loader,
            orig_shape=orig_shape,
            device=dev,
            t_min=ent_t_min,
            t_max=ent_t_max,
            num_t=ent_num_t,
            max_batches=ent_max_batches,
            save_dir=eval_dir,
            filename_prefix="entropy_profile",
            progress=True,
        )
        print(f"[Entropy] Saved entropy profile → {os.path.join(eval_dir, 'entropy_profile.png')}")

        if bool(geo_cfg.get("use_entropic_schedule", False)) and n_stages > 0:
            source = str(geo_cfg.get("entropic_source", "mmse")).lower()
            if source == "score":
                t_grid = ent["Phi_score_t"]; phiR = ent["Phi_rescaled_score"]
            else:
                t_grid = ent["Phi_mmse_t"];  phiR = ent["Phi_rescaled_mmse"]

            import numpy as _np
            t_lo = float(geo_cfg.get("ent_t_min", t_grid[0] if len(t_grid) else 1e-4))
            t_hi = float(geo_cfg.get("ent_t_max", t_grid[-1] if len(t_grid) else 1.0))
            entropic_sched = schedule_from_rescaled_entropic_time(
                t=_np.asarray(t_grid, dtype=_np.float64),
                Phi_rescaled=_np.asarray(phiR, dtype=_np.float64),
                n_stages=n_stages,
                t_lo=t_lo, t_hi=t_hi,
                descending=True,
            )
            geo_cfg["time_schedule"] = entropic_sched
            print(f"[Entropy] Replacing time_schedule with uniform-in-Φ̃ ({source}) → {entropic_sched}")

    res = run_geodesic_computation(geo_cfg, model, sde, dset, dev, eval_dir, visualize)

    li = res['loss_info']
    print("\n── Loss components (avg. over batch)")
    for k in ("total", "geodesic_energy", "smoothness", "monotonicity"):
        if k in li and isinstance(li[k], (int, float)):
            print(f"{k:>17}: {li[k]:.4e}")

    if not np.isnan(res['mean_error']):
        print("\n── Geodesic MSE vs GT (vectorized)")
        print(f"ours              : {res['mean_error']:.4e} ± {res['std_error']:.4e}")
        print(f"linear            : {res['linear_mean_error']:.4e} ± {res['linear_std_error']:.4e}")

    if res.get("avg_rmse_vs_gt") is not None:
        print(f"\n── Image-space RMSE vs GT: {res['avg_rmse_vs_gt']:.6f}")

    return res['path']


if __name__ == "__main__":
    P = ArgumentParser("Batch geodesic computation (ambient/image space)")
    P.add_argument("--config", required=True, help="Path to the main model/training config.")
    P.add_argument("--geo-config", required=True, help="Path to the geodesic-specific config.")
    P.add_argument("--visualize", action="store_true", help="Enable saving of plots and animations.")
    a = P.parse_args()

    diff_cfg = load_config(a.config)
    geo_cfg = load_geo_cfg(a.geo_config)

    log_dir = os.path.join(diff_cfg.base_log_dir, diff_cfg.experiment)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.pkl"), "wb") as f:
        pickle.dump(diff_cfg.to_dict(), f)

    torch.set_float32_matmul_precision("high")

    seed = int(geo_cfg.get("seed", geo_cfg.get("random_seed", 42)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    geodesic_batch_runner(diff_cfg, geo_cfg, visualize=a.visualize)
