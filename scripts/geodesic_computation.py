#!/usr/bin/env python
# =====================================================================
# Batch geodesic evaluation on a score-based manifold
# (refactored: imports DDIM init + plotting from debug_initialization.py)
# =====================================================================
from __future__ import annotations
import os, time, random, pickle
from argparse import ArgumentParser
from typing import Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

# ─── Project-local Imports ───────────────────────────────────────────
from data.data_utils import get_dataloaders
from models                  import get_model
from sde                     import configure_sde
from utils.train_utils       import prepare_training_dirs, EMA, load_model
from configs                 import load_config
from data_geometry.utils.visualization import visualize_riemannian_optimization_selector
from data_geometry.geodesics.fast_geodesic_computation import compute_geodesic

import imageio.v2 as imageio
import shutil

# ─── Import initialization/plot helpers from debugger ────────────────
from debug_initialization import (
    flatten, unflatten,
    get_score_fn,
    save_path_grid,
    generate_ode_initialized_path,
    make_ddim_post_denoiser,
    make_sharded_ddim_post_denoiser
)

# ─── Misc Helpers ────────────────────────────────────────────────────
def load_geo_cfg(path: str) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    with open(path, "r") as f:
        exec(compile(f.read(), path, "exec"), scope)
    if "CONFIG" not in scope:
        raise ValueError(f"Config file '{path}' must define a 'CONFIG' dictionary.")
    return scope["CONFIG"]

def _split_data(items: List[Any]) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Splits dataset items into image and rotation tensors."""
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

    # use a local copy of vis_kwargs so we can override log_dir
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

    # write gif
    with imageio.get_writer(output_filename, mode='I', duration=duration, loop=0) as w:
        for fp in frame_paths:
            if os.path.exists(fp):
                w.append_data(imageio.imread(fp))
    print(f"Saved optimization animation to {output_filename}")

    # cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

# ─── Main Geodesic Computation Routine ───────────────────────────────
def run_geodesic_computation(
        geo: Dict[str, Any], model: torch.nn.Module, sde: Any, dset: Dataset, 
        device: torch.device, eval_dir: str, visualize: bool):

    B = geo.get("num_pairs", 8)
    opt_settings = geo.get("optimizer_settings", {})
    micro_batch_size = opt_settings.get("optimizer_batch_size", B)

    g = torch.Generator().manual_seed(geo.get("random_seed", 42))
    idx = torch.randperm(len(dset), generator=g)[:2 * B]
    imgs, rots = _split_data([dset[i] for i in idx])

    p_img_full, q_img_full = imgs.chunk(2)
    p_rot_full, q_rot_full = (rots.chunk(2) if rots is not None else (None, None))
    
    all_paths, all_infos, all_path_histories = [], [], []
    initial_path_to_visualize = None
    first_batch_shape = None

    # Shared init settings
    end_time = geo.get("time_schedule", [0.03])[0]
    init_cfg  = geo.get("initialization", {})

    t0_wall = time.time()

    for i in range(0, B, micro_batch_size):
        print(f"\n── Processing Batch {i//micro_batch_size + 1}/{(B + micro_batch_size - 1)//micro_batch_size} (pairs {i} to {i+micro_batch_size-1}) ──")
        
        p_img = p_img_full[i:i+micro_batch_size].to(device)
        q_img = q_img_full[i:i+micro_batch_size].to(device)
        shp = p_img.shape[1:]
        first_batch_shape = first_batch_shape or shp
        p_flat, q_flat = flatten(p_img), flatten(q_img)

        with torch.no_grad():
            mse_diff = torch.mean((p_flat - q_flat)**2).item()
            print(f"[Debug] MSE between initial p and q: {mse_diff:.4e}")

        init_path_opt = None
        if init_cfg.get("method", "linear") == "ode":  # or "ddim", depending on your version
            init_path_opt = generate_ode_initialized_path(
            p_flat, q_flat, sde, model, shp,
            n_segments=geo.get("n_segments", 16),
            end_time=end_time,               # decode nodes to t_start
            num_solver_steps=init_cfg.get("ode_solver_steps", 250),
            t0=init_cfg.get("t0", 1e-3),
            device=device
        )
            # IMPORTANT: snapshot a deep copy for visualization BEFORE optimization mutates it
            if i == 0:
                initial_path_to_visualize = [t.detach().clone() for t in init_path_opt]

        # ---- New optimizer / line-search options pulled from config ----
        optimizer = geo.get("optimizer", "adam")  # keep config default = Adam
        line_search = geo.get("line_search", "armijo")

        # Adam & Wolfe don't pair well -> gently warn & coerce to armijo
        if optimizer.lower() == "adam" and line_search.lower() == "strong_wolfe":
            print("[driver] Warning: Adam + strong_wolfe is not supported; switching to 'armijo'.")
            line_search = "armijo"

        # Build CG kwargs only for the Jacobian metric
        mt = geo.get("metric_type", "stein").lower()
        if mt in {"jacobian", "jsm"}:
            cg_kwargs = dict(
                # We intentionally set reg_lambda from lam_metric;
                # JacobianMetric.apply_inverse does:
                #   {"reg_lambda": self.lam, **self.cg_kwargs}
                # so self.lam (== lam_metric) takes precedence — desired behavior.
                reg_lambda=geo.get("lam_metric", geo.get("reg_lambda", 1e-5)),
                max_iter=geo.get("cg_max_iter", 50),
                tol=geo.get("cg_tol", 1e-6),
                preconditioner=geo.get("cg_preconditioner", "diagonal"),
                precond_diag_samples=geo.get("cg_precond_diag_samples", 10),
            )
        else:
            cg_kwargs = {}


        pd_cfg = geo.get("post_denoise", None)
        post_denoise_fn = None
        if isinstance(pd_cfg, dict) and pd_cfg.get("method", "").lower() == "ddim":
            if len(geo.get("devices", [])) > 1 and pd_cfg.get("shard", False):
                post_denoise_fn = make_sharded_ddim_post_denoiser(
                    model, sde, shp, devices=geo["devices"],
                    to_time=pd_cfg.get("to_time", 1e-3),
                    steps=pd_cfg.get("steps", 50),
                )
            else:
                post_denoise_fn = make_ddim_post_denoiser(
                    model, sde, shp,
                    to_time=pd_cfg.get("to_time", 1e-3),
                    steps=pd_cfg.get("steps", 50),
                )


        path, info, path_history = compute_geodesic(
            p=p_flat, q=q_flat,
            initial_path=init_path_opt,
            model=model, sde=sde, orig_shape=shp,

            # discretisation & schedule
            n_segments=geo.get("n_segments", 16),
            time_schedule=geo.get("time_schedule", [0.03]),

            # regularisers
            lam_smooth=geo.get("lam_smooth", 1e-3),
            lam_mono=geo.get("lam_mono", 0.0),

            # learning / budget (shared)
            max_iters=geo.get("max_iters", 500),
            tol=geo.get("tol", 1e-6),
            patience=geo.get("patience", 30),

            # --- optimizer selection & params ---
            optimizer=optimizer,                                # "adam" | "rgd"
            adam_lr=geo.get("adam_lr", 1e-2),                   # used by Adam and fixed-step modes
            betas=geo.get("betas", (0.9, 0.999)),               # Adam only
            line_search=line_search,                            # "fixed" | "armijo" | "strong_wolfe"
            use_retraction_update=geo.get("use_retraction_update", True),  # RGD only

            # Armijo (used by Adam or RGD when line_search="armijo")
            armijo_rho=geo.get("armijo_rho", 1e-4),
            armijo_beta=geo.get("armijo_beta", 0.5),
            armijo_max_iter=geo.get("armijo_max_iter", 20),

            # Strong-Wolfe (RGD only when line_search="strong_wolfe")
            wolfe_c1=geo.get("wolfe_c1", 1e-4),
            wolfe_c2=geo.get("wolfe_c2", 0.9),
            wolfe_max_bracket=geo.get("wolfe_max_bracket", 10),
            wolfe_max_zoom=geo.get("wolfe_max_zoom", 10),
            wolfe_max_alpha=geo.get("wolfe_max_alpha", 50.0),

            # moment transport (Adam only; ignored by RGD as no momentum)
            transport_mode=geo.get("transport_mode", 'ad_hoc'),
            transport_steps=geo.get("transport_steps", 1),

            # metric
            metric_type=geo.get("metric_type", "stein"),
            lam_metric=geo.get("lam_metric", 1.0),

            # Jacobian metric CG kwargs
            cg_kwargs=cg_kwargs,

            devices=geo.get("devices", None),
            post_denoise_fn=post_denoise_fn,
            # endpoints policy (ignored if initial_path is provided)
            endpoint_mode=geo.get("endpoint_mode", "clean"),     # "clean" | "noisy"
            fixed_noise=None,                                    # or geo.get("fixed_noise", None)

            verbose=True,
            vis_kwargs=geo.get("visualization", {}),
        )
        
        all_paths.append(path)
        all_infos.append(info)
        all_path_histories.append(path_history)
        
    runtime = time.time() - t0_wall

    # Concatenate batches along B
    if B > 0:
        num_segments_total = len(all_paths[0])
        path = [torch.cat([batch[k] for batch in all_paths], dim=0) for k in range(num_segments_total)]
        info = all_infos[0]
        path_history = all_path_histories[0]
    else:
        path, info, path_history = [], {}, []

    # ── Comparison Path Calculation & Error Metrics ───────────
    T = geo.get("n_segments", 16) + 1
    tv = torch.linspace(0, 1, T)
    lin = (1 - tv.view(1, -1, 1)) * flatten(p_img_full).cpu().unsqueeze(1) + \
          tv.view(1, -1, 1) * flatten(q_img_full).cpu().unsqueeze(1)

    mean_err = std_err = lin_mean = lin_std = np.nan
    gt = None
    base = dset.dataset if isinstance(dset, Subset) else dset
    if hasattr(base, "compute_geodesic"):
        with torch.no_grad():
            p_rot, q_rot = (p_rot_full, q_rot_full) if p_rot_full is not None else (None, None)
            if p_rot is not None:
                gt_full = base.compute_geodesic(p_rot, q_rot, tv)
                gt = flatten(gt_full.flatten(0, 1)).view(gt_full.shape[0], T, -1)
            else:
                gt = base.compute_geodesic(flatten(p_img_full).cpu(), flatten(q_img_full).cpu(), tv)

            est = torch.stack(path, 0).permute(1, 0, 2).cpu()
            err = ((est - gt)**2).mean(dim=(1, 2)).numpy()
            mean_err, std_err = err.mean(), err.std()

            le = ((lin - gt)**2).mean(dim=(1, 2)).numpy()
            lin_mean, lin_std = le.mean(), le.std()

    # ── Visualisation ─────────────────────────────────────────
    if visualize:
        vis_cfg = geo.get("visualization", {})
        bg = min(2000, len(dset))
        im, _ = _split_data([dset[i] for i in range(bg)])
        
        # Use the decode target for vector-field visualizations
        t_pert = torch.tensor(end_time, device=device)

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
        base_filename = geo.get("plot_filename", "geodesics.png")

        # Initial path (first batch) – both styles
        if initial_path_to_visualize is not None:
            init_filename = base_filename.replace(".png", "_ode_initialization.png")
            print(f"Plotting ODE-initialized path to {os.path.join(eval_dir, init_filename)}...")
            visualize_riemannian_optimization_selector(
                **vis_kwargs,
                trajectories=[p.cpu() for p in initial_path_to_visualize],
                plot_filename=init_filename
            )
            # Raw grid like in the debugger
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

        if gt is not None:
            gt_filename = base_filename.replace(".png", "_ground_truth.png")
            gt_traj = [t for t in gt.permute(1, 0, 2)]
            print(f"Plotting ground truth geodesic to {os.path.join(eval_dir, gt_filename)}...")
            visualize_riemannian_optimization_selector(
                **vis_kwargs, trajectories=gt_traj, plot_filename=gt_filename
            )
            save_path_grid(
                [g.detach().cpu() for g in gt_traj],
                imgs.shape[1:],
                os.path.join(eval_dir, base_filename.replace(".png", "_ground_truth_raw.png"))
            )

        print("Plotting loss evolution...")
        _plot_loss(info, geo.get("lam_smooth", 0.), geo.get("lam_mono", 0.),
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

    return dict(path=path, loss_info=info,
                mean_error=mean_err, std_error=std_err,
                linear_mean_error=lin_mean, linear_std_error=lin_std,
                runtime_secs=runtime)

# ─── Main Execution Wrapper ──────────────────────────────────────────
def geodesic_batch_runner(diff_cfg: Dict[str, Any], geo_cfg: Dict[str, Any], visualize: bool):
    dev = torch.device(diff_cfg.training.device)
    _, _, eval_dir = prepare_training_dirs(diff_cfg)
    loader, _, _ = get_dataloaders(diff_cfg.data, seed=geo_cfg.get("random_seed", 42))
    dset = loader.dataset

    model = get_model(diff_cfg.model).to(dev)
    ema = EMA(model, diff_cfg.model.ema_decay)
    ckpt_path = os.path.join(diff_cfg.checkpoint_dir, diff_cfg.model.checkpoint)
    load_model(model, ema, ckpt_path, "Model", device=dev, is_ema=True)
    ema.apply_shadow()
    model.eval()
    sde = configure_sde(diff_cfg)

    res = run_geodesic_computation(geo_cfg, model, sde, dset, dev, eval_dir, visualize)

    li = res['loss_info']
    print("\n── Loss components (avg. over batch)")
    for k in ("total", "geodesic_energy", "smoothness", "monotonicity"):
        if k in li and isinstance(li[k], (int, float)):
            print(f"{k:>17}: {li[k]:.4e}")
    if not np.isnan(res['mean_error']):
        print("\n── Geodesic MSE vs GT")
        print(f"ours              : {res['mean_error']:.4e} ± {res['std_error']:.4e}")
        print(f"linear            : {res['linear_mean_error']:.4e} ± {res['linear_std_error']:.4e}")
    return res['path']

if __name__ == "__main__":
    P = ArgumentParser("Batch geodesic computation")
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

    seed = geo_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    geodesic_batch_runner(diff_cfg, geo_cfg, visualize=a.visualize)
