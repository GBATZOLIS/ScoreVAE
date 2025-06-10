#!/usr/bin/env python
"""
Batch geodesic computation on a **score-based data manifold**.

This script can be run directly from the command line for a single computation,
or its `run_geodesic_computation` function can be imported and used as a
library, for example, in a hyperparameter sweep.

--------------------------------------------------------------------------------
CLI Usage
--------------------------------------------------------------------------------
```bash
python geodesic_computation.py \
    --config      path/to/diffusion_config.py \
    --geo-config  path/to/geodesic_config.py
```
"""
from __future__ import annotations

import os
import time
import random
import numpy as np
import pickle
import torch
import torch.multiprocessing as mp
from torch.nn.functional import mse_loss
from torch.utils.data import Subset
from argparse import ArgumentParser
from typing import Tuple, Dict, Any, List, Callable

# --- Assume local imports are available in the project structure ---
# (Adjust these paths if your project structure differs)
from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import prepare_training_dirs, EMA, load_model
from configs import load_config
from data_geometry.utils.visualization import visualize_riemannian_optimization_selector

# --- UPDATED IMPORTS for the new modular structure ---
# Import the generalized computation function
from data_geometry.geodesics.fast_geodesic_computation import compute_geodesic, compute_geodesic_multistage
# Import the metric classes to instantiate them here
from data_geometry.metrics import SteinMetric, JacobianMetric


# --- Helper Utilities (Unchanged) ---

def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dims -> (B, d)."""
    return x.view(x.size(0), -1)

def unflatten_tensor(x: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    """Unflatten to original shape, keeping batch dim."""
    return x.view(x.size(0), *orig_shape)

def ensure_time_tensor(t: torch.Tensor, B: int) -> torch.Tensor:
    """Ensure time tensor `t` is broadcastable to batch size `B`."""
    if t.dim() == 0:
        return t.expand(B)
    if t.dim() == 1 and t.size(0) == B:
        return t
    return t.expand(B)


# --- Score / Denoiser Wrappers (Unchanged) ---

def get_score_fn(sde, model, t: torch.Tensor, orig_shape: Tuple[int, ...]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a function that computes the score of a batch of noised data."""
    sigma_fn = sde.get_sigma_fn()
    def score_fn(x_t: torch.Tensor):
        B = x_t.size(0)
        t_corr = ensure_time_tensor(t, B)
        sigma_t = sigma_fn(t_corr).view(B, 1)
        noise_pred = model(unflatten_tensor(x_t, orig_shape), None, t_corr)
        return -flatten_tensor(noise_pred) / sigma_t
    return score_fn

def get_denoiser_fn(sde, model, t: torch.Tensor, orig_shape: Tuple[int, ...]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a function that denoises a batch of data from time `t`."""
    alpha_fn, sigma_fn = sde.get_alpha_fn(), sde.get_sigma_fn()
    def denoise_fn(x_t: torch.Tensor):
        B = x_t.size(0)
        t_corr = ensure_time_tensor(t, B)
        alpha_t = alpha_fn(t_corr).view(B, 1)
        sigma_t = sigma_fn(t_corr).view(B, 1)
        noise_pred = model(unflatten_tensor(x_t, orig_shape), None, t_corr)
        return (flatten_tensor(x_t) - sigma_t * flatten_tensor(noise_pred)) / alpha_t
    return denoise_fn

def load_geo_config(path: str) -> Dict[str, Any]:
    """Loads a Python-based config file containing a 'CONFIG' dictionary."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Geodesic config file not found: {path}")
    scope: Dict[str, Any] = {}
    with open(path, "r") as f:
        exec(compile(f.read(), path, "exec"), scope)
    if "CONFIG" not in scope:
        raise ValueError("Geodesic config file must define a 'CONFIG' dictionary")
    return scope["CONFIG"]


# --- Core Logic (for Library Use) ---

def run_geodesic_computation(
    geo_cfg: Dict[str, Any],
    model: torch.nn.Module,
    sde: Any,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    eval_dir: str,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    Performs the core geodesic computation. Designed to be called by other scripts.
    """
    print(f"[Core] Running with config: { {k: v for k, v in geo_cfg.items() if 'lam' in k or 'metric' in k or 'lr' in k} }")

    # 1. Select data pairs using a fixed seed for reproducibility
    num_pairs = geo_cfg.get("num_pairs", 24)
    g = torch.Generator().manual_seed(geo_cfg.get("random_seed", 42))
    indices = torch.randperm(len(dataset), generator=g)[:2 * num_pairs]

    samples = torch.stack([dataset[i][0] if isinstance(dataset[i], tuple) else dataset[i] for i in indices])
    p_img, q_img = samples.chunk(2)
    p_img, q_img = p_img.to(device), q_img.to(device)
    orig_shape = p_img.shape[1:]
    p_flat, q_flat = flatten_tensor(p_img), flatten_tensor(q_img)

    # 2. Configure diffusion coefficients and necessary functions
    t_pert = torch.tensor(geo_cfg.get("time_for_perturbation", 0.01), device=device)
    alpha_t, sigma_t = sde.get_alpha_fn()(t_pert).item(), sde.get_sigma_fn()(t_pert).item()

    score_fn = get_score_fn(sde, model, t_pert, orig_shape)
    denoise_fn = get_denoiser_fn(sde, model, t_pert, orig_shape)

    # 3. Build the metric object based on the configuration
    metric_type = geo_cfg.get("metric_type", "stein").lower()
    lam_metric = geo_cfg.get("lam_metric", 1.0)

    if metric_type == "jacobian":
        print("[Core] Using JacobianMetric")
        # --- FIXED: Correctly map config keys to function argument names ---
        cg_kwargs = dict(
            preconditioner       = geo_cfg.get("cg_preconditioner", "diagonal"),
            precond_diag_samples = geo_cfg.get("cg_precond_diag_samples", 8),
            tol                  = geo_cfg.get("cg_tol", 5e-5),
            max_iter             = geo_cfg.get("cg_max_iter", 4),
        )
        metric = JacobianMetric(score_fn, lam=lam_metric, cg_kwargs=cg_kwargs)
    else:  # Default to Stein metric
        print("[Core] Using SteinMetric")
        metric = SteinMetric(score_fn, lam=lam_metric)

    # 4. Compute Geodesic Path using the new generalized function call
    print("[Core] Starting geodesic computation...")
    start = time.time()
    if geo_cfg.get("use_multistage", True):
        time_schedule = geo_cfg.get("time_schedule", [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
        metric_type = geo_cfg.get("metric_type", "stein")
        metric_lam = geo_cfg.get("lam_metric", 1.0)
        cg_kwargs = dict(
            preconditioner       = geo_cfg.get("cg_preconditioner", "diagonal"),
            precond_diag_samples = geo_cfg.get("cg_precond_diag_samples", 8),
            tol                  = geo_cfg.get("cg_tol", 5e-5),
            max_iter             = geo_cfg.get("cg_max_iter", 4),
        )

        print(f"[Core] Using MULTISTAGE geodesic computation with time_schedule = {time_schedule}")

        path, loss_info = compute_geodesic_multistage(
            p=p_flat,
            q=q_flat,
            model=model,
            sde=sde,
            n_segments=geo_cfg.get("n_segments", 16),
            time_schedule=time_schedule,
            orig_shape=orig_shape,
            lam_smooth=geo_cfg.get("lam_smooth", 1e-3),
            lam_mono=geo_cfg.get("lam_mono", 0.0),
            adam_lr=geo_cfg.get("adam_lr", 1e-2),
            max_iters=geo_cfg.get("max_iters", 500),
            tol=geo_cfg.get("tol", 1e-6),
            patience=geo_cfg.get("patience", 30),
            verbose=True,
            metric_type=metric_type,
            metric_lam=metric_lam,
            cg_kwargs=cg_kwargs,
        )
    else:
        print(f"[Core] Using SINGLE-STAGE geodesic computation")

        path, loss_info = compute_geodesic(
            p_flat,
            q_flat,
            metric,
            n_segments=geo_cfg.get("n_segments", 16),
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            lam_smooth=geo_cfg.get("lam_smooth", 1e-3),
            lam_mono=geo_cfg.get("lam_mono", 0.0),
            adam_lr=geo_cfg.get("adam_lr", 1e-2),
            max_iters=geo_cfg.get("max_iters", 500),
            tol=geo_cfg.get("tol", 1e-6),
            patience=geo_cfg.get("patience", 30),
            verbose=True,
            denoise_fn=denoise_fn,
        )

    elapsed = time.time() - start
    print(f"compute_geodesic finished in {elapsed:.2f} s")

    # 5. Compute Ground Truth error if available
    mean_error, std_error = np.nan, np.nan
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base_dataset, "compute_geodesic"):
        t_vals = torch.linspace(0, 1, geo_cfg.get("n_segments", 16) + 1, device=device)
        with torch.no_grad():
            gt_paths = base_dataset.compute_geodesic(p_flat.cpu(), q_flat.cpu(), t_vals).cpu()
        path_tensor = torch.stack(path, dim=0).permute(1, 0, 2).cpu()
        errors = np.array([mse_loss(path_tensor[i], gt_paths[i]).item() for i in range(num_pairs)])
        mean_error, std_error = errors.mean(), errors.std()

    # 6. Visualization (optional, usually disabled for sweeps)
    if visualize:
        with torch.no_grad():
            bg_cap = min(2000, len(dataset))
            bg = torch.stack([dataset[i][0] if isinstance(dataset[i], tuple) else dataset[i] for i in range(bg_cap)]).to(device)
            bg_noised = alpha_t * bg + sigma_t * torch.randn_like(bg)
            bg_flat = flatten_tensor(bg_noised)
        
        visualize_riemannian_optimization_selector(
            bg_flat.cpu(), lambda x: score_fn(x.to(device)).cpu(), t_pert.item(),
            trajectories=[p.cpu() for p in path], metrics={}, min_point=None,
            orig_shape=orig_shape, log_dir=eval_dir,
            plot_filename=geo_cfg.get("plot_filename", "geodesics.png"),
        )
    
    # 7. Return all relevant metrics
    return {
        "path": path,
        "loss_info": loss_info,
        "mean_error": mean_error,
        "std_error": std_error,
        "runtime_secs": elapsed
    }


# --- Standalone Script Runner ---

def geodesic_batch_runner(diff_cfg: Any, geo_cfg: Dict[str, Any], visualize: bool = True):
    """
    A wrapper that handles all setup and runs the geodesic computation.
    This is called when the script is executed from the command line.
    """
    device = torch.device(diff_cfg.training.device)
    print(f"[Geodesic-Batch] using device: {device}")

    # --- Setup: Dirs, Data, Model, SDE ---
    _, _, eval_dir = prepare_training_dirs(diff_cfg)
    train_loader, _, _ = get_dataloaders(diff_cfg.data, seed=geo_cfg.get("random_seed", 42))
    dataset = train_loader.dataset

    model = get_model(diff_cfg.model).to(device)
    ema = EMA(model=model, decay=diff_cfg.model.ema_decay)

    ckpt_name = diff_cfg.model.checkpoint
    if not ckpt_name.endswith(".pth"):
        ckpt_name += ".pth"
    ckpt_path = os.path.join(diff_cfg.checkpoint_dir, ckpt_name)

    load_model(model, ema, ckpt_path, "Model", device=device, is_ema=True)
    ema.apply_shadow()
    model.eval()

    sde = configure_sde(diff_cfg)

    # --- Run Core Computation ---
    results = run_geodesic_computation(geo_cfg, model, sde, dataset, device, eval_dir, visualize)

    # --- Print Formatted Results for CLI ---
    loss_info = results['loss_info']
    print((
        "┌─ Loss breakdown (averaged over batch) ───────────────────────────────┐\n"
        f"│  total            : {loss_info['total']:.4e}\n"
        f"│  geodesic energy  : {loss_info['geodesic_energy']:.4e}\n"
        f"│  smoothness pen.  : {loss_info['smoothness']:.4e}\n"
        f"│  monotonicity pen.: {loss_info['monotonicity']:.4e}\n"
        f"│  best iteration   : {loss_info['iter']}\n"
        "└───────────────────────────────────────────────────────────────────────┘"
    ))

    if not np.isnan(results['mean_error']):
        print("[Geodesic-Batch] Geodesic error statistics over batch:")
        print(f"    Mean       : {results['mean_error']:.4e}")
        print(f"    Std        : {results['std_error']:.4e}")
    else:
        print("[Geodesic-Batch] No GT geodesic method available in dataset.")

    return results['path']


# --- Entry-Point for Command-Line Execution ---

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    parser = ArgumentParser("Batch geodesic computation for diffusion models")
    parser.add_argument("--config", required=True, help="Path to the diffusion model config file (.py)")
    parser.add_argument("--geo-config", required=True, help="Path to the geodesic computation config file (.py)")
    args = parser.parse_args()

    # --- Load Configs ---
    diff_cfg = load_config(args.config)
    geo_cfg = load_geo_config(args.geo_config)

    # --- Save Master Config ---
    cfg_dir = os.path.join(diff_cfg.base_log_dir, diff_cfg.experiment)
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.pkl"), "wb") as f:
        pickle.dump(diff_cfg.to_dict(), f)

    # --- Set Reproducibility Seed ---
    seed = geo_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Execute the runner ---
    geodesic_batch_runner(diff_cfg, geo_cfg)
