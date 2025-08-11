"""ro_utils.py
================
Helper functions for Riemannian optimisation driver – refactored without any change in behaviour.
"""

import os
import random
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Tuple
from torch.utils.data import Subset

# ---------------------------------------------------------------------
# Reproducibility utilities
# ---------------------------------------------------------------------

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------

def load_py_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ns: Dict[str, Any] = {}
    exec(compile(open(path, "r").read(), path, "exec"), ns)
    if "CONFIG" not in ns:
        raise ValueError(f"{path} must define CONFIG = {{...}}")
    return ns["CONFIG"]

# ---------------------------------------------------------------------
# Tensor shaping utilities
# ---------------------------------------------------------------------

def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)

def unflatten(x: torch.Tensor, shape) -> torch.Tensor:
    return x.view(x.size(0), *shape)

def ensure_B(t: torch.Tensor, B: int) -> torch.Tensor:
    if t.dim() == 0:
        return t.expand(B)
    if t.dim() == 1:
        return t if t.size(0) == B else t.expand(B)
    return t.squeeze(-1) if t.size(-1) == 1 else t

# ---------------------------------------------------------------------
# Score & denoiser factories
# ---------------------------------------------------------------------

def get_score_fn(sde, model, t, orig_shape):
    sigma_fn = sde.get_sigma_fn()

    def score_fn(x_flat: torch.Tensor) -> torch.Tensor:
        B = x_flat.size(0)
        sig = sigma_fn(ensure_B(t, B)).view(B, 1)
        x = unflatten(x_flat, orig_shape)
        eps = model(x, None, ensure_B(t, B))
        return -flatten(eps) / sig

    return score_fn

def get_denoiser_fn(sde, model, t, orig_shape):
    alpha_fn, sigma_fn = sde.get_alpha_fn(), sde.get_sigma_fn()

    def denoiser_fn(x_flat: torch.Tensor) -> torch.Tensor:
        B = x_flat.size(0)
        sig = sigma_fn(ensure_B(t, B)).view(B, 1)
        alp = alpha_fn(ensure_B(t, B)).view(B, 1)
        x = unflatten(x_flat, orig_shape)
        eps = model(x, None, ensure_B(t, B))
        return (x_flat - sig * flatten(eps)) / alp

    return denoiser_fn

# ---------------------------------------------------------------------
# Deterministic sampler
# ---------------------------------------------------------------------

def sample_indices_excluding(
    N: int,
    k: int,
    seed: int,
    exclude_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    candidates = torch.arange(N)
    if exclude_idx is not None and len(exclude_idx) > 0:
        mask = torch.ones(N, dtype=torch.bool)
        mask[exclude_idx] = False
        candidates = candidates[mask]
    return candidates[torch.randperm(len(candidates), generator=g)[:k]]

# ---------------------------------------------------------------------
# Manifold-error helpers
# ---------------------------------------------------------------------

def build_projector(Q: torch.Tensor | None) -> torch.Tensor | None:
    if Q is None:
        return None
    with torch.no_grad():
        return Q @ Q.T

def manifold_errors(
    x: torch.Tensor,
    expected_radius: float,
    P: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if P is not None:
        x_proj = torch.matmul(x, P)
        radial = (x_proj.norm(dim=1) - expected_radius).abs()
        subsp = (x - x_proj).norm(dim=1)
        return radial, subsp
    radial = (x.norm(dim=1) - expected_radius).abs()
    return radial, None

# ---------------------------------------------------------------------
# Full trajectory + best-point evaluation
# ---------------------------------------------------------------------

def evaluate_trajectory_and_best_points(
    trajectory: List[torch.Tensor],
    opt_fn: Callable[[torch.Tensor], torch.Tensor],
    expected_radius: float,
    projector: torch.Tensor | None,
    device: torch.device,
    metrics: Dict[str, Any],
):
    with torch.no_grad():
        traj_tensor = torch.stack([p.to(device) for p in trajectory])  # (T+1, B, d)
        Tp1, B, d = traj_tensor.shape

        rad_all, sub_all = manifold_errors(traj_tensor.view(-1, d), expected_radius, projector)
        rad_all = rad_all.view(Tp1, B)
        sub_all = sub_all.view(Tp1, B) if sub_all is not None else None
        metrics["trajectory_errors"] = {
            "radial_all": rad_all,
            "subspace_all": sub_all,
            "radial_mean": rad_all.mean().item(),
            "radial_max": rad_all.max().item(),
            "subspace_mean": sub_all.mean().item() if sub_all is not None else None,
            "subspace_max": sub_all.max().item() if sub_all is not None else None,
        }
        print(
            f"[driver] trajectory radial error (mean / max) = "
            f"{metrics['trajectory_errors']['radial_mean']:.3e} / "
            f"{metrics['trajectory_errors']['radial_max']:.3e}"
        )
        if sub_all is not None:
            print(
                f"[driver] trajectory subspace_mean error (mean / max) = "
                f"{metrics['trajectory_errors']['subspace_mean']:.3e} / "
                f"{metrics['trajectory_errors']['subspace_max']:.3e}"
            )
            full_error = rad_all + sub_all
            print(
                f"[driver] trajectory total error (mean / max) = "
                f"{full_error.mean():.3e} / {full_error.max():.3e}"
            )

        # Compute objective values only once
        obj_tensor = torch.stack([opt_fn(p) for p in traj_tensor])  # (T+1, B)
        min_vals, min_idx = obj_tensor.min(dim=0)  # (B,)
        best_pts = traj_tensor[min_idx, torch.arange(B)]  # (B, d)

        rad_best = rad_all[min_idx, torch.arange(B)]
        sub_best = sub_all[min_idx, torch.arange(B)] if sub_all is not None else None

        metrics["best_point_errors"] = {
            "radial": {
                "mean": rad_best.mean().item(),
                "max": rad_best.max().item(),
                "all": rad_best.cpu(),
            },
            "subspace": {
                "mean": sub_best.mean().item() if sub_best is not None else None,
                "max": sub_best.max().item() if sub_best is not None else None,
                "all": sub_best.cpu() if sub_best is not None else None,
            },
            "min_loss_per_point": min_vals.cpu(),
            "min_loss_mean": min_vals.mean().item(),
        }
        print(
            f"[driver] best-point mean loss = {metrics['best_point_errors']['min_loss_mean']:.4f}"
        )

# ---------------------------------------------------------------------
# Dataset / manifold specific enrichment
# ---------------------------------------------------------------------


def _sphere_handler(ds):
    """Return additional riemannian config entries for a sphere dataset."""
    base = ds.dataset if isinstance(ds, Subset) else ds
    emb = getattr(base, "embedding_matrix", None)
    if emb is None:
        ambient_dim = base.data.size(1)
        manifold_dim = (
            base.manifold_dim
            if isinstance(base.manifold_dim, int)
            else base.manifold_dim[0]
        )
        emb = torch.eye(ambient_dim, manifold_dim + 1)
    return {"embedding_matrix": emb.cpu()}


## ---------------------------------------------------------------------
# Dataset / manifold specific enrichment (using decorator registration)
# ---------------------------------------------------------------------

_MANIFOLD_DISPATCH: Dict[str, Callable[[Any], Dict[str, Any]]] = {}

def register_manifold(name: str):
    def decorator(fn: Callable[[Any], Dict[str, Any]]):
        _MANIFOLD_DISPATCH[name] = fn
        return fn
    return decorator


@register_manifold("sphere")
def _sphere_handler(ds):
    base = ds.dataset if isinstance(ds, Subset) else ds
    emb = getattr(base, "embedding_matrix", None)
    if emb is None:
        ambient_dim = base.data.size(1)
        manifold_dim = (
            base.manifold_dim
            if isinstance(base.manifold_dim, int)
            else base.manifold_dim[0]
        )
        emb = torch.eye(ambient_dim, manifold_dim + 1)
    return {"embedding_matrix": emb.cpu()}


def enrich_riem_cfg_with_manifold_info(
    riem_cfg: Dict[str, Any],
    diff_cfg,
    dataset,
) -> Dict[str, Any]:
    name = getattr(diff_cfg.data, "dataset", None)
    if name is None or name not in _MANIFOLD_DISPATCH:
        return riem_cfg

    extra = _MANIFOLD_DISPATCH[name](dataset)
    for k, v in extra.items():
        riem_cfg.setdefault(k, v)
    return riem_cfg