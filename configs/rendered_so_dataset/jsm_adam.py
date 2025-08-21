"""
Configuration for geodesic computation (RenderedSO3Dataset • Teapots)
using the Jacobian metric and the Adam optimizer.
"""
from __future__ import annotations
import numpy as np  # optional; keep if your loader expects it

CONFIG = {
    # ---------------------------------------------------------------------
    # Reproducibility & Data
    # ---------------------------------------------------------------------
    "random_seed": 42,
    "num_pairs": 1,  # number of (p, q) image pairs

    # ---------------------------------------------------------------------
    # Diffusion Interface (coarse→fine)
    # ---------------------------------------------------------------------
    "time_schedule": [0.12, 0.10, 0.08],

    # ---------------------------------------------------------------------
    # Geodesic Path Initialization
    # ---------------------------------------------------------------------
    "initialization": {
        "method": "linear",         # {"ode", "linear"}
        "ode_solver_steps": 250,
        "t0": 1e-3,
    },

    # ---------------------------------------------------------------------
    # Metric Configuration
    # ---------------------------------------------------------------------
    "metric_type": "jacobian",
    "lam_metric":  1e-3,           # g = J^T J + λ I

    # ---------------------------------------------------------------------
    # Discretisation & Regularisers
    # ---------------------------------------------------------------------
    "n_segments": 13,
    "lam_smooth": 100.0,
    "lam_mono":   2.0,

    # ---------------------------------------------------------------------
    # Optimizer (ADAM)
    # ---------------------------------------------------------------------
    "optimizer": "adam",
    "adam_lr":   1e-4,             # your original small LR
    "betas":     (0.9, 0.999),
    "eps":       1e-8,

    # Line search for Adam: {"fixed","armijo"}
    "line_search": "armijo",
    "armijo_rho":  0.01,
    "armijo_beta": 0.7,
    "armijo_max_iter": 15,

    # Shared budget / stopping
    "max_iters": 150,
    "tol": 1e-6,
    "patience": 20,

    # Moment transport (used by Adam; RGD ignores)
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # ---------------------------------------------------------------------
    # Conjugate-Gradient (for Jacobian metric)
    # ---------------------------------------------------------------------
    "cg_kwargs": {
        "preconditioner": "diagonal",
        "precond_diag_samples": 10,
        "tol": 1e-5,
        "max_iter": 10,
    },
    # (Flat keys kept too, in case your driver reads this style)
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 10,
    "cg_tol": 1e-5,
    "cg_max_iter": 10,

    # ---------------------------------------------------------------------
    # Output / Logging
    # ---------------------------------------------------------------------
    "plot_filename": "geodesics_jacobian.png",
    "visualization": {
        "animate_optimization": True,
        "num_animation_frames": 20,
        "animation_duration_ms": 100,
    },
}
