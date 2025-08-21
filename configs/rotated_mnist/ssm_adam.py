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
    "num_pairs": 4,  # number of (p, q) image pairs
    "devices": ["cuda:0"],

    # ---------------------------------------------------------------------
    # Diffusion Interface (coarse→fine)
    # ---------------------------------------------------------------------
    "time_schedule": [0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05], #, 0.07, 0.06, 0.05, 0.04, 0.03],

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
    "metric_type": 'stein', #"jacobian",
    "lam_metric":  1., #1e-2,           # g = J^T J + λ I

    # ---------------------------------------------------------------------
    # Discretisation & Regularisers
    # ---------------------------------------------------------------------
    "n_segments": 16,
    "lam_smooth": 100.0,
    "lam_mono":   2.0,

    # ---------------------------------------------------------------------
    # Optimizer (ADAM)
    # ---------------------------------------------------------------------
    "optimizer": "adam",
    "adam_lr":   5e-4,             # your original small LR
    "betas":     (0.9, 0.999),
    "eps":       1e-8,

    # Line search for Adam: {"fixed","armijo"}
    "line_search": "fixed",
    "armijo_rho":  5e-4,
    "armijo_beta": 0.5,
    "armijo_max_iter": 15,

    # Shared budget / stopping
    "max_iters": 500,
    "tol": 1e-6,
    "patience": 20,

    # Moment transport (used by Adam; RGD ignores)
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # ---------------------------------------------------------------------
    # Conjugate-Gradient (for Jacobian metric)
    # ---------------------------------------------------------------------

    # (Flat keys kept too, in case your driver reads this style)
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 10,
    "cg_tol": 1e-5,
    "cg_max_iter": 10,

    "post_denoise": {
        "method": "ddim",
        "to_time": 1e-3,
        "steps": 20
    },

    "endpoint_mode": 'noisy', # {'clean', 'noisy'}

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
