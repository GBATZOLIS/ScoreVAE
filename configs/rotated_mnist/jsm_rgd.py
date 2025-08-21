"""
Configuration for geodesic computation (RenderedSO3Dataset • Teapots)
using the Jacobian metric and the Riemannian Gradient Descent (RGD) solver.
"""
from __future__ import annotations
import numpy as np  # optional; keep if your loader expects it

CONFIG = {
    # ---------------------------------------------------------------------
    # Reproducibility & Data
    # ---------------------------------------------------------------------
    "random_seed": 42,
    "num_pairs": 4,
    "devices": ["cuda:0"],

    # ---------------------------------------------------------------------
    # Diffusion Interface (coarse→fine)
    # ---------------------------------------------------------------------
    "time_schedule": [0.05], #, 0.07, 0.06, 0.05, 0.04, 0.03],

    # ---------------------------------------------------------------------
    # Geodesic Path Initialization
    # ---------------------------------------------------------------------
    "initialization": {
        "method": "linear",         # {"ode", "linear"}
        "ode_solver_steps": 250,
        "t0": 0.007,
    },

    # ---------------------------------------------------------------------
    # Metric Configuration
    # ---------------------------------------------------------------------
    "metric_type": "jacobian", # {"jacobian", "stein"}
    "lam_metric": 1e-2, # 1e-3,           # g = J^T J + λ I

    # ---------------------------------------------------------------------
    # Discretisation & Regularisers
    # ---------------------------------------------------------------------
    "n_segments": 16,
    "lam_smooth": 1.0,
    "lam_mono":   2.0,

    # ---------------------------------------------------------------------
    # Optimizer (RGD)
    # ---------------------------------------------------------------------
    "optimizer": "rgd",

    # Base LR is reused by RGD as alpha_init/init_step for line search
    # (you can bump to 5e-2 if steps feel too timid)
    "adam_lr":   5e-2,

    # Default LS for RGD: {"fixed","armijo","strong_wolfe"}
    "line_search": "armijo",
    "armijo_rho":  5e-4,
    "armijo_beta": 0.6,
    "armijo_max_iter": 15,

    # Retraction toggle for RGD updates
    "use_retraction_update": False,   # x <- x + Δ (set True to use metric.retraction_fn)

    # Strong-Wolfe params (only used if line_search == "strong_wolfe")
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.6,
    "wolfe_max_bracket": 12,
    "wolfe_max_zoom": 12,
    "wolfe_max_alpha": 20.0,

    # Shared budget / stopping
    "max_iters": 20,
    "tol": 1e-6,
    "patience": 5,

    # Moment transport (ignored by RGD; kept for API symmetry)
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # ---------------------------------------------------------------------
    # Conjugate-Gradient (for Jacobian metric)
    # ---------------------------------------------------------------------

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
