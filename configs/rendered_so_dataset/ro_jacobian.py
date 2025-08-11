#!/usr/bin/env python
"""
Riemannian-optimisation hyper-parameters for the RenderedSO3Dataset
(128 × 128 grayscale teapot images on the full SO(3) manifold).

Usage example:
    python riemannian_optimization.py \
        --config configs/rendered_so_dataset/config.py \
        --riem-config ro_jacobian.py
"""

import numpy as np

CONFIG = {
    # ──────────────────────────────────────────────────────────
    # Reproducibility & basic setup
    # ──────────────────────────────────────────────────────────
    "random_seed": 42,
    "num_points": 2,  # Matching the default used for geodesic evaluation

    # ──────────────────────────────────────────────────────────
    # Diffusion interface
    # ──────────────────────────────────────────────────────────
    "time_for_perturbation": 0.005,

    # ──────────────────────────────────────────────────────────
    # Riemannian optimiser – global hyper-parameters
    # ──────────────────────────────────────────────────────────
    "metric_type": "jacobian",      # Alternatives: {jacobian, stein}
    "reg_lambda": 1e-3,             # Regularisation λ for g = JᵀJ + λI
    "gradient_impl": "generic",    # Alternatives: {generic, classic}

    "riemannian_steps": 100,
    "riemannian_lr_init": 1,  # Initial learning rate for GD
    "optimizer_type": "gradient_descent",
    "use_momentum": False,
    "momentum_coeff": 0.6,

    # Line search parameters (used by GD)
    "line_search": "strong_wolfe",
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.5,
    "max_bracket": 15,
    "max_zoom": 10,
    "max_alpha": 10000,
    "armijo_rho": 1e-6,
    "armijo_beta": 0.1,

    # ──────────────────────────────────────────────────────────
    # Geometric ingredients
    # ──────────────────────────────────────────────────────────
    "retraction_operator": "denoiser",
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 6,
    "cg_tol": 1e-5,
    "cg_max_iter": 6,

    # ──────────────────────────────────────────────────────────
    # Optimisation function
    # ──────────────────────────────────────────────────────────
    "optim_type": "euclidean",
    "optim_lambda": 10.0,
    "n_minima": 1,
    "min_point": None, 

    # ──────────────────────────────────────────────────────────
    # Logging / visualisation
    # ──────────────────────────────────────────────────────────
    "log_dir": "logs",
    "plot_filename": "teapot_so3_riemopt.png",
    "visualize": True,
}
