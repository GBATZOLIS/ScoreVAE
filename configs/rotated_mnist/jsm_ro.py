#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Riemannian-optimisation config for latent path initialisation.

• Metric: Jacobian (JᵀJ + λI) with λ = 1e-3
• Diffusion time (for metric/denoiser): t = 0.01
• Strong-Wolfe line search with small init LR and capped expansion
• CG + denoiser retraction as requested

NOTE: `target_points` is injected at runtime (q_z per batch), do not set it here.
"""

CONFIG = {
    # How many pairs to pick from the eval loader
    "num_pairs": 10,

    # Geometry / metric time
    "riem_t": 0.04,
    "metric_type": "jacobian",
    "reg_lambda": 1e-3,   # J^T J + reg_lambda I inside CG/inverse

    # Geometric ingredients
    "retraction_operator": "denoiser",
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 6,
    "cg_tol": 1e-5,
    "cg_max_iter": 4,

    # Riemannian GD (GenericRiemannianGD)
    "optim_type": "pairwise_euclidean",  # we inject target_points dynamically
    "optim_lambda": 5.0,

    "riemannian_steps": 350,
    "riemannian_lr_init": 1e-2,      # small starting step
    "line_search": "strong_wolfe",   # curvature-aware steps
    "use_momentum": False,
    "momentum_coeff": 0.5,

    # Strong-Wolfe caps to prevent huge steps
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.8,
    "max_bracket": 13,
    "max_zoom": 10,
    "max_alpha": 48.0,                # hard cap on expansion

    # Armijo (unused here; left for quick switches)
    "armijo_rho": 1e-4,
    "armijo_beta": 0.5,

    # Early stopping helps keep the path short and smooth
    "patience": 70,
    "early_stop_delta": 0.0,

    # Logging
    "verbose": True,
}
