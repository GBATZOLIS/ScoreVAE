# geodesic_config_rendered_jacobian.py
"""
Configuration for geodesic computation on the RenderedSO3Dataset (Teapots)
using the Jacobian metric.
"""
import numpy as np

CONFIG = {
    # ---------------------------------------------------------------------
    # Reproducibility & Data
    # ---------------------------------------------------------------------
    "random_seed": 42,
    "num_pairs": 1,  # Number of (p,q) image pairs to process

    # ---------------------------------------------------------------------
    # Diffusion Interface
    # ---------------------------------------------------------------------
    "time_schedule": [0.12, 0.1, 0.08], #[0.06, 0.05, 0.04, 0.03],

    # Geodesic Path Initialization
    # ---------------------------------------------------------------------
    "initialization": {
        "method": "linear",              # 'ode' or 'linear'
        "ode_solver_steps": 250,       # Number of steps for the ODE solver
        "t0": 1e-3,
    },

    # ---------------------------------------------------------------------
    # Metric Configuration
    # ---------------------------------------------------------------------
    "metric_type": "jacobian",  # Use the Jacobian metric
    "lam_metric":  1e-3,        # λ for g = JᵀJ + λI

    # ---------------------------------------------------------------------
    # Geodesic Optimization Algorithm
    # ---------------------------------------------------------------------
    "n_segments": 13,
    "lam_smooth": 100.0,
    "lam_mono":   2.0,

    # ---------------------------------------------------------------------
    # Riemannian-Adam Optimizer
    # ---------------------------------------------------------------------
    "adam_lr":   1e-4,
    "betas":     (0.9, 0.999), #(0.8, 0.990),
    "max_iters": 400,
    "tol":       1e-6,
    "patience":  10,
    "line_search": 'armijo',
    "transport_mode": "ad_hoc",
    "transport_steps": 1,
    "armijo_rho": 0.01,
    "armijo_beta": 0.7,
    "armijo_max_iter": 15,

    # ---------------------------------------------------------------------
    # Conjugate-Gradient (for Jacobian metric)
    # ---------------------------------------------------------------------
    "cg_kwargs": {
        "preconditioner": "diagonal",
        "precond_diag_samples": 10,
        "tol": 1e-5,
        "max_iter": 10,
    },

    # ---------------------------------------------------------------------
    # Output / Logging
    # ---------------------------------------------------------------------
    "plot_filename": "geodesics_jacobian.png",
    "visualization": {
        "animate_optimization": True,       # True to generate a GIF of the optimization
        "num_animation_frames": 20,         # Total number of frames in the GIF
        "animation_duration_ms": 100,       # Duration per frame in milliseconds
    },
}