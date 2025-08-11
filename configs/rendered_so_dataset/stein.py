# geodesic_config_rendered_stein.py
"""
Configuration for geodesic computation on the RenderedSO3Dataset (Teapots)
using the Stein metric.
"""
import numpy as np

CONFIG = {
    # ---------------------------------------------------------------------
    # Reproducibility & Data
    # ---------------------------------------------------------------------
    "random_seed": 42,
    "num_pairs": 2,  # Number of (p,q) image pairs to process

    # ---------------------------------------------------------------------
    # Diffusion Interface
    # ---------------------------------------------------------------------
    # Time schedule for annealing. Start with a higher noise level (t) and
    # gradually decrease it to refine the path.
    "time_schedule": [0.07, 0.06], #[0.06, 0.05, 0.04, 0.03],

    # Geodesic Path Initialization
    # ---------------------------------------------------------------------
    "initialization": {
        "method": "linear",              # 'ode' or 'linear'
        "ode_solver_steps": 128,       # Number of steps for the ODE solver
        "t0": 1e-3,
    },
    
    # ---------------------------------------------------------------------
    # Metric Configuration
    # ---------------------------------------------------------------------
    "metric_type": "stein",  # Use the Stein metric
    "lam_metric":  2.0,      # λ for g = I + λ ssᵀ

    # ---------------------------------------------------------------------
    # Geodesic Optimization Algorithm
    # ---------------------------------------------------------------------
    "n_segments": 15,          # Number of segments in the discrete path
    "lam_smooth": 100.0,       # Curvature regularization
    "lam_mono":   2.0,         # Monotonicity penalty

    # ---------------------------------------------------------------------
    # Riemannian-Adam Optimizer
    # ---------------------------------------------------------------------
    "adam_lr":   2e-4, 
    "betas":     (0.9, 0.999),
    "max_iters": 250,
    "tol":       1e-6,
    "patience":  10,
    "line_search": 'armijo',
    "transport_mode": "ad_hoc",
    "transport_steps": 1,
    "armijo_rho": 0.01,
    "armijo_beta": 0.7,
    "armijo_max_iter": 15,

    # ---------------------------------------------------------------------
    # Output / Logging
    # ---------------------------------------------------------------------
    "plot_filename": "geodesics_stein.png",
    "visualization": {
        "animate_optimization": True,       # True to generate a GIF of the optimization
        "num_animation_frames": 40,         # Total number of frames in the GIF
        "animation_duration_ms": 100,       # Duration per frame in milliseconds
    },
}