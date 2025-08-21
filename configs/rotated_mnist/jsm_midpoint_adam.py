"""
Midpoint test (Adam) with noisy endpoints.
One interior node (n_segments=2) + single coarse time.
"""
from __future__ import annotations

CONFIG = {
    # ----- seeds / devices -----
    "random_seed": 42,
    "num_pairs": 4,
    "devices": ["cuda:0"],

    # ----- single coarse stage -----
    "time_schedule": [0.12, 0.10, 0.08],          # coarse → smoother metric

    # ----- init: let the solver *find* the midpoint -----
    "initialization": { "method": "linear" },  # NOT gt_noisy for this test

    # ----- metric -----
    "metric_type": "jacobian",
    "lam_metric": 1e-2,

    # ----- discretisation & regularisers -----
    "n_segments": 2,                  # ← exactly one interior node
    "lam_smooth": 300.0,               # moderate; stabilizes but doesn't dominate
    "lam_mono":   0.0,                # keep unbiased for midpoint

    # ----- optimizer: Adam (+ Armijo) -----
    "optimizer": "adam",
    "adam_lr":   1e-2,                # initial step for Armijo scaling
    "betas":     (0.9, 0.999),
    "eps":       1e-8,
    "line_search": "armijo",          # more robust than fixed here
    "armijo_rho":  5e-4,
    "armijo_beta": 0.6,
    "armijo_max_iter": 15,

    # ----- budget -----
    "max_iters": 60,                 # 1-node solve; usually converges fast
    "tol": 1e-6,
    "patience": 5,

    # ----- moment transport (Adam only) -----
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # ----- Jacobian-metric CG -----
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 10,
    "cg_tol": 1e-5,
    "cg_max_iter": 10,

    # ----- endpoints & post-denoise -----
    "endpoint_mode": "noisy",         # persistent xi handled inside compute_geodesic
    "post_denoise": {"method":"ddim", "to_time":1e-3, "steps":20},

    # ----- viz -----
    "plot_filename": "midpoint_adam_noisy.png",
    "visualization": {
        "animate_optimization": True,
        "num_animation_frames": 20,
        "animation_duration_ms": 100,
    },
}
