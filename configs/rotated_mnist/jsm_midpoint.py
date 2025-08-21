CONFIG = {
    "random_seed": 42,
    "num_pairs": 4,
    "devices": ["cuda:0"],

    # Single coarse stage so we're only testing the midpoint solve
    "time_schedule": [0.09, 0.07],

    # Linear init so the solver must *find* the midpoint
    "initialization": {
        "method": "linear"
    },

    "metric_type": "jacobian",
    "lam_metric": 1e-2,

    # One interior node (T = 3 nodes total)
    "n_segments": 2,

    # Keep smoothness moderate so it stabilizes but doesn't dominate
    "lam_smooth": 600.0,
    "lam_mono":   0.0,

    "optimizer": "rgd",
    "adam_lr":   5e-2,             # used as alpha_init for line search
    "line_search": "strong_wolfe",
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.7,               # a tad stricter curvature condition helps 1-node solves
    "wolfe_max_bracket": 12,
    "wolfe_max_zoom": 12,
    "wolfe_max_alpha": 20.0,
    "use_retraction_update": False,

    "max_iters": 7,               # give the midpoint a fair chance to settle
    "tol": 1e-6,
    "patience": 30,

    # IMPORTANT: noisy endpoints with fixed noise reused across stages
    # (our code already keeps the same xi internally for the whole schedule)
    "endpoint_mode": "noisy",

    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 10,
    "cg_tol": 1e-5,
    "cg_max_iter": 10,

    # (Denoising after final stage isn’t needed for the test, but harmless)
    "post_denoise": {"method": "ddim", "to_time": 1e-3, "steps": 20},

    "plot_filename": "midpoint_test_noisy.png",
    "visualization": {
        "animate_optimization": True,
        "num_animation_frames": 20,
        "animation_duration_ms": 100,
    },
}
