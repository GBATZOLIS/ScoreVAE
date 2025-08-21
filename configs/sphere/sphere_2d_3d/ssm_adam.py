# geodesic_config.py  (Stein)
CONFIG = {
    "random_seed": 42,
    "num_pairs": 100,

    "time_schedule": [0.06, 0.05, 0.04, 0.03],

    "metric_type": "stein",
    "lam_metric": 1.0,

    "n_segments": 15,
    "lam_smooth": 200.0,
    "lam_mono":   2.0,

    # === OPTIMIZER SELECTION (default: ADAM) =========================
    "optimizer": "adam",            # {"adam","rgd"}

    # --- Adam-specific ---
    "adam_lr":   1e-2,
    "betas":     (0.9, 0.999),
    "eps":       1e-8,

    # --- Line search (shared key) ---
    "line_search": "armijo",
    "armijo_rho":  0.05,
    "armijo_beta": 0.7,
    "armijo_max_iter": 15,

    # --- RGD-specific (safe defaults even if not used) ---------------
    "use_retraction_update": True,
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.5,
    "wolfe_max_bracket": 10,
    "wolfe_max_zoom": 10,
    "wolfe_max_alpha": 50.0,
    # "rgd_step_size": 1e-2,

    # --- Shared budget / stopping ---
    "max_iters": 2000,
    "tol": 1e-6,
    "patience": 50,

    # --- Moment transport (Adam only) --------------------------------
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # --- CG block ignored by Stein, but harmless ---------------------
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 6,
    "cg_tol": 5e-5,
    "cg_max_iter": 4,

    "plot_filename": "geodesics_stein.png",
}
