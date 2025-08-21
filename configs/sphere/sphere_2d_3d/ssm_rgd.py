# geodesic_config.py  (SSM • RGD)
CONFIG = {
    "random_seed": 42,
    "num_pairs": 100,

    # diffusion / schedule – coarse→fine stabilises RGD
    "time_schedule": [0.03],

    # metric
    "metric_type": "stein",      # {"stein", "jacobian"}
    "lam_metric": 1.,

    # discretisation & path regularisers
    "n_segments": 15,
    "lam_smooth": 200.0,
    "lam_mono":   2.0,

    # === OPTIMIZER SELECTION =========================================
    "optimizer": "rgd",             # {"adam","rgd"}

    # --- Base LR (used by RGD as alpha_init/init_step for LS) --------
    # NOTE: compute_geodesic reuses "adam_lr" for RGD's initial alpha.
    "adam_lr":   5e-2,

    # --- Line search -------------------------------------------------
    # For Adam: {"fixed","armijo"}; For RGD: {"fixed","armijo","strong_wolfe"}
    "line_search": "armijo",  

    # Armijo (if you switch LS to "armijo")
    "armijo_rho":  1e-4,
    "armijo_beta": 0.5,
    "armijo_max_iter": 15,

    # RGD-specific
    "use_retraction_update": False,   # apply retraction for updates
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.7,
    "wolfe_max_bracket": 12,
    "wolfe_max_zoom": 12,
    "wolfe_max_alpha": 20.0,         # cap to avoid runaway expansion

    # --- Shared budget / stopping -----------------------------------
    "max_iters": 1500,
    "tol": 1e-6,
    "patience": 50,

    # --- Moment transport (ignored by RGD) --------------------------
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # --- CG (Jacobian metric only) ----------------------------------
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 6,
    "cg_tol": 5e-5,
    "cg_max_iter": 4,

    # --- Viz ---------------------------------------------------------
    "plot_filename": "geodesics_stein.png",
}
