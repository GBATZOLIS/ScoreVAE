# geodesic_config.py  (Jacobian)
CONFIG = {
    "random_seed": 42,
    "num_pairs": 100,

    # diffusion / schedule
    "time_schedule": [0.06, 0.05, 0.04, 0.03],

    # metric
    "metric_type": "jacobian",      # {"stein", "jacobian"}
    "lam_metric": 0.05,

    # discretisation & path regularisers
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
    # For Adam: {"fixed","armijo"};  For RGD: {"fixed","armijo","strong_wolfe"}
    "line_search": "armijo",
    "armijo_rho":  0.01,
    "armijo_beta": 0.7,
    "armijo_max_iter": 15,   # used by Adam-Armijo and RGD-Armijo

    # --- RGD-specific (safe defaults even if not used) ---------------
    # If you switch to optimizer="rgd":
    # - recommended: line_search="strong_wolfe"
    # - if you choose "fixed", we'll use adam_lr as the fixed step (or add "rgd_step_size")
    "use_retraction_update": True,   # RGD steps via metric.retraction_fn(x, Δ)
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.5,
    "wolfe_max_bracket": 10,
    "wolfe_max_zoom": 10,
    "wolfe_max_alpha": 50.0,
    # Optional: uncomment if you want a separate fixed step just for RGD
    # "rgd_step_size": 1e-2,

    # --- Shared budget / stopping ---
    "max_iters": 2000,
    "tol": 1e-6,
    "patience": 50,

    # --- Moment transport (used by Adam only; RGD ignores) ----------
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # --- CG (Jacobian metric only) ----------------------------------
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 6,
    "cg_tol": 5e-5,
    "cg_max_iter": 4,

    # --- Viz ---------------------------------------------------------
    "plot_filename": "geodesics_jacobian.png",
}
