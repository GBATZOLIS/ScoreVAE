# geodesic_config_latent_tproject.py
# Geodesics in LATENT space of the AE using the latent diffusion model.
# Optimizer: Riemannian Adam (+ Armijo LS). Metric: Jacobian (J^T J + λI).
CONFIG = {
    "random_seed": 42,
    "num_pairs": 25,

    # diffusion schedule (descending = coarse → fine)
    "time_schedule": [0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05], #[0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10], #[0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.09, 0.07], #[0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.11, 0.09, 0.07, 0.05, 0.04],

    # (A) log and save profiles
    "entropic_profile": False,

    # (B) replace schedule with entropy-uniform (optional)
    "use_entropic_schedule": False,
    "entropic_source": "mmse",        # {"mmse","score"}
    "ent_num_t": 50,
    "ent_max_batches": 20,
    "ent_t_min": 0.01,
    "ent_t_max": 1.0,

    # endpoints at each stage
    "endpoint_mode": "noisy",         # {"clean","noisy"}

    # metric
    "metric_type": "jacobian",        # {"stein", "jacobian"}
    "lam_metric": 5e-2,

    # path discretisation & regularisers
    "n_segments": 30,
    "lam_smooth":  1.,
    "lam_mono":   1.,

    # === OPTIMISER ===================================================
    "optimizer": "adam",              # {"adam","rgd"}
    "adam_lr":   3e-2, #8e-2,
    "betas":     (0.9, 0.999),
    "line_search": "armijo",          # {"fixed","armijo"}
    "armijo_rho":  0.01,
    "armijo_beta": 0.7,
    "armijo_max_iter": 15,

    # shared budget / stopping
    "max_iters": 250, #500,
    "tol": 1e-6,
    "patience": 50,

    # moment transport (Adam only)
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # CG (Jacobian metric only)
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 6,
    "cg_tol": 1e-5,
    "cg_max_iter": 4,

    # ── NEW: initializer controls ────────────────────────────────────
    "init_method": "tproject",        # {"tproject","linear"}
    "init_add_noise": True,           # add SAME ξ to interiors after projecting back to time t

    # (optional) multi-GPU sharding for single path
    # "devices": ["cuda:0","cuda:1"],
}
