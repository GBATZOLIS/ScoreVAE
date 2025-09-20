# Geodesics in LATENT space of the AE using the latent diffusion model.
# Optimizer: Riemannian Adam (+ Armijo LS). Metric: Jacobian (J^T J + λI).
CONFIG = {
    "random_seed": 42,
    "num_pairs": 100,

    # diffusion schedule
    "time_schedule": [0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01], #[0.14, 0.12, 0.10, 0.08], #[0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.11, 0.09, 0.07, 0.05, 0.04], 

    # (A) just log and save profiles
    "entropic_profile": True,         # default True

    # (B) actually use the entropic schedule
    "use_entropic_schedule": False,    # default False (keeps old behaviour)
    "entropic_source": "mmse",        # {"mmse","score"}

    # (C) tuning for profile computation (optional)
    "ent_num_t": 60,                  # number of t samples for profiles
    "ent_max_batches": 15,            # dataloader batches to average at each t
    "ent_t_min": 0.01,                # clamp away from 0 if needed
    "ent_t_max": 1.,                 # usually 1.0 for VP

    "endpoint_mode": "noisy",  # {"clean","noisy"}

    # metric
    "metric_type": "jacobian",         # {"stein", "jacobian"}
    "lam_metric": 1e-2,

    # path discretisation & regularisers
    "n_segments": 30,
    "lam_smooth": 2.0, #0.65,
    "lam_mono":   2.0,

    # === OPTIMISER ===================================================
    "optimizer": "adam",               # {"adam","rgd"}
    "adam_lr":   5e-2,
    "betas":     (0.9, 0.999),
    # line search for Adam: {"fixed","armijo"}
    "line_search": "armijo",
    "armijo_rho":  0.01,
    "armijo_beta": 0.7,
    "armijo_max_iter": 15,

    # shared budget / stopping
    "max_iters": 1500,
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
}
