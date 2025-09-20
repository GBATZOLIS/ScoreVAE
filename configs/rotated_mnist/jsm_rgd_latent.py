import numpy as np

# Geodesics in LATENT space of the AE using the latent diffusion model.
# Optimizer: RGD (+ Armijo LS). Metric: Jacobian (J^T J + λI) built from the score.
CONFIG = {
    "random_seed": 42,
    "num_pairs": 10,                          # how many (p,q) pairs from the eval set

    # diffusion / coarse→fine schedule (stabilises optimisation)
    "time_schedule": [0.3],

    # (A) just log and save profiles
    "entropic_profile": True,         # default True

    # (B) actually use the entropic schedule
    "use_entropic_schedule": False,    # default False (keeps old behaviour)
    "entropic_source": "mmse",        # {"mmse","score"}

    # (C) tuning for profile computation (optional)
    "ent_num_t": 200,                  # number of t samples for profiles
    "ent_max_batches": 20,            # dataloader batches to average at each t
    "ent_t_min": 5e-2,                # clamp away from 0 if needed
    "ent_t_max": 0.5,                 # usually 1.0 for VP

    "endpoint_mode": "noisy",  # {"clean","noisy"}

    # metric
    "metric_type": "jacobian",                # {"stein", "jacobian"}
    "lam_metric": 1e-2,                       # ~0.01–0.1 works well

    # path discretisation & regularisers
    "n_segments": 15,
    "lam_smooth": 0.3,
    "lam_mono":   2.0,

    # === OPTIMISER ===================================================
    "optimizer": "rgd",                       # {"adam","rgd"}

    # base LR (used as initial step for LS)
    "adam_lr":   1e-2, #5e-2,

    # line search (RGD supports: {"fixed","armijo","strong_wolfe"})
    "line_search": "armijo",
    "armijo_rho":  0.01,
    "armijo_beta": 0.7,
    "armijo_max_iter": 15,

    # RGD-specific retraction step
    "use_retraction_update": False,

    # Strong-Wolfe params (kept for easy switching)
    "wolfe_c1": 1e-3,
    "wolfe_c2": 0.4,
    "wolfe_max_bracket": 12,
    "wolfe_max_zoom": 12,
    "wolfe_max_alpha": 20.0,

    # shared budget / stopping
    "max_iters": 1200,
    "tol": 1e-6,
    "patience": 50,

    # moment transport (ignored by RGD)
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # CG (Jacobian metric only)
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 3,
    "cg_tol": 1e-5,
    "cg_max_iter": 4,

    # optional multi-GPU shard for a single path (rarely needed for latent d=16)
    # "devices": ["cuda:0", "cuda:1"],
}
