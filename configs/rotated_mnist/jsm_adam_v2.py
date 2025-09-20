"""
Ambient/image-space geodesics (e.g., RotMNIST), aligned with latent geodesic policies.

Key choices:
- No external init (let the solver build the t-projected init when endpoint_mode="noisy")
- Single final denoise (no DDIM chain)
- Jacobian metric (J^T J + λI)
- Riemannian Adam + Armijo line search
- Entropy profile logging (+ optional Φ̃-uniform schedule)
"""
from __future__ import annotations

CONFIG = {
    # ---------------------------------------------------------------------
    # Reproducibility & Data
    # ---------------------------------------------------------------------
    "random_seed": 42,
    "num_pairs":   15,                 # match latent experiments where possible
    "devices":     ["cuda:1"],

    # ---------------------------------------------------------------------
    # Diffusion Interface (coarse→fine)
    # ---------------------------------------------------------------------
    # You can also enable Φ̃-uniform schedule below to overwrite this.
    "time_schedule": [0.25, 0.2, 0.15], #[0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],

    # ---------------------------------------------------------------------
    # Entropy Profile (analysis and/or schedule selection)
    # ---------------------------------------------------------------------
    "entropic_profile": False,           # compute & save HR and Φ̃ profiles
    "use_entropic_schedule": False,     # if True, replace time_schedule with Φ̃-uniform
    "entropic_source": "mmse",          # {"mmse","score"} source for Φ̃
    "ent_num_t": 60,                    # # of t samples
    "ent_max_batches": 15,              # batches per t
    "ent_t_min": 0.01,                  # avoid t≈0 blow-ups
    "ent_t_max": 1.0,

    # ---------------------------------------------------------------------
    # Endpoints & Initialization Policy
    # ---------------------------------------------------------------------
    "endpoint_mode": "noisy",           # {"clean","noisy"} → enables t-projected init internally
    "initialization": {
        "method": "none"                # {"none","linear","ode"}; "none" recommended (t-project init)
    },

    # ---------------------------------------------------------------------
    # Metric Configuration
    # ---------------------------------------------------------------------
    "metric_type": "jacobian",          # {"stein","jacobian"}
    "lam_metric":  1e-2,                # g = J^T J + λ I

    # ---------------------------------------------------------------------
    # Discretisation & Regularisers
    # ---------------------------------------------------------------------
    "n_segments": 14,                   # T = n_segments + 1
    "lam_smooth": 10.0,
    "lam_mono":   10.0,

    # ---------------------------------------------------------------------
    # Optimizer (Riemannian Adam + Armijo)
    # ---------------------------------------------------------------------
    "optimizer": "adam",
    "adam_lr":   5e-3,                  # tune per dataset; try 5e-3 if too aggressive
    "betas":     (0.9, 0.999),
    "line_search": "armijo",            # {"fixed","armijo"}
    "armijo_rho":  5e-4,
    "armijo_beta": 0.5,
    "armijo_max_iter": 15,

    # ---------------------------------------------------------------------
    # Shared budget / stopping
    # ---------------------------------------------------------------------
    "max_iters": 750,
    "tol": 1e-6,
    "patience": 50,

    # ---------------------------------------------------------------------
    # Moment transport (Adam only)
    # ---------------------------------------------------------------------
    "transport_mode": "ad_hoc",
    "transport_steps": 1,

    # ---------------------------------------------------------------------
    # Conjugate-Gradient (for Jacobian metric)
    # ---------------------------------------------------------------------
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 10,
    "cg_tol": 1e-5,
    "cg_max_iter": 10,

    # ---------------------------------------------------------------------
    # Output / Logging
    # ---------------------------------------------------------------------
    "plot_filename": "geodesics_jacobian.png",
    "visualization": {
        "animate_optimization": True,
        "num_animation_frames": 20,
        "animation_duration_ms": 100,
    },
}
