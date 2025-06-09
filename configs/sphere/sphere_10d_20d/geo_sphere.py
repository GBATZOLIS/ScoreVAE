# geodesic_config.py
"""Configuration file for *geodesic_batch.py*.

Copy / edit as needed for your own experiments.  The values below reproduce the
settings we have discussed so far:
*   **time_for_perturbation** is the diffusion time *t* \in [0,1] that defines
    (α_t, σ_t) used to noise the endpoints before running Algorithm‑2.
*   Change **metric_type** to "jacobian" to use the JᵀJ metric; when you do,
    the CG‑related keys are forwarded to the inner solver.
"""

import numpy as np

CONFIG = {
    # ---------------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------------
    "random_seed": 42,

    # ---------------------------------------------------------------------
    # Data batching / pair selection
    # ---------------------------------------------------------------------
    "num_pairs": 200,               # number of (p,q) pairs to process together

    # ---------------------------------------------------------------------
    # Diffusion interface
    # ---------------------------------------------------------------------
    # Forward‑noise level t  (0 < t ≤ 1).  Pick the smallest t such that
    # optimisation is numerically stable (SNR ≳ 5 dB).
    "time_for_perturbation": 0.04, # 0.2 is typical for SD‑2.1 latents

    # ---------------------------------------------------------------------
    # Metric
    # ---------------------------------------------------------------------
    #stein,lam_metric->1
    #jacobian,lam_metric->0.05
    "metric_type": "jacobian",       # {"stein", "jacobian"}
    "lam_metric":   0.05,          # λ in  g = I + λ ssᵀ  or  JᵀJ + λI
    #stein,lam_metric->1
    #jacobian,lam_metric->0.01-0.1
    # ---------------------------------------------------------------------
    # Geodesic algorithm (Algorithm‑2)
    # ---------------------------------------------------------------------
    "n_segments": 15,             # resolution of the discretised path
    "lam_smooth": 100.,           # curvature penalty
    "lam_mono":   2.,            # "monotonic progress" penalty (set >0 if
                                   # path tends to double back)

    # ---------------------------------------------------------------------
    # Riemannian‑Adam optimiser (Algorithm‑1)
    # ---------------------------------------------------------------------
    "adam_lr":   1e-2,            # learning‑rate α
    "betas":     (0.9, 0.999),    # momentum coefficients β₁, β₂
    "eps":       1e-8,            # numerical stability ε
    "max_iters": 1000,             # optimisation budget
    "tol":       1e-6,            # stop when max‖∇_Riem‖ < tol
    "patience": 40,  # stop if no improvement in last 20 steps

    # ---------------------------------------------------------------------
    # Conjugate‑Gradient (Jacobian metric only)
    # ---------------------------------------------------------------------
    "cg_preconditioner":        "diagonal", # {"none", "diagonal"}
    "cg_precond_diag_samples": 6,           # #samples for diag pre‑cond
    "cg_tol":       5e-5,                   # relative residual tol
    "cg_max_iter":  4,                      # hard cap on CG iterations

    # ---------------------------------------------------------------------
    # Output / logging
    # ---------------------------------------------------------------------
    "plot_filename": "geodesics.png",
}
