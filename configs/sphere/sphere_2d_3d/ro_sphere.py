#!/usr/bin/env python
"""Riemannian‑optimisation hyper‑parameters for the 2‑sphere dataset.

Save this file (e.g. as *riem_config_sphere.py*) and point the optimisation
launcher at it:

    python riemannian_optimization.py \
        --config path/to/sphere_diffusion_config.py \
        --riem-config riem_config_sphere.py

Tweak *time_for_perturbation* or *min_point* as needed; most other settings
match those that worked for the Earth‑latitude example.
"""

import numpy as np

def unit_sphere_xyz(theta_deg: float, phi_deg: float):
    """Return (x, y, z) on the unit sphere given spherical angles in degrees.

    θ (theta) is the inclination from the +z axis ∈ [0°, 180°].
    φ (phi)   is the azimuth around the z‑axis        ∈ [0°, 360°).
    """
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return [x, y, z]

CONFIG = {
    # ------------------------------------------------------------------
    # Reproducibility & basic setup
    # ------------------------------------------------------------------
    "random_seed": 42,
    "num_points": 40,

    # ------------------------------------------------------------------
    # Diffusion interface
    # ------------------------------------------------------------------
    # Pick the lowest *t* that still gives a comfortable SNR (≈ 5 dB).
    "time_for_perturbation": 0.006,

    # ------------------------------------------------------------------
    # Riemannian optimiser – global hyper‑parameters
    # ------------------------------------------------------------------
    "metric_type"  : "stein", #"jacobian"     ,  # {jacobian, stein}
    "reg_lambda": 1, #5e-4,              # Regularisation λ for the metric g = JᵀJ + λI or I + λssᵀ for Stein.
    "gradient_impl": "generic",      # {generic, classic} 

    "riemannian_steps": 25,          # Outer iterations (Algorithm‑1)
    "riemannian_lr_init": 1e-2,      # Initial learning rate (if GD)
    "optimizer_type": "gradient_descent",  # {gradient_descent, trust_region}
    "use_momentum": False,
    "momentum_coeff": 0.6,

    # Trust‑region (ignored for GD)
    "trust_region_delta0": 0.1,
    "trust_region_eta_success": 0.75,
    "trust_region_eta_fail": 0.25,
    "trust_region_gamma_inc": 2.0,
    "trust_region_gamma_dec": 0.5,

    # Line search (used by GD variant)
    "line_search": "strong_wolfe",
    "wolfe_c1": 1e-5,
    "wolfe_c2": 0.8,
    "max_bracket": 15,
    "max_zoom": 10,
    "max_alpha": 300,
    "armijo_rho": 1e-6,
    "armijo_beta": 0.1,

    # ------------------------------------------------------------------
    # Geometric ingredients
    # ------------------------------------------------------------------
    "retraction_operator": "denoiser",  # {identity, denoiser}

    # Conjugate‑gradient acceleration of ∇_Riem
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 6,
    "cg_tol": 5e-5,
    "cg_max_iter": 4,

    # ------------------------------------------------------------------
    # Euclidean objective f(x)  (soft‑minimum of quadratics)
    # ------------------------------------------------------------------
    # Six symmetric minima on the unit sphere: ±x, ±y, ±z.
    "optim_type"   : "softmin",  # {softmin, euclidean}
    "min_point": [
        unit_sphere_xyz(0,   0),   # +z  (north pole)
        unit_sphere_xyz(180, 0),   # −z  (south pole)
        #unit_sphere_xyz(90,  0),   # +x  (equator, φ=0°)
        #unit_sphere_xyz(90,  90),  # +y  (equator, φ=90°)
        #unit_sphere_xyz(90, 180),  # −x  (equator, φ=180°)
        #unit_sphere_xyz(90, 270),  # −y  (equator, φ=270°)
    ],
    "optim_lambda": 10.0,         # Sharpness of the soft minimum

    # ------------------------------------------------------------------
    # Logging / visualisation
    # ------------------------------------------------------------------
    "log_dir": "logs",
    "plot_filename": "sphere_riemopt.png",
}
