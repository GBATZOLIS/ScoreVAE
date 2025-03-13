import numpy as np

def sph_from_deg(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return [x, y, z]

CONFIG = {
    # Random seed for reproducibility
    "random_seed": 42,

    # Riemannian optimization parameters
    "time_for_perturbation": 0.01,
    "reg_lambda": 5e-4,
    "riemannian_steps": 18,
    "riemannian_lr_init": 1e-2,
    
    # Optimizer selection:
    "optimizer_type": "gradient_descent",  # Choices:["gradient_descent", "trust_region"]

    # Trust-region parameters
    "trust_region_delta0": 0.1,
    "trust_region_eta_success": 0.75,
    "trust_region_eta_fail": 0.25,
    "trust_region_gamma_inc": 2.0,
    "trust_region_gamma_dec": 0.5,

    # Line search parameters (used by gradient descent branch)
    "line_search": "strong_wolfe",
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.7,
    "max_bracket": 15,
    "max_zoom": 10,
    "max_alpha": 300,
    "armijo_rho": 1e-6,
    "armijo_beta": 0.1,

    # Retraction operator options: "identity" or "denoiser"
    "retraction_operator": "denoiser",

    # Momentum settings (if used in gradient descent)
    "use_momentum": True,  # (set to False for now)
    "momentum_coeff": 0.6,

    # Settings for fast calculation of Riemannian gradient via CG
    "cg_preconditioner": 'diagonal',
    "cg_precond_diag_samples": 8, 
    "cg_tol": 5e-5, 
    "cg_max_iter": 4,

    # Objective function parameters
    "min_point": [
        sph_from_deg(51.5, -0.1),     # London ~ [0.6235, -0.0011, 0.7818]
        sph_from_deg(39.9, 116.4),      # Beijing ~ [-0.3447, 0.6830, 0.6428]
        sph_from_deg(30.0, 31.2),       # Cairo   ~ [0.7410, 0.4460, 0.5000]
        sph_from_deg(-23.5, -46.6),     # São Paulo ~ [0.6300, -0.6660, -0.3980]
        sph_from_deg(-33.9, 151.2),      # Sydney  ~ [-0.7290, 0.3990, -0.5540]
        sph_from_deg(37.8, -122.4)      # San Francisco ~ [0.4180, -0.7280, 0.5440]
    ],

    "optim_lambda": 10,

    # Logging
    "log_dir": "logs",
    "plot_filename": "combined_plot.png",
}
