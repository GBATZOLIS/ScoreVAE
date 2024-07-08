from .sde import VPSDE, VESDE, subVPSDE, SNRSDE
from .sde_utils import get_named_beta_schedule
from scipy.interpolate import PchipInterpolator
import torch
import numpy as np

def configure_sde(config):
    sampling_eps = 1e-3  # Default sampling epsilon
    sde = None

    if config.training.sde.lower() == 'vpsde':
        sde = VPSDE(beta_min=0.1, beta_max=20., N=1000)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = subVPSDE(beta_min=0.1, beta_max=20., N=1000)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=0.01, sigma_max=50, N=1000)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'snrsde':
        sampling_eps = 1e-3

        if hasattr(config.training, 'beta_schedule'):
            # DISCRETE QUANTITIES
            N = 1000
            betas = get_named_beta_schedule('linear', N)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            discrete_snrs = alphas_cumprod / (1.0 - alphas_cumprod)

            # Monotonic Bicubic Interpolation
            snr = PchipInterpolator(np.linspace(sampling_eps, 1, len(discrete_snrs)), discrete_snrs)
            d_snr = snr.derivative(nu=1)

            def logsnr(t):
                device = t.device
                snr_val = torch.from_numpy(snr(t.cpu().numpy())).float().to(device)
                return torch.log(snr_val)

            def d_logsnr(t):
                device = t.device
                dsnr_val = torch.from_numpy(d_snr(t.cpu().numpy())).float().to(device)
                snr_val = torch.from_numpy(snr(t.cpu().numpy())).float().to(device)
                return dsnr_val / snr_val

            sde = SNRSDE(N=N, gamma=logsnr, dgamma=d_logsnr)
        else:
            sde = SNRSDE(N=1000)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    sde.sampling_eps = sampling_eps
    return sde
