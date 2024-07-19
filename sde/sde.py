import abc
import torch
import numpy as np

class SDE(abc.ABC):
    def __init__(self, N):
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    def perturb(self, x_0, t):
        z = torch.randn_like(x_0)
        mean, std = self.marginal_prob(x_0, t)
        perturbed_data = mean + std[(...,) + (None,) * len(x_0.shape[1:])] * z
        return perturbed_data

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        pass

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, y, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, y, t)
                drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, y, t):
                f, G = discretize_fn(x, t)
                rev_f = f - G[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score_fn(x, y, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min))).type_as(t))
        return drift, diffusion

    def marginal_prob(self, x, t):
        sigma_min = torch.tensor(self.sigma_min).type_as(t)
        sigma_max = torch.tensor(self.sigma_max).type_as(t)
        std = sigma_min * (sigma_max / sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        dims_to_reduce = tuple(range(len(z.shape))[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=dims_to_reduce) / (2 * self.sigma_max ** 2)

class SNRSDE(SDE):
  def __init__(self, N, gamma=None, dgamma=None, a=2, b=3, c=6, minus_log_SNR_0 = -10, minus_log_SNR_1 = 5):
    super().__init__(N)
    if gamma is None:
      gamma = lambda t: a * t + b * t**c
      d_gamma = lambda t: a + b*c * t**(c-1)
      
      # Gamma has to be normalized to have correct start and end points (cf. Appendix D of VDM paper)
      normalizing_consant = (minus_log_SNR_1 - minus_log_SNR_0)/(gamma(1)-gamma(0))
      log_SNR = lambda t: - (minus_log_SNR_0 +  normalizing_consant * (gamma(t) - gamma(0)))
      self.d_log_SNR = lambda t: -normalizing_consant * d_gamma(t)
      self.log_SNR = log_SNR

    else:
        self.log_SNR = gamma
        self.d_log_SNR = dgamma
  
  @property
  def T(self):
    return 1
  
  def perturbation_coefficients(self, t):
    SNR = lambda t: torch.exp(self.log_SNR(t))
    alpha = torch.sqrt(SNR(t) / (1 + SNR(t)))
    a_t = alpha
    sigma_t = torch.sqrt(1 / (1 + SNR(t)))
    return a_t, sigma_t 
  
  def sde(self, x, t, return_f=False):
    SNR = lambda t: torch.exp(self.log_SNR(t))
    d_log_SNR = self.d_log_SNR
    std = torch.sqrt(1 / (1 + SNR(t)))
    diffusion_squared = - std**2 * d_log_SNR(t)
    diffusion = torch.sqrt(diffusion_squared)
    f = 0.5 * std[(...,)+(None,)*len(x.shape[1:])]**2 * d_log_SNR(t)[(...,)+(None,)*len(x.shape[1:])]

    if return_f:
      return f, diffusion
    else:
      drift = f * x
      return drift, diffusion
    

  def marginal_prob(self, x, t): 
    SNR = lambda t: torch.exp(self.log_SNR(t))
    alpha = torch.sqrt(SNR(t) / (1 + SNR(t)))[(...,)+(None,)*len(x.shape[1:])]
    mean = alpha * x
    std = torch.sqrt(1 / (1 + SNR(t)))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    dims_to_reduce=tuple(range(len(z.shape))[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=dims_to_reduce) / 2.
    return logps

class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def edm_coefficients(self, t):
     beta_d = self.beta_1 - self.beta_0
     edm_s_t = torch.exp(-0.25 * beta_d * t ** 2 - 0.5 * t * self.beta_0)
     edm_sigma_t = torch.sqrt(torch.exp(0.5 * beta_d * t ** 2 + self.beta_0 * t) - 1)
     return edm_s_t, edm_sigma_t

  def perturbation_coefficients(self, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    a_t = torch.exp(log_mean_coeff)
    sigma_t = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return a_t, sigma_t 

  def snr(self, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t**2/std**2

  def sde(self, x, t, return_f=False):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[(...,)+(None,)*len(x.shape[1:])] * x
    diffusion = torch.sqrt(beta_t)
    if return_f:
      return -0.5 * beta_t[(...,)+(None,)*len(x.shape[1:])], diffusion
    else:
      return drift, diffusion

  def marginal_prob(self, x, t): #perturbation kernel
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[(...,)+(None,)*len(x.shape[1:])]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[(...,)+(None,)*len(x.shape[1:])] * x - x
    G = sqrt_beta
    return f, G

class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[(...,)+(None,)*len(x.shape[1:])] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[(...,)+(None,)*len(x.shape[1:])] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])

    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
