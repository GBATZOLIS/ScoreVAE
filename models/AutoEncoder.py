# models/autoencoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SimpleGroupNorm2d(nn.Module):
    """
    Drop-in replacement for nn.GroupNorm (no running stats).
    Normalizes over groups across (C_group, H, W) with affine per-channel.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        assert num_channels % num_groups == 0, "channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        G = self.num_groups
        xg = x.reshape(N, G, C // G, H, W)
        mean = xg.mean(dim=(2, 3, 4), keepdim=True)
        var  = xg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        y = xg.reshape(N, C, H, W)
        if self.affine:
            y = y * self.weight + self.bias
        return y


class ConvEncoder(nn.Module):
    """
    Convolutional encoder with:
      • feature(x): conv → flatten
      • forward(x): deterministic AE projection (z_proj(feature))
      • posterior_stats / heads for VAE (used by AutoEncoder when enabled)
    """
    def __init__(self, in_ch: int, z_dim: int, base: int = 32, levels: int = 3, img_size: int = 32):
        super().__init__()
        chs = [in_ch, base, base * 2, base * 4][:levels + 1]
        self.levels = levels
        self.img_size = img_size
        enc = []
        for i in range(levels):
            enc += [
                nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1),
                SimpleGroupNorm2d(8, chs[i + 1]),
                nn.SiLU(),
                nn.Conv2d(chs[i + 1], chs[i + 1], 3, 1, 1),
                SimpleGroupNorm2d(8, chs[i + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*enc)
        self.z_dim = z_dim
        self._chs_out = chs[levels]

        # spatial size after 'levels' downsamples
        s = max(1, img_size // (2 ** levels))
        self._spatial = s
        flat_dim = self._chs_out * s * s

        # AE deterministic projection
        self.z_proj = nn.Linear(flat_dim, self.z_dim)

        # VAE heads (always defined; cheap if unused)
        self.mu_head     = nn.Linear(flat_dim, self.z_dim)
        self.logvar_head = nn.Linear(flat_dim, self.z_dim)
        # start with modest posterior variance
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.reshape(y.size(0), -1)  # safe for channels_last & compile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        return self.z_proj(h)

    def posterior_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    Decoder that mirrors the encoder’s downsampling with `levels` upsampling blocks.
    """
    def __init__(self, out_ch: int, z_dim: int, base: int = 32, levels: int = 3, img_size: int = 32):
        super().__init__()
        chs = [base, base * 2, base * 4][:levels]
        self.levels = levels
        self.img_size = img_size
        s = max(1, img_size // (2 ** levels))
        self.spatial = s
        self.stem_ch = chs[-1]
        self.fc = nn.Linear(z_dim, self.stem_ch * s * s)

        outs = list(reversed(chs[:-1])) + [chs[0]]
        blocks = []
        prev_ch = self.stem_ch
        for next_ch in outs:
            blocks += [
                nn.ConvTranspose2d(prev_ch, next_ch, kernel_size=4, stride=2, padding=1),
                SimpleGroupNorm2d(8, next_ch),
                nn.SiLU(),
                nn.Conv2d(next_ch, next_ch, kernel_size=3, padding=1),
                SimpleGroupNorm2d(8, next_ch),
                nn.SiLU(),
            ]
            prev_ch = next_ch
        self.dec = nn.Sequential(*blocks)
        self.out = nn.Conv2d(chs[0], out_ch, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(z.size(0), self.stem_ch, self.spatial, self.spatial)
        h = self.dec(h)
        x = self.out(h)
        return torch.sigmoid(x)  # dataset is in [0, 1]


class AutoEncoder(nn.Module):
    """
    Minimal AE/VAE with encode/decode methods and latent normalization buffers.

    Backward-compat:
      • When VAE disabled: identical to a standard AE.
      • When VAE enabled: encode(x) returns µ(x) (deterministic) for stable JVP/VJP.
    """
    def __init__(self, cfg):
        super().__init__()
        self.in_ch = cfg.in_channels
        self.out_ch = cfg.out_channels
        self.z_dim = cfg.latent_dim
        base = getattr(cfg, "base_channels", 32)
        levels = getattr(cfg, "num_down_levels", 3)
        img_size = getattr(cfg, "image_size", 32)

        # VAE flags from cfg.model.vae.*
        vae_cfg = getattr(cfg, "vae", None)
        self.as_vae = bool(getattr(vae_cfg, "enabled", False)) if vae_cfg is not None else False
        self.learnable_prior_diag = bool(getattr(vae_cfg, "learnable_prior_diag", False)) if vae_cfg is not None else False
        prior_logvar_init = float(getattr(vae_cfg, "prior_logvar_init", 0.0)) if vae_cfg is not None else 0.0

        self.encoder = ConvEncoder(self.in_ch, self.z_dim, base=base, levels=levels, img_size=img_size)
        self.decoder = ConvDecoder(self.out_ch, self.z_dim, base=base, levels=levels, img_size=img_size)

        # Latent normalization buffers (non-learnable; updated via callback)
        self.register_buffer("latent_norm_mean", torch.zeros(self.z_dim))  # μ
        self.register_buffer("latent_norm_std",  torch.ones(self.z_dim))   # σ (clamped to ≥eps at use)

        # Optional learnable diagonal PRIOR: N(0, diag(exp(prior_logvar)))
        if self.as_vae and self.learnable_prior_diag:
            self.prior_logvar = nn.Parameter(torch.full((self.z_dim,), float(prior_logvar_init)))
        else:
            self.register_parameter("prior_logvar", None)

        # slots for last forward (used by loss)
        self._last_mu: Optional[torch.Tensor] = None
        self._last_logvar: Optional[torch.Tensor] = None
        self._last_z: Optional[torch.Tensor] = None

    # ---- latent normalization API (non-learnable) ----
    def set_latent_normalization(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        """Update μ, σ buffers from tensors shaped (z_dim,) or broadcastable."""
        mean = mean.detach().to(self.latent_norm_mean.device, dtype=self.latent_norm_mean.dtype)
        std  = std.detach().to(self.latent_norm_std.device,  dtype=self.latent_norm_std.dtype)
        std  = std.clamp_min(eps)
        if mean.numel() == 1: mean = mean.expand_as(self.latent_norm_mean)
        if std.numel()  == 1: std  = std.expand_as(self.latent_norm_std)
        self.latent_norm_mean.copy_(mean)
        self.latent_norm_std.copy_(std)

    def normalize_latent(self, z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """ẑ = (z - μ) / σ per-dimension; safe with broadcasting."""
        mu = self.latent_norm_mean
        sd = self.latent_norm_std.clamp_min(eps)
        return (z - mu) / sd

    def denormalize_latent(self, z_hat: torch.Tensor) -> torch.Tensor:
        """z = ẑ * σ + μ"""
        return z_hat * self.latent_norm_std + self.latent_norm_mean

    # ---------- API used by losses / regularisers ----------
    # Deterministic encoder mapping (µ when VAE is on)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.as_vae:
            mu, _ = self.encoder.posterior_stats(x)
            return mu
        return self.encoder(x)

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # posterior µ, logσ^2
        return self.encoder.posterior_stats(x)

    def sample_z(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample_prior(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        if self.prior_logvar is None:
            std = torch.ones(self.z_dim, device=device)
        else:
            std = (0.5 * self.prior_logvar).exp()
        return torch.randn(n, self.z_dim, device=device) * std

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor):
        if not self.as_vae:
            z = self.encode(x)
            x_hat = self.decode(z)
            return x_hat, z

        # VAE path: reparameterize
        z, mu, logvar = self.sample_z(x)
        x_hat = self.decode(z)
        # expose to the loss (avoids recomputing)
        self._last_mu, self._last_logvar, self._last_z = mu, logvar, z
        return x_hat, z

    # parity with your models
    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[AE] Total params: {total:,}  (trainable: {trainable:,})")
