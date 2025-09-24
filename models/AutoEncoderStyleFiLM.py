# models/ae_stylefilm_decoder_only.py
from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── utils ─────────────────────────── #

class SimpleGroupNorm2d(nn.Module):
    """GroupNorm with optional affine; we usually keep affine=False and modulate via FiLM."""
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        assert num_channels % max(1, num_groups) == 0, "channels must be divisible by num_groups"
        self.num_groups, self.num_channels, self.eps, self.affine = num_groups, num_channels, eps, affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        G = self.num_groups
        xg   = x.reshape(N, G, C // G, H, W)
        mean = xg.mean(dim=(2, 3, 4), keepdim=True)
        var  = xg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        y    = ((xg - mean) / torch.sqrt(var + self.eps)).reshape(N, C, H, W)
        if self.affine:
            y = y * self.weight + self.bias
        return y


def _choose_gn_groups(C: int, preferred: int) -> int:
    g = min(preferred, C)
    g = math.gcd(g, C)
    return max(1, g)


def _gn_for(C: int, preferred_groups: int, affine: bool = False) -> SimpleGroupNorm2d:
    return SimpleGroupNorm2d(_choose_gn_groups(C, preferred_groups), C, affine=affine)


def init_deconv_like_bilinear(convT: nn.ConvTranspose2d):
    if not isinstance(convT, nn.ConvTranspose2d) or convT.kernel_size[0] != 4 or convT.stride[0] != 2:
        return
    with torch.no_grad():
        k = convT.kernel_size[0]
        factor = (k + 1) // 2
        center = factor - 1 if k % 2 == 1 else factor - 0.5
        og = torch.arange(k, dtype=torch.float32)
        filt_1d = (1 - torch.abs(og - center) / factor).clamp_min(0)
        ker = filt_1d.unsqueeze(0) * filt_1d.unsqueeze(1)
        w = torch.zeros_like(convT.weight.data)
        oc, ic, _, _ = w.shape
        for i in range(oc):
            w[i, i % ic, :, :] = ker
        convT.weight.copy_(w)
        if convT.bias is not None:
            convT.bias.zero_()


# ───────────────────── Coordinate Grid Generation ───────────────────── #

class CoordinateGridGenerator:
    """
    Encapsulates logic for creating coordinate grids, with optional Fourier features.
    """
    def __init__(self, cfg):
        m = cfg if hasattr(cfg, "in_channels") else cfg.model
        self.use_fourier = bool(getattr(m, "use_fourier_features", False))
        self.num_freqs = int(getattr(m, "fourier_num_freqs", 4))
        self.max_freq_log2 = int(getattr(m, "fourier_max_freq_log2", 5))
        self.include_self = bool(getattr(m, "fourier_include_self", True))
        self._cache = {}

    @property
    def grid_channels(self) -> int:
        if not self.use_fourier or self.num_freqs <= 0:
            return 2
        base_ch = 2 if self.include_self else 0
        fourier_ch = 2 * 2 * self.num_freqs  # (sin,cos) × (x,y) × num_freqs
        return base_ch + fourier_ch

    def _make_grid_simple(self, H: int, W: int, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=0)

    def _make_grid_fourier(self, H: int, W: int, dtype: torch.dtype) -> torch.Tensor:
        simple_grid = self._make_grid_simple(H, W, dtype)
        freq_bands = 2.0 ** torch.linspace(0.0, float(self.max_freq_log2), steps=self.num_freqs, dtype=dtype)
        grid_for_encoding = simple_grid.unsqueeze(0)
        freqs_for_encoding = freq_bands.view(self.num_freqs, 1, 1, 1)
        encoded_grid = grid_for_encoding * freqs_for_encoding
        sincos_features = torch.cat([torch.sin(encoded_grid), torch.cos(encoded_grid)], dim=0)
        sincos_features = sincos_features.reshape(2 * 2 * self.num_freqs, H, W)
        if self.include_self:
            return torch.cat([simple_grid, sincos_features], dim=0)
        else:
            return sincos_features

    def generate(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (H, W, str(device), str(dtype),
               self.use_fourier, self.num_freqs, self.max_freq_log2, self.include_self)
        if key in self._cache:
            return self._cache[key]
        grid = (self._make_grid_fourier if self.use_fourier else self._make_grid_simple)(H, W, dtype).to(device)
        self._cache[key] = grid
        return grid


# ───────────────────── BlurPool (anti-aliased downsample) ───────────────────── #

class BlurPool2d(nn.Module):
    """Fixed low-pass (binomial) prefilter + stride-2 downsample."""
    def __init__(self, channels: int, filt_size: int = 5, stride: int = 2, pad_mode: str = "reflect"):
        super().__init__()
        assert filt_size in (3, 5, 7) and stride in (1, 2)
        self.channels, self.stride, self.pad_mode = channels, stride, pad_mode

        k1 = {3: [1., 2., 1.], 5: [1., 4., 6., 4., 1.], 7: [1., 6., 15., 20., 15., 6., 1.]}[filt_size]
        k1 = torch.tensor(k1, dtype=torch.float32)
        k1 = k1 / k1.sum()
        k2 = (k1[:, None] * k1[None, :])[None, None, :, :]

        self.register_buffer("kernel_2d", k2, persistent=False)
        self._pad = (filt_size // 2,) * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self._pad, mode=self.pad_mode)
        # IMPORTANT: use keyword args (PyTorch 2.x stricter)
        k = self.kernel_2d.to(dtype=x.dtype, device=x.device).expand(self.channels, 1, -1, -1)
        return F.conv2d(x, k, bias=None, stride=self.stride, padding=0, groups=self.channels)


# ─────────────────────── encoder (ORIGINAL, unchanged logic) ─────────────────────── #

class ConvEncoderClassicCC(nn.Module):
    def __init__(self, in_ch: int, z_dim: int, base: int, levels: int,
                 groups_gn: int, use_coordconv_encoder: bool,
                 grid_generator: CoordinateGridGenerator,
                 downsample_mode: str = "blur", blur_filt_size: int = 5, img_size: int = 32):
        super().__init__()
        assert downsample_mode in ("avg", "blur")
        self.levels, self.use_cc = levels, use_coordconv_encoder
        self.grid_generator = grid_generator

        coord_ch = self.grid_generator.grid_channels if self.use_cc else 0
        in_eff = in_ch + coord_ch
        chs = [in_eff] + [base * (2**i) for i in range(levels)]

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(levels):
            in_c, out_c = chs[i], chs[i+1]
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), _gn_for(out_c, groups_gn), nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), _gn_for(out_c, groups_gn), nn.SiLU(),
            ))
            if downsample_mode == "avg":
                self.downs.append(nn.AvgPool2d(2))
            else:
                self.downs.append(BlurPool2d(out_c, filt_size=blur_filt_size))

        # final flat dim
        s = img_size // (2 ** levels)
        flat_dim = chs[levels] * s * s
        self.z_proj = nn.Linear(flat_dim, z_dim)

        self.apply(self._init_conv_silu_orthogonal)
        nn.init.orthogonal_(self.z_proj.weight, gain=1.0)
        nn.init.zeros_(self.z_proj.bias)

    @staticmethod
    def _init_conv_silu_orthogonal(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            try:
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
            except Exception:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.use_cc:
            N, _, H, W = x.shape
            grid = self.grid_generator.generate(H, W, device=x.device, dtype=x.dtype)
            h = torch.cat([h, grid.unsqueeze(0).expand(N, -1, -1, -1)], dim=1)

        for stage, down in zip(self.blocks, self.downs):
            h = stage(h)
            h = down(h)

        h_flat = h.reshape(h.size(0), -1)
        return self.z_proj(h_flat)


# ───────────────────── StyleGAN-inspired decoder (GN+FiLM) ───────────────────── #

class StyleganDecoder(nn.Module):
    """
    StyleGAN-inspired decoder tuned for isometry + curvature:
      - Learned constant (no giant FC stem).
      - Mapping MLP z→w (small).
      - GN + FiLM modulation (data-independent norm → cleaner Jacobians/Hessians).
      - Optional Coord/Fourier injection at bottleneck or at all levels.
    """
    def __init__(
        self,
        out_ch: int,
        z_dim: int,
        base: int,
        levels: int,
        img_size: int,
        groups_gn: int,
        grid_generator: Optional[CoordinateGridGenerator],
        use_cc_bot: bool,
        use_cc_all: bool,
        upsample_mode: str = "resize_conv",
        output_activation: str = "linear",
        deconv_bilinear_init: bool = True,
        use_spectral_norm: bool = False,
        output_head_scale: float = 0.1,
        w_dim: int = 128,
        film_bias_init: float = 0.0,
    ):
        super().__init__()
        self.levels = int(levels)
        assert self.levels >= 1
        self.upsample_mode = upsample_mode
        self.output_activation = output_activation
        self.grid_generator = grid_generator
        self.use_cc_bot = bool(use_cc_bot)
        self.use_cc_all = bool(use_cc_all)

        chs = [base * (2 ** i) for i in range(levels)]
        self.stem_ch = chs[-1]
        self.spatial = img_size // (2 ** levels)

        # learned constant seed
        self.const = nn.Parameter(torch.randn(1, self.stem_ch, self.spatial, self.spatial))

        # mapping MLP
        self.map = nn.Sequential(
            nn.Linear(z_dim, w_dim), nn.SiLU(),
            nn.Linear(w_dim, w_dim), nn.SiLU(),
            nn.Linear(w_dim, w_dim),
        )

        # per-level modules
        self.up_blocks = nn.ModuleList()
        self.conv1s, self.conv2s = nn.ModuleList(), nn.ModuleList()
        self.norms1, self.norms2 = nn.ModuleList(), nn.ModuleList()
        self.film1,  self.film2  = nn.ModuleList(), nn.ModuleList()
        self.act = nn.SiLU()

        ch_pairs = list(zip(reversed(chs), reversed([chs[0]] + chs[:-1])))

        for i, (in_c, out_c) in enumerate(ch_pairs):
            # upsample
            if upsample_mode == "resize_conv":
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_c, out_c, 3, padding=1),
                )
            else:
                up = nn.Sequential(
                    nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
                )
                if deconv_bilinear_init:
                    init_deconv_like_bilinear(up[0])
            self.up_blocks.append(up)

            coord_ch = 0
            inject = (self.use_cc_all or (i == 0 and self.use_cc_bot))
            if (self.grid_generator is not None) and inject:
                coord_ch = self.grid_generator.grid_channels

            c1_in = out_c + coord_ch
            conv1 = nn.Conv2d(c1_in, out_c, 3, padding=1)
            conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
            self.conv1s.append(conv1); self.conv2s.append(conv2)

            self.norms1.append(_gn_for(out_c, groups_gn, affine=False))
            self.norms2.append(_gn_for(out_c, groups_gn, affine=False))

            f1 = nn.Linear(w_dim, 2 * out_c)
            f2 = nn.Linear(w_dim, 2 * out_c)
            nn.init.zeros_(f1.bias); f1.bias.data[out_c:] = film_bias_init
            nn.init.zeros_(f2.bias); f2.bias.data[out_c:] = film_bias_init
            self.film1.append(f1); self.film2.append(f2)

        self.out = nn.Conv2d(chs[0], out_ch, 3, padding=1)
        with torch.no_grad():
            self.out.weight.mul_(output_head_scale)
            if self.out.bias is not None: self.out.bias.zero_()

        self.apply(self._init_conv_silu_orthogonal)

    @staticmethod
    def _init_conv_silu_orthogonal(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.weight.requires_grad:
            try:
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
            except Exception:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _film(x: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
        gamma, beta = gb.chunk(2, dim=1)
        gamma = (1.0 + gamma).unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        w = self.map(z)
        h = self.const.expand(B, -1, -1, -1)

        for i in range(self.levels):
            h = self.up_blocks[i](h)

            inject = (self.use_cc_all or (i == 0 and self.use_cc_bot))
            if (self.grid_generator is not None) and inject:
                H, W = h.shape[-2:]
                grid = self.grid_generator.generate(H, W, device=h.device, dtype=h.dtype)
                h = torch.cat([h, grid.unsqueeze(0).expand(B, -1, -1, -1)], dim=1)

            h = self.conv1s[i](h)
            h = self._film(self.norms1[i](h), self.film1[i](w))
            h = self.act(h)

            h = self.conv2s[i](h)
            h = self._film(self.norms2[i](h), self.film2[i](w))
            h = self.act(h)

        x = self.out(h)
        if self.output_activation == "tanh":    return torch.tanh(x)
        if self.output_activation == "sigmoid": return torch.sigmoid(x)
        return x


# ─────────────────────────── AutoEncoder: OLD ENC + NEW DEC ─────────────────────────── #

class AutoEncoderStyleFiLM(nn.Module):
    """
    Autoencoder with:
      - Encoder: ORIGINAL ConvEncoderClassicCC (exact structure kept)
      - Decoder: StyleganDecoder (GN+FiLM, learned constant)
      - Latent normalization buffers (same API)
    """
    def __init__(self, cfg):
        super().__init__()
        m = cfg if hasattr(cfg, "in_channels") else cfg.model

        # Shared coord/Fourier grid generator
        self.grid_generator = CoordinateGridGenerator(cfg)

        # Original encoder
        self.encoder = ConvEncoderClassicCC(
            in_ch=int(m.in_channels), z_dim=int(m.latent_dim), base=int(m.base_channels),
            levels=int(m.num_down_levels), groups_gn=int(m.groups_gn),
            use_coordconv_encoder=bool(getattr(m, "use_coordconv_encoder", False)),
            grid_generator=self.grid_generator,
            downsample_mode=str(getattr(m, "encoder_downsample_mode", "blur")),
            blur_filt_size=int(getattr(m, "encoder_blur_filt_size", 5)),
            img_size=int(m.image_size)
        )

        # New decoder
        self.decoder = StyleganDecoder(
            out_ch=int(m.out_channels), z_dim=int(m.latent_dim),
            base=int(m.base_channels), levels=int(m.num_down_levels), img_size=int(m.image_size),
            groups_gn=int(m.groups_gn),
            grid_generator=self.grid_generator,
            use_cc_bot=bool(getattr(m, "use_coordconv_decoder_bottleneck", False)),
            use_cc_all=bool(getattr(m, "use_coordconv_decoder_all_levels", False)),
            upsample_mode=str(getattr(m, "decoder_upsample_mode", "resize_conv")),
            output_activation=str(getattr(m, "output_activation", "linear")),
            deconv_bilinear_init=bool(getattr(m, "deconv_bilinear_init", True)),
            use_spectral_norm=bool(getattr(m, "use_spectral_norm", False)),
            output_head_scale=float(getattr(m, "output_head_scale", 0.1)),
            w_dim=int(getattr(m, "style_w_dim", 128)),
            film_bias_init=float(getattr(m, "style_film_bias_init", 0.0)),
        )

        # latent normalization (compat)
        self.z_dim = int(m.latent_dim)
        self.register_buffer("latent_norm_mean", torch.zeros(self.z_dim))
        self.register_buffer("latent_norm_std",  torch.ones(self.z_dim))

    # latent norm API
    def set_latent_normalization(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        mean = mean.detach().to(self.latent_norm_mean.device, dtype=self.latent_norm_mean.dtype)
        std  = std.detach().to(self.latent_norm_std.device,  dtype=self.latent_norm_std.dtype).clamp_min(eps)
        if mean.numel() == 1: mean = mean.expand_as(self.latent_norm_mean)
        if std.numel()  == 1: std  = std.expand_as(self.latent_norm_std)
        self.latent_norm_mean.copy_(mean)
        self.latent_norm_std.copy_(std)

    def normalize_latent(self, z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (z - self.latent_norm_mean) / self.latent_norm_std.clamp_min(eps)

    def denormalize_latent(self, z_hat: torch.Tensor) -> torch.Tensor:
        return z_hat * self.latent_norm_std + self.latent_norm_mean

    # standard AE API
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[AE] Total params: {total:,} (trainable: {trainable:,})")
        print(f"[AE] Coordinate grid channels: {self.grid_generator.grid_channels}")
