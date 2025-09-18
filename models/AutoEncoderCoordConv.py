# models/autoencoder_coordconv.py
from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ─────────────────────────── utils ─────────────────────────── #

class SimpleGroupNorm2d(nn.Module):
    """
    GroupNorm with per-channel affine. Construct via _gn_for to guarantee divisibility.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        assert num_channels % num_groups == 0, "channels must be divisible by num_groups"
        self.num_groups, self.num_channels, self.eps, self.affine = num_groups, num_channels, eps, affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        G = self.num_groups
        xg = x.reshape(N, G, C // G, H, W)
        mean = xg.mean(dim=(2, 3, 4), keepdim=True)
        var = xg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        y = xg.reshape(N, C, H, W)
        if self.affine:
            y = y * self.weight + self.bias
        return y


def _choose_gn_groups(C: int, preferred: int) -> int:
    """
    Robust GN group chooser: enforce divisibility, fallback to gcd, clamp to [1, C].
    """
    g = min(preferred, C)
    g = math.gcd(g, C)
    if g == 0:
        g = 1
    return max(1, min(g, C))


def _gn_for(C: int, preferred_groups: int) -> SimpleGroupNorm2d:
    return SimpleGroupNorm2d(_choose_gn_groups(C, preferred_groups), C)


def _make_grid(H: int, W: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build a (2, H, W) grid with x,y in [-1, 1].
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0).contiguous()


def init_deconv_like_bilinear(convT: nn.ConvTranspose2d):
    """
    Initialize a ConvTranspose2d (stride=2, k=4) to approximate bilinear upsampling.
    """
    if not isinstance(convT, nn.ConvTranspose2d):
        return
    k = convT.kernel_size[0] if isinstance(convT.kernel_size, tuple) else convT.kernel_size
    s = convT.stride[0] if isinstance(convT.stride, tuple) else convT.stride
    if k != 4 or s != 2:
        return
    with torch.no_grad():
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


# ─────────────────── Orthogonal Stem Module ─────────────────── #

class OrthogonalLinearPad(nn.Module):
    """
    Map z ∈ R^{in_d} → R^{out_d} by zero-padding then an orthogonal transform Q,
    blended via α with the padded identity. Q is a product of Householder reflections.
    Efficient: rank-1 updates, small fixed loop over K reflections.
    """
    def __init__(self, in_d: int, out_d: int, num_reflections: int = 4, init_alpha: float = 0.0):
        super().__init__()
        assert out_d >= in_d, "out_d must be >= in_d for zero-padding"
        self.in_d = int(in_d)
        self.out_d = int(out_d)
        self.num_reflections = int(max(1, min(num_reflections, out_d)))

        # Reflection parameters: (K, out_d)
        scale = (1.0 / max(1, out_d)**0.5)
        self.v = nn.Parameter(torch.randn(self.num_reflections, out_d) * scale)

        # α = sigmoid(alpha_logit); start near 0 (identity-ish)
        with torch.no_grad():
            init_alpha = float(init_alpha)
            init_alpha = min(max(init_alpha, 1e-6), 1 - 1e-6)
            init_logit = torch.logit(torch.tensor(init_alpha))
        self.alpha_logit = nn.Parameter(init_logit.clone())

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit).detach()

    def _apply_Q(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply Q = H_K ... H_1 to y, with H(v) = I - 2 v v^T (||v||=1).
        y: (B, out_d)
        """
        v_norm = F.normalize(self.v, dim=1, eps=1e-8)  # (K, out_d)
        Qy = y
        # small loop over K reflections; each step is a rank-1 update
        for k in range(self.num_reflections):
            vk = v_norm[k]                 # (out_d,)
            proj = torch.matmul(Qy, vk)    # (B,)
            Qy = Qy - 2.0 * proj.unsqueeze(-1) * vk.unsqueeze(0)
        return Qy

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 1) zero-pad to out_d (no-op if in_d==out_d)
        y = z if self.out_d == self.in_d else F.pad(z, (0, self.out_d - self.in_d), mode="constant", value=0.0)
        # 2) orthogonal stack
        Qy = self._apply_Q(y)
        # 3) blend with α∈(0,1)
        alpha = torch.sigmoid(self.alpha_logit)
        return y + alpha * (Qy - y)  # (1-α)·y + α·Qy


# ─────────────────────── encoder (CoordConv) ─────────────────────── #

class ConvEncoderClassicCC(nn.Module):
    """
    Convolutional encoder with optional CoordConv on input.
    """
    def __init__(
        self,
        in_ch: int,
        z_dim: int,
        base: int,
        levels: int,
        img_size: int,
        groups_gn: int,
        use_coordconv_encoder: bool,
    ):
        super().__init__()
        self.levels = levels
        self.img_size = img_size
        self.use_cc = use_coordconv_encoder

        in_eff = in_ch + (2 if self.use_cc else 0)
        chs = [in_eff, base, base * 2, base * 4][:levels + 1]

        blocks = []
        for i in range(levels):
            out_c = chs[i + 1]
            blocks += [
                nn.Conv2d(chs[i], out_c, 3, stride=2, padding=1),
                _gn_for(out_c, groups_gn),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
                _gn_for(out_c, groups_gn),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*blocks)

        s = max(1, img_size // (2 ** levels))
        self._spatial = s
        flat_dim = chs[levels] * s * s
        self.z_proj = nn.Linear(flat_dim, z_dim)

        if self.use_cc:
            # register once at nominal size; rebuild on drift with correct device
            self.register_buffer("enc_grid", _make_grid(img_size, img_size), persistent=False)

    def _coord_grid(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        g: torch.Tensor = self.enc_grid
        if (H, W) != g.shape[-2:]:
            # rebuild on the SAME device as x
            g_new = _make_grid(H, W, dtype=x.dtype).to(device=x.device)
            self.register_buffer("enc_grid", g_new, persistent=False)
            g = self.enc_grid
        # ensure dtype+device match even when not rebuilt
        return g.to(device=x.device, dtype=x.dtype)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cc:
            N, _, H, W = x.shape
            grid = self._coord_grid(x, H, W)  # (2,H,W)
            x = torch.cat([x, grid.unsqueeze(0).expand(N, -1, -1, -1)], dim=1)
        y = self.net(x)
        return y.reshape(y.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.z_proj(self.feature(x))


# ────────────────────── decoder (smooth upsample) ───────────────────── #

class ConvDecoderClassicCC(nn.Module):
    """
    Decoder with two upsample modes:
      - 'resize_conv' (default): bilinear upsample ×2 → 3×3 conv (smoothest)
      - 'deconv'               : ConvTranspose2d (bilinear init available)

    CoordConv can be injected at the bottleneck stage and/or at all stages.
    Output head is linear by default (avoid sigmoid saturation for geometry regs).
    Optional orthogonal stem and spectral_norm on selected convs.
    """
    def __init__(
        self,
        out_ch: int,
        z_dim: int,
        base: int,
        levels: int,
        img_size: int,
        groups_gn: int,
        use_cc_bot: bool,
        use_cc_all: bool,
        upsample_mode: str = "resize_conv",    # 'resize_conv' | 'deconv'
        output_activation: str = "linear",     # 'linear' | 'tanh' | 'sigmoid'
        deconv_bilinear_init: bool = True,
        use_orthogonal_stem: bool = True,
        stem_reflections: int = 4,
        stem_init_alpha: float = 0.0,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        assert upsample_mode in ("resize_conv", "deconv")
        assert output_activation in ("linear", "tanh", "sigmoid")
        self.levels = levels
        self.img_size = img_size
        self.use_cc_bot = use_cc_bot
        self.use_cc_all = use_cc_all
        self.upsample_mode = upsample_mode
        self.output_activation = output_activation
        self.deconv_bilinear_init = deconv_bilinear_init
        self.use_spectral_norm = use_spectral_norm

        chs = [base, base * 2, base * 4][:levels]
        s = max(1, img_size // (2 ** levels))
        self.spatial = s
        self.stem_ch = chs[-1]
        stem_dim = self.stem_ch * s * s

        # Stem: orthogonal (Householder) or plain linear
        if use_orthogonal_stem:
            self.stem = OrthogonalLinearPad(z_dim, stem_dim, stem_reflections, stem_init_alpha)
        else:
            self.stem = nn.Linear(z_dim, stem_dim)

        self.up_blocks = nn.ModuleList()
        self.conv2s = nn.ModuleList()
        self.norm2s = nn.ModuleList()
        self.act2s = nn.ModuleList()

        prev_ch = self.stem_ch
        outs = list(reversed(chs[:-1])) + [chs[0]]

        for stage_idx, next_ch in enumerate(outs):
            # Upsample half
            if self.upsample_mode == "resize_conv":
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(prev_ch, next_ch, kernel_size=3, padding=1, stride=1),
                    _gn_for(next_ch, groups_gn),
                    nn.SiLU(),
                )
            else:
                up = nn.Sequential(
                    nn.ConvTranspose2d(prev_ch, next_ch, kernel_size=4, stride=2, padding=1),
                    _gn_for(next_ch, groups_gn),
                    nn.SiLU(),
                )
            self.up_blocks.append(up)

            # Second half: optional CoordConv then 3×3 conv (+ optional SN)
            inject = (self.use_cc_all or (stage_idx == 0 and self.use_cc_bot))
            conv_in = next_ch + (2 if inject else 0)
            conv2 = nn.Conv2d(conv_in, next_ch, kernel_size=3, padding=1)
            if self.use_spectral_norm:
                conv2 = spectral_norm(conv2, n_power_iterations=1)
            self.conv2s.append(conv2)
            self.norm2s.append(_gn_for(next_ch, groups_gn))
            self.act2s.append(nn.SiLU())

            # Stage grid buffer (after the *first* upsample in this stage)
            Hs = s * (2 ** (stage_idx + 1))
            self.register_buffer(f"grid_stage_{stage_idx}", _make_grid(Hs, Hs), persistent=False)

            prev_ch = next_ch

        out_conv = nn.Conv2d(chs[0], out_ch, kernel_size=3, padding=1)
        self.out = spectral_norm(out_conv, n_power_iterations=1) if self.use_spectral_norm else out_conv

        # Optional: initialize deconvs to bilinear
        if self.upsample_mode == "deconv" and self.deconv_bilinear_init:
            for m in self.up_blocks.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    init_deconv_like_bilinear(m)

        # Initialize convs orthogonally, zero biases
        self.apply(self._init_conv_orthogonal)

    @staticmethod
    def _init_conv_orthogonal(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            try:
                nn.init.orthogonal_(m.weight)
            except Exception:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _stage_grid(self, i: int, dtype: torch.dtype, device: torch.device, H: int, W: int) -> torch.Tensor:
        g: torch.Tensor = getattr(self, f"grid_stage_{i}")
        if g.shape[-2:] != (H, W):
            g_new = _make_grid(H, W, dtype=dtype).to(device=device)
            self.register_buffer(f"grid_stage_{i}", g_new, persistent=False)
            g = getattr(self, f"grid_stage_{i}")
        return g.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.stem(z).reshape(B, self.stem_ch, self.spatial, self.spatial)

        for i in range(self.levels):
            # upsample block
            h = self.up_blocks[i](h)

            # optional CoordConv before the 3×3 conv
            inject = (self.use_cc_all or (i == 0 and self.use_cc_bot))
            if inject:
                Hs, Ws = h.shape[-2:]
                grid = self._stage_grid(i, dtype=h.dtype, device=h.device, H=Hs, W=Ws)  # (2, Hs, Ws)
                h = torch.cat([h, grid.unsqueeze(0).expand(B, -1, -1, -1)], dim=1)

            # 3×3 + GN + SiLU
            h = self.conv2s[i](h)
            h = self.act2s[i](self.norm2s[i](h))

        x = self.out(h)

        # output activation policy
        if self.output_activation == "tanh":
            return torch.tanh(x)
        elif self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        else:
            return x  # 'linear' (recommended for geometry regs)


# ─────────────────────────── top-level AE ─────────────────────────── #

class AutoEncoderCoordConv(nn.Module):
    """
    AE with:
      • Encoder: optional CoordConv on input
      • Decoder: CoordConv at bottleneck and/or all stages
      • Upsample: 'resize_conv' (default) or 'deconv'
      • Output: linear (default), tanh, or sigmoid
      • Orthogonal stem and optional spectral norm for geometric stability
    """
    def __init__(self, cfg):
        super().__init__()
        # tolerate cfg or cfg.model
        m = cfg if hasattr(cfg, "in_channels") else cfg.model

        in_ch = int(getattr(m, "in_channels"))
        out_ch = int(getattr(m, "out_channels"))
        z_dim = int(getattr(m, "latent_dim"))
        base = int(getattr(m, "base_channels", 32))
        levels = int(getattr(m, "num_down_levels", 3))
        img_size = int(getattr(m, "image_size", 32))
        groups_gn = int(getattr(m, "groups_gn", 8))

        use_cc_enc = bool(getattr(m, "use_coordconv_encoder", False))
        use_cc_bot = bool(getattr(m, "use_coordconv_decoder_bottleneck", True))
        use_cc_all = bool(getattr(m, "use_coordconv_decoder_all_levels", False))
        upsample_mode = str(getattr(m, "decoder_upsample_mode", "resize_conv"))
        out_act = str(getattr(m, "output_activation", "linear"))
        deconv_bilin_init = bool(getattr(m, "deconv_bilinear_init", True))

        # geometry-friendly stem + SN (disable SN if compiling, unless you know it’s fine)
        compile_flag = bool(getattr(m, "compile", False))
        use_orthogonal_stem = bool(getattr(m, "use_orthogonal_stem", True))
        stem_reflections = int(getattr(m, "stem_reflections", 4))
        stem_init_alpha = float(getattr(m, "stem_init_alpha", 0.0))
        use_spectral_norm = bool(getattr(m, "use_spectral_norm", True))
        if compile_flag:
            # some PT versions have compile slowdowns with spectral_norm reparam
            use_spectral_norm = bool(getattr(m, "use_spectral_norm_when_compiled", False))

        self.encoder = ConvEncoderClassicCC(
            in_ch=in_ch, z_dim=z_dim, base=base, levels=levels, img_size=img_size,
            groups_gn=groups_gn, use_coordconv_encoder=use_cc_enc,
        )
        self.decoder = ConvDecoderClassicCC(
            out_ch=out_ch, z_dim=z_dim, base=base, levels=levels, img_size=img_size,
            groups_gn=groups_gn, use_cc_bot=use_cc_bot, use_cc_all=use_cc_all,
            upsample_mode=upsample_mode, output_activation=out_act,
            deconv_bilinear_init=deconv_bilin_init,
            use_orthogonal_stem=use_orthogonal_stem,
            stem_reflections=stem_reflections, stem_init_alpha=stem_init_alpha,
            use_spectral_norm=use_spectral_norm,
        )

        # latent norm buffers for your callbacks
        self.z_dim = z_dim
        self.register_buffer("latent_norm_mean", torch.zeros(self.z_dim))
        self.register_buffer("latent_norm_std", torch.ones(self.z_dim))

        # VAE compatibility slots (unused by default)
        self.register_parameter("prior_logvar", None)
        self._last_mu = None
        self._last_logvar = None
        self._last_z = None

    # parity with your previous API
    def set_latent_normalization(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        mean = mean.detach().to(self.latent_norm_mean.device, dtype=self.latent_norm_mean.dtype)
        std = std.detach().to(self.latent_norm_std.device, dtype=self.latent_norm_std.dtype).clamp_min(eps)
        if mean.numel() == 1:
            mean = mean.expand_as(self.latent_norm_mean)
        if std.numel() == 1:
            std = std.expand_as(self.latent_norm_std)
        self.latent_norm_mean.copy_(mean)
        self.latent_norm_std.copy_(std)

    def normalize_latent(self, z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (z - self.latent_norm_mean) / self.latent_norm_std.clamp_min(eps)

    def denormalize_latent(self, z_hat: torch.Tensor) -> torch.Tensor:
        return z_hat * self.latent_norm_std + self.latent_norm_mean

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
