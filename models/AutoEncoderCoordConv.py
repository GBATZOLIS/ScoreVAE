# models/autoencoder_coordconv.py
from __future__ import annotations
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn


# ─────────────────────────── utils ─────────────────────────── #

class SimpleGroupNorm2d(nn.Module):
    """
    GroupNorm with per-channel affine, implemented so it's torch.compile-friendly.
    Construct it via _gn_for(C, preferred_groups) to guarantee divisibility.
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
    Robust GN group chooser:
    - try 'preferred' but enforce divisibility,
    - fall back to gcd(preferred, C),
    - ensure 1 <= groups <= C.
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
    Build a (2, H, W) grid with x,y in [-1, 1]. Registered as buffers so it
    travels with the module across devices; forward only casts dtype.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0).contiguous()


def _bilinear_kernel_2d(k: int) -> torch.Tensor:
    """
    Standard 2D bilinear upsampling kernel of size k×k (float32).
    """
    factor = (k + 1) // 2
    if k % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = torch.arange(k, dtype=torch.float32)
    filt_1d = (1 - torch.abs(og - center) / factor).clamp_min(0)
    kernel = filt_1d.unsqueeze(0) * filt_1d.unsqueeze(1)
    return kernel


def init_deconv_like_bilinear(convT: nn.ConvTranspose2d):
    """
    Initialize a ConvTranspose2d (stride=2, k=4) to approximate bilinear upsampling.
    This helps reduce checkerboard artifacts if you keep the 'deconv' path.
    """
    if not isinstance(convT, nn.ConvTranspose2d):
        return
    k = convT.kernel_size
    s = convT.stride
    if isinstance(k, tuple): k = k[0]
    if isinstance(s, tuple): s = s[0]
    if k != 4 or s != 2:
        return  # only handle the common case used below
    with torch.no_grad():
        ker = _bilinear_kernel_2d(k)  # (4,4)
        w = torch.zeros_like(convT.weight.data)
        oc, ic, _, _ = w.shape
        for i in range(oc):
            j = i % ic
            w[i, j, :, :] = ker
        convT.weight.copy_(w)
        if convT.bias is not None:
            convT.bias.zero_()


# ─────────────────────── encoder (CoordConv) ─────────────────────── #

class ConvEncoderClassicCC(nn.Module):
    """
    Convolutional encoder with optional CoordConv on input:
      - if use_coordconv_encoder=True, concatenates (x,y) grid to the input.
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
            self.register_buffer("enc_grid", _make_grid(img_size, img_size), persistent=False)

    def _coord_grid(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if (H, W) != (self.img_size, self.img_size):
            # rare dynamic shape: rebuild on the fly and store (still a buffer)
            self.register_buffer("enc_grid", _make_grid(H, W), persistent=False)
        return self.enc_grid.to(dtype=x.dtype)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cc:
            N, _, H, W = x.shape
            grid = self._coord_grid(x, H, W)  # (2,H,W), already on device
            x = torch.cat([x, grid.unsqueeze(0).expand(N, -1, -1, -1)], dim=1)
        y = self.net(x)  # <- (fixed indentation) always runs
        return y.reshape(y.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.z_proj(self.feature(x))


# ────────────────────── decoder (smooth upsample) ───────────────────── #

class ConvDecoderClassicCC(nn.Module):
    """
    Decoder with two upsample modes:
      - 'resize_conv' (default): bilinear upsample ×2 → 3×3 conv (smoothest)
      - 'deconv'               : ConvTranspose2d (kept for ablations; bilinear init available)

    CoordConv can be injected at the bottleneck stage and/or at all stages.
    Output head is linear by default (avoid sigmoid saturation for geometry regs).
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

        chs = [base, base * 2, base * 4][:levels]
        s = max(1, img_size // (2 ** levels))
        self.spatial = s
        self.stem_ch = chs[-1]

        self.fc = nn.Linear(z_dim, self.stem_ch * s * s)

        # stage grids as buffers: size after the *first* upsample in each stage
        for i in range(levels):
            Hs = s * (2 ** (i + 1))
            self.register_buffer(f"grid_stage_{i}", _make_grid(Hs, Hs), persistent=False)

        self.up_blocks = nn.ModuleList()
        self.conv2s = nn.ModuleList()
        self.norm2s = nn.ModuleList()
        self.act2s = nn.ModuleList()

        prev_ch = self.stem_ch
        outs = list(reversed(chs[:-1])) + [chs[0]]

        for stage_idx, next_ch in enumerate(outs):
            # First half: upsample
            if self.upsample_mode == "resize_conv":
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(prev_ch, next_ch, kernel_size=3, padding=1, stride=1),
                    _gn_for(next_ch, groups_gn),
                    nn.SiLU(),
                )
            else:  # 'deconv'
                up = nn.Sequential(
                    nn.ConvTranspose2d(prev_ch, next_ch, kernel_size=4, stride=2, padding=1),
                    _gn_for(next_ch, groups_gn),
                    nn.SiLU(),
                )
            self.up_blocks.append(up)

            # Second half: optional CoordConv then 3×3 conv
            inject = (self.use_cc_all or (stage_idx == 0 and self.use_cc_bot))
            conv_in = next_ch + (2 if inject else 0)
            self.conv2s.append(nn.Conv2d(conv_in, next_ch, kernel_size=3, padding=1))
            self.norm2s.append(_gn_for(next_ch, groups_gn))
            self.act2s.append(nn.SiLU())

            prev_ch = next_ch

        self.out = nn.Conv2d(chs[0], out_ch, kernel_size=3, padding=1)

        # Optional: make deconvs approximate bilinear upsampling at init
        if self.upsample_mode == "deconv" and self.deconv_bilinear_init:
            for m in self.up_blocks.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    init_deconv_like_bilinear(m)

    def _stage_grid(self, i: int, dtype: torch.dtype, H: int, W: int) -> torch.Tensor:
        g: torch.Tensor = getattr(self, f"grid_stage_{i}")
        if g.shape[-2:] != (H, W):
            # very rare: shape drift — rebuild and replace buffer
            self.register_buffer(f"grid_stage_{i}", _make_grid(H, W), persistent=False)
            g = getattr(self, f"grid_stage_{i}")
        return g.to(dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.fc(z).reshape(B, self.stem_ch, self.spatial, self.spatial)

        for i in range(self.levels):
            # upsample block
            h = self.up_blocks[i](h)

            # optional CoordConv before the 3×3 conv
            inject = (self.use_cc_all or (i == 0 and self.use_cc_bot))
            if inject:
                Hs, Ws = h.shape[-2:]
                grid = self._stage_grid(i, h.dtype, Hs, Ws)  # (2,Hs,Ws)
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

        self.encoder = ConvEncoderClassicCC(
            in_ch=in_ch, z_dim=z_dim, base=base, levels=levels, img_size=img_size,
            groups_gn=groups_gn, use_coordconv_encoder=use_cc_enc,
        )
        self.decoder = ConvDecoderClassicCC(
            out_ch=out_ch, z_dim=z_dim, base=base, levels=levels, img_size=img_size,
            groups_gn=groups_gn, use_cc_bot=use_cc_bot, use_cc_all=use_cc_all,
            upsample_mode=upsample_mode, output_activation=out_act,
            deconv_bilinear_init=deconv_bilin_init,
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
