# models/autoencoder_coordconv.py
from __future__ import annotations
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# ---------- utils ----------

class SimpleGroupNorm2d(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        assert num_channels % num_groups == 0
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
        xg = x.reshape(N, G, C // G, H, W)
        mean = xg.mean(dim=(2, 3, 4), keepdim=True)
        var  = xg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        y = xg.reshape(N, C, H, W)
        if self.affine:
            y = y * self.weight + self.bias
        return y

@torch.no_grad()
def _make_grid_cpu(H: int, W: int) -> torch.Tensor:
    """
    CPU float32 grid (2,H,W) with x,y in [-1,1], cached by the decoder.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, dtype=torch.float32, device="cpu"),
        torch.linspace(-1.0, 1.0, W, dtype=torch.float32, device="cpu"),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0).contiguous()  # (2,H,W)

# ---------- classic encoder with optional CoordConv at input ----------

class ConvEncoderClassicCC(nn.Module):
    def __init__(self, in_ch: int, z_dim: int, base: int, levels: int, img_size: int,
                 groups_gn: int, use_coordconv_encoder: bool):
        super().__init__()
        in_eff = in_ch + (2 if use_coordconv_encoder else 0)
        chs = [in_eff, base, base * 2, base * 4][:levels + 1]
        self.levels = levels
        self.img_size = img_size
        self.use_cc = use_coordconv_encoder

        enc = []
        for i in range(levels):
            enc += [
                nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1),
                SimpleGroupNorm2d(groups_gn, chs[i + 1]),
                nn.SiLU(),
                nn.Conv2d(chs[i + 1], chs[i + 1], 3, 1, 1),
                SimpleGroupNorm2d(groups_gn, chs[i + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*enc)

        s = max(1, img_size // (2 ** levels))
        self._spatial = s
        flat_dim = chs[levels] * s * s
        self.z_proj = nn.Linear(flat_dim, z_dim)

    def _coord_grid(self, H: int, W: int, device, dtype):
        g = _make_grid_cpu(H, W)
        return g.to(device=device, dtype=dtype, non_blocking=True).detach()

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cc:
            N, C, H, W = x.shape
            grid = self._coord_grid(H, W, x.device, x.dtype)    # (2,H,W)
            x = torch.cat([x, grid.unsqueeze(0).expand(N, -1, -1, -1)], dim=1)
        y = self.net(x)
        return y.reshape(y.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        return self.z_proj(h)

# ---------- classic decoder with CoordConv at bottleneck / all levels ----------

class ConvDecoderClassicCC(nn.Module):
    """
    Mirrors the old ConvDecoder:
      [ConvT2d -> GN -> SiLU -> (CoordConv?) -> Conv2d -> GN -> SiLU] x levels
    CoordConv is inserted BETWEEN the ConvTranspose2d and the 3x3 Conv2d.
    """
    def __init__(self, out_ch: int, z_dim: int, base: int, levels: int, img_size: int,
                 groups_gn: int, use_cc_bot: bool, use_cc_all: bool):
        super().__init__()
        self.levels = levels
        self.img_size = img_size
        self.use_cc_bot = use_cc_bot
        self.use_cc_all = use_cc_all

        # Tiny CPU cache for grids: {(H,W): (2,H,W)}
        self._grid_cache = {}

        chs = [base, base * 2, base * 4][:levels]
        s = max(1, img_size // (2 ** levels))
        self.spatial = s
        self.stem_ch = chs[-1]

        self.fc = nn.Linear(z_dim, self.stem_ch * s * s)

        # Build per-stage modules
        self.deconvs = nn.ModuleList()
        self.norm1s  = nn.ModuleList()
        self.acts1   = nn.ModuleList()
        self.convs   = nn.ModuleList()
        self.norm2s  = nn.ModuleList()
        self.acts2   = nn.ModuleList()

        prev_ch = self.stem_ch
        outs = list(reversed(chs[:-1])) + [chs[0]]
        for stage_idx, next_ch in enumerate(outs):
            self.deconvs.append(nn.ConvTranspose2d(prev_ch, next_ch, kernel_size=4, stride=2, padding=1))
            self.norm1s.append(SimpleGroupNorm2d(groups_gn, next_ch))
            self.acts1.append(nn.SiLU())

            inject = (self.use_cc_all or (stage_idx == 0 and self.use_cc_bot))
            conv_in = next_ch + (2 if inject else 0)
            self.convs.append(nn.Conv2d(conv_in, next_ch, kernel_size=3, padding=1))
            self.norm2s.append(SimpleGroupNorm2d(groups_gn, next_ch))
            self.acts2.append(nn.SiLU())

            prev_ch = next_ch

        self.out = nn.Conv2d(chs[0], out_ch, kernel_size=3, padding=1)

    @torch.no_grad()
    def _grid_cpu_cached(self, H: int, W: int) -> torch.Tensor:
        key = (H, W)
        g = self._grid_cache.get(key, None)
        if g is None:
            g = _make_grid_cpu(H, W)
            self._grid_cache[key] = g
        return g

    def _coord_grid(self, H: int, W: int, device, dtype):
        with torch.no_grad():
            g = self._grid_cpu_cached(H, W)
            return g.to(device=device, dtype=dtype, non_blocking=True).detach()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.fc(z).reshape(B, self.stem_ch, self.spatial, self.spatial)

        for i in range(self.levels):
            # ConvTranspose2d upsample
            h = self.deconvs[i](h)
            h = self.acts1[i](self.norm1s[i](h))

            # Optional CoordConv before the 3x3 conv
            inject = (self.use_cc_all or (i == 0 and self.use_cc_bot))
            if inject:
                Hs, Ws = h.size(2), h.size(3)
                grid = self._coord_grid(Hs, Ws, h.device, h.dtype)   # (2,Hs,Ws), no grad edge
                h = torch.cat([h, grid.unsqueeze(0).expand(B, -1, -1, -1)], dim=1)

            h = self.convs[i](h)
            h = self.acts2[i](self.norm2s[i](h))

        x = self.out(h)
        return torch.sigmoid(x)  # images in [0,1]

# ---------- top-level (cfg-friendly) ----------

class AutoEncoderCoordConv(nn.Module):
    """
    Classic AE with CoordConv:
      • encoder: optional CoordConv at input (off by default)
      • decoder: CoordConv at bottleneck (on by default) and optional at all levels
    Constructor matches your factory: __init__(cfg)
    """
    def __init__(self, cfg):
        super().__init__()
        # pull from cfg.model
        m = cfg if hasattr(cfg, "in_channels") else cfg.model  # tolerate passing cfg or cfg.model
        in_ch       = int(getattr(m, "in_channels"))
        out_ch      = int(getattr(m, "out_channels"))
        z_dim       = int(getattr(m, "latent_dim"))
        base        = int(getattr(m, "base_channels", 32))
        levels      = int(getattr(m, "num_down_levels", 3))
        img_size    = int(getattr(m, "image_size", 32))
        groups_gn   = int(getattr(m, "groups_gn", 8))

        use_cc_enc  = bool(getattr(m, "use_coordconv_encoder", False))
        use_cc_bot  = bool(getattr(m, "use_coordconv_decoder_bottleneck", True))
        use_cc_all  = bool(getattr(m, "use_coordconv_decoder_all_levels", False))

        # modules
        self.encoder = ConvEncoderClassicCC(
            in_ch=in_ch, z_dim=z_dim, base=base, levels=levels, img_size=img_size,
            groups_gn=groups_gn, use_coordconv_encoder=use_cc_enc,
        )
        self.decoder = ConvDecoderClassicCC(
            out_ch=out_ch, z_dim=z_dim, base=base, levels=levels, img_size=img_size,
            groups_gn=groups_gn, use_cc_bot=use_cc_bot, use_cc_all=use_cc_all,
        )

        # latent norm buffers (kept for your callbacks)
        self.z_dim = z_dim
        self.register_buffer("latent_norm_mean", torch.zeros(self.z_dim))
        self.register_buffer("latent_norm_std",  torch.ones(self.z_dim))

        # VAE compatibility knobs (unused by default)
        self.register_parameter("prior_logvar", None)
        self._last_mu = None
        self._last_logvar = None
        self._last_z = None

    # API parity with old model
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
        print(f"[AE] Total params: {total:,}  (trainable: {trainable:,})")
