# models/autoencoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SimpleGroupNorm2d(nn.Module):
    """
    Drop-in replacement for nn.GroupNorm (no running stats).
    Normalizes over groups across (C_group, H, W) with affine per-channel.
    Implemented with reshape/reductions (no .view), so it plays nicely with
    channels_last + torch.compile + functorch (vmap/jvp).
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        assert num_channels % num_groups == 0, "channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            # per-channel scale/shift
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in x.dtype; if you want extra stability under AMP, uncomment:
        # orig_dtype = x.dtype; x = x.float()
        N, C, H, W = x.shape
        G = self.num_groups
        xg = x.reshape(N, G, C // G, H, W)                      # (N, G, Cg, H, W)
        mean = xg.mean(dim=(2, 3, 4), keepdim=True)             # (N, G, 1, 1, 1)
        var  = xg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        y = xg.reshape(N, C, H, W)
        if self.affine:
            y = y * self.weight + self.bias
        # return y.to(orig_dtype)
        return y


class ConvEncoder(nn.Module):
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

        self._chs_out = chs[levels]
        # compute spatial after 'levels' stride-2 downsamples
        s = max(1, img_size // (2 ** levels))
        flat_dim = self._chs_out * s * s
        # define the projection in __init__ so it moves to CUDA with the model
        self.z_proj = nn.Linear(flat_dim, self.z_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = y.reshape(y.size(0), -1)  # safe for channels_last & compile
        return self.z_proj(y)


class ConvDecoder(nn.Module):
    """
    Decoder that mirrors the encoder’s downsampling with `levels` upsampling blocks.
    Fixes channel schedule so each upsample transitions prev_ch -> next_ch.
    """
    def __init__(self, out_ch: int, z_dim: int, base: int = 32, levels: int = 3, img_size: int = 32):
        super().__init__()
        # Channel schedule used during encoding (lowest -> highest)
        # e.g., [32, 64, 128] when levels=3
        chs = [base, base * 2, base * 4][:levels]

        self.levels = levels
        self.img_size = img_size

        # Spatial size after `levels` stride-2 downsamples
        s = max(1, img_size // (2 ** levels))
        self.spatial = s

        # Stem channels at the bottleneck match the last encoder level
        self.stem_ch = chs[-1]

        # Project latent vector to (stem_ch, s, s)
        self.fc = nn.Linear(z_dim, self.stem_ch * s * s)

        # Build upsampling blocks:
        # For levels=3 and chs=[32, 64, 128], we want transitions:
        # 128 -> 64, 64 -> 32, 32 -> 32
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

        # Final output conv
        self.out = nn.Conv2d(chs[0], out_ch, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(z.size(0), self.stem_ch, self.spatial, self.spatial)
        h = self.dec(h)
        x = self.out(h)
        return torch.sigmoid(x)  # dataset is in [0, 1]



class AutoEncoder(nn.Module):
    """
    Minimal AE with encode/decode methods for Jacobian regularization.
    """
    def __init__(self, cfg):
        super().__init__()
        self.in_ch = cfg.in_channels
        self.out_ch = cfg.out_channels
        self.z_dim = cfg.latent_dim
        base = getattr(cfg, "base_channels", 32)
        levels = getattr(cfg, "num_down_levels", 3)
        img_size = getattr(cfg, "image_size", 32)

        self.encoder = ConvEncoder(self.in_ch, self.z_dim, base=base, levels=levels, img_size=img_size)
        self.decoder = ConvDecoder(self.out_ch, self.z_dim, base=base, levels=levels, img_size=img_size)

    # API used by loss/regularisers
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # parity with your models
    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[AE] Total params: {total:,}  (trainable: {trainable:,})")
