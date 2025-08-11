"""
UNet-style backbone for score-based diffusion that supports arbitrary image resolutions.

Key points
----------
* Residual Conv + GroupNorm + SiLU with FiLM conditioning.
* Self-attention at user-configurable resolutions (e.g. 16 × 16).
* Skip connections saved **before** the channel-doubling ResBlock (fixes concat mismatch).
* `Up` blocks follow the encoder’s true channel schedule (`skip_channels`).
* **New fix →** Attention permutation bug squashed (no more *IndexError*).
* Head dimension now *always* `ch // heads` → uniform across levels.
* BatchNorm-free ⇒ vmap-safe.
* Works on any `base_channels` you set (32–128 tested).
* Resolution-agnostic: no hard-coded spatial dims.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # <- use functional alias for SiLU & SDPA


# ─────────────────────────────── helpers ────────────────────────────────

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal position embedding identical to DDPM / Stable-Diffusion."""
    half  = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half,
                                                      dtype=t.dtype,
                                                      device=t.device) / (half - 1))
    emb = torch.cat([torch.sin(t[:, None] * freqs),
                     torch.cos(t[:, None] * freqs)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, in_dim: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(in_dim, out_ch * 2))

    def forward(self, x: torch.Tensor, emb: torch.Tensor):  # type: ignore[override]
        scale, shift = self.net(emb).chunk(2, 1)
        return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """(Conv → GN → SiLU) × 2 with FiLM in between."""

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 emb_dim: int,
                 p: float = 0.0,
                 groups: int = 8):
        super().__init__()
        g = min(groups, out_ch) if out_ch % groups == 0 else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.film  = FiLM(emb_dim, out_ch)
        self.act   = nn.SiLU()
        self.drop  = nn.Dropout(p)
        self.skip  = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):  # type: ignore[override]
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, emb)
        h = self.drop(h)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head spatial self-attention using PyTorch’s SDP kernel."""

    def __init__(self, ch: int, heads: int, _cfg_head_dim: int, p: float):
        super().__init__()
        if ch % heads != 0:
            raise ValueError(f"Channels ({ch}) not divisible by heads ({heads}).")
        self.heads    = heads
        self.head_dim = ch // heads  # enforce consistency
        self.qkv   = nn.Conv1d(ch, ch * 3, 1)
        self.proj  = nn.Conv1d(ch, ch, 1)
        self.drop  = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, H, W)
        B, C, H, W = x.shape
        N    = H * W
        hdim = self.head_dim
        y    = x.view(B, C, N)                                   # (B, C, N)

        qkv  = self.qkv(y).view(B, 3, self.heads, hdim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                # (B, h, d, N)

        q = q.permute(0, 1, 3, 2).reshape(B * self.heads, N, hdim)  # (B·h, N, d)
        k = k.permute(0, 1, 3, 2).reshape(B * self.heads, N, hdim)
        v = v.permute(0, 1, 3, 2).reshape(B * self.heads, N, hdim)

        # functional alias ensures ∀ Torch versions with SDPA
        out = F.scaled_dot_product_attention(
                  q, k, v,
                  dropout_p=self.drop.p if self.training else 0.0   # <<< add
              )

        out = (out.reshape(B, self.heads, N, hdim)
                   .permute(0, 1, 3, 2)
                   .reshape(B, C, N))
        return x + self.proj(out).view(B, C, H, W)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1)

    def forward(self, x):  # type: ignore[override]
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):  # type: ignore[override]
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ───────────────────────────── main UNet ────────────────────────────────

class DDPM(nn.Module):
    """Flexible UNet backbone for continuous-time score-based diffusion."""

    # ─── constructor ───
    def __init__(self, cfg):
        super().__init__()

        base      = cfg.base_channels
        L         = cfg.num_blocks
        rbl       = cfg.res_blocks_per_level
        emb_dim   = cfg.time_embed_dim
        attn_res  = set(cfg.attn_resolutions)
        heads     = cfg.num_heads
        d_conv    = cfg.dropout_conv
        d_attn    = cfg.dropout_attn

        # ───── stems ─────
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.stem = nn.Conv2d(cfg.in_channels, base, 3, 1, 1)

        # channel schedule *before* doubling
        self.skip_channels: List[int] = [base * (2 ** lv) for lv in range(L)]

        # ───── encoder ─────
        self.enc, self.downs = nn.ModuleList(), nn.ModuleList()
        ch = base
        for lv in range(L):
            blk = nn.ModuleList([ResBlock(ch, ch, emb_dim, d_conv) for _ in range(rbl)])
            if 2 ** lv in attn_res:
                blk.append(AttentionBlock(ch, heads, cfg.head_dim, d_attn))
            next_ch = ch if lv == L - 1 else ch * 2
            blk.append(ResBlock(ch, next_ch, emb_dim, d_conv))
            self.enc.append(blk)
            if lv < L - 1:
                self.downs.append(Down(next_ch, next_ch))
            ch = next_ch

        # ───── bottleneck ─────
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, emb_dim, d_conv),
            AttentionBlock(ch, heads, cfg.head_dim, d_attn),
            ResBlock(ch, ch, emb_dim, d_conv),
        ])

        # ───── decoder ─────
        self.up_convs = nn.ModuleList([
            Up(self.skip_channels[-i - 1], self.skip_channels[-i - 2])
            for i in range(L - 1)
        ])
        self.dec = nn.ModuleList()
        ch = self.skip_channels[-1]
        for lv in range(L):
            blk = nn.ModuleList()
            blk.append(ResBlock(ch * 2, ch, emb_dim, d_conv))          # concat → ch
            for _ in range(rbl - 1):
                blk.append(ResBlock(ch, ch, emb_dim, d_conv))
            if 2 ** (L - lv - 1) in attn_res:
                blk.append(AttentionBlock(ch, heads, cfg.head_dim, d_attn))
            self.dec.append(blk)
            if lv < L - 1:
                ch = self.skip_channels[-lv - 2]

        # ───── output ─────
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, cfg.out_channels, 3, 1, 1)

    # ─────────────────────────── forward ───────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t is None:
            raise ValueError("timestep `t` required")

        emb = self.time_mlp(timestep_embedding(t,
                                               self.time_mlp[0].in_features))

        # encoder
        h, skips = self.stem(x), []
        for blk, down in zip(self.enc, self.downs + [None]):
            for b in blk[:-1]:
                h = b(h, emb) if isinstance(b, ResBlock) else b(h)
            skips.append(h)
            h = blk[-1](h, emb)
            if down:
                h = down(h)

        # bottleneck
        for b in self.mid:
            h = b(h, emb) if isinstance(b, ResBlock) else b(h)

        # decoder
        for lv, blk in enumerate(self.dec):
            if lv > 0:
                h = self.up_convs[lv - 1](h)
            h = torch.cat([h, skips[-lv - 1]], dim=1)
            for b in blk:
                h = b(h, emb) if isinstance(b, ResBlock) else b(h)

        # final projection  (torch.silu → F.silu fix)
        return self.out_conv(F.silu(self.out_norm(h)))

    # ─────────────────────────── utils ───────────────────────────

    def get_score_fn(self, sde):
        """
        Returns a function that computes the score.
        
        Args:
            sde: The SDE object that provides the marginal probability.
            train: Boolean flag indicating whether in training mode.
            
        Returns:
            score_fn: A function that computes the score based on the diffusion model's noise prediction.
        """
        def score_fn(x, y, t):
            noise_prediction = self.forward(x, y, t)
            _, std = sde.marginal_prob(x, t)
            std = std.view(std.shape[0], *[1 for _ in range(len(x.shape) - 1)])  # Expand std to match the shape of noise_prediction
            score = -noise_prediction / std
            return score
        
        return score_fn
    
    def get_denoiser_fn(self, sde):
        # Infer the alpha and sigma functions from the SDE
        alpha_fn = sde.get_alpha_fn()
        sigma_fn = sde.get_sigma_fn()
        def denoiser_fn(x_t, y, t):
            sigma_t, alpha_t = sigma_fn(t), alpha_fn(t)
            noise_pred = self.forward(x_t, y, t)
            x_denoised = (x_t - sigma_t * noise_pred) / alpha_t
            return x_denoised
        return denoiser_fn
    
    def print_model_summary(self) -> None:
        """Print parameter counts."""
        tot = sum(p.numel() for p in self.parameters())
        tr  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {tot:,}\n  … trainable: {tr:,}\n  … frozen: {tot - tr:,}")
