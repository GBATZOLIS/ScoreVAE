# loss/isometry.py
from __future__ import annotations
from typing import Callable, Optional
import torch
from torch import func as func  # torch >= 2.0

def _qr_frame(B: int, dim: int, r: int, *, device, dtype) -> torch.Tensor:
    r = min(r, dim)
    with torch.amp.autocast("cuda", enabled=False):
        M32 = torch.randn(B, dim, r, device=device, dtype=torch.float32)
        Q32, _ = torch.linalg.qr(M32, mode="reduced")  # (B, dim, r)
    return Q32.to(dtype=dtype).permute(2, 0, 1)       # (r, B, dim)

@torch.enable_grad()
def encoder_isometry_regularisation(
    encode: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    num_v: int = 8,
    train: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Local isometry for the encoder: J_e(x)^T U should have orthonormal rows,
    where U are Euclidean-orthonormal in the latent space.
    """
    if not train: torch.set_grad_enabled(True)
    x = x.contiguous().requires_grad_(True)

    def f(inp): return encode(inp)          # (B, ...) -> (B, k)
    y, vjp_fn = func.vjp(f, x)              # y: (B, k)
    B, k = y.shape
    r = min(num_v, k)

    # Simplified: Always build a Euclidean orthonormal frame
    U = _qr_frame(B, k, r, device=device, dtype=y.dtype)

    def single_vjp(u_single):  # (B, k) -> (B, ...)
        (vx,) = vjp_fn(u_single); return vx

    V = func.vmap(single_vjp)(U)            # (r, B, ...)
    if not train: torch.set_grad_enabled(False)

    V = V.reshape(V.shape[0], V.shape[1], -1).permute(1, 0, 2)  # (B, r, n)
    G = torch.bmm(V.float(), V.float().transpose(1, 2))         # (B, r, r)
    I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype).expand_as(G)
    return (G - I).pow(2).sum(dim=(1, 2)).mean()

@torch.enable_grad()
def decoder_isometry_regularisation(
    decode: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    *,
    num_v: int = 16,
    train: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Local isometry for the decoder: Gram(J_d(z) u_i) ≈ I for Euclidean latent/data metrics.
    Uses JVPs with orthonormal latent directions.
    """
    B, d = z.shape[0], z[1:].numel() if z.dim() > 2 else z.shape[1]
    z = z.contiguous().requires_grad_(True)

    # build orthonormal u_i in latent (Euclidean)
    U = _qr_frame(B, z.shape[1], min(num_v, z.shape[1]), device=device, dtype=z.dtype)  # (r,B,d)
    U = U.reshape(U.shape[0], B, *z.shape[1:])  # match shape for decode JVPs

    def jvp_single(u_single):  # (B,d) or (B,...) -> (B, *xshape)
        return func.jvp(decode, (z,), (u_single.contiguous(),))[1]

    if not train: torch.set_grad_enabled(True)
    Jv = func.vmap(jvp_single)(U)    # (r, B, *xshape)
    if not train: torch.set_grad_enabled(False)

    Jv = Jv.reshape(Jv.shape[0], B, -1).permute(1, 0, 2)  # (B, r, n_x)
    G = torch.bmm(Jv, Jv.transpose(1, 2))                 # (B, r, r)
    I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype).expand_as(G)
    return (G - I).pow(2).sum(dim=(1, 2)).mean()