# loss/isometry.py
from __future__ import annotations
import torch
from torch import func as func  # torch>=2.0
from typing import Callable

@torch.enable_grad()
def approximate_orthogonal_jacobian_regularisation(
    phi: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    num_v: int = 16,
    train: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Encourages local isometry by matching Gram(J v_i) ≈ I for random orthonormal {v_i}.
    Uses JVPs; QR is computed in float32 (AMP-safe), then cast to x.dtype.
    """
    B = x.shape[0]
    flat_dim = x[0].numel()

    # ensure good layout for norms/conv during JVP
    if x.ndim == 4:
        x = x.contiguous(memory_format=torch.contiguous_format)
    else:
        x = x.contiguous()
    x = x.requires_grad_(True)

    # ---- AMP-safe orthonormal directions (QR in fp32) ----
    with torch.amp.autocast("cuda", enabled=False):
        rand32 = torch.randn(B, flat_dim, num_v, device=device, dtype=torch.float32)
        q32, _ = torch.linalg.qr(rand32, mode="reduced")  # (B, n, num_v), fp32
    q = q32.to(dtype=x.dtype)
    v = q.permute(2, 0, 1).reshape(num_v, B, *x.shape[1:])  # (num_v, B, ...)

    if v.ndim == 5:  # image tensors
        v = v.contiguous(memory_format=torch.contiguous_format)
    else:
        v = v.contiguous()

    # ---- JVP over v directions ----
    def jvp_single(v_single):
        if v_single.ndim == 4:
            v_single = v_single.contiguous(memory_format=torch.contiguous_format)
        else:
            v_single = v_single.contiguous()
        return func.jvp(phi, (x,), (v_single,))[1]

    if not train:
        torch.set_grad_enabled(True)
    Jv = func.vmap(jvp_single)(v)  # (num_v, B, *out)
    if not train:
        torch.set_grad_enabled(False)

    # (B, num_v, m)
    Jv = Jv.reshape(num_v, B, -1).permute(1, 0, 2)

    # Gram ≈ I
    G = torch.bmm(Jv, Jv.transpose(1, 2))
    I = torch.eye(num_v, device=Jv.device, dtype=Jv.dtype).expand(B, num_v, num_v)
    diff = G - I
    return (diff ** 2).sum(dim=(1, 2)).mean()

def _orthonormal_frame(B: int, dim: int, r: int, *, device, dtype):
    r = min(r, dim)
    with torch.amp.autocast("cuda", enabled=False):
        M32 = torch.randn(B, dim, r, device=device, dtype=torch.float32)
        Q32, _ = torch.linalg.qr(M32, mode="reduced")
    Q = Q32.to(dtype=dtype)
    return Q.permute(2, 0, 1)  # (r, B, dim)

@torch.enable_grad()
def encoder_row_orthogonality_regularisation(
    encode: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    num_v: int = 8,
    train: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    # Turn grad on even in eval so VJP can be formed; turn it back off after.
    if not train:
        torch.set_grad_enabled(True)

    x = x.contiguous().requires_grad_(True)

    def f(inp):  # (B, C, H, W) -> (B, k)
        return encode(inp)

    y, vjp_fn = func.vjp(f, x)              # y: (B, k)
    B, k = y.shape

    U = _orthonormal_frame(B, k, num_v, device=device, dtype=y.dtype)  # (r, B, k)

    def single_vjp(u_single):               # u_single: (B, k)
        (vx,) = vjp_fn(u_single)            # (B, ...)
        return vx

    V = func.vmap(single_vjp)(U)            # (r, B, ...)
    if not train:
        torch.set_grad_enabled(False)

    V = V.reshape(V.shape[0], V.shape[1], -1).permute(1, 0, 2)   # (B, r, n)

    # Compute Gram in fp32 for stability under AMP
    Vf = V.float()
    G  = torch.bmm(Vf, Vf.transpose(1, 2))                       # (B, r, r)
    I  = torch.eye(G.size(-1), device=G.device, dtype=G.dtype).expand_as(G)
    return (G - I).pow(2).sum(dim=(1, 2)).mean()
