# loss/geom_isometry.py
from __future__ import annotations
from typing import Callable, Optional, Union
import torch
from torch import func as func

# Soft import for type hints / optional use
try:
    from data_geometry.metrics import BaseMetric
except Exception:  # pragma: no cover
    BaseMetric = object  # type: ignore[misc,assignment]


# ────────────────────────── small helpers ───────────────────────────

def _to_contig(t: torch.Tensor) -> torch.Tensor:
    """Make tensor contiguous; keep 4D tensors in their default contiguous format."""
    if t.ndim == 4:
        return t.contiguous(memory_format=torch.contiguous_format)
    return t.contiguous()


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten non-batch dims: (B, ...) -> (B, -1)."""
    return x.reshape(x.shape[0], -1)


def _metric_inner(
    metric: Optional[BaseMetric],
    x_flat_detached: torch.Tensor,  # (B, n) — MUST be detached
    u_flat: torch.Tensor,           # (B, n) — keep grad
    v_flat: torch.Tensor,           # (B, n) — keep grad
) -> torch.Tensor:
    """
    Inner product ⟨u,v⟩_g(x). Gradients flow through u,v; *not* through the metric
    or x, because x must be passed detached by the caller.
    """
    if metric is None:
        return (u_flat * v_flat).sum(dim=-1)
    return metric.inner_product(x_flat_detached, u_flat, v_flat)


def _metric_gram(
    metric: Optional[BaseMetric],
    x_flat_detached: torch.Tensor,  # (B, n) — MUST be detached
    V: torch.Tensor,                # (B, r, n) — keep grad
) -> torch.Tensor:
    """
    Gram matrix G_ij = ⟨V_i, V_j⟩_g(x). Gradients flow only through V.
    """
    if metric is None:
        # Euclidean: full gradient through V
        return V @ V.transpose(1, 2)

    B, r, n = V.shape
    # Use float32 for metric arithmetic for stability (keeps grad through V)
    V32 = V.to(dtype=torch.float32)
    G32 = V32.new_zeros(B, r, r)
    x_det = x_flat_detached  # already detached by caller

    for i in range(r):
        ui = V32[:, i, :]
        for j in range(r):
            vj = V32[:, j, :]
            G32[:, i, j] = _metric_inner(metric, x_det, ui, vj)  # (B,)

    return G32.to(dtype=V.dtype)


@torch.no_grad()
def _metric_orthonormal_frame(
    metric: Optional[BaseMetric],
    x: torch.Tensor,                 # (B, ...)
    r: int,
    *,
    device: Union[torch.device, str],
    dtype: torch.dtype,
    project_first: bool = True,
    debug: bool = False,
) -> torch.Tensor:
    """
    Return an (r, B, ...) frame that is orthonormal w.r.t. *metric* at x.
    - If metric is None: Euclidean QR (float32) then cast back.
    - If metric is provided: metric-aware Gram–Schmidt in float32, isolating
      the whole computation from AMP/mixed precision.

    No gradients are recorded through this construction by design.
    """
    B = x.shape[0]
    n = x[0].numel()
    x_flat = _flatten(x)

    # Euclidean fallback: QR in float32 for stability, then cast back
    if metric is None:
        with torch.amp.autocast("cuda", enabled=False):
            R = torch.randn(B, n, r, device=device, dtype=torch.float32)
            Q32, _ = torch.linalg.qr(R, mode="reduced")  # (B, n, r)
        Q = Q32.to(dtype=dtype)  # (B, n, r)
        return Q.permute(2, 0, 1).reshape(r, B, *x.shape[1:])

    if debug:
        print(
            f"[metric frame] B={B}, n={n}, r={r}, "
            f"x.dtype={x.dtype}, target_dtype={dtype}"
        )

    # Metric-aware Gram–Schmidt in float32 (AMP disabled)
    with torch.amp.autocast("cuda", enabled=False):
        x_flat_f32 = x_flat.float()
        # Start with random directions
        V = torch.randn(B, r, n, device=device, dtype=torch.float32)
        Q = torch.zeros_like(V)  # (B, r, n)

        for i in range(r):
            vi = V[:, i, :]

            # Optionally project a random vector to the metric tangent via inverse op
            if project_first:
                # This call is in float32 and under no_grad — safe for most metrics
                vi = metric.apply_inverse(x_flat_f32, vi)

            # Remove components along previous q_j using *metric* inner products
            for j in range(i):
                qj = Q[:, j, :]
                alpha = _metric_inner(metric, x_flat_f32, vi, qj)  # (B,)
                vi = vi - alpha.unsqueeze(-1) * qj

            # Normalise under the metric
            denom = torch.sqrt(_metric_inner(metric, x_flat_f32, vi, vi) + 1e-8)
            qi = vi / denom.unsqueeze(-1)
            Q[:, i, :] = qi

    # Cast the frame back to the model dtype and original shape
    Q = Q.to(dtype=dtype)  # (B, r, n) in model dtype
    return Q.permute(1, 0, 2).reshape(r, B, *x.shape[1:])


# ─────────────────────── encoder (data → latent) ───────────────────────

@torch.enable_grad()
def encoder_tangent_isometry_regularisation(
    encode: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    data_metric: Optional[BaseMetric] = None,
    latent_metric: Optional[BaseMetric] = None,
    num_v: int = 8,
    train: bool = True,
    device: Union[torch.device, str, None] = None,
    project_first: bool = True,
) -> torch.Tensor:
    """
    Build a g_x-orthonormal (or Euclidean if None) tangent frame {v_i} at x,
    push through the encoder via JVPs, then encourage latent inner products to be
    preserved in the latent metric h_y:

        G_ij = ⟨ J_e(x) v_i , J_e(x) v_j ⟩_{h(encode(x))}  ≈ δ_ij

    • No gradients flow through metric definitions (x is detached for metric ops).
    • Gradients DO flow through the encoder via the JVP outputs.
    """
    x = _to_contig(x).requires_grad_(True)
    dev = x.device if device is None else torch.device(device)

    # 1) metric-orthonormal frame in data space (no grad)
    r = min(num_v, x[0].numel())
    U = _metric_orthonormal_frame(
        data_metric, x, r,
        device=dev, dtype=x.dtype, project_first=project_first, debug=not train
    )  # (r, B, ...)

    # 2) JVPs through the encoder (gradients must flow to encoder params)
    def jvp_single(v_single: torch.Tensor) -> torch.Tensor:
        v_single = _to_contig(v_single)
        return func.jvp(encode, (x,), (v_single,))[1]  # (B, k)

    Jv = func.vmap(jvp_single)(U)  # (r, B, k)

    # 3) Gram in latent metric at y = encode(x); block grad through metric inputs
    with torch.no_grad():
        y = encode(x.detach())
        y_flat = _flatten(y).float()  # detached

    V_lat = Jv.permute(1, 0, 2).contiguous()  # (B, r, k)
    G = _metric_gram(latent_metric, y_flat, V_lat.float())  # (B, r, r)

    I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype).expand_as(G)
    return (G - I).pow(2).sum(dim=(1, 2)).mean()


# ─────────────────────── decoder (latent → data) ───────────────────────

@torch.enable_grad()
def decoder_tangent_isometry_regularisation(
    decode: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    *,
    data_metric: Optional[BaseMetric] = None,
    latent_metric: Optional[BaseMetric] = None,
    num_v: int = 8,
    train: bool = True,
    device: Union[torch.device, str, None] = None,
    project_first: bool = True,
) -> torch.Tensor:
    """
    Build an h_z-orthonormal (or Euclidean if None) tangent frame {u_i} at z,
    push through the decoder via JVPs, then encourage data inner products to be
    preserved in the data metric g_x:

        G_ij = ⟨ J_d(z) u_i , J_d(z) u_j ⟩_{g(decode(z))}  ≈ δ_ij

    • No gradients flow through metric definitions.
    • Gradients DO flow through the decoder via the JVP outputs.
    """
    z = _to_contig(z).requires_grad_(True)
    dev = z.device if device is None else torch.device(device)

    # 1) metric-orthonormal frame in latent space (no grad)
    r = min(num_v, z[0].numel())
    U = _metric_orthonormal_frame(
        latent_metric, z, r,
        device=dev, dtype=z.dtype, project_first=project_first, debug=not train
    )  # (r, B, ...)

    # 2) JVPs through the decoder (gradients flow to decoder params)
    def jvp_single(u_single: torch.Tensor) -> torch.Tensor:
        u_single = _to_contig(u_single)
        return func.jvp(decode, (z,), (u_single,))[1]  # (B, *x_shape)

    Ju = func.vmap(jvp_single)(U)  # (r, B, *x_shape)

    V_data = Ju.reshape(Ju.shape[0], Ju.shape[1], -1).permute(1, 0, 2).contiguous()  # (B, r, n_x)

    # 3) Gram in data metric at x = decode(z); block grad through metric inputs
    with torch.no_grad():
        x = decode(z.detach())
        x_flat = _flatten(x).float()  # detached

    G = _metric_gram(data_metric, x_flat, V_data.float())  # (B, r, r)

    I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype).expand_as(G)
    return (G - I).pow(2).sum(dim=(1, 2)).mean()


# ───────────── composition check: D(e∘d) ≈ I in latent metric ─────────

@torch.enable_grad()
def latent_composition_isometry_regularisation(
    encode: Callable[[torch.Tensor], torch.Tensor],
    decode: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    *,
    latent_metric: Optional[BaseMetric] = None,
    num_v: int = 8,
    train: bool = True,
    device: Union[torch.device, str, None] = None,
    project_first: bool = True,
) -> torch.Tensor:
    """
    Check that the differential of f = encode ∘ decode at z acts like identity
    in the latent metric:

        G_ij = ⟨ Df(z) u_i , Df(z) u_j ⟩_{h(f(z))}  ≈ δ_ij

    • No gradients flow through metric definitions.
    • Gradients DO flow through (encode, decode) via the JVP outputs.
    """
    z = _to_contig(z).requires_grad_(True)
    dev = z.device if device is None else torch.device(device)

    # 1) latent-metric-orthonormal frame (no grad)
    r = min(num_v, z[0].numel())
    U = _metric_orthonormal_frame(
        latent_metric, z, r,
        device=dev, dtype=z.dtype, project_first=project_first, debug=not train
    )  # (r, B, ...)

    # 2) JVPs through the composition f = e∘d (gradients flow to both)
    def f(zz: torch.Tensor) -> torch.Tensor:
        return encode(decode(zz))

    def jvp_single(u_single: torch.Tensor) -> torch.Tensor:
        u_single = _to_contig(u_single)
        return func.jvp(f, (z,), (u_single,))[1]  # (B, k)

    Ju = func.vmap(jvp_single)(U)  # (r, B, k)

    # 3) Gram in latent metric at y = f(z); block grad through metric inputs
    with torch.no_grad():
        y = f(z.detach())
        y_flat = _flatten(y).float()  # detached

    V_lat = Ju.permute(1, 0, 2).contiguous()  # (B, r, k)
    G = _metric_gram(latent_metric, y_flat, V_lat.float())  # (B, r, r)

    I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype).expand_as(G)
    return (G - I).pow(2).sum(dim=(1, 2)).mean()
