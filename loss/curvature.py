# loss/curvature.py
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from torch.func import jvp, vjp, vmap, grad

# ====================================================================================
# Small helpers
# ====================================================================================

def _as_single_map(f_batched: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Wrap a batched decode so it accepts (d,) and returns flat (D,)."""
    def f_single(z1d: Tensor) -> Tensor:
        y = f_batched(z1d.unsqueeze(0))
        return y.reshape(1, -1).squeeze(0)
    return f_single

def _Jv(f_single: Callable[[Tensor], Tensor], z: Tensor, p: Tensor) -> Tensor:
    """J(z) @ p  -> (D,)"""
    return jvp(f_single, (z,), (p,))[1]

def _JT(f_single: Callable[[Tensor], Tensor], z: Tensor, y: Tensor) -> Tensor:
    """J(z)^T @ y  -> (d,)"""
    _, vjp_fn = vjp(f_single, z)
    return vjp_fn(y)[0]

def _chol_solve_batch(L: Tensor, b: Tensor) -> Tensor:
    """
    Solve (L L^T) x = b in batch.
      L: (B,d,d) lower Cholesky
      b: (B,d) or (B,d,k)
    """
    add_dim = (b.dim() == 2)
    if add_dim:
        b = b.unsqueeze(-1)
    x = torch.cholesky_solve(b, L)
    return x.squeeze(-1) if add_dim else x

def _solve_upper_tri_batch(U: Tensor, B_: Tensor) -> Tensor:
    """
    Solve U X = B with U upper-triangular.
      U: (B,d,d), B: (B,d) or (B,d,k)
    """
    add_dim = (B_.dim() == 2)
    if add_dim:
        B_ = B_.unsqueeze(-1)
    try:
        X = torch.linalg.solve_triangular(U, B_, upper=True)
    except AttributeError:
        X, _ = torch.triangular_solve(B_, U, upper=True)
    return X.squeeze(-1) if add_dim else X

def _sample_rademacher(shape, device, dtype):
    return (torch.randint(0, 2, shape, device=device) * 2 - 1).to(dtype)

# ====================================================================================
# Mixed second derivative D^2 f[z][u, w]
# ====================================================================================

def _H_mixed_exact_batched(
    f_single: Callable[[Tensor], Tensor],
    z: Tensor,   # (B,d)
    u: Tensor,   # (B,d)
    w: Tensor,   # (B,d)
) -> Tensor:
    """
    Exact batched mixed derivative using nested forward-mode JVPs:
      D^2 f[z][u, w] = D( J(z) u )[z] · w
    Returns (B, D).
    """
    def single(zi, ui, wi):
        g = lambda zz: jvp(f_single, (zz,), (ui,))[1]       # (D,)
        return jvp(g, (zi,), (wi,))[1]                      # (D,)
    return vmap(single)(z, u, w)

def _H_mixed_fd_batched(
    f_single: Callable[[Tensor], Tensor],
    z: Tensor, u: Tensor, w: Tensor, eps: float = 1e-3
) -> Tensor:
    def jv_(zz, uu):
        return jvp(f_single, (zz,), (uu,))[1]
    Jv_pos = vmap(jv_)(z + eps * w, u)
    Jv_now = vmap(jv_)(z,            u)
    return (Jv_pos - Jv_now) / eps

# ====================================================================================
# Build J, G, L in batch
# ====================================================================================

def _build_J_and_G_batched(decode, z: Tensor, lam: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build per-sample Jacobians and pullback metrics in batch.
      z: (B,d)
    Returns:
      J: (B, D, d)
      G: (B, d, d)   with G = J^T J + lam I
      L: (B, d, d)   lower Cholesky of G
    """
    B, d = z.shape
    f_single = _as_single_map(decode)

    with torch.no_grad():
        D = decode(z[:1]).reshape(1, -1).size(1)

    I = torch.eye(d, device=z.device, dtype=z.dtype)

    def J_cols_for_sample(zi: Tensor) -> Tensor:              # -> (d, D)
        return vmap(lambda e: _Jv(f_single, zi, e))(I)

    J_cols = vmap(J_cols_for_sample)(z)                       # (B,d,D)
    J = J_cols.transpose(1, 2).contiguous()                   # (B,D,d)

    JT = J.transpose(1, 2)                                    # (B,d,D)
    G = JT @ J                                                # (B,d,d)
    if lam != 0.0:
        G = G + lam * torch.eye(d, device=z.device, dtype=z.dtype).expand(B, d, d)
    L = torch.linalg.cholesky(G)                              # (B,d,d)
    return J, G, L

# ====================================================================================
# Directional derivative of (T v) in batch   (MECAE core)
# ====================================================================================

def _dir_dTv_batched(
    decode,
    z: Tensor,          # (B,d)
    J: Tensor,          # (B,D,d)
    L: Tensor,          # (B,d,d)  (Cholesky of G)
    v: Tensor,          # (B,D)
    w: Tensor,          # (B,d)    direction (either G^{-1/2} w or w)
    use_exact_hessian: bool = True,
    fd_eps: float = 1e-3,
) -> Tensor:
    """
    Compute d_w := (w · ∇)(T v) in batch.
    T(z)v = J G^{-1} J^T v,  with G = J^T J + lam I.
    Returns (B,D).
    """
    B, d = z.shape
    f_single = _as_single_map(decode)

    # α = G^{-1} J^T v  (batched)
    b = (J.transpose(1, 2) @ v.unsqueeze(-1)).squeeze(-1)     # (B,d)
    alpha = _chol_solve_batch(L, b)                           # (B,d)

    # term1: (∂J · w) α
    if use_exact_hessian:
        term1 = _H_mixed_exact_batched(f_single, z, alpha, w)     # (B,D)
    else:
        term1 = _H_mixed_fd_batched(f_single, z, alpha, w, fd_eps)# (B,D)

    # helper ops for ∂α
    def F_const_alpha_single(zz, a):
        J_alpha = _Jv(f_single, zz, a)          # (D,)
        return _JT(f_single, zz, J_alpha)       # (d,)
    dG_alpha = vmap(
        lambda zi, wi, ai: jvp(lambda zz: F_const_alpha_single(zz, ai), (zi,), (wi,))[1]
    )(z, w, alpha)  # (B,d)

    def JT_v_single(zz, vv):
        return _JT(f_single, zz, vv)            # (d,)
    dJT_v = vmap(
        lambda zi, wi, vv: jvp(lambda zz: JT_v_single(zz, vv), (zi,), (wi,))[1]
    )(z, w, v)  # (B,d)

    # ∂α = - G^{-1} (∂G · w) α  +  G^{-1} (∂J^T · w) v
    termA = _chol_solve_batch(L, dJT_v)         # (B,d)
    termB = _chol_solve_batch(L, dG_alpha)      # (B,d)
    dalpha = termA - termB                      # (B,d)

    # term2: J (∂α)
    def jv_single(zi, ui): return _Jv(f_single, zi, ui)  # (D,)
    term2 = vmap(jv_single)(z, dalpha)                  # (B,D)

    return term1 + term2

# ====================================================================================
# MECAE estimator (batched, non-negative default)
# ====================================================================================

def mecae_extrinsic_decoder(
    *,
    decode: Callable[[Tensor], Tensor],
    z: Tensor,                 # (B,d)
    lam: float = 1e-6,
    K_v: int = 1,              # output probes in ℝ^D
    K_w: int = 1,              # latent probes in ℝ^d
    use_rademacher: bool = True,
    estimator: str = "square",  # "square" (≥0) or "bilinear" (unbiased, can be <0)
    use_exact_hessian: bool = True,
    fd_eps: float = 1e-3,
    return_aux: bool = False,   # also return J, G, L
) -> Dict[str, Tensor]:
    """
    Batched MECAE extrinsic curvature estimator at points z for a decoder f=decode.

    Paper quantity (Dirichlet energy of T):
        EEC(z) = 1/2 * Tr( (∇T)^T (∇T) G^{-1} ) ≥ 0

    Estimators:
      - "square": E[ || (∂(T v) · G^{-1/2} w) ||^2 ] / 2     (non-negative; recommended)
      - "bilinear": E[ ⟨ ∂(T v)·w , ∂(T v)·(G^{-1} w) ⟩ ]/2  (unbiased per probe; can be negative)
    """
    assert z.dim() == 2, "z must be (B,d)"
    dev, dt = z.device, z.dtype
    B, d = z.shape

    # Build J, G, L once per sample (batched)
    J, G, L = _build_J_and_G_batched(decode, z, lam)  # J:(B,D,d), L:(B,d,d)

    # Ambient dimension D
    with torch.no_grad():
        D = decode(z[:1]).reshape(1, -1).size(1)

    # Probes
    if use_rademacher:
        V = _sample_rademacher((B, K_v, D), dev, dt)   # (B,K_v,D)
        W = _sample_rademacher((B, K_w, d), dev, dt)   # (B,K_w,d)
    else:
        V = torch.randn(B, K_v, D, device=dev, dtype=dt)
        W = torch.randn(B, K_w, d, device=dev, dtype=dt)

    # Precompute G^{-1/2} w_k via triangular solves: s = L^{-T} w
    U = L.transpose(1, 2)                               # (B,d,d) upper
    W_T = W.transpose(1, 2)                             # (B,d,K_w)
    S_T = _solve_upper_tri_batch(U, W_T)                # (B,d,K_w)
    S = S_T.transpose(1, 2).contiguous()                # (B,K_w,d)

    # Accumulate energies across probes (batched across samples)
    eec_acc = torch.zeros(B, device=dev, dtype=dt)

    for kv in range(K_v):
        v = V[:, kv, :]                                 # (B,D)
        if estimator == "square":
            for kw in range(K_w):
                s = S[:, kw, :]                         # (B,d) = G^{-1/2} w
                ddir = _dir_dTv_batched(
                    decode, z, J, L, v, s,
                    use_exact_hessian=use_exact_hessian,
                    fd_eps=fd_eps,
                )                                       # (B,D)
                eec_acc = eec_acc + (ddir * ddir).sum(dim=-1)
        elif estimator == "bilinear":
            for kw in range(K_w):
                w = W[:, kw, :]
                d_w  = _dir_dTv_batched(
                    decode, z, J, L, v, w,
                    use_exact_hessian=use_exact_hessian,
                    fd_eps=fd_eps,
                )
                wtil = _chol_solve_batch(L, w)          # (B,d) = G^{-1} w
                d_wt = _dir_dTv_batched(
                    decode, z, J, L, v, wtil,
                    use_exact_hessian=use_exact_hessian,
                    fd_eps=fd_eps,
                )
                eec_acc = eec_acc + (d_w * d_wt).sum(dim=-1)
        else:
            raise ValueError(f"Unknown estimator '{estimator}'. Use 'square' or 'bilinear'.")

    eec = 0.5 * eec_acc / (K_v * K_w)                   # (B,)
    if return_aux:
        return {"EEC": eec, "J": J, "G": G, "L": L}
    else:
        return {"EEC": eec}

# ====================================================================================
# MICAE (intrinsic curvature via Gauss equation + Hutchinson)
# ====================================================================================

@torch.no_grad()
def _build_normal_projector(J: Tensor, G: Tensor, L: Tensor) -> Tensor:
    """
    Normal projector N = I - J G^{-1} J^T (batched).
    J:(B,D,m), G:(B,m,m), L:(B,m,m) is Cholesky(G)
    """
    B, D, m = J.shape
    JT = J.transpose(1, 2)                        # (B,m,D)
    X  = torch.cholesky_solve(JT, L)              # (B,m,D) = G^{-1} JT
    P  = J @ X                                    # (B,D,D)
    I  = torch.eye(D, device=J.device, dtype=J.dtype).expand(B, D, D)
    return I - P                                  # (B,D,D)

def _project_and_normalize(N: Tensor, Xi: Tensor, ref: Optional[Tensor] = None, eps: float = 1e-8) -> Tensor:
    """
    Project ambient directions Xi onto the normal space via N and normalise.
    N: (B,D,D), Xi: (R_a,B,D) -> (R_a,B,D)
    """
    A = torch.einsum('bdd,rbd->rbd', N, Xi)        # (R_a,B,D)
    if ref is not None:
        pref = torch.einsum('bdd,bd->bd', N, ref)  # (B,D)
        while pref.dim() < A.dim():
            pref = pref.unsqueeze(0)
        nrm = A.norm(dim=-1, keepdim=True)
        A = torch.where(nrm > eps, A, A + pref)
    return A / (A.norm(dim=-1, keepdim=True) + eps)

def _II_in_normal_dir_batched(
    decode, z: Tensor, a: Tensor,
    use_exact_hessian: bool = True, fd_eps: float = 1e-3
) -> Tensor:
    """
    II^(a) = ∇^2_z <a, f(z)> for each sample (B,m,m), symmetric.
    Uses nested JVPs (exact) or a simple FD fallback.
    """
    f_single = _as_single_map(decode)
    B, m = z.shape
    I = torch.eye(m, device=z.device, dtype=z.dtype)  # (m,m)

    def g_of_a(ai):
        def g_local(zz):
            return (ai * f_single(zz)).sum()
        return g_local

    if use_exact_hessian:
        def H_for_normal(ai: Tensor, zi: Tensor) -> Tensor:
            grad_g = grad(g_of_a(ai))
            cols = vmap(lambda ej: jvp(grad_g, (zi,), (ej,))[1])(I)   # (m,m)
            return 0.5 * (cols + cols.transpose(0, 1))
    else:
        def H_for_normal(ai: Tensor, zi: Tensor) -> Tensor:
            eps = fd_eps
            def jv_(zz, vv): return jvp(f_single, (zz,), (vv,))[1]
            Jv_pos = vmap(jv_, in_dims=(None, 0))(zi + eps * I, I)    # (m,D)
            Jv_now = vmap(jv_, in_dims=(None, 0))(zi, I)
            cols = (Jv_pos - Jv_now) / eps
            return 0.5 * (cols + cols.transpose(0, 1))

    return vmap(H_for_normal)(a, z)                # (B,m,m)

def _II_multi_in_normals(
    decode, z: Tensor, A: Tensor,
    use_exact_hessian: bool = True, fd_eps: float = 1e-3
) -> Tensor:
    """
    Vectorised II for many normals at once.
      A: (R_a,B,D)  -> returns (R_a,B,m,m)
    """
    return vmap(
        lambda a_b: _II_in_normal_dir_batched(decode, z, a_b, use_exact_hessian, fd_eps)
    )(A)  # (R_a,B,m,m)

def micae_intrinsic_decoder(
    *,
    decode: Callable[[Tensor], Tensor],
    z: Tensor,                  # (B,m)
    lam: float = 1e-6,
    R_a: int = 2,               # normals to sample
    use_rademacher: bool = True,
    use_exact_hessian: bool = True,
    fd_eps: float = 1e-3,
    JGL: Optional[Tuple[Tensor, Tensor, Tensor]] = None,   # (J,G,L) reuse
    normalize_codim: bool = True,  # average over normals instead of multiplying by codim
    return_aux: bool = False,
) -> Dict[str, Tensor]:
    """
    MICAE: intrinsic curvature penalty using the Gauss equation in Euclidean ambient.

    Scalar curvature per sample (Gauss):
        R(z) = sum_{a=1}^{D-m} [ (tr S_a)^2 - ||S_a||_F^2 ],  where S_a = G^{-1} II^(a).

    We estimate the sum over an orthonormal basis of normals via Hutchinson:
        sum_a ≈ codim * E_a[ (tr S_a)^2 - ||S_a||_F^2 ].
    If `normalize_codim=True`, we average instead of multiplying by codim (more stable for large codim).

    The MICAE loss uses R(z)^2 (squared scalar curvature) per sample: EIC(z) = R(z)^2.
    """
    assert z.dim() == 2, "z must be (B,m)"
    dev, dt = z.device, z.dtype
    B, m = z.shape

    # Build or reuse J, G, L
    if JGL is None:
        J, G, L = _build_J_and_G_batched(decode, z, lam)
    else:
        J, G, L = JGL

    # Ambient & codim
    with torch.no_grad():
        D = decode(z[:1]).reshape(1, -1).size(1)
    codim = max(0, int(D) - int(m))

    # Trivial intrinsic curvature cases
    if (m < 2) or (codim == 0):
        out = {"EIC": torch.zeros(B, device=dev, dtype=dt)}
        if return_aux:
            out.update({"J": J, "G": G, "L": L})
        return out

    # Normal projector
    with torch.no_grad():
        N = _build_normal_projector(J, G, L)  # (B,D,D)

    # Sample normals in the normal space
    if use_rademacher:
        Xi = _sample_rademacher((R_a, B, D), dev, dt)
    else:
        Xi = torch.randn(R_a, B, D, device=dev, dtype=dt)
    ref = torch.zeros(B, D, device=dev, dtype=dt); ref[:, 0] = 1.0
    A = _project_and_normalize(N, Xi, ref=ref)  # (R_a,B,D)

    # Second fundamental forms and shape operators
    II_all = _II_multi_in_normals(decode, z, A, use_exact_hessian, fd_eps)  # (R_a,B,m,m)
    L_exp  = L.unsqueeze(0).expand(II_all.shape[0], -1, -1, -1)             # (R_a,B,m,m)
    Sa_all = torch.cholesky_solve(II_all, L_exp)                             # (R_a,B,m,m)
    Sa_all = 0.5 * (Sa_all + Sa_all.transpose(-1, -2))

    # Hutchinson estimate of scalar curvature
    trSa  = Sa_all.diagonal(dim1=-1, dim2=-2).sum(-1)        # (R_a,B)
    fro2  = (Sa_all * Sa_all).sum(dim=(-1, -2))              # (R_a,B)
    est   = (trSa.pow(2) - fro2).mean(dim=0)                 # (B,)
    if normalize_codim:
        R_est = est                                          # average over normals
    else:
        R_est = codim * est                                  # sum over orthonormal normals

    EIC   = R_est.pow(2)                                     # squared scalar curvature (B,)

    out = {"EIC": EIC}
    if return_aux:
        out.update({"J": J, "G": G, "L": L})
    return out
