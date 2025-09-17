# loss/curvature.py
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from torch.func import jvp, vjp, vmap, grad
from torch.func import jacfwd, jacrev

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

def _build_J_and_G_batched(
    decode,
    z: Tensor,
    lam: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build per-sample Jacobians and pullback metrics in batch using jacfwd/jacrev.

    Heuristic:
      - If d <= D, use jacfwd (forward-mode): cost ~ O(d)
      - Else       use jacrev (reverse-mode): cost ~ O(D)
    """
    B, d = z.shape
    f_single = _as_single_map(decode)

    # Peek D once to choose the mode
    with torch.no_grad():
        D = decode(z[:1]).reshape(1, -1).size(1)

    # Choose forward- or reverse-mode Jacobian
    use_fwd = (d <= D)
    jac_fn = jacfwd(f_single) if use_fwd else jacrev(f_single)

    # J: (B, D, d)
    J = vmap(jac_fn)(z)

    JT = J.transpose(1, 2)               # (B, d, D)
    G = JT @ J                            # (B, d, d)
    if lam != 0.0:
        G = G + lam * torch.eye(d, device=z.device, dtype=z.dtype).expand(B, d, d)
    L = torch.linalg.cholesky(G)          # (B, d, d)
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

def _dir_dTv_many_w(
    decode,
    z: Tensor,          # (B,d)
    J: Tensor,          # (B,D,d)
    L: Tensor,          # (B,d,d)
    v: Tensor,          # (B,D)
    W: Tensor,          # (B,K_w,d)
    use_exact_hessian: bool = True,
    fd_eps: float = 1e-3,
) -> Tensor:
    """
    Vectorized version of _dir_dTv_batched over multiple latent directions.
    W: (B, K_w, d) -> returns (B, K_w, D)
    """
    return vmap(
        lambda w_single: _dir_dTv_batched(
            decode, z, J, L, v, w_single,
            use_exact_hessian=use_exact_hessian,
            fd_eps=fd_eps,
        ),
        in_dims=1,  # map over K_w
        out_dims=1
    )(W)

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

    # Precompute G^{-1/2} W via triangular solves: S = L^{-T} W
    U = L.transpose(1, 2)                               # (B,d,d) upper
    W_T = W.transpose(1, 2)                             # (B,d,K_w)
    S_T = _solve_upper_tri_batch(U, W_T)                # (B,d,K_w)
    S   = S_T.transpose(1, 2).contiguous()              # (B,K_w,d)

    if estimator == "square":
        # Vectorize across K_v and K_w:
        # For each kv: D_many = ∂(T v_kv)·(G^{-1/2} W)  -> (B,K_w,D)
        # Stack across kv -> (B,K_v,K_w,D)
        D_many_all = vmap(
            lambda v_row: _dir_dTv_many_w(
                decode, z, J, L, v_row, S,
                use_exact_hessian=use_exact_hessian,
                fd_eps=fd_eps,
            ),
            in_dims=1,  # map over K_v
            out_dims=1
        )(V)  # (B,K_v,K_w,D)

        # Sum ||·||^2 over D, then average over K_v * K_w
        eec_acc = (D_many_all * D_many_all).sum(dim=-1).sum(dim=-1).sum(dim=-2)  # (B,)
        eec = 0.5 * eec_acc / (K_v * K_w)

    elif estimator == "bilinear":
        # Compute Wtil = G^{-1} W  -> (B,K_w,d)
        Wtil_T = _chol_solve_batch(L, W_T)  # (B,d,K_w)
        Wtil   = Wtil_T.transpose(1, 2).contiguous()  # (B,K_w,d)

        def per_kv(v_row: Tensor) -> Tensor:
            # (B,K_w,D) each
            dW    = _dir_dTv_many_w(
                decode, z, J, L, v_row, W,
                use_exact_hessian=use_exact_hessian,
                fd_eps=fd_eps,
            )
            dWtil = _dir_dTv_many_w(
                decode, z, J, L, v_row, Wtil,
                use_exact_hessian=use_exact_hessian,
                fd_eps=fd_eps,
            )
            # ⟨d_w, d_{wtil}⟩ over D -> (B,K_w)
            return (dW * dWtil).sum(dim=-1)

        # Map over K_v -> (B,K_v,K_w)
        bil_terms = vmap(per_kv, in_dims=1, out_dims=1)(V)
        # Sum over K_w and K_v
        eec_acc = bil_terms.sum(dim=-1).sum(dim=-1)  # (B,)
        eec = 0.5 * eec_acc / (K_v * K_w)

    else:
        raise ValueError(f"Unknown estimator '{estimator}'. Use 'square' or 'bilinear'.")

    if return_aux:
        return {"EEC": eec, "J": J, "G": G, "L": L}
    else:
        return {"EEC": eec}

# ====================================================================================
# MICAE (intrinsic curvature via Gauss equation + Hutchinson)
# ====================================================================================

# Project Xi into the normal space without forming N
def _project_normals_functionally(J, L, Xi, ref=None, eps: float = 1e-8) -> Tensor:
    # J: (B, D, m), L: (B, m, m), Xi: (R_a, B, D)
    Xi_BDR = Xi.permute(1, 2, 0)                         # (B, D, R_a)

    # --- FIX 1: keep 'm' = latent dim, 'd' = ambient dim consistently ---
    JT_Xi  = torch.einsum('bmd,bdk->bmk', J.transpose(1, 2), Xi_BDR)  # (B, m, R_a)

    Y      = _chol_solve_batch(L, JT_Xi)                  # (B, m, R_a)

    # --- FIX 2: J is (B, D, m) so use 'bdm' here, not 'bmd' ---
    JY     = torch.einsum('bdm,bmk->bdk', J, Y)           # (B, D, R_a)

    A_BDR  = Xi_BDR - JY                                  # (B, D, R_a)
    A      = A_BDR.permute(2, 0, 1).contiguous()          # (R_a, B, D)

    if ref is not None:
        JT_ref = torch.einsum('bmd,bd->bm', J.transpose(1, 2), ref)   # (B, m)
        Y_ref  = _chol_solve_batch(L, JT_ref)                          # (B, m)
        # J is (B, D, m)  → use 'bdm'
        JY_ref = torch.einsum('bdm,bm->bd', J, Y_ref)                  # (B, D)
        pref   = ref - JY_ref                                          # (B, D)
        while pref.dim() < A.dim():
            pref = pref.unsqueeze(0)
        nrm = A.norm(dim=-1, keepdim=True)
        A   = torch.where(nrm > eps, A, A + pref)

    return A / (A.norm(dim=-1, keepdim=True) + eps)



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

    # Trivial intrinsic curvature cases: still emit debug scalars so logs exist
    if (m < 2) or (codim == 0):
        diagL = L.diagonal(dim1=-1, dim2=-2)  # (B,m)
        out = {
            "EIC": torch.zeros(B, device=dev, dtype=dt),
            "reg/G_min_diagL":    torch.as_tensor(diagL.min().detach().item(), device=dev),
            "reg/G_mean_diagL":   torch.as_tensor(diagL.mean().detach().item(), device=dev),
            "reg/micae_R_mean":       torch.tensor(0.0, device=dev),
            "reg/micae_R_absmean":    torch.tensor(0.0, device=dev),
            "reg/micae_trSa2_mean":   torch.tensor(0.0, device=dev),
            "reg/micae_fro2_mean":    torch.tensor(0.0, device=dev),
            "reg/micae_II_fro2_mean": torch.tensor(0.0, device=dev),
        }
        if return_aux:
            out.update({"J": J, "G": G, "L": L})
        return out

    # Probes for normals
    Xi = _sample_rademacher((R_a, B, D), dev, dt) if use_rademacher else \
         torch.randn(R_a, B, D, device=dev, dtype=dt)

    # Safety ref for projection
    ref = torch.zeros(B, D, device=dev, dtype=dt); ref[:, 0] = 1.0
    A = _project_normals_functionally(J, L, Xi, ref=ref)  # (R_a,B,D)

    # Second fundamental forms and shape operators
    II_all = _II_multi_in_normals(decode, z, A, use_exact_hessian, fd_eps)  # (R_a,B,m,m)
    L_exp  = L.unsqueeze(0).expand(II_all.shape[0], -1, -1, -1)             # (R_a,B,m,m)
    Sa_all = torch.cholesky_solve(II_all, L_exp)                             # (R_a,B,m,m)
    Sa_all = 0.5 * (Sa_all + Sa_all.transpose(-1, -2))

    # Hutchinson estimate of scalar curvature
    trSa  = Sa_all.diagonal(dim1=-1, dim2=-2).sum(-1)        # (R_a,B)
    fro2  = (Sa_all * Sa_all).sum(dim=(-1, -2))              # (R_a,B)
    est   = (trSa.pow(2) - fro2).mean(dim=0)                 # (B,)
    R_est = est if normalize_codim else codim * est          # (B,)
    EIC   = R_est.pow(2)                                     # (B,)

    # Debug scalars (as tensors so generic loggers can ingest)
    diagL = L.diagonal(dim1=-1, dim2=-2)                     # (B,m)
    R_mean        = R_est.mean()
    R_absmean     = R_est.abs().mean()
    trSa2_mean    = (trSa.pow(2)).mean()
    fro2_mean     = fro2.mean()
    II_fro2_mean  = (II_all * II_all).sum(dim=(-1, -2)).mean()

    out = {
        "EIC": EIC,
        "reg/G_min_diagL":    torch.as_tensor(diagL.min().detach().item(), device=dev),
        "reg/G_mean_diagL":   torch.as_tensor(diagL.mean().detach().item(), device=dev),
        "reg/micae_R_mean":       torch.as_tensor(R_mean.detach().item(), device=dev),
        "reg/micae_R_absmean":    torch.as_tensor(R_absmean.detach().item(), device=dev),
        "reg/micae_trSa2_mean":   torch.as_tensor(trSa2_mean.detach().item(), device=dev),
        "reg/micae_fro2_mean":    torch.as_tensor(fro2_mean.detach().item(), device=dev),
        "reg/micae_II_fro2_mean": torch.as_tensor(II_fro2_mean.detach().item(), device=dev),
    }
    if return_aux:
        out.update({"J": J, "G": G, "L": L})
    return out

