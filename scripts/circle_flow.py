#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, argparse, csv
import torch
import matplotlib.pyplot as plt

# ============================== VPSDE ==============================
class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000, device="cpu", dtype=torch.float64):
        self.beta_0 = float(beta_min)
        self.beta_1 = float(beta_max)
        self.N = int(N)
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.no_grad()
    def get_alpha_fn(self):
        def alpha_fn(t):
            t = t.to(self.device, self.dtype)
            log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            return torch.exp(log_mean_coeff)
        return alpha_fn

    @torch.no_grad()
    def get_sigma_fn(self):
        def sigma_fn(t):
            t = t.to(self.device, self.dtype)
            log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            val = 1.0 - torch.exp(2.0 * log_mean_coeff)
            return torch.sqrt(val.clamp_min(0.0))
        return sigma_fn

# ===================== Stable Bessel ratio I1/I0 ===================
def ratio_I1_over_I0(kappa: torch.Tensor, eps: float = 1e-15):
    # vmap/jacrev-safe, branchless I1/I0 via scaled Bessels; stable for all real kappa
    r = torch.special.i1e(kappa) / torch.special.i0e(kappa)
    return r.clamp(min=-(1.0 - eps), max=(1.0 - eps))

# ============== Closed-form circle score (with radius R) ===========
def score_circle_vpsde(
    x: torch.Tensor, a_t: torch.Tensor, sigma_t: torch.Tensor,
    radius: float = 1.0, sigma_scaled: bool = False, eps_sigma2: float = 1e-18
):
    B, d = x.shape
    assert d == 2, "This demo assumes 2D."
    R = torch.as_tensor(radius, dtype=x.dtype, device=x.device)
    r = x.norm(dim=1).clamp_min(1e-12)
    sigma2 = (sigma_t**2).clamp_min(eps_sigma2)
    kappa = (a_t * R * r) / sigma2
    ratio = ratio_I1_over_I0(kappa)
    coeff = (a_t * R * ratio / r) - 1.0
    s = (coeff / sigma2).unsqueeze(1) * x
    if sigma_scaled:
        s = sigma_t * s
    return s

# =========================== Geometry ==============================
def _finite_all(x): return torch.isfinite(x).all().item()

def _safe_cholesky_psd(G, *, abs_jitter=1e-5, rel_jitter=1e-2, max_tries=5):
    B, m, _ = G.shape
    I = torch.eye(m, device=G.device, dtype=G.dtype).expand_as(G)
    diag_mean = torch.diagonal(G, dim1=1, dim2=2).mean(dim=1).view(B,1,1)
    a, r = float(abs_jitter), float(rel_jitter)
    for _ in range(max_tries):
        G_ = 0.5*(G + G.transpose(1,2)) + a*I + r*diag_mean*I
        try:
            L = torch.linalg.cholesky(G_)
            if _finite_all(L): return L, a, r, G_
        except RuntimeError:
            pass
        a *= 10.0; r *= 10.0
    base = 0.5*(G + G.transpose(1,2))
    evals, Q = torch.linalg.eigh(base)
    floor = max(abs_jitter, rel_jitter * float(diag_mean.mean().item()))
    evals_clamped = evals.clamp_min(floor)
    G_repair = Q @ torch.diag_embed(evals_clamped) @ Q.transpose(1,2)
    L = torch.linalg.cholesky(G_repair)
    return L, a, r, G_repair

def _as_single_map(f_batched):
    def f_single(u1d: torch.Tensor) -> torch.Tensor:
        y = f_batched(u1d.unsqueeze(0))
        return y.reshape(1, -1).squeeze(0)
    return f_single

def build_JGL(s_fn, u, *, chol_abs=1e-5, chol_rel=1e-2, tikhonov=0.0):
    f_single = _as_single_map(s_fn)
    jac = torch.func.jacrev(f_single)
    J = torch.func.vmap(jac)(u)                       # (B,m,m)
    G = torch.einsum('bim,bjm->bij', J, J)            # (B,m,m)
    G = 0.5*(G + G.transpose(1,2))
    if tikhonov and tikhonov > 0.0:
        m = G.size(-1)
        I = torch.eye(m, device=G.device, dtype=G.dtype).expand_as(G)
        G = G + float(tikhonov) * I
    L, aj, rj, G_used = _safe_cholesky_psd(G, abs_jitter=chol_abs, rel_jitter=chol_rel)
    return J, G_used, L, aj, rj

def _solve_tri_upper(U, B_):
    add_dim = (B_.dim()==2)
    if add_dim: B_=B_.unsqueeze(-1)
    out = torch.linalg.solve_triangular(U, B_, upper=True)
    return out.squeeze(-1) if add_dim else out

# (DG[w]) v for pullback G=J^T J
def _dG_times_vec_single(f_single, ui, Ji, v, w):
    a = Ji @ v
    def g_local(z): return (a * f_single(z)).sum()
    grad_g = torch.func.grad(g_local)
    term1 = torch.func.jvp(grad_g, (ui,), (w,))[1]
    def jv_local(z): return torch.func.jvp(f_single, (z,), (v,))[1]
    dv = torch.func.jvp(jv_local, (ui,), (w,))[1]
    term2 = Ji.transpose(0,1) @ dv
    return term1 + term2

# ====== Q invariant (using precomputed J,G,L; modes "uv" and "vv") ======
def _sample_G_unit_vectors(L, K, *, device, dtype):
    B, m, _ = L.shape
    z = torch.randn(K, B, m, device=device, dtype=dtype)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    z_perm = z.permute(1,2,0)                 # (B,m,K)
    a_perm = _solve_tri_upper(L.transpose(1,2), z_perm)
    a = a_perm.permute(2,0,1).contiguous()    # (K,B,m)
    return a

def Q_given_JGL(s_fn, u, J, G, L, *, K_pairs=64, mode: str = "uv"):
    """
    mode == 'uv': Q_uv = E_{||u||_G=||v||_G=1} <H(u,v), G^{-1} H(u,v)>
    mode == 'vv': Q_vv = E_{||v||_G=1}         <H(v,v), G^{-1} H(v,v)>
    """
    f_single = _as_single_map(s_fn)
    A = _sample_G_unit_vectors(L, K_pairs, device=u.device, dtype=u.dtype)
    Bv = A if mode == "vv" else _sample_G_unit_vectors(L, K_pairs, device=u.device, dtype=u.dtype)

    def term_DG_times_vec_all(V, W):
        def body(v_row, w_row):
            return torch.func.vmap(_dG_times_vec_single, in_dims=(None,0,0,0,0))(f_single, u, J, v_row, w_row)
        return torch.func.vmap(body, in_dims=(0,0))(V, W)  # (K,B,m)

    DGuv = term_DG_times_vec_all(Bv, A)   # (DG[u]) v
    DGvu = term_DG_times_vec_all(A,  Bv)  # (DG[v]) u

    def grad_ab_inner_all(V, W):
        def single_pair(a_row, b_row):
            def grad_one(ui, ai, bi):
                def ab_inner(z):
                    Ja = torch.func.jvp(f_single, (z,), (ai,))[1]
                    Jb = torch.func.jvp(f_single, (z,), (bi,))[1]
                    return (Ja * Jb).sum()
                return torch.func.grad(ab_inner)(ui)
            return torch.func.vmap(grad_one, in_dims=(0,0,0))(u, a_row, b_row)
        return torch.func.vmap(single_pair, in_dims=(0,0))(V, W)

    DGdot = grad_ab_inner_all(A, Bv)  # (K,B,m)
    H = 0.5 * (DGuv + DGvu - DGdot)   # (K,B,m)

    RHS  = H.permute(1,2,0)           # (B,m,K)
    Vsol = torch.cholesky_solve(RHS, L)
    energy_per_pair = (Vsol * RHS).sum(dim=1)  # (B,K)
    return energy_per_pair.mean(dim=1)         # (B,)

# ===== Deterministic tangent-direction curvature estimator ========
def kappa2_tangent_estimator(s_fn, u, J, G, L):
    B, m = u.shape
    u = u.to(torch.float64)
    f_single = _as_single_map(s_fn)
    evals, evecs = torch.linalg.eigh(G)
    lam_t = evals[:, 0].clamp_min(1e-300)
    e_t   = evecs[:, :, 0]
    a = e_t / lam_t.sqrt().unsqueeze(-1)
    DGaa = torch.func.vmap(_dG_times_vec_single, in_dims=(None,0,0,0,0))(f_single, u, J, a, a)
    def grad_inner(ui, ai):
        def inner(z):
            Ja = torch.func.jvp(f_single, (z,), (ai,))[1]
            return (Ja * Ja).sum()
        return torch.func.grad(inner)(ui)
    DGdot = torch.func.vmap(grad_inner, in_dims=(0,0))(u, a)
    H    = DGaa - 0.5 * DGdot
    RHS  = H.unsqueeze(-1)
    Vsol = torch.cholesky_solve(RHS, L).squeeze(-1)
    return (Vsol * H).sum(dim=-1)  # (B,)

# ============== Sampling x_t on the perturbed circle ==============
@torch.no_grad()
def sample_xt_from_circle(B, a_t, sigma_t, radius, device, dtype=torch.float64):
    theta = torch.rand(B, device=device, dtype=dtype) * (2*math.pi)
    R = torch.as_tensor(radius, dtype=dtype, device=device)
    x0 = torch.stack([R*torch.cos(theta), R*torch.sin(theta)], dim=1)
    z  = torch.randn_like(x0)
    x_t = a_t.unsqueeze(1) * x0 if a_t.ndim == 1 else a_t * x0
    noise = sigma_t.unsqueeze(1) * z if sigma_t.ndim == 1 else sigma_t * z
    return x_t + noise

# ============================ Plotting ============================
def save_lineplot(x, ys, labels, title, out_path, xlabel="time t", ylabel="value"):
    plt.figure(figsize=(6.8,4.4))
    for y, lab in zip(ys, labels):
        plt.plot(x, y, marker='o', linewidth=1.8, markersize=3.5, label=lab)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(True, ls='--', alpha=0.35)
    if len(labels) > 1: plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

# ================================ Main ============================
def main():
    parser = argparse.ArgumentParser("Invariant Γ-magnitude sweep (both Q_uv & Q_vv) on radius-R circle; robust near t->0")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save plots/CSV.")
    parser.add_argument("--radius", type=float, default=1.0, help="Circle radius R (>0).")
    parser.add_argument("--sigma_scaled", action="store_true", help="Use σ*score (noise field).")
    parser.add_argument("--tikhonov", type=float, default=0.0, help="Tikhonov λ for G: add λI.")
    parser.add_argument("--batch", type=int, default=8192, help="Samples per t (increases stability).")
    parser.add_argument("--Kpairs", type=int, default=64, help="(u,v) pairs for MC estimates.")
    parser.add_argument("--times", type=str,
                        default="0.0025,0.003,0.004,0.005,0.006,0.008,0.010,0.020,0.030,0.050,0.100,0.150,0.200,0.300,0.400,0.500,0.600,0.700,0.800",
                        help="Comma-separated diffusion times.")
    parser.add_argument("--eps_sigma2", type=float, default=1e-18, help="Floor for sigma(t)^2 (bump to 1e-16 if needed).")
    parser.add_argument("--chol_abs", type=float, default=1e-5, help="Absolute jitter for Cholesky.")
    parser.add_argument("--chol_rel", type=float, default=1e-2, help="Relative jitter for Cholesky (× mean diag).")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir); os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)

    sde = VPSDE(beta_min=0.1, beta_max=20.0, N=1000, device=device, dtype=torch.float64)
    alpha_fn = sde.get_alpha_fn(); sigma_fn = sde.get_sigma_fn()

    t_list = [float(t.strip()) for t in args.times.split(",") if t.strip()]
    B = int(args.batch); Kpairs = int(args.Kpairs)
    R = float(args.radius)
    use_sigma_scaled = bool(args.sigma_scaled)
    tikh = float(args.tikhonov)
    eps_sigma2 = float(args.eps_sigma2)
    chol_abs = float(args.chol_abs); chol_rel = float(args.chol_rel)

    # Targets (m = 2 here; keep general just in case)
    m = 2
    target_Q_uv = (1.0 / (R**2)) / (m**2)                 # → as t→0
    target_Q_vv = (3.0 / (m*(m+2))) * (1.0 / (R**2))      # → as t→0

    # containers
    Quv_means, Qvv_means = [], []
    Quv_norm_means, Qvv_norm_means, kappa2_tan_norm_means = [], [], []
    trG_means, lam_min_means, lam_max_means, cond_means = [], [], [], []
    jitter_abs, jitter_rel = [], []

    print(f"Saving outputs to: {out_dir}")
    print(f"Radius R = {R}")
    print(f"Targets as t→0:  Q_uv → {target_Q_uv:.6g}  ,  Q_vv → {target_Q_vv:.6g}")
    print(f"Rescaled (should → 1):  uv_norm = m^2·Q_uv·R^2  ,  vv_norm = (m(m+2)/3)·Q_vv·R^2  ,  tan_norm = tan·R^2")
    header = (
        f"{'t':>7} | {'Q_uv':>12} {'Q_vv':>12} || {'uv_norm':>9} {'vv_norm':>9} {'tan_norm':>9} "
        f"|| {'tr(G)':>10} {'lam_min':>10} {'lam_max':>10} {'cond':>9} | jitter(abs,rel)"
    )
    print("-"*len(header))
    print(header)
    print("-"*len(header))

    for t in t_list:
        t_vec = torch.full((B,), float(t), device=device, dtype=torch.float64)
        a_t   = alpha_fn(t_vec)
        sig_t = sigma_fn(t_vec)
        x_t   = sample_xt_from_circle(B, a_t, sig_t, R, device, torch.float64)

        a_s   = a_t[0]; sig_s = sig_t[0]
        def s_fn_batch(x):
            return score_circle_vpsde(x, a_s, sig_s, radius=R, sigma_scaled=use_sigma_scaled, eps_sigma2=eps_sigma2)

        with torch.autocast(device_type=('cuda' if device=='cuda' else 'cpu'), enabled=False):
            # Precompute geometry once
            J, G, L, aj, rj = build_JGL(s_fn_batch, x_t, tikhonov=tikh, chol_abs=chol_abs, chol_rel=chol_rel)

            # Q_uv and Q_vv (reuse J,G,L)
            Q_uv_vals = Q_given_JGL(s_fn_batch, x_t, J, G, L, K_pairs=Kpairs, mode="uv")
            Q_vv_vals = Q_given_JGL(s_fn_batch, x_t, J, G, L, K_pairs=Kpairs, mode="vv")

            # Tangent deterministic estimator
            kappa2_tan_vals = kappa2_tangent_estimator(s_fn_batch, x_t, J, G, L)

            # diag
            evals = torch.linalg.eigvalsh(G)
            lam_min = evals[:,0].clamp_min(0.0)
            lam_max = evals[:,1].clamp_min(0.0)
            trG     = evals.sum(dim=-1)
            cond    = lam_max / lam_min.clamp_min(1e-38)

        Q_uv_m  = float(Q_uv_vals.mean().item());  Quv_means.append(Q_uv_m)
        Q_vv_m  = float(Q_vv_vals.mean().item());  Qvv_means.append(Q_vv_m)

        # Rescaled (→ 1)
        Q_uv_norm  = (m**2) * Q_uv_m * (R**2)
        Q_vv_norm  = (m*(m+2)/3.0) * Q_vv_m * (R**2)
        tan_norm   = float(kappa2_tan_vals.mean().item()) * (R**2)

        Quv_norm_means.append(Q_uv_norm)
        Qvv_norm_means.append(Q_vv_norm)
        kappa2_tan_norm_means.append(tan_norm)

        trG_m  = float(trG.mean().item());  trG_means.append(trG_m)
        lamn_m = float(lam_min.mean().item()); lam_min_means.append(lamn_m)
        lamx_m = float(lam_max.mean().item()); lam_max_means.append(lamx_m)
        cond_m = float(cond.mean().item()); cond_means.append(cond_m)
        jitter_abs.append(aj); jitter_rel.append(rj)

        print(
            f"{t:7.4f} | {Q_uv_m:12.6e} {Q_vv_m:12.6e} || {Q_uv_norm:9.3f} {Q_vv_norm:9.3f} {tan_norm:9.3f} "
            f"|| {trG_m:10.4e} {lamn_m:10.4e} {lamx_m:10.4e} {cond_m:9.3e} | ({aj:.1e},{rj:.1e})"
        )

    # ---- save CSV
    csv_path = os.path.join(out_dir, "circle_Q_both_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t","Q_uv","Q_vv","uv_norm","vv_norm","tan_norm",
            "traceG","lam_min","lam_max","cond","jitter_abs","jitter_rel",
            "radius","batch","Kpairs","eps_sigma2","chol_abs","chol_rel","sigma_scaled","tikhonov",
            "target_Q_uv","target_Q_vv"
        ])
        for i, t in enumerate(t_list):
            w.writerow([
                t, Quv_means[i], Qvv_means[i], Quv_norm_means[i], Qvv_norm_means[i], kappa2_tan_norm_means[i],
                trG_means[i], lam_min_means[i], lam_max_means[i], cond_means[i], jitter_abs[i], jitter_rel[i],
                R, B, Kpairs, eps_sigma2, chol_abs, chol_rel, int(use_sigma_scaled), tikh,
                target_Q_uv, target_Q_vv
            ])
    print(f"[Saved] {csv_path}")

    # ---- plots
    save_lineplot(
        t_list, [Quv_means, Qvv_means], ["Q_uv", "Q_vv"],
        title=f"Raw objectives vs t (R={R})",
        out_path=os.path.join(out_dir, "Q_raw_vs_t.png"),
        ylabel=r"$\mathbb{E}[\langle H, G^{-1}H\rangle]$",
    )
    save_lineplot(
        t_list, [Quv_norm_means, Qvv_norm_means, kappa2_tan_norm_means],
        [r"$m^2\cdot Q_{uv}\cdot R^2$", r"$\frac{m(m+2)}{3}Q_{vv}\cdot R^2$", r"$\text{tan}\cdot R^2$"],
        title=f"Rescaled estimates (→ 1) vs t (R={R})",
        out_path=os.path.join(out_dir, "Q_rescaled_vs_t.png"),
        ylabel="normalized to 1",
    )
    save_lineplot(
        t_list, [trG_means, lam_min_means, lam_max_means, cond_means],
        ["trace(G)", "λ_min(G)", "λ_max(G)", "cond(G)"],
        title=f"G diagnostics vs t (R={R})",
        out_path=os.path.join(out_dir, "diag_G_vs_t.png"),
        ylabel="value",
    )

if __name__ == "__main__":
    main()
