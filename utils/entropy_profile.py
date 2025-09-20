from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, Optional
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ───────────────────────── helpers ─────────────────────────

def _t_like(x: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
    return t_scalar.expand(x.shape[0]) if t_scalar.ndim == 0 else t_scalar

def _prod(shape) -> int:
    p = 1
    for s in shape: p *= int(s)
    return p

def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out

def _split_batch(b):
    if isinstance(b, (tuple, list)):
        if len(b) == 1: return b[0], None
        return b[0], b[1]
    return b, None

def _g2_at(sde, x_probe: torch.Tensor, t: torch.Tensor) -> float:
    try:
        _, g = sde.sde(x_probe, t, return_f=True)
    except TypeError:
        _, g = sde.sde(x_probe, t)
    return float((g**2).reshape(-1)[0].item())

def _sigma_marg_at(sde, x_probe: torch.Tensor, t: torch.Tensor) -> float:
    std = sde.marginal_prob(x_probe, t)[1]
    return float(std.reshape(-1)[0].item())

def _edm_pack(sde, t_scalar: torch.Tensor, x_probe: torch.Tensor) -> tuple[float,float,float,float]:
    """
    Return (s_edm, sigma_edm, sigma_dot_edm, sigma_marginal) at time t.
    Uses analytic dσ/dt for VPSDE; numeric fallback otherwise.
    """
    if not hasattr(sde, "edm_coefficients"):
        # fall back: use marginal sigma as EDM sigma if not provided
        s_edm = 1.0
        sigma_edm = _sigma_marg_at(sde, x_probe, t_scalar)
        sigma_marg = sigma_edm
        h = max(1e-4, 5e-3 * float(t_scalar.item()))
        tp = torch.tensor(min(1.0, float(t_scalar.item()) + h), device=t_scalar.device)
        tm = torch.tensor(max(0.0, float(t_scalar.item()) - h), device=t_scalar.device)
        sp = _sigma_marg_at(sde, x_probe, tp)
        sm = _sigma_marg_at(sde, x_probe, tm)
        sigma_dot = (sp - sm) / max(1e-12, float(tp.item() - tm.item()))
        return s_edm, sigma_edm, sigma_dot, sigma_marg

    s_t, sig_t = sde.edm_coefficients(t_scalar)  # [B] or scalar
    s_edm     = float(s_t.reshape(-1)[0].item())
    sigma_edm = float(sig_t.reshape(-1)[0].item())

    # analytic dσ/dt for VPSDE
    sigma_dot = None
    if type(sde).__name__.lower().startswith("vpsde"):
        b0 = float(sde.beta_0); bd = float(sde.beta_1 - sde.beta_0)
        t  = float(t_scalar.item())
        exp_term = np.exp(0.5*bd*t*t + b0*t)
        # σ = sqrt(exp_term - 1);  σ' = (exp_term*(bd*t + b0)) / (2σ)
        sigma_dot = 0.0 if sigma_edm <= 0 else 0.5 * exp_term * (bd*t + b0) / sigma_edm
    if sigma_dot is None:
        h = max(1e-4, 5e-3 * float(t_scalar.item()))
        tp = torch.tensor(min(1.0, float(t_scalar.item()) + h), device=t_scalar.device)
        tm = torch.tensor(max(0.0, float(t_scalar.item()) - h), device=t_scalar.device)
        _, sp = sde.edm_coefficients(tp)
        _, sm = sde.edm_coefficients(tm)
        sigma_dot = float((sp - sm).reshape(-1)[0].item()) / max(1e-12, float(tp.item() - tm.item()))

    sigma_marg = _sigma_marg_at(sde, x_probe, t_scalar)
    return s_edm, sigma_edm, sigma_dot, sigma_marg


# ─────────────────────── main computation ───────────────────────

@torch.no_grad()
def compute_entropy_profile(
    *,
    model: torch.nn.Module,
    sde: Any,
    loader: Iterable,
    orig_shape: Tuple[int, ...],
    device: torch.device,
    t_min: float | None = None,
    t_max: float = 1.0,
    num_t: int = 96,
    max_batches: Optional[int] = 12,
    save_dir: Optional[str] = None,
    filename_prefix: str = "entropy_profile",
    progress: bool = True,
    sigma_floor: float = 1e-6,      # avoid σ≈0 blow-ups
    enforce_nonneg: bool = True,    # clip tiny negatives due to noise
) -> Dict[str, Any]:
    """
    Computes the conditional-entropy rate in two ways:

      (1) **MMSE / EDM (paper Eq. in your screenshot; λ=1)**
          Ḣ = (σ̇_edm / σ_edm^3) · ε²,
          where ε² = E[ || x0 − x̂0(xt) ||² ] and x̂0 is obtained from
          the model's *denoiser function*.

      (2) **Score / Song**
          Ḣ = g(t)^2 [ D/σ_marg(t)^2 − E||score_t(xt)||² ].

    Both are returned along with their *rescaled* counterparts σ·Ḣ and
    the cumulative entropic times Φ(t) = ∫ Ḣ and  Φ̃(t) = ∫ σ·Ḣ.
    """
    model.eval()
    D = _prod(orig_shape)

    # build model callables (must exist in your repo)
    score_fn   = model.get_score_fn(sde)     # (xt, y, t[B]) -> score like xt
    denoiser   = model.get_denoiser_fn(sde)  # (xt, y, t[B]) -> x0_hat

    # time grid (log spacing improves readability near t≈0)
    eps = getattr(sde, "sampling_eps", 1e-5)
    if t_min is None: t_min = max(1e-4, float(eps))
    assert 0 < t_min < t_max <= 1.0
    t_grid = np.geomspace(t_min, t_max, num_t)

    # cache a few batches once
    cached: list[tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    it = tqdm(loader, desc="Prefetch", leave=False) if progress else loader
    for k, batch in enumerate(it):
        x0, y = _split_batch(batch)
        cached.append((x0.cpu(), None if y is None else y.cpu()))
        if max_batches is not None and (k + 1) >= max_batches:
            break
    if not cached:
        raise RuntimeError("Empty dataloader.")

    x0_probe = cached[0][0][:1].to(device)

    # accumulators
    Hdot_score, Hdot_resc_score = [], []
    Hdot_mmse,  Hdot_resc_mmse  = [], []
    E_score_sq, g2s = [], []
    sigma_marg, sigma_edm, s_edm = [], [], []

    # main loop
    for t in tqdm(t_grid, desc="Entropy profile (t)", disable=not progress):
        t_sc = torch.tensor(float(t), device=device)

        sE, sEDM, sEDM_dot, sM = _edm_pack(sde, t_sc, x0_probe)
        s_edm.append(sE); sigma_edm.append(sEDM); sigma_marg.append(sM)

        g2 = _g2_at(sde, x0_probe, t_sc)
        g2s.append(g2)

        # batch expectations at fixed t
        sum_s2, sum_eps2, n = 0.0, 0.0, 0
        for x0_cpu, y_cpu in (tqdm(cached, leave=False, disable=not progress,
                                   desc=f"batches@t={t:.4g}") if progress else cached):
            x0 = x0_cpu.to(device, non_blocking=True).float()
            y  = None if y_cpu is None else y_cpu.to(device, non_blocking=True)
            tb = _t_like(x0, t_sc)

            # sample x_t ~ N(s(t)x0, σ(t)^2 I) via the *marginal kernel*
            mean, std = sde.marginal_prob(x0, tb)              # mean like x0, std [B]
            noise = torch.randn_like(x0)
            xt = mean + std[(...,) + (None,)*(x0.ndim-1)] * noise

            # score path
            sc = score_fn(xt, y, tb)
            sum_s2 += sc.flatten(1).square().sum(1).mean().item()

            # MMSE path (use *model denoiser*)
            xhat = denoiser(xt, y, tb)
            sum_eps2 += (xhat - x0).flatten(1).square().sum(1).mean().item()
            n += 1

        E_s2   = sum_s2 / max(n, 1)
        eps2_t = sum_eps2 / max(n, 1)
        E_score_sq.append(E_s2)

        # (2) score/Song estimator (uses marginal σ and g² from the SDE)
        h_score = g2 * (D / (sM**2) - E_s2)
        if enforce_nonneg and h_score < 0: h_score = 0.0
        Hdot_score.append(h_score)
        Hdot_resc_score.append(sM * h_score)

        # (1) MMSE/EDM estimator (λ=1): Ḣ = (σ̇_edm / σ_edm^3) · ε²
        if sEDM <= sigma_floor:
            h_mmse = np.nan
        else:
            h_mmse = (sEDM_dot / (sEDM**3)) * eps2_t
            if enforce_nonneg and h_mmse < 0: h_mmse = 0.0
        Hdot_mmse.append(h_mmse)
        Hdot_resc_mmse.append(np.nan if np.isnan(h_mmse) else sEDM * h_mmse)

    # vectorize + integrate
    t = np.asarray(t_grid, dtype=np.float64)

    def _pack_and_integrate(h: list[float], resc: list[float]):
        h = np.asarray(h, dtype=np.float64)
        resc = np.asarray(resc, dtype=np.float64)
        m = np.isfinite(h)
        tt = t[m]; hh = h[m]; rr = resc[m]
        return h, resc, tt, _cumtrapz(hh, tt), _cumtrapz(rr, tt), m

    Hs, HRs, ts, Phi_s, PhiR_s, m_s = _pack_and_integrate(Hdot_score, Hdot_resc_score)
    Hm, HRm, tm, Phi_m, PhiR_m, m_m = _pack_and_integrate(Hdot_mmse,  Hdot_resc_mmse)

    # discrepancy (common valid region)
    common = m_s & m_m
    if np.any(common):
        a, b = Hs[common], Hm[common]
        rel_L2  = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
        max_rel = float(np.nanmax(np.abs((a - b) / (np.abs(a) + 1e-12))))
        bias    = float(np.nanmean(b - a))
    else:
        rel_L2 = max_rel = bias = float("nan")

    out: Dict[str, Any] = {
        "t": t,
        "sigma_marg": np.asarray(sigma_marg, dtype=np.float64),
        "sigma_edm":  np.asarray(sigma_edm,  dtype=np.float64),
        "s_edm":      np.asarray(s_edm,      dtype=np.float64),
        "g2":         np.asarray(g2s,        dtype=np.float64),

        "Hdot_score": Hs,           "Hdot_rescaled_score": HRs,
        "Phi_score_t": ts,          "Phi_score": Phi_s,            "Phi_rescaled_score": PhiR_s,

        "Hdot_mmse":  Hm,           "Hdot_rescaled_mmse": HRm,
        "Phi_mmse_t": tm,           "Phi_mmse": Phi_m,             "Phi_rescaled_mmse": PhiR_m,

        "discrepancy": {"rel_L2": rel_L2, "max_rel_err": max_rel, "signed_bias": bias},
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_dir, f"{filename_prefix}.npz"), **out)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # entropy rates (log-x so early-time structure is readable)
        #ax[0].plot(t, Hs, label=r"$\dot H$ (score/Song)")
        #ax[0].plot(t, Hm, ":", label=r"$\dot H$ (MMSE/EDM)")
        ax[0].plot(t, HRs, "--", label=r"$\sigma_{\mathrm{Song}}\dot H$")
        ax[0].plot(t, HRm, "-.", label=r"$\sigma_{\mathrm{EDM}}\dot H$")
        ax[0].set_xscale("log"); ax[0].grid(True, ls=":")
        ax[0].set_xlabel("t (log)"); ax[0].set_title("Entropy rate"); ax[0].legend()

        # cumulative entropic times
        #ax[1].plot(ts, Phi_s, label=r"$\Phi$ (score)")
        #ax[1].plot(tm, Phi_m, ":", label=r"$\Phi$ (MMSE)")
        ax[1].plot(ts, PhiR_s, "--", label=r"$\tilde{\Phi}$ (score)")
        ax[1].plot(tm, PhiR_m, "-.", label=r"$\tilde{\Phi}$ (MMSE)")
        ax[1].set_xscale("log"); ax[1].grid(True, ls=":")
        ax[1].set_xlabel("t (log)"); ax[1].set_title("Cumulative entropic times"); ax[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{filename_prefix}.png"), dpi=220)
        plt.close(fig)

    return out


def schedule_from_rescaled_entropic_time(
    t: np.ndarray,
    Phi_rescaled: np.ndarray,
    n_stages: int,
    t_lo: float | None = None,
    t_hi: float | None = None,
    descending: bool = True,
) -> list[float]:
    """Return n_stages times uniformly spaced in rescaled entropic time,
    INCLUDING both endpoints t_lo and t_hi."""
    assert t.ndim == 1 and Phi_rescaled.ndim == 1 and t.shape == Phi_rescaled.shape

    # sort by t if needed
    if not np.all(np.diff(t) > 0):
        idx = np.argsort(t)
        t, Phi_rescaled = t[idx], Phi_rescaled[idx]

    # clip to [t_lo, t_hi]
    if t_lo is None: t_lo = float(t[0])
    if t_hi is None: t_hi = float(t[-1])
    m = (t >= t_lo) & (t <= t_hi)
    tt, phi = t[m], Phi_rescaled[m]

    # normalize Φ̃ to [0,1]
    denom = float(phi[-1] - phi[0]) if len(phi) > 1 else 1.0
    phi = (phi - phi[0]) / max(1e-12, denom)

    # n_stages points including both endpoints (0 and 1)
    targets = np.linspace(0.0, 1.0, int(n_stages))
    sched = np.interp(targets, phi, tt)  # [t_lo, ..., t_hi]
    if descending:
        sched = sched[::-1]
    return [float(x) for x in sched]