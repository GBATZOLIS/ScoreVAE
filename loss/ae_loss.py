# loss/ae_loss.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
from contextlib import contextmanager, nullcontext
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch._dynamo as dynamo
except Exception:
    dynamo = None

from .isometry import (
    encoder_isometry_regularisation,
    decoder_isometry_regularisation,
)
from .curvature import (
    mecae_extrinsic_decoder,
    micae_intrinsic_decoder,
    metric_smoothness_decoder_Ginv_fast,
    metric_smoothness_latent_score_Ginv_fd,  # NEW (you add this in loss/curvature.py)
)


@contextmanager
def freeze_params(module: Optional[nn.Module]):
    """Temporarily sets a module to eval and disables gradients on its params."""
    if module is None:
        yield
        return
    was_training = module.training
    prev_reqgrad = [p.requires_grad for p in module.parameters()]
    try:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        module.train(was_training)
        for p, rg in zip(module.parameters(), prev_reqgrad):
            p.requires_grad_(rg)


def _guess_decoder_module(model: nn.Module) -> Optional[nn.Module]:
    """Best-effort to guess a decoder submodule (used to freeze when targeting encoder)."""
    for name in ["decoder", "dec", "decode_net", "generator", "g", "dec_block", "decode_block", "decoder_net"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                return mod
    decode_attr = getattr(model, "decode", None)
    if callable(decode_attr) and hasattr(decode_attr, "__self__"):
        bound_self = getattr(decode_attr, "__self__")
        if isinstance(bound_self, nn.Module) and bound_self is not model:
            return bound_self

    warnings.warn(
        "[ae_loss] Could not infer a decoder submodule to freeze; "
        "curvature target='encoder' will not freeze decoder params.",
        RuntimeWarning,
    )
    return None


def _get_latent_t_star(cfg, default: float = 0.05) -> float:
    """Pull t_value from cfg.loss.geom.latent.t_value if present."""
    try:
        geom = getattr(cfg.loss, "geom", None)
        lat = getattr(geom, "latent", None) if geom is not None else None
        t = getattr(lat, "t_value", default) if lat is not None else default
        return float(t)
    except Exception:
        return float(default)


def ae_loss(
    model: nn.Module,
    batch,
    cfg,
    device: torch.device,
    train: bool = True,
    *,
    # NEW: needed for latent score-metric smoothness (encoder gets gradients; latent params stay frozen)
    latent_model: Optional[nn.Module] = None,
    latent_ema=None,      # EMA wrapper (optional), must support apply_shadow()/restore()
    latent_sde=None,      # latent SDE (required for score-metric smoothness)
) -> Tuple[torch.Tensor, Dict[str, Optional[float]]]:
    """
    Autoencoder loss with optional:
      - isometry (encoder/decoder)
      - MECAE (extrinsic curvature)
      - MICAE (intrinsic via Gauss)
      - decoder metric smoothness (G^{-1}-normalised)
      - NEW: latent score-metric smoothness based on g_score = J_s^T J_s

    NOTE: In your new config you will set all other weights to 0
          and only keep reconstruction + score_metric_smoothness.
    """
    x = batch[0].to(device)

    # ---------------- reconstruction ----------------
    use_vae = bool(getattr(getattr(cfg, "model", None), "vae", {}).get("enabled", False))
    x_hat, _ = model(x)
    rec_type = str(getattr(cfg.loss, "reconstruction", "mse")).lower()
    rec = F.mse_loss(x_hat, x) if rec_type == "mse" else F.l1_loss(x_hat, x)

    # ---------------- KL (VAE) ----------------
    kl = torch.tensor(0.0, device=device)
    if use_vae:
        mu, logvar = getattr(model, "_last_mu"), getattr(model, "_last_logvar")
        prior_logvar = torch.zeros_like(mu) if getattr(model, "prior_logvar", None) is None \
            else model.prior_logvar.view(1, -1).expand_as(mu)
        s2, sp2 = logvar.exp(), prior_logvar.exp()
        kl = (0.5 * (prior_logvar - logvar - 1.0 + (s2 + mu.pow(2)) / sp2)).sum(dim=1).mean()
    beta_kl = float(getattr(cfg.loss, "beta_kl", 1.0))

    # ---------------- isometry ----------------
    enc_iso_w = float(getattr(cfg.loss, "enc_iso_weight", 0.0))
    dec_iso_w = float(getattr(cfg.loss, "dec_iso_weight", 0.0))
    num_v     = int(getattr(cfg.loss, "num_v", 2))

    enc_iso = torch.tensor(0.0, device=device)
    if enc_iso_w > 0.0:
        enc_iso = encoder_isometry_regularisation(model.encode, x, num_v=num_v, train=train, device=device)

    dec_iso = torch.tensor(0.0, device=device)
    if dec_iso_w > 0.0:
        z_for_dec = model.encode(x).detach()
        dec_iso = decoder_isometry_regularisation(
            model.decode, z_for_dec, num_v=int(getattr(cfg.loss, "dec_num_v", num_v)),
            train=train, device=device
        )

    total = rec + beta_kl * kl + enc_iso_w * enc_iso + dec_iso_w * dec_iso

    # ---------------- curvature / smoothness weights ----------------
    curv_w = float(getattr(cfg.loss, "curvature_weight", 0.0))
    intr_w = float(getattr(cfg.loss, "intrinsic_weight", 0.0))
    ms_w   = float(getattr(cfg.loss, "metric_smooth_weight", 0.0))

    # NEW: latent score-metric smoothness weight
    score_ms_w   = float(getattr(cfg.loss, "score_metric_smooth_weight", 0.0))

    # outputs (logged as scalars)
    me_iter_val: Optional[float] = None
    mi_iter_val: Optional[float] = None
    ms_iter_val: Optional[float] = None
    score_ms_iter_val: Optional[float] = None

    metrics: Dict[str, Optional[float]] = {}

    # step-based gating (shared counter)
    step_attr = "_curv_step"
    step = int(getattr(model, step_attr, 0))

    curv_cfg = getattr(cfg.loss, "curvature", {}) or {}
    intr_cfg = getattr(cfg.loss, "intrinsic", {}) or {}
    ms_cfg   = getattr(cfg.loss, "metric_smoothness", {}) or {}
    score_ms_cfg = getattr(cfg.loss, "score_metric_smoothness", {}) or {}

    every_n = int(getattr(curv_cfg, "every_n_steps", 1))
    do_curv = train and (curv_w > 0.0) and (every_n <= 1 or (step % max(every_n, 1) == 0))

    intr_every = int(getattr(intr_cfg, "every_n_steps", every_n))
    do_intr = train and (intr_w > 0.0) and (intr_every <= 1 or (step % max(intr_every, 1) == 0))

    ms_every = int(getattr(ms_cfg, "every_n_steps", every_n))
    do_ms = train and (ms_w > 0.0) and (ms_every <= 1 or (step % max(ms_every, 1) == 0))

    score_ms_every = int(getattr(score_ms_cfg, "every_n_steps", every_n))
    do_score_ms = (
        train
        and (score_ms_w > 0.0)
        and (latent_model is not None)
        and (latent_sde is not None)
        and (score_ms_every <= 1 or (step % max(score_ms_every, 1) == 0))
    )

    # optional warmup gate for score-ms
    start_after = int(getattr(score_ms_cfg, "start_after_steps", 0))
    if do_score_ms and step < start_after:
        do_score_ms = False

    # ---------------- shared freeze policy for decoder-based regs ----------------
    # (kept for compatibility; your new config sets those weights to 0 anyway)
    target = str(getattr(curv_cfg, "target", "encoder")).lower()

    # heavy geometry terms should run in fp32 without autocast
    if do_curv or do_intr or do_ms or do_score_ms:
        if dynamo is not None:
            try:
                dynamo.graph_break()
            except Exception:
                pass

        # Choose freeze policy for decoder-based terms
        # Score-ms targets the encoder, so we do NOT want to freeze encoder;
        # it needs grads. Decoder-freezing only matters for decoder-based regs.
        if target == "encoder":
            dec_mod = _guess_decoder_module(model)
            freeze_ctx = freeze_params(dec_mod)
        else:
            freeze_ctx = nullcontext()

        with torch.amp.autocast("cuda", enabled=False), freeze_ctx:
            # ---- latent codes in fp32
            # We recompute encode here in fp32 (autocast disabled) for stability.
            z_full32 = model.encode(x).to(torch.float32)

            # ---- sub-batch for decoder-based geometry regs
            B = z_full32.size(0)
            B_curv = int(getattr(curv_cfg, "B_curv", 8))
            B_curv = max(1, min(B_curv, B))
            idx = torch.randperm(B, device=z_full32.device)[:B_curv] if (B_curv < B) \
                else torch.arange(B, device=z_full32.device)
            z_curv = z_full32[idx]

            JGL = None

            # ---- MECAE (extrinsic) ----
            if do_curv:
                me_out = mecae_extrinsic_decoder(
                    decode=model.decode, z=z_curv,
                    lam=float(getattr(curv_cfg, "reg_lambda", 1e-6)),
                    K_v=int(getattr(curv_cfg, "K_v", 1)),
                    K_w=int(getattr(curv_cfg, "K_w", 1)),
                    use_rademacher=bool(getattr(curv_cfg, "use_rademacher", True)),
                    estimator=str(getattr(curv_cfg, "estimator", "square")),
                    use_exact_hessian=bool(getattr(curv_cfg, "use_exact_hessian", True)),
                    fd_eps=float(getattr(curv_cfg, "fd_eps", 1e-3)),
                    return_aux=True,
                )
                eec = me_out["EEC"].mean().to(x.dtype)
                total = total + curv_w * eec
                me_iter_val = float(eec.detach().item())
                JGL = (me_out["J"], me_out["G"], me_out["L"])

            # ---- MICAE (intrinsic) ----
            if do_intr:
                mi_out = micae_intrinsic_decoder(
                    decode=model.decode, z=z_curv,
                    lam=float(getattr(intr_cfg, "reg_lambda", getattr(curv_cfg, "reg_lambda", 1e-6))),
                    R_a=int(getattr(intr_cfg, "R_a", 2)),
                    use_rademacher=bool(getattr(intr_cfg, "use_rademacher", True)),
                    use_exact_hessian=bool(getattr(intr_cfg, "use_exact_hessian", True)),
                    fd_eps=float(getattr(intr_cfg, "fd_eps", 1e-3)),
                    JGL=JGL,
                    normalize_codim=bool(getattr(intr_cfg, "normalize_codim", True)),
                    return_aux=False,
                )
                eic = mi_out["EIC"].mean().to(x.dtype)
                total = total + intr_w * eic
                mi_iter_val = float(eic.detach().item())
                for k, v in mi_out.items():
                    if k == "EIC":
                        continue
                    if isinstance(v, torch.Tensor) and v.ndim == 0 and torch.isfinite(v):
                        metrics[f"reg/micae/{k}"] = float(v.detach().item())
                metrics["reg/micae/EIC_mean"] = mi_iter_val

            # ---- Decoder metric smoothness (G^{-1}-normalised) ----
            if do_ms:
                target_ms = str(getattr(ms_cfg, "target", target)).lower()
                if target_ms == "encoder":
                    dec_mod_ms = _guess_decoder_module(model)
                    freeze_ctx_ms = freeze_params(dec_mod_ms)
                    z_for_ms = z_full32[idx]
                elif target_ms == "both":
                    freeze_ctx_ms = nullcontext()
                    z_for_ms = z_full32[idx]
                else:  # "decoder"
                    freeze_ctx_ms = nullcontext()
                    z_for_ms = z_full32[idx].detach()

                with freeze_ctx_ms:
                    ms_out = metric_smoothness_decoder_Ginv_fast(
                        decode=model.decode,
                        z=z_for_ms,
                        K_w=int(getattr(ms_cfg, "K_w", 1)),
                        use_rademacher=bool(getattr(ms_cfg, "use_rademacher", True)),
                        use_exact_hessian=bool(getattr(ms_cfg, "use_exact_hessian", True)),
                        fd_eps=float(getattr(ms_cfg, "fd_eps", 1e-3)),
                        JGL=JGL,
                        normalize_by_dim=bool(getattr(ms_cfg, "normalize_by_dim", True)),
                    )
                    msm = ms_out["MSM_G"].mean().to(x.dtype)
                    total = total + ms_w * msm
                    ms_iter_val = float(msm.detach().item())

            # ---- NEW: Latent score-metric smoothness (encoder-targeted) ----
            if do_score_ms:
                # sub-batch size for score-ms (can be different from B_curv)
                B_score = int(getattr(score_ms_cfg, "B_score", 16))
                B_score = max(1, min(B_score, B))
                idx_score = torch.randperm(B, device=z_full32.device)[:B_score] if (B_score < B) \
                    else torch.arange(B, device=z_full32.device)
                z_score = z_full32[idx_score]  # IMPORTANT: NOT detached -> grads flow to encoder

                # Use latent EMA weights for the metric (recommended)
                use_lat_ema = bool(getattr(score_ms_cfg, "use_latent_ema_for_metric", True))
                t_star = _get_latent_t_star(cfg, default=0.05)

                # Freeze latent params, but keep grads wrt input z_score.
                # Also temporarily swap in EMA weights if requested.
                lat_mod = latent_model
                with freeze_params(lat_mod):
                    if use_lat_ema and latent_ema is not None:
                        latent_ema.apply_shadow()
                    try:
                        out = metric_smoothness_latent_score_Ginv_fd(
                            z0=z_score,
                            latent_model=lat_mod,
                            latent_sde=latent_sde,
                            t_value=float(t_star),
                            delta_std=float(getattr(score_ms_cfg, "delta_std", 1e-2)),
                            eps_metric=float(getattr(score_ms_cfg, "eps_metric", 1e-3)),
                        )
                        msm_score = out["MSM_score"].to(x.dtype)
                        total = total + score_ms_w * msm_score
                        score_ms_iter_val = float(msm_score.detach().item())
                    finally:
                        if use_lat_ema and latent_ema is not None:
                            latent_ema.restore()

    if train:
        setattr(model, step_attr, step + 1)

    # stem alpha (if present)
    stem_alpha = None
    try:
        stem_alpha = float(getattr(getattr(model, "decoder", None), "stem", None).alpha.item())
    except Exception:
        pass

    metrics.update({
        "loss/total":                float(total.detach().item()),
        "loss/reconstruction":       float(rec.detach().item()),
        "loss/kl":                   float(kl.detach().item()),
        "reg/enc_iso":               float(enc_iso.detach().item()),
        "reg/dec_iso":               float(dec_iso.detach().item()),
        "reg/mecae_eec":             me_iter_val,
        "reg/micae_eic":             mi_iter_val,
        "reg/metric_smoothness":     ms_iter_val,
        "reg/score_metric_smoothness": score_ms_iter_val,  # NEW
        "model/stem_alpha":          stem_alpha,
    })

    return total, metrics
