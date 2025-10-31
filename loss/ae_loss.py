from __future__ import annotations
from typing import Dict, Tuple, Optional
from contextlib import contextmanager, nullcontext
import warnings
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


def ae_loss(
    model: nn.Module,
    batch,
    cfg,
    device: torch.device,
    train: bool = True,
    weight_override: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Optional[float]]]:
    """
    Autoencoder loss with optional isometry + MECAE (extrinsic) + MICAE (intrinsic) +
    Metric Smoothness (invariant, G^{-1}-normalised).
    """
    x = batch[0].to(device)
    wo = weight_override or {}

    # ---- reconstruction ----
    use_vae = bool(getattr(getattr(cfg, "model", None), "vae", {}).get("enabled", False))
    x_hat, _ = model(x)
    rec_type = str(getattr(cfg.loss, "reconstruction", "mse")).lower()
    rec = F.mse_loss(x_hat, x) if rec_type == "mse" else F.l1_loss(x_hat, x)

    # ---- KL (VAE) ----
    kl = torch.tensor(0.0, device=device)
    if use_vae:
        mu, logvar = getattr(model, "_last_mu"), getattr(model, "_last_logvar")
        prior_logvar = torch.zeros_like(mu) if getattr(model, "prior_logvar", None) is None \
            else model.prior_logvar.view(1, -1).expand_as(mu)
        s2, sp2 = logvar.exp(), prior_logvar.exp()
        kl = (0.5 * (prior_logvar - logvar - 1.0 + (s2 + mu.pow(2)) / sp2)).sum(dim=1).mean()
    beta_kl = float(getattr(cfg.loss, "beta_kl", 1.0))

    # ---- isometry ----
    enc_iso_w = float(wo.get("enc_iso_weight", getattr(cfg.loss, "enc_iso_weight", 0.0)))
    dec_iso_w = float(wo.get("dec_iso_weight", getattr(cfg.loss, "dec_iso_weight", 0.0)))
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

    # ---- curvature regs (MECAE / MICAE) ----
    me_iter_val: Optional[float] = None
    mi_iter_val: Optional[float] = None
    ms_iter_val: Optional[float] = None

    metrics: Dict[str, Optional[float]] = {}

    curv_w = float(wo.get("curvature_weight", getattr(cfg.loss, "curvature_weight", 0.0)))
    intr_w = float(wo.get("intrinsic_weight", getattr(cfg.loss, "intrinsic_weight", 0.0)))
    ms_w   = float(wo.get("metric_smooth_weight", getattr(cfg.loss, "metric_smooth_weight", 0.0)))

    step_attr = "_curv_step"
    step = getattr(model, step_attr, 0)

    curv_cfg = getattr(cfg.loss, "curvature", {}) or {}
    intr_cfg = getattr(cfg.loss, "intrinsic", {}) or {}
    ms_cfg   = getattr(cfg.loss, "metric_smoothness", {}) or {}

    every_n     = int(getattr(curv_cfg, "every_n_steps", 1))
    do_curv     = train and (curv_w > 0.0) and (every_n <= 1 or (step % max(every_n, 1) == 0))
    intr_every  = int(getattr(intr_cfg, "every_n_steps", every_n))
    do_intr     = train and (intr_w > 0.0) and (intr_every <= 1 or (step % max(intr_every, 1) == 0))
    ms_every    = int(getattr(ms_cfg, "every_n_steps", every_n))
    do_ms       = train and (ms_w > 0.0) and (ms_every <= 1 or (step % max(ms_every, 1) == 0))

    target     = str(getattr(curv_cfg, "target", "encoder")).lower()
    z_full32 = model.encode(x).to(torch.float32)

    if target == "encoder":
        dec_mod    = _guess_decoder_module(model)
        freeze_ctx = freeze_params(dec_mod)
        z_in = z_full32
    elif target == "both":
        freeze_ctx = nullcontext()
        z_in = z_full32
    else:  # "decoder"
        freeze_ctx = nullcontext()
        z_in = z_full32.detach()

    B = z_in.size(0)
    B_curv = int(getattr(curv_cfg, "B_curv", 8))
    B_curv = max(1, min(B_curv, B))
    idx = torch.randperm(B, device=z_in.device)[:B_curv] if (B_curv < B) \
        else torch.arange(B, device=z_in.device)
    z_curv = z_in[idx]

    if (do_curv or do_intr or do_ms):
        if dynamo is not None:
            try:
                dynamo.graph_break()
            except Exception:
                pass

        with torch.amp.autocast("cuda", enabled=False), freeze_ctx:
            JGL = None
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
                total += curv_w * eec
                me_iter_val = float(eec.detach().item())
                JGL = (me_out["J"], me_out["G"], me_out["L"])

            if do_intr:
                mi_out = micae_intrinsic_decoder(
                    decode=model.decode, z=z_curv,
                    lam=float(getattr(intr_cfg, "reg_lambda", curv_cfg.get("reg_lambda", 1e-6))),
                    R_a=int(getattr(intr_cfg, "R_a", 2)),
                    use_rademacher=bool(getattr(intr_cfg, "use_rademacher", True)),
                    use_exact_hessian=bool(getattr(intr_cfg, "use_exact_hessian", True)),
                    fd_eps=float(getattr(intr_cfg, "fd_eps", 1e-3)),
                    JGL=JGL,
                    normalize_codim=bool(getattr(intr_cfg, "normalize_codim", True)),
                    return_aux=False,
                )
                eic = mi_out["EIC"].mean().to(x.dtype)
                total += intr_w * eic
                mi_iter_val = float(eic.detach().item())
                for k, v in mi_out.items():
                    if k == "EIC": continue
                    if isinstance(v, torch.Tensor) and v.ndim == 0 and torch.isfinite(v):
                        metrics[f"reg/micae/{k}"] = float(v.detach().item())
                metrics["reg/micae/EIC_mean"] = mi_iter_val

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
                    total += ms_w * msm
                    ms_iter_val = float(msm.detach().item())

    if train:
        setattr(model, step_attr, int(step) + 1)

    stem_alpha = None
    try:
        stem_alpha = float(getattr(getattr(model, "decoder", None), "stem", None).alpha.item())
    except Exception:
        pass

    metrics.update({
        "loss/total": float(total.detach().item()),
        "loss/reconstruction": float(rec.detach().item()),
        "loss/kl": float(kl.detach().item()),
        "reg/enc_iso": float(enc_iso.detach().item()),
        "reg/dec_iso": float(dec_iso.detach().item()),
        "reg/mecae_eec": me_iter_val,
        "reg/micae_eic": mi_iter_val,
        "reg/metric_smoothness": ms_iter_val,
        "model/stem_alpha": stem_alpha,
        "weights/enc_iso": enc_iso_w,
        "weights/dec_iso": dec_iso_w,
        "weights/curv": curv_w,
        "weights/intr": intr_w,
        "weights/ms": ms_w,
    })

    return total, metrics

