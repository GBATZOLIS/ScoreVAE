# loss/ae_loss.py
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

from .isometry import encoder_isometry_regularisation, decoder_isometry_regularisation
from .curvature import mecae_extrinsic_decoder, micae_intrinsic_decoder


@contextmanager
def freeze_params(module: Optional[nn.Module]):
    """
    Temporarily sets a module to eval and disables gradients on its params.
    Restores mode and requires_grad flags after the block.
    """
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
    """
    Best-effort to guess a submodule corresponding to the decoder block,
    used only to freeze during curvature targeting the encoder.
    """
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
) -> Tuple[torch.Tensor, Dict[str, Optional[float]]]:
    """
    Autoencoder loss with optional isometry + MECAE (extrinsic) + MICAE (intrinsic).
    DDP-safe: operates on the local batch; DDP will reduce grads as usual.
    """
    x = batch[0].to(device)

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

    # ---- curvature regs ----
    me_iter_val: Optional[float] = None   # MECAE EEC
    mi_iter_val: Optional[float] = None   # MICAE EIC

    curv_w = float(getattr(cfg.loss, "curvature_weight", 0.0))  # MECAE weight
    intr_w = float(getattr(cfg.loss, "intrinsic_weight", 0.0))  # MICAE weight

    # use a per-model step counter to gate curvature frequency
    step_attr = "_curv_step"
    step = getattr(model, step_attr, 0)

    curv_cfg = getattr(cfg.loss, "curvature", {}) or {}
    intr_cfg = getattr(cfg.loss, "intrinsic", {}) or {}

    # schedule
    every_n  = int(getattr(curv_cfg, "every_n_steps", 1))
    do_curv  = train and (every_n <= 1 or (step % max(every_n, 1) == 0))
    intr_every = int(getattr(intr_cfg, "every_n_steps", every_n))
    do_intr = train and (intr_every <= 1 or (step % max(intr_every, 1) == 0))

    # target & freeze policy (shared)
    target   = str(getattr(curv_cfg, "target", "encoder")).lower()
    z_full32 = model.encode(x).to(torch.float32)
    if target == "encoder":
        dec_mod  = _guess_decoder_module(model)
        freeze_ctx = freeze_params(dec_mod)
        z_in = z_full32
    elif target == "both":
        freeze_ctx = nullcontext()
        z_in = z_full32
    else:  # "decoder"
        freeze_ctx = nullcontext()
        z_in = z_full32.detach()

    # sub-batch for curvature (memory-friendly)
    B = z_in.size(0)
    B_curv = int(getattr(curv_cfg, "B_curv", 8))
    B_curv = max(1, min(B_curv, B))
    idx = torch.randperm(B, device=z_in.device)[:B_curv] if (B_curv < B) else torch.arange(B, device=z_in.device)
    z_curv = z_in[idx]

    if (curv_w > 0.0 or intr_w > 0.0) and (do_curv or do_intr):
        if dynamo is not None:
            try:
                dynamo.graph_break()
            except Exception:
                pass

        # Compute curvature in full precision for stability
        with torch.amp.autocast("cuda", enabled=False), freeze_ctx:
            JGL = None

            # ---- MECAE (extrinsic) ----
            if curv_w > 0.0 and do_curv:
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

            # ---- MICAE (intrinsic via Gauss) ----
            if intr_w > 0.0 and do_intr:
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

    if train:
        setattr(model, step_attr, int(step) + 1)

    metrics: Dict[str, Optional[float]] = {
        "loss/total":          float(total.detach().item()),
        "loss/reconstruction": float(rec.detach().item()),
        "loss/kl":             float(kl.detach().item()),
        "reg/enc_iso":         float(enc_iso.detach().item()),
        "reg/dec_iso":         float(dec_iso.detach().item()),
        "reg/mecae_eec":       me_iter_val,
        "reg/micae_eic":       mi_iter_val,
    }
    return total, metrics
