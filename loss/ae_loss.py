# loss/ae_loss.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from .isometry import approximate_orthogonal_jacobian_regularisation, encoder_row_orthogonality_regularisation

def ae_loss(
    model,
    batch,
    cfg,
    device: torch.device,
    train: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns total loss and metrics dict.
    """
    x = batch[0].to(device)
    x_hat, z = model(x)

    rec_type = getattr(cfg.loss, "reconstruction", "mse").lower()
    if rec_type == "mse":
        rec = F.mse_loss(x_hat, x)
    elif rec_type == "l1":
        rec = F.l1_loss(x_hat, x)
    else:
        raise ValueError(f"Unknown reconstruction loss '{rec_type}'")

    # encoder iso (R^{image} -> R^{z})
    enc_w = float(getattr(cfg.loss, "enc_iso_weight", 0.0))
    enc_iso = torch.tensor(0.0, device=device)
    if enc_w > 0:
        enc_iso = encoder_row_orthogonality_regularisation(
            model.encode, x, num_v=int(getattr(cfg.loss, "num_v", 16)),
            train=train, device=device
        )

    # decoder iso (R^{z} -> R^{image})
    dec_w = float(getattr(cfg.loss, "dec_iso_weight", 0.0))
    dec_iso = torch.tensor(0.0, device=device)
    if dec_w > 0:
        z_in = z.detach() if bool(getattr(cfg.loss, "dec_iso_detach_encoder", True)) else z
        dec_iso = approximate_orthogonal_jacobian_regularisation(
            model.decode, z_in, num_v=int(getattr(cfg.loss, "num_v", 16)),
            train=train, device=device
        )

    total = rec + enc_w * enc_iso + dec_w * dec_iso

    metrics = {
        "loss/total": float(total.detach().item()),
        "loss/reconstruction": float(rec.detach().item()),
        "reg/enc_iso": float(enc_iso.detach().item()),
        "reg/dec_iso": float(dec_iso.detach().item()),
    }
    return total, metrics
