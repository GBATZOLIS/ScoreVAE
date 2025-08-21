#!/usr/bin/env python3
# eval_autoencoder.py

import os
import argparse
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

from configs import load_config
from data.data_utils_fast import get_dataloaders
from models import get_model
from utils.train_utils import (
    prepare_training_dirs,
    EMA,
    load_model,
)
from utils.ae_utils import get_reconstruction_callback, get_latent_scatter_callback

# NEW: import the isometry regularizer
from loss.isometry import approximate_orthogonal_jacobian_regularisation


def evaluate_decoder_isometry(
    model,
    loader,
    device,
    *,
    num_v: int,
    detach_encoder_output: bool = True,
    max_batches: int | None = None,
    writer: SummaryWriter | None = None,
    epoch: int = 0,
    tag_prefix: str = "AE",
):
    """
    Compute the decoder's isometry closeness over loader batches by:
      x -> z = encode(x)  (no_grad)
      reg = approximate_orthogonal_jacobian_regularisation(decode, z, train=False)
    Returns the mean of the metric across processed batches.
    """
    model.eval()
    totals = []
    n_processed = 0

    for b_idx, (x, *_) in enumerate(loader):
        if (max_batches is not None) and (b_idx >= max_batches):
            break

        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            z = model.encode(x)

        if detach_encoder_output:
            z = z.detach()

        # The regularizer toggles grad internally; pass train=False here.
        reg = approximate_orthogonal_jacobian_regularisation(
            model.decode, z, num_v=num_v, train=False, device=device
        )
        totals.append(reg.detach().item())
        n_processed += 1

        # Optional: per-batch histogram (comment out if you want fewer TB writes)
        if writer is not None:
            writer.add_scalar(f"{tag_prefix}/eval/dec_iso_batch", float(reg.item()), epoch * 10_000 + b_idx)

    mean_val = float(sum(totals) / max(1, n_processed))
    if writer is not None:
        writer.add_scalar(f"{tag_prefix}/eval/dec_iso_mean", mean_val, epoch)
    return mean_val


def eval_autoencoder(cfg, *, use_test: bool, num_batches: int, max_points: int, also_recon: bool,
                     iso_batches: int | None, iso_num_v: int | None, iso_detach_encoder: bool | None):
    """
    Loads EMA weights, runs latent scatter & (optional) recon callbacks, logs to eval_dir,
    and computes decoder isometry closeness over the eval set.
    """
    # Ensure dirs exist (and get eval_dir)
    _, checkpoint_dir, eval_dir = prepare_training_dirs(cfg)
    writer = SummaryWriter(log_dir=eval_dir)

    # Device
    device = torch.device(cfg.training.device)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(cfg.data, seed=cfg.random_seed)
    eval_loader = test_loader if use_test else val_loader

    # Model + EMA
    model = get_model(cfg.model).to(device, memory_format=torch.channels_last)
    ema = EMA(model=model, decay=cfg.model.ema_decay)

    # Resolve checkpoint path(s)
    ckpt_path = cfg.model.checkpoint
    if ckpt_path is None:
        raise ValueError(
            "config.model.checkpoint is None. Please set it to a checkpoint name/path "
            "(e.g. 'AE_epoch010.pth') in your config or pass --checkpoint."
        )

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_path)
    if not ckpt_path.endswith(".pth"):
        ckpt_path += ".pth"

    # Load weights (both raw and EMA if present)
    load_model(model, ema, ckpt_path, "AE", device=device, is_ema=False)
    ema_path = ckpt_path.replace(".pth", "_EMA.pth")
    if os.path.exists(ema_path):
        load_model(model, ema, ema_path, "AE", device=device, is_ema=True)
        ema.apply_shadow()
    else:
        print(f"[Eval] EMA checkpoint not found at {ema_path}. Using non-EMA weights.")

    model.eval()

    # Callbacks
    latent_cb = get_latent_scatter_callback(num_batches=num_batches, max_points=max_points)
    if also_recon:
        recon_cb = get_reconstruction_callback()

    # Run callbacks once (epoch=0 for logging)
    epoch = 0
    latent_cb(eval_loader, writer, model, device, epoch, tag_prefix="AE")
    if also_recon:
        batch = next(iter(eval_loader))
        recon_cb(batch, writer, model, device, epoch, tag_prefix="AE")

    # --- Decoder isometry evaluation ---
    # Defaults fall back to config if not given via CLI
    num_v_eff = int(iso_num_v if iso_num_v is not None else getattr(cfg.loss, "num_v", 16))
    detach_eff = bool(iso_detach_encoder if iso_detach_encoder is not None
                      else getattr(cfg.loss, "dec_iso_detach_encoder", True))
    max_batches_eff = iso_batches  # None -> run through entire eval_loader

    mean_iso = evaluate_decoder_isometry(
        model=model,
        loader=eval_loader,
        device=device,
        num_v=num_v_eff,
        detach_encoder_output=detach_eff,
        max_batches=max_batches_eff,
        writer=writer,
        epoch=epoch,
        tag_prefix="AE",
    )
    print(f"[Eval] Decoder isometry (mean over batches): {mean_iso:.6f}")

    # Restore non-EMA weights if we applied EMA
    if os.path.exists(ema_path):
        ema.restore()

    writer.close()


def main():
    parser = argparse.ArgumentParser("AutoEncoder evaluation (latent scatter + optional recon + isometry)")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration .py")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: override config.model.checkpoint (name or absolute path).")
    parser.add_argument("--use_test", action="store_true",
                        help="If set, evaluate on the test loader instead of validation.")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="How many batches to aggregate for the scatter.")
    parser.add_argument("--max_points", type=int, default=3000,
                        help="Maximum number of points to plot.")
    parser.add_argument("--also_recon", action="store_true",
                        help="Also log a reconstruction grid.")
    # NEW CLI for the isometry evaluation
    parser.add_argument("--iso_batches", type=int, default=None,
                        help="How many batches to use for isometry eval (default: all).")
    parser.add_argument("--iso_num_v", type=int, default=None,
                        help="Number of random directions v for the JVPs (default: cfg.loss.num_v or 16).")
    parser.add_argument("--iso_detach_encoder", action="store_true",
                        help="Detach z from encoder when evaluating decoder isometry.")
    parser.add_argument("--no_iso_detach_encoder", dest="iso_detach_encoder", action="store_false")
    parser.set_defaults(iso_detach_encoder=None)

    args = parser.parse_args()

    cfg = load_config(args.config)

    # If user passed a checkpoint on CLI, override the config
    if args.checkpoint is not None:
        cfg.model.checkpoint = args.checkpoint

    # Save a copy of the eval config
    out_dir = os.path.join(cfg.base_log_dir, cfg.experiment)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_eval_autoencoder.pkl"), "wb") as f:
        pickle.dump(cfg.to_dict(), f)

    eval_autoencoder(
        cfg,
        use_test=bool(args.use_test),
        num_batches=int(args.num_batches),
        max_points=int(args.max_points),
        also_recon=bool(args.also_recon),
        iso_batches=args.iso_batches,
        iso_num_v=args.iso_num_v,
        iso_detach_encoder=args.iso_detach_encoder,
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
