# scripts/train_autoencoder.py
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import load_config
from models import get_model
from utils.train_utils import (
    prepare_training_dirs, prepare_batch, EMA, save_model, load_model, resume_training
)
from utils.ae_utils import get_reconstruction_callback, get_latent_scatter_callback
from utils.optim_utils import get_optimizer_and_scheduler
from data.data_utils_fast import get_dataloaders
from loss.ae_loss import ae_loss

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def compile_model(model: nn.Module, do_compile: bool = True) -> nn.Module:
    if not do_compile:
        return model
    try:
        # CUDA graphs can be touchy; use safe mode like your fast trainer
        from torch._inductor import config as inductor_cfg
        inductor_cfg.triton.cudagraphs = False
        return torch.compile(model, mode="max-autotune-no-cudagraphs")
    except Exception:
        return torch.compile(model)

def train(cfg):
    tb_dir, ckpt_dir, _ = prepare_training_dirs(cfg)
    writer = SummaryWriter(log_dir=tb_dir)
    device = torch.device(cfg.training.device)

    train_loader, val_loader, test_loader = get_dataloaders(cfg.data, seed=cfg.random_seed)

    model = get_model(cfg.model).to(device, memory_format=torch.channels_last)
    model.print_model_summary()
    model = compile_model(model, bool(getattr(cfg.model, "compile", True)))
    ema = EMA(model, decay=cfg.model.ema_decay)

    optimizer, scheduler = get_optimizer_and_scheduler(model, cfg)
    (
        start_epoch, global_step, best_ckpts, best_val_loss, epochs_no_improve,
        optimizer, scheduler
    ) = resume_training(cfg, model, ema, load_model, get_optimizer_and_scheduler)

    recon_cb = get_reconstruction_callback()
    latent_cb = get_latent_scatter_callback(num_batches=5, max_points=3000)

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16))

    for epoch in range(start_epoch, cfg.training.epochs):
        # ---------------- train ----------------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[AE] Epoch {epoch+1}/{cfg.training.epochs}")
        for data in pbar:
            batch = prepare_batch(data, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss, metrics = ae_loss(model, batch, cfg, device, train=True)

            scaler.scale(loss).backward()
            if cfg.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()

            running += loss.item()
            writer.add_scalar("AE/Loss/train_iter", loss.item(), global_step)
            for k, v in metrics.items():
                writer.add_scalar(f"AE/{k}_iter", v, global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_epoch = running / len(train_loader)
        writer.add_scalar("AE/Loss/train_epoch", train_epoch, epoch)

        # ---------------- val ----------------
        ema.apply_shadow()
        model.eval()
        total = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            for data in val_loader:
                loss, _ = ae_loss(model, prepare_batch(data, device), cfg, device, train=False)
                total += loss.item()
        ema.restore()
        val_loss = total / len(val_loader)
        writer.add_scalar("AE/Loss/val", val_loss, epoch)

        # ---------------- visualize ----------------
        if (epoch + 1) % cfg.training.vis_frequency == 0:
            batch = next(iter(val_loader))
            recon_cb(batch, writer, model, device, epoch, tag_prefix="AE")
            latent_cb(val_loader, writer, model, device, epoch, tag_prefix="AE")

        # ---------------- early stop / ckpts ----------------
        improved = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        epochs_no_improve = 0 if improved else epochs_no_improve + 1
        if epochs_no_improve >= cfg.training.patience_epochs:
            print(f"[AE] Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % cfg.training.checkpoint_frequency == 0:
            save_model(
                model, ema, epoch, val_loss, "AE", ckpt_dir, best_ckpts,
                global_step, best_val_loss, epochs_no_improve, optimizer, scheduler
            )

    writer.close()

def main():
    P = argparse.ArgumentParser("Isometry-regularized AutoEncoder trainer")
    P.add_argument("--config", required=True, help="Path to the configuration file (python .py).")
    args = P.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.base_log_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.pkl", "wb") as f:
        pickle.dump(cfg.to_dict(), f)

    set_seed(cfg.random_seed)
    train(cfg)

if __name__ == "__main__":
    main()
