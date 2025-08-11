# -------------------------------------------------------------------------
# FAST single‑GPU trainer for ScoreVAE / DDPM – feature‑parity with train.py
# -------------------------------------------------------------------------
# This version disables CUDA Graph capture inside Torch‑Inductor to avoid
# the "assert data_ptr == new_inputs[idx].data_ptr()" crash that appears
# with dynamic batch sizes or mixed autocast contexts.
# -------------------------------------------------------------------------
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Disable CUDA graphs *properly* for Torch‑Inductor (must be done before
# the first torch.compile call and ideally before importing anything that
# triggers Inductor setup).
# -------------------------------------------------------------------------
import os
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Uniform
from tqdm import tqdm

# Official switch – works for PyTorch >= 2.1
from torch._inductor import config as inductor_cfg
inductor_cfg.triton.cudagraphs = False  # ⇐ THIS actually blocks CUDAGraphs
# (The env‑var TORCHINDUCTOR_DISABLE_CUDAGRAPHS you tried earlier is ignored.)

# ──────────────────────────────────────────────────────────────────────────
# project imports
# -------------------------------------------------------------------------
from data.data_utils_fast import get_dataloaders  # fast dataloader implementation
from models import get_model
from sde import configure_sde
from loss import get_loss_fn
from configs import load_config
from evaluation.fid import fid_evaluation_callback

from utils.optim_utils import get_optimizer_and_scheduler
from utils.sampling_utils import get_generation_callback
from utils.train_utils import (
    prepare_training_dirs,
    prepare_batch,
    EMA,
    save_model,
    load_model,
    resume_training,
)

# ---------------------------------------------------------------------- helpers

def set_seed(seed: int):
    """Set all the RNGs we know about for full reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compile_model(model: nn.Module) -> nn.Module:
    """Return a torch.compile‑d model **without** CUDA graphs.

    • On PyTorch ≥ 2.6 the preset "max-autotune-no-cudagraphs" is available.
    • For slightly older 2.x releases we just fall back to "max-autotune" –
      the global flag we set above already prevents graph capture.
    """
    try:
        return torch.compile(model, mode="max-autotune-no-cudagraphs")
    except (TypeError, ValueError):
        # Fallback for Torch versions that don't know the newer preset.
        return torch.compile(model, mode="max-autotune")

# ---------------------------------------------------------------------- train loop

def train(cfg):
    # ---------------- configuration‑dependent dirs & writer ----------------
    tb_dir, ckpt_dir, _ = prepare_training_dirs(cfg)
    writer = SummaryWriter(log_dir=tb_dir)

    # ---------------- hardware knobs ----------------
    device = torch.device(cfg.training.device)
    torch.backends.cudnn.benchmark = True  # heuristic autotune convs
    torch.set_float32_matmul_precision("high")  # enable TF32 on A100/RTX 30xx

    # ---------------- data ----------------
    train_loader, val_loader, test_loader = get_dataloaders(cfg.data, seed=cfg.random_seed)

    # ---------------- model ----------------
    base_model = get_model(cfg.model).to(device, memory_format=torch.channels_last)
    base_model.print_model_summary()
    model = compile_model(base_model)  # compiled model is still nn.Module

    # ---------------- diffusion process / SDE ----------------
    sde = configure_sde(cfg)

    # ---------------- log‑space EMA ----------------
    ema = EMA(model, decay=cfg.model.ema_decay)

    # ---------------- optimiser & LR schedule ----------------
    optimizer, scheduler = get_optimizer_and_scheduler(model, cfg)

    # ---------------- checkpoint / resume ----------------
    (
        start_epoch,
        global_step,
        best_ckpts,
        best_val_loss,
        epochs_no_improve,
        optimizer,
        scheduler,
    ) = resume_training(cfg, model, ema, load_model, get_optimizer_and_scheduler)

    # ---------------- loss fn ----------------
    t_dist = Uniform(sde.sampling_eps, 1.0)
    loss_fn = get_loss_fn(cfg, sde, t_dist)

    # ---------------- callbacks ----------------
    gen_cb = get_generation_callback(cfg.training.vis_callback)

    # ---------------- AMP settings ----------------
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype is torch.float16)

    # ====================================================================
    #                           EPOCH LOOP
    # ====================================================================
    for epoch in range(start_epoch, cfg.training.epochs):
        # ================================================= train phase ===
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        running_train = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{cfg.training.epochs}")
        for data in pbar:
            batch = prepare_batch(data, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                loss = loss_fn(model, batch, train=True)

            scaler.scale(loss).backward()
            if cfg.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()

            running_train += loss.item()
            writer.add_scalar("Loss/Train", loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_train / len(train_loader)
        writer.add_scalar("Loss/Train_epoch", epoch_train_loss, epoch)

        # ================================================= val phase ===
        ema.apply_shadow()  # swap to EMA params
        model.eval()
        val_total = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
            for data in val_loader:
                val_total += loss_fn(model, prepare_batch(data, device), train=False).item()
        ema.restore()  # swap back

        val_loss = val_total / len(val_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # ---------------- visualisation callback ----------------
        if (epoch + 1) % cfg.training.vis_frequency == 0:
            shape = (cfg.training.num_samples, *cfg.data.shape)
            batch = prepare_batch(next(iter(val_loader)), device)
            gen_cb(batch, writer, sde, model, cfg.training.steps, shape, device, epoch)

        # ---------------- FID evaluation ----------------
        if (epoch + 1) % cfg.training.fid_eval_frequency == 0:
            shape = (cfg.training.num_samples, *cfg.data.shape)
            steps = cfg.training.steps

            # FID on (train+val)
            fid_evaluation_callback(
                writer, sde, model, steps, shape, device, epoch,
                dataloaders=[train_loader, val_loader], train=True,
            )
            # FID on test
            fid_evaluation_callback(
                writer, sde, model, steps, shape, device, epoch,
                dataloaders=[test_loader], train=False,
            )

        # ---------------- early stopping / bookkeeping ----------------
        improved = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        epochs_no_improve = 0 if improved else epochs_no_improve + 1

        if epochs_no_improve >= cfg.training.patience_epochs:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # ---------------- checkpointing ----------------
        if (epoch + 1) % cfg.training.checkpoint_frequency == 0:
            save_model(
                model, ema, epoch, val_loss, "Model", ckpt_dir, best_ckpts,
                global_step, best_val_loss, epochs_no_improve, optimizer, scheduler,
            )

    # ====================================================================
    writer.close()
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        print(f"Training complete. Peak GPU RAM: {peak:.2f} GiB")

# ---------------------------------------------------------------------- entry‑point

def main():
    parser = argparse.ArgumentParser(
        description="FAST trainer with torch.compile + AMP – functional parity"
    )
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # persist config for later reproducibility
    out_dir = Path(cfg.base_log_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.pkl", "wb") as f:
        pickle.dump(cfg.to_dict(), f)

    set_seed(cfg.random_seed)
    train(cfg)


if __name__ == "__main__":
    main()
