# -----------------------------------------------------------------------------
# FAST multi‑GPU trainer (single‑node) for ScoreVAE / DDPM
# -----------------------------------------------------------------------------
# * Launch with:  torchrun --standalone --nproc_per_node N train_fast_ddp.py --config <cfg>
# * DistributedDataParallel (DDP)
# * Torch‑compile + channels_last + AMP (bfloat16/float16)
# * Feature‑parity with original train.py / train_fast.py (EMA, vis, FID, early‑stop)
# -----------------------------------------------------------------------------
from __future__ import annotations

import os, random, argparse, pickle
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch, torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Uniform
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────── helpers

def init_distributed() -> tuple[bool, int, int, int]:
    """Initialise torch.distributed for a *single* node (NCCL backend).

    Returns
    -------
    is_dist   : bool   – True if running under `torchrun` / DDP.
    rank      : int    – Global rank (0‑based).
    world_size: int    – #processes (GPUs).
    local_rank: int    – Rank within the node (also the GPU index).
    """
    if not torch.cuda.is_available() or "RANK" not in os.environ:
        # Fallback to single‑process / single‑GPU (or CPU) mode
        return False, 0, 1, 0

    rank        = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def set_seed(seed: int, rank: int):
    """Deterministic seed — offset by rank so that dataloader shuffling differs per GPU."""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def compile_model(model: nn.Module):
    """Compile with torch 2.x"""
    try:
        return torch.compile(model, mode="max-autotune")
    except Exception as e:  # pragma: no cover (older PyTorch / unsupported HW)
        if dist.is_available() and dist.get_rank() == 0:
            print("[warn] torch.compile disabled →", e)
        return model

# ───────────────────────────────────────────────────────────────────────── project imports
from data.data_utils_fast import get_dataloaders  # our dataloader util (supports DistributedSampler)
from models               import get_model
from sde                  import configure_sde
from loss                 import get_loss_fn
from configs              import load_config
from evaluation.fid       import fid_evaluation_callback

from utils.optim_utils     import get_optimizer_and_scheduler
from utils.sampling_utils  import get_generation_callback
from utils.train_utils     import (
    prepare_training_dirs,
    prepare_batch,
    EMA,
    save_model,
    load_model,
    resume_training,
)

# ────────────────────────────────────────────────────────────────────────── train

def train(cfg):
    # ---------------- distributed init ----------------
    is_dist, rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    # ---------------- logging dirs --------------------
    tb_dir, ckpt_dir, _ = prepare_training_dirs(cfg)
    writer = SummaryWriter(tb_dir) if is_main else nullcontext()

    # ---------------- device & perf knobs -------------
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+

    # ---------------- reproducibility -----------------
    set_seed(cfg.random_seed, rank)

    # ---------------- data ----------------------------
    train_loader, val_loader, test_loader = get_dataloaders(cfg.data, seed=cfg.random_seed)

    if is_dist:
        # Swap to DistributedSampler ‑ one per rank
        def make_loader(ds, shuffle: bool, drop_last: bool = False):
            return DataLoader(
                ds,
                batch_size        = cfg.data.batch_size,
                sampler           = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle),
                shuffle           = False,  # sampler handles shuffling
                drop_last         = drop_last,
                num_workers       = getattr(cfg.data, "n_workers", 4),
                pin_memory        = torch.cuda.is_available(),
                persistent_workers= True,
                prefetch_factor   = 4,
            )

        train_loader = make_loader(train_loader.dataset, shuffle=True,  drop_last=True)
        val_loader   = make_loader(val_loader.dataset,   shuffle=False)
        test_loader  = make_loader(test_loader.dataset,  shuffle=False)

    # ---------------- model ---------------------------
    base = get_model(cfg.model).to(device, memory_format=torch.channels_last)
    if is_main:
        base.print_model_summary()
    base = compile_model(base)

    # DDP wrapper (keeps param memory shared)
    model = (
        nn.parallel.DistributedDataParallel(base, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        if is_dist else base
    )

    # IMPORTANT ↴  we keep a *direct* handle to the underlying nn.Module for EMA
    net = model.module if is_dist else model

    # ---------------- diffusion / loss / optimiser ----
    sde        = configure_sde(cfg)
    ema        = EMA(net, decay=cfg.model.ema_decay)         # EMA sees *real* params, not DDP wrapper
    optimizer, scheduler = get_optimizer_and_scheduler(model, cfg)
    loss_fn    = get_loss_fn(cfg, sde, Uniform(sde.sampling_eps, 1.0))

    # ---------------- resume --------------------------
    (
        start_epoch,
        global_step,
        best_ckpts,
        best_val_loss,
        epochs_no_improve,
        optimizer,
        scheduler,
    ) = resume_training(cfg, model, ema, load_model, get_optimizer_and_scheduler)

    # ---------------- mixed‑precision -----------------
    use_bf16  = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler    = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16))

    # ---------------- callbacks -----------------------
    gen_cb = get_generation_callback(cfg.training.vis_callback)

    # ===================================================================
    #                             EPOCH LOOP
    # ===================================================================
    for epoch in range(start_epoch, cfg.training.epochs):
        # shuffle seed for DistributedSampler
        if is_dist:
            train_loader.sampler.set_epoch(epoch)

        # ---------------- TRAIN -----------------------------------
        model.train(); torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
        running_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}", disable=not is_main)
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
            ema.update()  # ↖ EMA *after* optimizer step

            running_sum += loss.item()
            if is_main:
                writer.add_scalar("Loss/Train", loss.item(), global_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        train_loss_epoch = running_sum / len(train_loader)
        if is_main:
            writer.add_scalar("Loss/Train_epoch", train_loss_epoch, epoch)

        # ---------------- VALIDATION (EMA weights) -----------------
        ema.apply_shadow(); model.eval()
        val_accum = torch.zeros(1, device=device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
            for data in val_loader:
                v = loss_fn(model, prepare_batch(data, device), train=False).item()
                val_accum += v
        if is_dist:
            dist.all_reduce(val_accum, op=dist.ReduceOp.SUM)
        ema.restore()

        val_loss = (val_accum.item() / world_size) / len(val_loader)
        if is_main:
            writer.add_scalar("Loss/Validation", val_loss, epoch)

        # ---------------- VIS / FID (rank‑0 only) ------------------
        if is_main and ((epoch + 1) % cfg.training.vis_frequency == 0):
            shape = (cfg.training.num_samples, *cfg.data.shape)
            batch = prepare_batch(next(iter(val_loader)), device)
            gen_cb(batch, writer, sde, model, cfg.training.steps, shape, device, epoch)

        if is_main and ((epoch + 1) % cfg.training.fid_eval_frequency == 0):
            shape = (cfg.training.num_samples, *cfg.data.shape)
            steps = cfg.training.steps
            fid_evaluation_callback(writer, sde, model, steps, shape, device, epoch,
                                     dataloaders=[train_loader, val_loader], train=True)
            fid_evaluation_callback(writer, sde, model, steps, shape, device, epoch,
                                     dataloaders=[test_loader], train=False)

        # ---------------- EARLY‑STOP / CHECKPOINT ------------------
        improved          = val_loss < best_val_loss
        best_val_loss     = min(best_val_loss, val_loss)
        epochs_no_improve = 0 if improved else epochs_no_improve + 1

        stop_tensor = torch.tensor([int(epochs_no_improve >= cfg.training.patience_epochs)], device=device)
        if is_dist:
            dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            if is_main:
                print(f"Early stopping at epoch {epoch+1}")
            break

        if is_main and ((epoch + 1) % cfg.training.checkpoint_frequency == 0):
            save_model(model, ema, epoch, val_loss, "Model", ckpt_dir, best_ckpts,
                       global_step, best_val_loss, epochs_no_improve, optimizer, scheduler)

    # ===================================================================
    #                             CLEAN‑UP
    # ===================================================================
    if is_main:
        writer.close()
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            print(f"Training complete. Peak GPU RAM: {peak:.2f} GiB")
    if is_dist:
        dist.destroy_process_group()

# ───────────────────────────────────────────────────────────────────────── CLI

def main():
    parser = argparse.ArgumentParser(description="DDP trainer (single‑node)")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # persist cfg once (main rank)
    is_dist = "RANK" in os.environ
    if (not is_dist) or (int(os.environ["RANK"]) == 0):
        out_dir = Path(cfg.base_log_dir) / cfg.experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.pkl", "wb") as f:
            pickle.dump(cfg.to_dict(), f)

    train(cfg)


if __name__ == "__main__":
    main()
