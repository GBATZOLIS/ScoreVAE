#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import math
import os
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import ml_collections

from configs import load_config
from models import get_model
from utils.train_utils import (
    prepare_training_dirs, prepare_batch, EMA, save_model, load_model, resume_training
)
from utils.ae_utils import (
    get_reconstruction_callback, get_latent_scatter_callback,
    get_update_latent_normalizer_callback, get_generation_callback
)
from utils.optim_utils import get_optimizer_and_scheduler
from data.data_utils_ddp import get_dataloaders
from loss.ae_loss import ae_loss
from sde import configure_sde

# Enable TF32 paths on Ampere+ GPUs (harmless elsewhere).
if torch.cuda.is_available():
    # cuDNN (convs, batchnorm, etc.)
    torch.backends.cudnn.allow_tf32 = True
    # matmul/linear ops
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        # older PyTorch fallback
        torch.backends.cuda.matmul.allow_tf32 = True
        
# ---------------- DDP helpers ----------------

def ddp_is_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_active() else 0

def ddp_world_size() -> int:
    return dist.get_world_size() if ddp_is_active() else 1

def setup_distributed(devices_cfg: Optional[List[int] | str]) -> Tuple[torch.device, int, int, int]:
    """
    Initialize DDP if launched with torchrun. Map LOCAL_RANK to selected devices if provided.
    """
    use_ddp = ("RANK" in os.environ) or ("LOCAL_RANK" in os.environ) or ("WORLD_SIZE" in os.environ)
    rank = 0
    world_size = 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Map selected devices
    selected_devices: Optional[List[int]] = None
    if devices_cfg is not None:
        if isinstance(devices_cfg, str):
            if devices_cfg.strip().lower() == "auto":
                selected_devices = list(range(torch.cuda.device_count()))
            else:
                selected_devices = [int(x) for x in devices_cfg.split(",") if x.strip() != ""]
        elif isinstance(devices_cfg, (list, tuple)):
            selected_devices = [int(x) for x in devices_cfg]
        else:
            selected_devices = None

    if use_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # default mapping if not provided
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank % max(1, torch.cuda.device_count()))))

    # Device select
    if torch.cuda.is_available():
        if selected_devices:
            assert local_rank < len(selected_devices), (
                f"LOCAL_RANK={local_rank} but selected devices={selected_devices}"
            )
            real_idx = selected_devices[local_rank]
        else:
            real_idx = local_rank
        torch.cuda.set_device(real_idx)
        device = torch.device(f"cuda:{real_idx}")
    else:
        device = torch.device("cpu")

    torch.backends.cudnn.benchmark = True
    return device, rank, world_size, local_rank

def finalize_distributed(
    device=None,
    _ddp_is_active=lambda: dist.is_available() and dist.is_initialized(),
    _rank=lambda: dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
):
    try:
        if _ddp_is_active():
            try:
                if device is not None and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
            except Exception:
                pass
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
    except Exception as e:
        print(f"[rank{_rank()}] finalize_distributed warning: {e}")


# ---------------- small utils ----------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def compile_model(model: nn.Module, do_compile: bool = True) -> nn.Module:
    if not do_compile:
        return model
    try:
        from torch._inductor import config as inductor_cfg
        inductor_cfg.triton.cudagraphs = False
        return torch.compile(model, mode="max-autotune-no-cudagraphs")
    except Exception:
        return torch.compile(model)

def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

@contextmanager
def freeze_eval(module: nn.Module | None):
    if module is None:
        yield
        return
    was_training = module.training
    prev_reqgrad = [p.requires_grad for p in module.parameters()]
    try:
        module.eval()
        _set_requires_grad(module, False)
        yield
    finally:
        module.train(was_training)
        for p, rg in zip(module.parameters(), prev_reqgrad):
            p.requires_grad_(rg)

def _make_latent_dsm_loss(sde):
    """
    Basic DSM-like objective in latent space.
    Expects latent_model(zt, y=None, t=t) signature; sde exposes marginal_prob().
    """
    def loss_fn(model, z_batch):
        B = z_batch.size(0)
        t = torch.empty(B, device=z_batch.device).uniform_(sde.sampling_eps, 1.0)
        noise = torch.randn_like(z_batch)
        mean, std = sde.marginal_prob(z_batch, t)
        std_view = std.view(B, *([1] * (z_batch.ndim - 1))) if std.ndim == 1 else std
        zt = mean + std_view * noise
        pred = model(zt, y=None, t=t)
        return torch.mean((noise - pred) ** 2)
    return loss_fn


# ---------------- Null writer for non-rank0 ----------------

class _NullWriter:
    def add_scalar(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def add_histogram(self, *a, **k): pass
    def add_figure(self, *a, **k): pass
    def close(self): pass


# ---------------- Post-training for latent diffusion ----------------

def post_train_latent(
    *,
    latent_model,                 # DDP-wrapped or plain latent score model
    latent_ema,                   # EMA wrapper for latent_model (has apply_shadow/restore/update)
    ae_mod,                       # your AE module
    ema,                          # EMA wrapper for AE (has apply_shadow/restore)
    train_loader,                 # training DataLoader
    val_loader=None,              # optional validation DataLoader
    train_sampler=None,           # optional DistributedSampler for train_loader
    lat_cfg=None,                 # latent config (with .optim and optional .post_train)
    latent_sde=None,              # latent SDE object used by the loss/sampling
    writer=None,                  # TensorBoard SummaryWriter
    device=None,
    tb_dir=None,                  # for gen_cb save dir
    ckpt_dir=None,                # for save_model
    global_step=0,                # passthrough to save_model
    best_val_loss=float("inf"),   # passthrough to save_model (unchanged here)
    epochs_no_improve=0,          # passthrough to save_model (unchanged here)
    gen_cb=None,                  # your generation callback (optional)
    save_tag="LatentDiffPost",    # checkpoint tag
    _make_latent_dsm_loss=None,   # factory returning loss closure: fn(latent_model, z_hat)->loss
    get_optimizer_and_scheduler=None,  # factory: (model, cfg, global_step)->(opt,sched)
    prepare_batch=None,           # function to move batch to device, etc.
    save_model=None,              # checkpoint saver
    ddp_is_active=lambda: dist.is_available() and dist.is_initialized(),
    ddp_rank=lambda: (dist.get_rank() if dist.is_available() and dist.is_initialized() else 0),
):
    assert latent_model is not None, "latent_model is required"
    assert _make_latent_dsm_loss is not None and get_optimizer_and_scheduler is not None
    assert prepare_batch is not None and save_model is not None
    assert latent_sde is not None, "latent_sde is required for the latent loss"

    # ---- config knobs (with safe defaults)
    post_cfg         = getattr(lat_cfg, "post_train", torch.nn.Module()) if lat_cfg is not None else torch.nn.Module()
    post_steps       = int(getattr(post_cfg, "steps", 20_000))
    post_lr          = getattr(post_cfg, "lr", None)       # None -> reuse lat_cfg.optim.lr
    post_warmup      = int(getattr(post_cfg, "warmup", min(2_000, max(100, post_steps // 10))))
    post_grad_clip   = float(getattr(lat_cfg.optim, "grad_clip", 1.0)) if lat_cfg is not None else 1.0
    base_epoch_seed  = int(getattr(post_cfg, "epoch_seed", 10_000))  # seed offset for post phase
    post_val_every   = int(getattr(post_cfg, "val_every", 1))        # validate every N post-epochs
    post_val_batches = getattr(post_cfg, "val_batches", None)        # limit val batches (None=full)
    post_gen_every   = int(getattr(post_cfg, "gen_every", 1))        # generate every N epochs (default 1)
    overfit_patience = int(getattr(post_cfg, "overfit_patience", 2))
    overfit_delta    = float(getattr(post_cfg, "overfit_delta", 0.0))

    # ---- latent optimizer/scheduler (fresh)
    lat_mod = latent_model.module if ddp_is_active() else latent_model
    lat_cfg_post = None
    if lat_cfg is not None:
        import copy as _copy
        lat_cfg_post = _copy.deepcopy(lat_cfg)
        lat_cfg_post.optim.total_steps = post_steps
        lat_cfg_post.optim.warmup = post_warmup
        if post_lr is not None:
            lat_cfg_post.optim.lr = float(post_lr)
    post_opt, post_sched = get_optimizer_and_scheduler(lat_mod, lat_cfg_post, global_step=0)

    # ---- keep AE in EMA mode for the entire post-phase (EMA-encoded latents + EMA-decoder for gen_cb)
    ema.apply_shadow()
    ae_mod.eval()

    if ddp_rank() == 0:
        lr_show = getattr(lat_cfg_post.optim, "lr", None) if lat_cfg_post else getattr(lat_cfg.optim, "lr", None)
        print(f"[Latent Post-Train] starting {post_steps} iterations. "
              f"lr={lr_show}, warmup={post_warmup}, clip={post_grad_clip}")

    # ---- optional tqdm (rank 0 only)
    try:
        from tqdm.auto import tqdm as _tqdm
        use_tqdm = (ddp_rank() == 0)
    except Exception:
        _tqdm = None
        use_tqdm = False

    # -------- helpers
    def _all_reduce_sum(t):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    @torch.no_grad()
    def _run_validation(epoch_idx: int) -> float:
        if val_loader is None:
            return float("nan")

        # temporarily evaluate with EMA latent model
        if latent_ema is not None:
            latent_ema.apply_shadow()
        was_training = latent_model.training
        latent_model.eval()

        total_loss_sum = torch.tensor(0.0, device=device)
        total_count    = torch.tensor(0.0, device=device)

        vbar = None
        if use_tqdm and _tqdm is not None:
            total = (post_val_batches if post_val_batches is not None else len(val_loader))
            vbar = _tqdm(total=total, leave=False, desc=f"Val (epoch {epoch_idx+1})")

        for b_idx, batch in enumerate(val_loader):
            if post_val_batches is not None and b_idx >= int(post_val_batches):
                break
            batch = prepare_batch(batch, device)
            x = batch[0]
            z_hat = ae_mod.normalize_latent(ae_mod.encode(x))
            loss  = _make_latent_dsm_loss(latent_sde)(latent_model, z_hat)
            bs    = torch.tensor(z_hat.shape[0], device=device, dtype=torch.float32)
            total_loss_sum += loss.detach() * bs
            total_count    += bs
            if vbar is not None:
                vbar.update(1)
                vbar.set_postfix(val_loss=f"{loss.item():.4f}")

        if vbar is not None:
            vbar.close()

        total_loss_sum = _all_reduce_sum(total_loss_sum)
        total_count    = _all_reduce_sum(total_count)
        mean_val = (total_loss_sum / torch.clamp_min(total_count, 1.0)).item()

        if was_training:
            latent_model.train()
        if latent_ema is not None:
            latent_ema.restore()
        return mean_val
    # --------

    # try estimate steps/epoch; fall back to large epoch count
    try:
        steps_per_epoch = len(train_loader)
        max_epochs = math.ceil(post_steps / max(1, steps_per_epoch))
    except Exception:
        steps_per_epoch = None
        max_epochs = 1_000_000_000

    lat_mod.train()
    post_global_step = 0
    best_val = float("inf")
    worse_epochs = 0

    for post_epoch in range(max_epochs):
        # reshuffle per epoch (DDP)
        if ddp_is_active() and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(base_epoch_seed + post_epoch)

        pbar = None
        if use_tqdm and _tqdm is not None:
            desc = f"Latent Post-Train | Epoch {post_epoch+1}/{max_epochs if max_epochs < 10**9 else '∞'}"
            pbar = _tqdm(total=steps_per_epoch, leave=False, desc=desc) if steps_per_epoch is not None else _tqdm(leave=False, desc=desc)

        # accumulate train loss (DDP-averaged later)
        train_epoch_sum = torch.tensor(0.0, device=device)
        train_epoch_cnt = torch.tensor(0.0, device=device)

        for batch in train_loader:
            batch = prepare_batch(batch, device)
            x = batch[0]

            # AE-EMA encode & normalize
            with torch.no_grad():
                z_hat = ae_mod.normalize_latent(ae_mod.encode(x).detach())

            # one latent step
            post_opt.zero_grad(set_to_none=True)
            loss_lat = _make_latent_dsm_loss(latent_sde)(latent_model, z_hat)
            loss_lat.backward()
            clip_grad_norm_(lat_mod.parameters(), post_grad_clip)
            post_opt.step()
            if post_sched is not None:
                post_sched.step()
            latent_ema.update()

            # per-step logging
            if writer is not None and ddp_rank() == 0:
                writer.add_scalar("LatentDiffPost/Loss_iter", float(loss_lat.detach().item()), post_global_step)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss_lat.item():.4f}", step=post_global_step)

            # accumulate epoch mean (weighted by batch size)
            bs = torch.tensor(z_hat.shape[0], device=device, dtype=torch.float32)
            train_epoch_sum += loss_lat.detach() * bs
            train_epoch_cnt += bs

            post_global_step += 1
            if post_global_step >= post_steps:
                break  # mid-epoch stop

        if pbar is not None:
            pbar.close()

        # DDP average train epoch mean
        train_epoch_sum = _all_reduce_sum(train_epoch_sum)
        train_epoch_cnt = _all_reduce_sum(train_epoch_cnt)
        train_mean = (train_epoch_sum / torch.clamp_min(train_epoch_cnt, 1.0)).item()

        # validation
        val_mean = float("nan")
        do_val = (post_epoch % max(1, post_val_every) == 0)
        if do_val:
            val_mean = _run_validation(post_epoch)
            if writer is not None and ddp_rank() == 0:
                writer.add_scalar("LatentDiffPost/ValLoss_epoch", val_mean, post_epoch)

        # epoch-level logs + simple overfitting flag
        if writer is not None and ddp_rank() == 0:
            writer.add_scalar("LatentDiffPost/TrainLoss_epoch", train_mean, post_epoch)
            if do_val and not math.isnan(val_mean):
                if val_mean < best_val - 1e-12:
                    best_val = val_mean
                    worse_epochs = 0
                elif (val_mean - best_val) > overfit_delta:
                    worse_epochs += 1
                writer.add_scalar("LatentDiffPost/OverfitFlag", int(worse_epochs >= overfit_patience), post_epoch)
                writer.add_scalar("LatentDiffPost/TrainValGap", train_mean - val_mean, post_epoch)

        # generation callback (rank 0), every post_gen_every epochs, and always on final step
        do_gen = ((post_epoch + 1) % max(1, post_gen_every) == 0) or (post_global_step >= post_steps)
        if (ddp_rank() == 0 and do_gen and gen_cb is not None and latent_sde is not None):
            try:
                gen_cb(
                    writer=writer,
                    model=ae_mod,  # AE already in EMA mode globally
                    latent_model=(latent_model.module if ddp_is_active() else latent_model),
                    latent_sde=latent_sde,
                    device=device,
                    save_dir=tb_dir,
                    epoch=post_epoch,
                    ema_latent=latent_ema,   # sample with EMA latent model
                    ema_ae=None              # decoder already EMA via ema.apply_shadow()
                )
            except Exception as e:
                print(f"[Latent Post-Train] generation callback failed: {e}")

        if post_global_step >= post_steps:
            break

    # ---- Save final latent model (rank 0)
    if ddp_rank() == 0:
        save_model(
            lat_mod, latent_ema, -1, 0.0, save_tag, ckpt_dir,
            [], global_step, best_val_loss, epochs_no_improve, post_opt, post_sched
        )

    # ---- restore AE params after post-phase
    ema.restore()

    # return a few stats in case caller wants them
    return {
        "post_steps_done": post_global_step,
        "best_val": best_val,
        "overfit_warnings": worse_epochs,
    }


# ---------------- main training ----------------

def train(cfg):
    # DDP setup
    devices_cfg = getattr(cfg.training, "devices", None)  # e.g., "auto" or "0,1,2,3" or [0,1,2,3]
    device, rank, world_size, _ = setup_distributed(devices_cfg)

    # dirs + writer (rank0 only)
    tb_dir, ckpt_dir, _ = prepare_training_dirs(cfg)
    writer = SummaryWriter(log_dir=tb_dir) if rank == 0 else _NullWriter()

    # data (DDP-aware loaders + samplers)
    train_loader, val_loader, test_loader, samplers = get_dataloaders(
        cfg.data, seed=cfg.random_seed, distributed=ddp_is_active(), rank=rank, world_size=world_size, return_samplers=True
    )
    train_sampler: Optional[DistributedSampler] = samplers.get("train", None)

    # AE model (+DDP)
    base_model = get_model(cfg.model).to(device)
    if rank == 0 and hasattr(base_model, "print_model_summary"):
        base_model.print_model_summary()
    base_model = compile_model(base_model, bool(getattr(cfg.model, "compile", True)))

    if ddp_is_active():
        model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=True,  # decoder may be frozen on some steps
        )
        ema = EMA(model.module, decay=cfg.model.ema_decay)
    else:
        model = base_model
        ema = EMA(model, decay=cfg.model.ema_decay)

    # Optim/sched + (optional) resume
    ae_mod = model.module if ddp_is_active() else model
    optimizer, scheduler = get_optimizer_and_scheduler(ae_mod, cfg)
    start_epoch, global_step, best_ckpts, best_val_loss, epochs_no_improve, optimizer, scheduler = \
        resume_training(cfg, ae_mod, ema, load_model, get_optimizer_and_scheduler)

    # callbacks
    recon_cb  = get_reconstruction_callback()
    latent_cb = get_latent_scatter_callback(num_batches=20, max_points=5000, mode="both")
    latent_norm_cb = get_update_latent_normalizer_callback(min_count=5000, max_batches=None, tag_prefix="AE")

    # AMP setup
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype is torch.float16))
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()

    # -------- Latent diffusion (DDP-wrapped as well) --------
    geom_block = getattr(cfg.loss, "geom", None)
    gen_cb = None
    latent_model = latent_opt = latent_sched = latent_sde = latent_dsm_loss = latent_ema = None
    lat_cfg = None
    latent_geom_cfg = getattr(geom_block, "latent", None) if geom_block else None
    if latent_geom_cfg and bool(getattr(latent_geom_cfg, "enabled", False)):
        lat_cfg = load_config(latent_geom_cfg.diffusion_config)
        # ensure device/shape info
        lat_cfg.training.device = f"cuda:{device.index}" if device.type == "cuda" else "cpu"
        lat_cfg.data.latent_dim = cfg.model.latent_dim
        lat_cfg.data.shape = [cfg.model.latent_dim]
        lat_cfg.model.state_size = cfg.model.latent_dim

        steps_per_epoch_lat = len(train_loader) * int(lat_cfg.training.steps_per_ae)
        lat_cfg.optim.total_steps = steps_per_epoch_lat * cfg.training.epochs

        latent_sde = configure_sde(lat_cfg)
        latent_base = get_model(lat_cfg.model).to(device)
        latent_base.train()

        if ddp_is_active():
            latent_model = nn.parallel.DistributedDataParallel(
                latent_base,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=False
            )
            latent_ema = EMA(latent_model.module, decay=float(lat_cfg.model.ema_decay))
        else:
            latent_model = latent_base
            latent_ema = EMA(latent_model, decay=float(lat_cfg.model.ema_decay))

        latent_opt, latent_sched = get_optimizer_and_scheduler(
            latent_model.module if ddp_is_active() else latent_model, lat_cfg, global_step=0
        )
        latent_dsm_loss = _make_latent_dsm_loss(latent_sde)

        smp_steps = int(getattr(getattr(lat_cfg, "sampling", ml_collections.ConfigDict()), "steps", 250))
        smp_count = int(getattr(getattr(lat_cfg, "sampling", ml_collections.ConfigDict()), "num_samples", 36))
        smp_nrow  = getattr(getattr(lat_cfg, "sampling", ml_collections.ConfigDict()), "grid_nrow", None)
        gen_cb = get_generation_callback(sample_steps=smp_steps, sample_count=smp_count, grid_nrow=smp_nrow)

    # ============ Training loop ============
    for epoch in range(start_epoch, cfg.training.epochs):
        if ddp_is_active() and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        ae_mod.train()
        it = tqdm(train_loader, desc=f"[AE] Epoch {epoch+1}/{cfg.training.epochs}") if ddp_rank() == 0 else train_loader

        for data in it:
            batch = prepare_batch(data, device)
            x = batch[0]

            # ---- latent diffusion steps (k per AE step) ----
            if latent_model:
                lat_mod = latent_model.module if ddp_is_active() else latent_model
                lat_mod.train()
                _set_requires_grad(lat_mod, True)
                with torch.no_grad():
                    z_cur = ae_mod.encode(x).detach()
                    z_cur_hat = ae_mod.normalize_latent(z_cur)
                k_steps = int(lat_cfg.training.steps_per_ae)
                for _ in range(k_steps):
                    latent_opt.zero_grad(set_to_none=True)
                    loss_lat = latent_dsm_loss(latent_model, z_cur_hat)
                    loss_lat.backward()
                    nn.utils.clip_grad_norm_(lat_mod.parameters(), lat_cfg.optim.grad_clip)
                    latent_opt.step()
                    if latent_sched: latent_sched.step()
                    latent_ema.update()
                    if ddp_rank() == 0:
                        writer.add_scalar("LatentDiff/Loss_iter", float(loss_lat.detach().item()), global_step)

            # ---- AE step ----
            optimizer.zero_grad(set_to_none=True)
            ctx = freeze_eval(latent_model.module if (latent_model and ddp_is_active()) else latent_model) if latent_model else nullcontext()
            with autocast_ctx, ctx:
                loss, metrics = ae_loss(ae_mod, batch, cfg, device, train=True)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(ae_mod.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()

            if ddp_rank() == 0:
                writer.add_scalar("AE/Loss/train_iter", float(loss.detach().item()), global_step)
                for k, v in metrics.items():
                    if v is None: 
                        continue
                    try:
                        v = float(v)
                    except Exception:
                        continue
                    if not math.isfinite(v):
                        continue
                    writer.add_scalar(f"AE/{k}_iter", v, global_step)
                if isinstance(it, tqdm):
                    it.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        # ---- validation ----
        ema.apply_shadow()
        if latent_model: latent_ema.apply_shadow()
        ae_mod.eval()

        val_loss_sum = torch.tensor(0.0, device=device)
        val_batches  = torch.tensor(0.0, device=device)
        with torch.no_grad(), autocast_ctx:
            for data in val_loader:
                loss, _ = ae_loss(ae_mod, prepare_batch(data, device), cfg, device, train=False)
                val_loss_sum += loss.detach()
                val_batches  += torch.tensor(1.0, device=device)

        if ddp_is_active():
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches,  op=dist.ReduceOp.SUM)

        val_loss = (val_loss_sum / torch.clamp_min(val_batches, 1.0)).item()
        if ddp_rank() == 0:
            writer.add_scalar("AE/Loss/val", val_loss, epoch)

        if latent_model: latent_ema.restore()
        ema.restore()

        # ---- end-of-epoch latent μ/σ update (DDP-aware callback) ----
        # Get the update frequency from the config
        update_norm_freq = getattr(cfg.training, "update_norm_frequency", 1)

        if latent_model is not None and ((epoch + 1) % update_norm_freq == 0):
            latent_norm_cb(val_loader, writer if ddp_rank() == 0 else _NullWriter(), ae_mod, device, epoch)
            # Broadcast updated buffers so all ranks agree
            if ddp_is_active():
                with torch.no_grad():
                    mu = ae_mod.latent_norm_mean.clone().to(device)
                    sd = ae_mod.latent_norm_std.clone().to(device)
                    dist.broadcast(mu, src=0)
                    dist.broadcast(sd, src=0)
                    ae_mod.set_latent_normalization(mu, sd)

        # ---- visualization (rank 0) ----
        if ddp_rank() == 0 and ((epoch + 1) % cfg.training.vis_frequency == 0):
            batch = next(iter(val_loader))
            recon_cb(batch, writer, ae_mod, device, epoch, tag_prefix="AE")
            latent_cb(val_loader, writer, ae_mod, device, epoch, tag_prefix="AE")
            if gen_cb is not None and latent_model is not None and latent_sde is not None:
                gen_cb(
                    writer=writer,
                    model=ae_mod,
                    latent_model=(latent_model.module if ddp_is_active() else latent_model),
                    latent_sde=latent_sde,
                    device=device,
                    save_dir=tb_dir,
                    epoch=epoch,
                )

        # ---- early stopping / ckpt (rank 0 decides) ----
        stop_flag = torch.tensor(0, device=device, dtype=torch.int32)
        if ddp_rank() == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.patience_epochs:
                print(f"[AE] Early stopping at epoch {epoch+1}")
                stop_flag.fill_(1)
            if (epoch + 1) % cfg.training.checkpoint_frequency == 0:
                save_model(ae_mod, ema, epoch, val_loss, "AE", ckpt_dir, best_ckpts,
                           global_step, best_val_loss, epochs_no_improve, optimizer, scheduler)
                if latent_model:
                    lat_mod = latent_model.module if ddp_is_active() else latent_model
                    save_model(lat_mod, latent_ema, epoch, val_loss, "LatentDiff", ckpt_dir,
                               [], global_step, best_val_loss, epochs_no_improve, latent_opt, latent_sched)
        if ddp_is_active():
            dist.broadcast(stop_flag, src=0)
        if int(stop_flag.item()) == 1:
            break

    # ===================== POST LATENT-ONLY TRAINING =====================
    # Continue training the latent diffusion model AFTER AE finishes
    if latent_model is not None:
        _ = post_train_latent(
            latent_model=latent_model,
            latent_ema=latent_ema,
            ae_mod=ae_mod,
            ema=ema,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            lat_cfg=lat_cfg,
            latent_sde=latent_sde,
            writer=writer,
            device=device,
            tb_dir=tb_dir,
            ckpt_dir=ckpt_dir,
            global_step=global_step,
            best_val_loss=best_val_loss,
            epochs_no_improve=epochs_no_improve,
            gen_cb=gen_cb,
            _make_latent_dsm_loss=_make_latent_dsm_loss,
            get_optimizer_and_scheduler=get_optimizer_and_scheduler,
            prepare_batch=prepare_batch,
            save_model=save_model,
        )

    if ddp_rank() == 0 and writer is not None:
        writer.close()
    finalize_distributed(device=device)


def main():
    P = argparse.ArgumentParser("AutoEncoder trainer (DDP) + latent diffusion post-train")
    P.add_argument("--config", required=True, help="Path to the AE configuration (.py).")
    args = P.parse_args()

    cfg = load_config(args.config)

    # Optional multi-GPU selection field; keeps single-GPU backward-compat
    if not hasattr(cfg.training, "devices"):
        setattr(cfg.training, "devices", None)

    out_dir = Path(cfg.base_log_dir) / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.pkl", "wb") as f:
        pickle.dump(cfg.to_dict(), f)

    set_seed(cfg.random_seed)
    train(cfg)


if __name__ == "__main__":
    main()
