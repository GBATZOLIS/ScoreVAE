#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from typing import Optional
from contextlib import contextmanager, nullcontext
import math

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import ml_collections

from configs import load_config
from models import get_model
from utils.train_utils import (
    prepare_training_dirs, prepare_batch, EMA, save_model, load_model, resume_training
)
from utils.ae_utils import get_reconstruction_callback, get_latent_scatter_callback, get_update_latent_normalizer_callback, get_generation_callback
from utils.optim_utils import get_optimizer_and_scheduler
from data.data_utils_fast import get_dataloaders
from loss.ae_loss import ae_loss

# latent diffusion utilities
from data_geometry.utils.diffusion_utils import get_score_fn as dg_get_score_fn, get_denoiser_fn as dg_get_denoiser_fn
from data_geometry.metrics import create_metric
from sde import configure_sde


# ---------------- utils ----------------

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
    """Temporarily set module to eval and disable requires_grad on its parameters."""
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

def _load_pretrained_diffusion(diff_cfg, device: torch.device):
    """Load a diffusion model for DATA-space metric if enabled (not used by default)."""
    from utils.train_utils import EMA as _EMA, load_model as _load_model
    from models import get_model as _get_model

    model = _get_model(diff_cfg.model).to(device)
    sde   = configure_sde(diff_cfg)
    ema   = _EMA(model, decay=diff_cfg.model.ema_decay)

    ckpt_path = diff_cfg.model.checkpoint
    if not Path(ckpt_path).is_absolute():
        ckpt_path = str(Path(diff_cfg.checkpoint_dir) / ckpt_path)
    if not ckpt_path.endswith(".pth"):
        ckpt_path += ".pth"

    _load_model(model, ema, ckpt_path, "Model", device=device, is_ema=True)
    ema.apply_shadow()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        from torch._inductor import config as inductor_cfg
        inductor_cfg.triton.cudagraphs = False
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        print("[diffusion] compiled ✓")
    except Exception as e:
        print(f"[diffusion] compile fallback: {e}")

    return model, sde

def _build_metric_from_diffusion(
    *,
    metric_type: str,
    lam_metric: float,
    t_value: float,
    sde,
    diff_model: nn.Module,
    orig_shape: tuple[int, ...],
    device: torch.device,
    use_denoiser_retraction: bool = True,
    cg_kwargs: Optional[dict] = None,
):
    t_tensor = torch.tensor(float(t_value), dtype=torch.float32, device=device)
    score_fn    = dg_get_score_fn(sde, diff_model, t_tensor, orig_shape)
    denoiser_fn = dg_get_denoiser_fn(sde, diff_model, t_tensor, orig_shape) if use_denoiser_retraction else None
    return create_metric(
        metric_type=metric_type,
        score_fn=score_fn,
        lam_metric=float(lam_metric),
        denoiser_fn=denoiser_fn,
        cg_kwargs=cg_kwargs or {},
    )

def _make_latent_dsm_loss(sde):
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

# ---------------- main train ----------------

def train(cfg):
    tb_dir, ckpt_dir, _ = prepare_training_dirs(cfg)
    writer = SummaryWriter(log_dir=tb_dir)
    device = torch.device(cfg.training.device)

    # data
    train_loader, val_loader, test_loader = get_dataloaders(cfg.data, seed=cfg.random_seed)

    # AE model
    model = get_model(cfg.model).to(device, memory_format=torch.channels_last)
    model.print_model_summary()
    model = compile_model(model, bool(getattr(cfg.model, "compile", True)))
    ema = EMA(model, decay=cfg.model.ema_decay)

    optimizer, scheduler = get_optimizer_and_scheduler(model, cfg)
    start_epoch, global_step, best_ckpts, best_val_loss, epochs_no_improve, optimizer, scheduler = \
        resume_training(cfg, model, ema, load_model, get_optimizer_and_scheduler)

    recon_cb  = get_reconstruction_callback()
    latent_cb = get_latent_scatter_callback(num_batches=20, max_points=5000, mode="both")
    latent_norm_cb = get_update_latent_normalizer_callback(
        min_count=5000, max_batches=None, tag_prefix="AE" )

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16))
    
    # Latent diffusion
    geom_block = getattr(cfg.loss, "geom", None)
    gen_cb = None
    latent_model = latent_opt = latent_sched = latent_sde = latent_dsm_loss = latent_ema = None
    latent_geom_cfg = getattr(geom_block, "latent", None) if geom_block else None
    if latent_geom_cfg and bool(getattr(latent_geom_cfg, "enabled", False)):
        lat_cfg = load_config(latent_geom_cfg.diffusion_config)
        lat_cfg.training.device = cfg.training.device
        lat_cfg.data.latent_dim = cfg.model.latent_dim
        lat_cfg.data.shape = [cfg.model.latent_dim]
        lat_cfg.model.state_size = cfg.model.latent_dim

        steps_per_epoch_lat = len(train_loader) * int(lat_cfg.training.steps_per_ae)
        lat_cfg.optim.total_steps = steps_per_epoch_lat * cfg.training.epochs

        latent_sde = configure_sde(lat_cfg)
        latent_model = get_model(lat_cfg.model).to(device)
        latent_model.train()
        latent_opt, latent_sched = get_optimizer_and_scheduler(latent_model, lat_cfg, global_step=0)
        latent_ema = EMA(latent_model, decay=float(lat_cfg.model.ema_decay))
        latent_dsm_loss = _make_latent_dsm_loss(latent_sde)

        # Optional sampling defaults from the latent config (harmless if missing)
        smp_steps = int(getattr(getattr(lat_cfg, "sampling", ml_collections.ConfigDict()), "steps", 250))
        smp_count = int(getattr(getattr(lat_cfg, "sampling", ml_collections.ConfigDict()), "num_samples", 36))
        smp_nrow  = getattr(getattr(lat_cfg, "sampling", ml_collections.ConfigDict()), "grid_nrow", None)
        gen_cb = get_generation_callback(sample_steps=smp_steps, sample_count=smp_count, grid_nrow=smp_nrow)


    # ============ Training loop ============
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[AE] Epoch {epoch+1}/{cfg.training.epochs}")

        for data in pbar:
            batch = prepare_batch(data, device)
            x = batch[0]

            # ---- latent diffusion steps ----
            if latent_model:
                latent_model.train()
                _set_requires_grad(latent_model, True)
                with torch.no_grad():
                    z_cur = model.encode(x).detach()
                    z_cur_hat = model.normalize_latent(z_cur)   # ← normalize for latent diffusion
                k_steps = int(lat_cfg.training.steps_per_ae)
                for _ in range(k_steps):
                    latent_opt.zero_grad(set_to_none=True)
                    loss_lat = latent_dsm_loss(latent_model, z_cur_hat)  # ← normalized
                    loss_lat.backward()
                    nn.utils.clip_grad_norm_(latent_model.parameters(), lat_cfg.optim.grad_clip)
                    latent_opt.step()
                    if latent_sched: latent_sched.step()
                    latent_ema.update()
                    writer.add_scalar("LatentDiff/Loss_iter", float(loss_lat.detach().item()), global_step)


            # ---- AE step ----
            optimizer.zero_grad(set_to_none=True)
            ctx = freeze_eval(latent_model) if latent_model else nullcontext()
            with torch.amp.autocast("cuda", dtype=amp_dtype), ctx:
                loss, metrics = ae_loss(model, batch, cfg, device, train=True)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()

            running += loss.item()
            writer.add_scalar("AE/Loss/train_iter", float(loss.detach().item()), global_step)
            for k, v in metrics.items():
                if v is None:
                    continue  # skip on steps where the metric wasn't computed
                try:
                    v = float(v)
                except Exception:
                    continue
                if not math.isfinite(v):
                    continue

                writer.add_scalar(f"AE/{k}_iter", v, global_step)  # use global_step for all        
            
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---- validation ----
        ema.apply_shadow()
        if latent_model: latent_ema.apply_shadow()
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            for data in val_loader:
                loss, _ = ae_loss(model, prepare_batch(data, device), cfg, device, train=False)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar("AE/Loss/val", val_loss, epoch)
        if latent_model: latent_ema.restore()
        ema.restore()

        # ---- end-of-epoch μ/σ update for latent normalization (if latent diffusion is used) ----
        if latent_model is not None:
            latent_norm_cb(val_loader, writer, model, device, epoch)

        # ---- visualization ----
        if (epoch + 1) % cfg.training.vis_frequency == 0:
            batch = next(iter(val_loader))
            recon_cb(batch, writer, model, device, epoch, tag_prefix="AE")
            latent_cb(val_loader, writer, model, device, epoch, tag_prefix="AE")

            # NEW: latent sampling → denormalize → decode → grid
            if gen_cb is not None and latent_model is not None and latent_sde is not None:
                gen_cb(
                    writer=writer,
                    model=model,
                    latent_model=latent_model,
                    latent_sde=latent_sde,
                    device=device,
                    save_dir=tb_dir,          # or a subfolder like f"{tb_dir}/samples"
                    epoch=epoch,
                    #ema_latent=latent_ema,    # optional; nice for smooth previews
                    # ema_ae=ema,             # enable if you want EMA decode during training
                )

        # ---- early stopping / ckpt ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= cfg.training.patience_epochs:
            print(f"[AE] Early stopping at epoch {epoch+1}")
            break
        if (epoch + 1) % cfg.training.checkpoint_frequency == 0:
            save_model(model, ema, epoch, val_loss, "AE", ckpt_dir, best_ckpts,
                       global_step, best_val_loss, epochs_no_improve, optimizer, scheduler)
            if latent_model:
                save_model(latent_model, latent_ema, epoch, val_loss, "LatentDiff", ckpt_dir,
                           [], global_step, best_val_loss, epochs_no_improve, latent_opt, latent_sched)

    writer.close()


def main():
    P = argparse.ArgumentParser("AutoEncoder trainer with curvature regularization")
    P.add_argument("--config", required=True, help="Path to the AE configuration (.py).")
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
