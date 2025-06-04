import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import pickle
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import prepare_training_dirs, prepare_batch, EMA, save_model, get_score_fn, eval_callback
from utils.sampling_utils import get_generation_callback
from utils.optim_utils import get_optimizer_and_scheduler
from torch.distributions import Uniform
from loss import get_loss_fn
from configs import load_config
from evaluation.fid import fid_evaluation_callback

def save_config(config):
    config_dir = os.path.join(config.base_log_dir, config.experiment)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config.to_dict(), f)

def setup_ddp(rank, world_size):
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    return torch.device(f'cuda:{rank}')

def cleanup_ddp():
    dist.destroy_process_group()

def get_distributed_loader(dataset, batch_size, world_size, rank, shuffle=True, drop_last=True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last)
    return loader, sampler

def train_one_epoch_ddp(
    model, ema_model, train_loader, train_sampler, optimizer, scheduler,
    loss_fn, sde, device, writer, global_step, grad_clip, epoch, total_epochs, rank
):
    model.train()
    train_loss = 0
    train_sampler.set_epoch(epoch)  # shuffle
    for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{total_epochs}", disable=rank != 0):
        batch = prepare_batch(data, device)
        optimizer.zero_grad()
        loss = loss_fn(model, batch, train=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        ema_model.update()
        scheduler.step()
        train_loss += loss.item()
        if rank == 0:
            writer.add_scalar('Loss/Train', loss.item(), global_step)
        global_step += 1
    train_loss /= len(train_loader)
    if rank == 0:
        writer.add_scalar('Loss/Train_epoch', train_loss, epoch)
    return train_loss, global_step

def validate_one_epoch_ddp(model, ema_model, val_loader, loss_fn, sde, device, writer, epoch, total_epochs, rank):
    ema_model.apply_shadow()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{total_epochs}", disable=rank != 0):
            batch = prepare_batch(data, device)
            loss = loss_fn(model, batch, train=False)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    if rank == 0:
        writer.add_scalar('Loss/Validation', val_loss, epoch)
    ema_model.restore()
    return val_loss

def run_callbacks_ddp(config, model, ema_model, val_loader, train_loader, test_loader, writer,
                  sde, epoch, device, rank):
    if rank != 0:
        return
    if (epoch + 1) % config.training.vis_frequency == 0:
        steps = config.training.steps
        num_samples = config.training.num_samples
        shape = (num_samples, *config.data.shape)
        data = next(iter(val_loader))
        batch = prepare_batch(data, device)
        generation_callback = get_generation_callback(
            config.training.vis_callback,
            G=config.training.guidance_weight,
            guidance_interval=config.training.guidance_interval,
            )
        generation_callback(batch, writer, sde, model, steps, shape, device, epoch)

def train_ddp(rank, world_size, config):
    device = setup_ddp(rank, world_size)
    tensorboard_dir, checkpoint_dir, eval_dir = prepare_training_dirs(config)
    writer = SummaryWriter(log_dir=tensorboard_dir) if rank == 0 else None

    train_loader, val_loader, test_loader = get_dataloaders(config.data)
    train_loader, train_sampler = get_distributed_loader(
        train_loader.dataset, config.data.batch_size, world_size, rank
    )

    model = get_model(config.model).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    # if rank == 0:
    #     print_model_size(model)

    sde = configure_sde(config)
    ema_model = EMA(model=model, decay=config.model.ema_decay)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    global_step = 0
    best_checkpoints = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    t_dist = Uniform(sde.sampling_eps, 1)
    loss_fn = get_loss_fn(config, sde, t_dist)

    for epoch in range(config.training.epochs):
        train_loss, global_step = train_one_epoch_ddp(
            model, ema_model, train_loader, train_sampler, optimizer, scheduler,
            loss_fn, sde, device, writer, global_step, config.optim.grad_clip, epoch, config.training.epochs, rank
        )
        val_loss = validate_one_epoch_ddp(
            model, ema_model, val_loader, loss_fn, sde, device, writer, epoch, config.training.epochs, rank
        )
        run_callbacks_ddp(
            config, model, ema_model, val_loader, train_loader, test_loader, writer,
            sde, epoch, device, rank
        )

        # Early stopping and checkpointing (only on rank 0)
        if rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= config.training.patience_epochs:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            if (epoch + 1) % config.training.checkpoint_frequency == 0:
                save_model(
                    model, ema_model, epoch, val_loss, "Model", checkpoint_dir,
                    best_checkpoints
                )
    if rank == 0:
        writer.close()
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description="Distributed Training Script for Diffusion/Score Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    config = load_config(args.config)
    save_config(config)
    world_size = min(config.training.gpus, torch.cuda.device_count())

    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        train_ddp(rank, world_size, config)
    else:
        mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
    