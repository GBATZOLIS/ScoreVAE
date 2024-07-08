import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import argparse
import os
import pickle
import torch.distributed as dist
import torch.multiprocessing as mp

from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import prepare_training_dirs, prepare_batch, print_model_size, EMA, save_model, get_score_fn, eval_callback
from utils.sampling_utils import generation_callback
from utils.optim_utils import get_optimizer_and_scheduler
from torch.distributions import Uniform
from loss import get_loss_fn
from configs import load_config

def train(rank, world_size, config):
    # Initialize process group for distributed training
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    tensorboard_dir, checkpoint_dir, eval_dir = prepare_training_dirs(config)
    writer = SummaryWriter(log_dir=tensorboard_dir) if rank == 0 else None

    train_loader, val_loader, test_loader = get_dataloaders(config.data)
    
    # Use DistributedSampler for training data
    train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_loader.dataset, batch_size=config.data.batch_size, sampler=train_sampler)

    # Create the model
    model = get_model(config).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    if rank == 0:
        print_model_size(model)

    sde = configure_sde(config)
    ema_model = EMA(model=model, decay=config.model.ema_decay)

    # Setup the optimizer and the scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)

    global_step = 0
    best_checkpoints = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    t_dist = Uniform(sde.sampling_eps, 1)
    loss_fn = get_loss_fn(config.training.loss, sde, t_dist, likelihood_weighting=config.training.likelihood_weighting)

    for epoch in range(config.training.epochs):
        model.train()
        train_loss = 0
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling for each epoch

        for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{config.training.epochs}", disable=rank != 0):
            batch = prepare_batch(data, device)

            optimizer.zero_grad()
            score_fn = get_score_fn(sde, model)
            loss = loss_fn(score_fn, batch)

            loss.backward()
            if config.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
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

        ema_model.apply_shadow()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{config.training.epochs}", disable=rank != 0):
                batch = prepare_batch(data, device)
                score_fn = get_score_fn(sde, model)
                loss = loss_fn(score_fn, batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        if rank == 0:
            writer.add_scalar('Loss/Validation', val_loss, epoch)

        if rank == 0 and (epoch + 1) % config.training.vis_frequency == 0:
            steps = config.training.steps
            num_samples = config.training.num_samples
            shape = (num_samples, config.data.ambient_dim)

            data = next(iter(val_loader))
            batch = prepare_batch(data, device)
            _, y = batch

            generation_callback(y, writer, sde, model, steps, shape, device, epoch)

        ema_model.restore()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.training.patience_epochs:
            if rank == 0:
                print(f"Early stopping at epoch {epoch + 1}")
            break

        if rank == 0 and (epoch + 1) % config.training.checkpoint_frequency == 0:
            save_model(model, ema_model, epoch, val_loss, "Model", checkpoint_dir, best_checkpoints)

    if rank == 0:
        writer.close()

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Training Script for Diffusion/Score Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    world_size = min(config.training.gpus, torch.cuda.device_count())
    if 'SLURM_PROCID' in os.environ:
        # Running on a SLURM cluster
        rank = int(os.environ['SLURM_PROCID'])
        train(rank, world_size, config)
    else:
        # Running locally with multiple GPUs
        mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
