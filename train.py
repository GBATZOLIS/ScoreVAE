import os
import torch
import argparse
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import (
    prepare_training_dirs, 
    prepare_batch, 
    EMA, 
    save_model, 
    load_model, 
    resume_training
)
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

def train_one_epoch(
    model, ema_model, train_loader, optimizer, scheduler, 
    loss_fn, device, writer, global_step, grad_clip, epoch, total_epochs
):
    model.train()
    train_loss = 0
    for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{total_epochs}"):
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
        writer.add_scalar('Loss/Train', loss.item(), global_step)
        global_step += 1
    train_loss /= len(train_loader)
    writer.add_scalar('Loss/Train_epoch', train_loss, epoch)
    return train_loss, global_step

def validate_one_epoch(model, ema_model, val_loader, loss_fn, device, writer, epoch, total_epochs):
    ema_model.apply_shadow()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{total_epochs}"):
            batch = prepare_batch(data, device)
            loss = loss_fn(model, batch, train=False)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    ema_model.restore()
    return val_loss

def run_callbacks(
    config, model, ema_model, val_loader, train_loader, test_loader, writer, 
    sde, epoch, device
):
    # Visualization callback
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

    # FID evaluation callback
    if (epoch + 1) % config.training.fid_eval_frequency == 0:
        steps = config.training.steps
        num_samples = config.training.num_samples
        shape = (num_samples, *config.data.shape)
        fid_evaluation_callback(writer, sde, model, steps, shape, device, epoch, [train_loader, val_loader], train=True)
        fid_evaluation_callback(writer, sde, model, steps, shape, device, epoch, [test_loader], train=False)

def train_loop(config):
    tensorboard_dir, checkpoint_dir, eval_dir = prepare_training_dirs(config)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    device = torch.device(config.training.device)

    train_loader, val_loader, test_loader = get_dataloaders(config.data)
    model = get_model(config.model).to(device)
    model.print_model_summary()
    sde = configure_sde(config)
    ema_model = EMA(model=model, decay=config.model.ema_decay)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    epoch, global_step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler = resume_training(
        config, model, ema_model, load_model, get_optimizer_and_scheduler
    )
    t_dist = Uniform(sde.sampling_eps, 1)
    loss_fn = get_loss_fn(config, sde, t_dist)

    for epoch in range(epoch, config.training.epochs):
        train_loss, global_step = train_one_epoch(
            model, ema_model, train_loader, optimizer, scheduler, 
            loss_fn, device, writer, global_step, config.optim.grad_clip, epoch, config.training.epochs
        )
        val_loss = validate_one_epoch(
            model, ema_model, val_loader, loss_fn, device, writer, epoch, config.training.epochs
        )
        run_callbacks(
            config, model, ema_model, val_loader, train_loader, test_loader, writer, 
            sde, epoch, device
        )

        # Early stopping and checkpointing
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
                best_checkpoints, global_step, best_val_loss, epochs_no_improve, optimizer, scheduler
            )
    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Training Script for Diffusion/Score Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    config = load_config(args.config)
    save_config(config)
    train_loop(config)

if __name__ == "__main__":
    main()
