import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import pickle

from data import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import prepare_training_dirs, prepare_batch, print_model_size, EMA, save_model, load_model, get_score_fn, eval_callback, resume_training
from utils.sampling_utils import generation_callback
from utils.optim_utils import get_optimizer_and_scheduler
from torch.distributions import Uniform
from loss import get_loss_fn
from configs import load_config
from evaluation.fid import fid_evaluation_callback

def train(config):
    tensorboard_dir, checkpoint_dir, eval_dir = prepare_training_dirs(config)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    device = torch.device(config.training.device)

    train_loader, val_loader, test_loader = get_dataloaders(config.data)
    
    # Create the model
    model = get_model(config.model)
    model = model.to(device)
    print_model_size(model)

    sde = configure_sde(config)
    ema_model = EMA(model=model, decay=config.model.ema_decay)

    # Setup the optimizer and the scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)

    # Resuming: Load from checkpoint if provided and reinitialize optimizer and scheduler
    epoch, global_step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler = resume_training(config, model, ema_model, load_model, get_optimizer_and_scheduler)

    t_dist = Uniform(sde.sampling_eps, 1)
    loss_fn = get_loss_fn(config.training.loss, sde, t_dist, likelihood_weighting=config.training.likelihood_weighting)

    for epoch in range(epoch, config.training.epochs):
        model.train()
        train_loss = 0
        for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{config.training.epochs}"):
            batch = prepare_batch(data, device)

            optimizer.zero_grad()
            loss = loss_fn(model, batch)

            loss.backward()
            if config.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
            optimizer.step()
            ema_model.update()
            scheduler.step()

            train_loss += loss.item()
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            global_step += 1

        train_loss /= len(train_loader)
        writer.add_scalar('Loss/Train_epoch', train_loss, epoch)

        ema_model.apply_shadow()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{config.training.epochs}"):
                batch = prepare_batch(data, device)
                loss = loss_fn(model, batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        if (epoch + 1) % config.training.vis_frequency == 0:
            steps = config.training.steps
            num_samples = config.training.num_samples
            shape = (num_samples, *config.data.shape)

            data = next(iter(val_loader))
            batch = prepare_batch(data, device)
            _, y = batch

            generation_callback(y, writer, sde, model, steps, shape, device, epoch)

        if (epoch + 1) % config.training.fid_eval_frequency == 0:
            steps = config.training.steps
            num_samples = config.training.num_samples
            shape = (num_samples, *config.data.shape)

            dataloaders = [train_loader, val_loader]
            fid_evaluation_callback(writer, sde, model, steps, shape, device, epoch, dataloaders, train=True)
            
            dataloaders = [test_loader]
            fid_evaluation_callback(writer, sde, model, steps, shape, device, epoch, dataloaders, train=False)

        ema_model.restore()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.training.patience_epochs:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % config.training.checkpoint_frequency == 0:
            save_model(model, ema_model, epoch, val_loss, "Model", checkpoint_dir, best_checkpoints, global_step, best_val_loss, epochs_no_improve, optimizer, scheduler)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script for Diffusion/Score Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Save configuration to a file
    config_dir = os.path.join(config.base_log_dir, config.experiment)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config.to_dict(), f)

    train(config)
