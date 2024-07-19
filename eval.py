import os
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import pickle

from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import prepare_training_dirs, EMA, load_model
from utils.sampling_utils import generation_callback
from configs import load_config
from evaluation.fid import fid_evaluation_callback

def eval_fid(config):
    _, checkpoint_dir, eval_dir = prepare_training_dirs(config)
    writer = SummaryWriter(log_dir=eval_dir)
    device_ids = config.evaluation.devices

    train_loader, val_loader, test_loader = get_dataloaders(config.data)

    model = get_model(config.model)
    sde = configure_sde(config)
    ema_model = EMA(model=model, decay=config.model.ema_decay)

    checkpoint_path = config.model.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    if not checkpoint_path.endswith('.pth'):
        checkpoint_path += '.pth'
    
    load_model(model, ema_model, checkpoint_path, "Model", is_ema=False)
    ema_checkpoint_path = checkpoint_path.replace(".pth", "_EMA.pth")
    load_model(model, ema_model, ema_checkpoint_path, "Model", is_ema=True)

    ema_model.apply_shadow()

    steps = config.training.steps
    samples_per_batch = config.training.num_samples
    shape = (samples_per_batch, *config.data.shape)

    # Generate the plot and save it in eval_dir
    #practical_infer_timesteps(sde, steps)

    # Generate samples and save to TensorBoard
    #generation_callback(None, writer, sde, model, steps, shape, torch.device(f'cuda:{device_ids[0]}'), 0)

    dataloaders = [train_loader, val_loader]
    fid_evaluation_callback(writer, sde, model, steps, shape, device_ids, 0, dataloaders, train=True)   

    dataloaders = [test_loader]
    fid_evaluation_callback(writer, sde, model, steps, shape, device_ids, 0, dataloaders, train=False)

    ema_model.restore()
    writer.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Use 'spawn' start method
    parser = ArgumentParser(description="FID Evaluation Script for Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    config_dir = os.path.join(config.base_log_dir, config.experiment)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config.to_dict(), f)

    eval_fid(config)
