#!/usr/bin/env python
import os
import sys
import time
import pickle
import torch
import torch.multiprocessing as mp
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

# ---------------------------
# Diffusion code imports (as in your eval_fid script)
# ---------------------------
from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import prepare_training_dirs, EMA, load_model
from configs import load_config  # This loads an ml_collections.ConfigDict from a Python file.
from evaluation.fid import fid_evaluation_callback

# ---------------------------
# Riemannian optimization imports (from your external data_geometry library)
# ---------------------------
from data_geometry.optim_function import get_optim_function
from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from data_geometry.riemannian_optimization.optimizers import get_riemannian_optimizer
from data_geometry.utils.visualization import visualize_riemannian_optimization_selector

def flatten_tensor(x):
    """Flattens a tensor across all dimensions except the batch dimension."""
    return x.view(x.size(0), -1)

def unflatten_tensor(x, orig_shape):
    """Unflattens a tensor to the specified shape (excluding batch dimension).
    
    Args:
      x (Tensor): Tensor of shape (B, prod(orig_shape)).
      orig_shape (tuple): The desired shape for each sample, e.g. (C, W, H)
    
    Returns:
      Tensor of shape (B, *orig_shape)
    """
    return x.view(x.size(0), *orig_shape)


def ensure_time_tensor(t, batch_size):
    """
    Ensures that the time tensor `t` is a 1-D tensor of shape (batch_size,).
    
    Parameters:
      t (torch.Tensor): The original time tensor (could be scalar or 1D).
      batch_size (int): The desired batch size.
    
    Returns:
      torch.Tensor: A 1-D tensor of shape (batch_size,) populated with the same t value.
    """
    if t.dim() == 0:
        return t.expand(batch_size)
    elif t.dim() == 1:
        return t if t.size(0) == batch_size else t.expand(batch_size)
    else:
        # If t has an extra singleton dimension, squeeze it.
        if t.size(-1) == 1:
            return t.squeeze(-1)
        return t

def get_score_fn(sde, model, t, orig_shape):
    sigma_fn = sde.get_sigma_fn()
        
    def score_fn(x_t):  # x_t is expected to be flattened: shape (B, d)
        batch_size = x_t.size(0)
        t_corrected = ensure_time_tensor(t, batch_size)
        
        sigma_t = sigma_fn(t_corrected)
        # Ensure sigma_t has shape [B, 1] for correct broadcasting
        sigma_t = sigma_t.view(batch_size, 1)
        
        # Unflatten x_t using the provided original shape.
        x_t_tensor_shape = unflatten_tensor(x_t, orig_shape)
        # Pass the corrected time tensor to the model
        noise_pred = model(x_t_tensor_shape, None, t_corrected)
        noise_pred_flat = flatten_tensor(noise_pred)
        score = -noise_pred_flat / sigma_t
        return score
        
    return score_fn

def get_denoiser_fn(sde, model, t, orig_shape):
    alpha_fn = sde.get_alpha_fn()
    sigma_fn = sde.get_sigma_fn()
    
    def denoiser_fn(x_t):  # x_t is expected to be flattened: shape (B, d)
        batch_size = x_t.size(0)
        # Ensure t is a 1-D tensor of shape [B]
        t_corrected = ensure_time_tensor(t, batch_size)
        
        sigma_t = sigma_fn(t_corrected)
        alpha_t = alpha_fn(t_corrected)
        # Reshape sigma_t and alpha_t to (B, 1) for correct broadcasting.
        sigma_t_flat = sigma_t.view(batch_size, 1)
        alpha_t_flat = alpha_t.view(batch_size, 1)
        
        # Unflatten x_t for the model's forward pass.
        x_t_tensor_shape = unflatten_tensor(x_t, orig_shape)
        # Pass the corrected time tensor to the model
        noise_pred = model(x_t_tensor_shape, None, t_corrected)
        noise_pred_flat = flatten_tensor(noise_pred)
        
        x_denoised_flat = (x_t - sigma_t_flat * noise_pred_flat) / alpha_t_flat
        return x_denoised_flat
    
    return denoiser_fn


def load_riemannian_config(path):
    """
    Load a Python-based config file (e.g., 'riem_config.py') by executing it
    into a temporary dictionary, and return the global CONFIG dictionary.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    config_dict = {}
    with open(path, "r") as f:
        code = compile(f.read(), path, 'exec')
        exec(code, config_dict)
    if "CONFIG" not in config_dict:
        raise ValueError(f"No 'CONFIG' dictionary found inside {path}")
    return config_dict["CONFIG"]

def get_custom_optim_function(x_target):
    def opt_fn(x):
        """
        Compute the objective as the scaled squared L2 norm between x and the target.
        
        Args:
          x (Tensor): A flattened tensor of shape (B, d).
          
        Returns:
          Tensor: A tensor of shape (B,) containing the objective value for each sample.
        """
        return torch.sum((x - x_target) ** 2, dim=1)
    return opt_fn

# ---------------------------
# Main riemannian optimization function.
# Receives two separate configs: one for diffusion and one for riemannian optimization.
# ---------------------------
def riemannian_optimization(diff_config, riem_config):
    device = torch.device(diff_config.training.device)
    print(f'Optimization takes place on device:{device}')

    # Prepare logging directories (from the diffusion config)
    _, checkpoint_dir, eval_dir = prepare_training_dirs(diff_config)
    #writer = SummaryWriter(log_dir=eval_dir)

    # Load the diffusion dataset using standard dataloaders.
    train_loader, val_loader, test_loader = get_dataloaders(diff_config.data)
    
    # Select initial point(s) from the **entire** training dataset.
    dataset = train_loader.dataset
    num_points = riem_config.get("num_points", 2)
    
    # Convert dataset to a full tensor if necessary
    if isinstance(dataset, torch.utils.data.Dataset):
        dataset_tensor = torch.stack([dataset[i][0] if isinstance(dataset[i], tuple) else dataset[i] for i in range(len(dataset))])
    else:
        dataset_tensor = dataset  # If dataset is already a tensor

    # Select the first `num_points` samples
    x0 = dataset_tensor[:num_points].to(device)
    orig_shape = x0.shape[1:]  
    x0_flat = flatten_tensor(x0) #the Riemannian optimization library expects flattened tensors.
    riem_config["initial_point"] = x0_flat
    # ---------------------------
    # Load the diffusion model, SDE, and EMA (from the diffusion config)
    # ---------------------------
    model = get_model(diff_config.model)
    model = model.to(device)
    sde = configure_sde(diff_config)
    ema_model = EMA(model=model, decay=diff_config.model.ema_decay)
    
    checkpoint_path = diff_config.model.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(diff_config.checkpoint_dir, checkpoint_path)
    if not checkpoint_path.endswith('.pth'):
        checkpoint_path += '.pth'
    
    #load_model(model, ema_model, checkpoint_path, "Model", is_ema=False)
    #ema_checkpoint_path = checkpoint_path.replace(".pth", "_EMA.pth")
    load_model(model, ema_model, checkpoint_path, "Model", device=device, is_ema=True)
    ema_model.apply_shadow()
    model.eval()


    # ---------------------------
    # Set the diffusion time and obtain SDE functions.
    # Note that the riemannian config holds parameters like 'time_for_perturbation'
    # ---------------------------
    t_val = riem_config["time_for_perturbation"]
    t_diff = torch.tensor(t_val, dtype=torch.float32, device=device)
    SNR = sde.snr(t_diff)
    print(f'Riemannian optimisation at SNR:{SNR.item()}')

    alpha_fn = sde.get_alpha_fn()
    sigma_fn = sde.get_sigma_fn()

    # Create the score function using the diffusion model.
    score_fn = get_score_fn(sde, model, t_diff, orig_shape)

    # Create the retraction function using the provided helper.
    # (This uses the denoiser-based retraction if so specified in riem_config.)
    '''
    retraction_fn = create_retraction_fn(
        retraction_type=riem_config.get("retraction_operator", "identity"),
        score_fn=score_fn,
        alpha_fn=alpha_fn,
        sigma_fn=sigma_fn,
        t=t_diff
    )
    '''
    
    denoiser_fn = get_denoiser_fn(sde, model, t_diff, orig_shape)
    retraction_fn = create_retraction_fn(
        retraction_type=riem_config.get("retraction_operator", "identity"),
        denoiser_fn=denoiser_fn)
    
    # ---------------------------
    # Define the Euclidean optimization objective from the riemannian config.
    # ---------------------------
    #opt_fn = get_optim_function(riem_config)
    x_target = flatten_tensor(dataset_tensor[-1].unsqueeze(0)).to(device)
    opt_fn = get_custom_optim_function(x_target)

    # ---------------------------
    # Instantiate and run the riemannian optimizer.
    # ---------------------------
    riemannian_opt = get_riemannian_optimizer(score_fn, opt_fn, riem_config, retraction_fn)
    start_time = time.time()
    trajectory, metrics = riemannian_opt.run()
    elapsed_time = time.time() - start_time
    print(f"Riemannian optimization took {elapsed_time:.2f} seconds.")

    
    #Visualisation of the Riemannian optimisation.
    data_tensor = dataset_tensor[:250].to(device)
    alpha_t = alpha_fn(t_diff)
    sigma_t = sigma_fn(t_diff)
    perturbed_points = alpha_t * data_tensor + sigma_t * torch.randn_like(data_tensor)
    trajectory = [t.detach().cpu() for t in trajectory]
    min_point = torch.tensor(riem_config["min_point"], dtype=torch.float32, device=device).detach().cpu().numpy()

    visualize_riemannian_optimization_selector(
        perturbed_points,  # only used for low-dim cases
        score_fn,
        t_val,
        trajectory,
        metrics,
        min_point,
        orig_shape,
        log_dir=eval_dir,
        plot_filename=riem_config["plot_filename"]
    )

    #writer.close()

# ---------------------------
# Main script entry point.
# ---------------------------
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Use 'spawn' start method
    parser = ArgumentParser(description="Riemannian Optimization Evaluation Script for Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the diffusion configuration file.")
    parser.add_argument("--riem-config", type=str, required=True, help="Path to the riemannian optimization configuration file.")
    args = parser.parse_args()

    # Load the diffusion config using your standard loader.
    diff_config = load_config(args.config)
    # Save the diffusion config (as done in your other evaluation scripts).
    config_dir = os.path.join(diff_config.base_log_dir, diff_config.experiment)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(diff_config.to_dict(), f)

    # Load the riemannian config using the helper.
    riem_config = load_riemannian_config(args.riem_config)

    # Call the main optimization routine with the two separate configs.
    riemannian_optimization(diff_config, riem_config)
