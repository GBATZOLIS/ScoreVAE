import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import os
import numpy as np
import abc
from contextlib import contextmanager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def prepare_training_dirs(config):
    # Set up logging directories
    tensorboard_dir = config.tensorboard_dir
    checkpoint_dir = config.checkpoint_dir
    eval_dir = config.eval_dir
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return tensorboard_dir, checkpoint_dir, eval_dir

def prepare_batch(data, device):
    if isinstance(data, torch.Tensor):
        return [data.to(device), None]
    elif isinstance(data, list):
        if len(data) == 1:
            return [data[0].to(device), None]
        else:
            return [item.to(device) for item in data]
    else:
        raise ValueError("Unsupported data type.")

def print_model_summary(model):
    total_trainable_params = 0

    for module in model.modules():
        num_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_trainable_params += num_trainable_params

    print(f'Total number of trainable parameters: {total_trainable_params}')


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def save_model(model, ema_model, epoch, loss, model_name, checkpoint_dir, best_checkpoints, global_step, best_val_loss, epochs_no_improve, optimizer, scheduler):
    def write_model(model, path, epoch, loss, is_ema=False):
        state_dict = model.state_dict() if not is_ema else ema_model.shadow
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'loss': loss,
            'global_step': global_step,
            'best_checkpoints': best_checkpoints,
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve,
            'optimizer_state_dict': optimizer.state_dict()
        }
        if hasattr(scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, path)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    last_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_last.pth")
    write_model(model, last_checkpoint_path, epoch, loss)

    last_ema_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_last_EMA.pth")
    write_model(model, last_ema_checkpoint_path, epoch, loss, is_ema=True)
    
    if len(best_checkpoints) < 3:
        new_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}.pth")
        new_ema_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}_EMA.pth")
        best_checkpoints.append((new_checkpoint_path, new_ema_checkpoint_path, loss))
        write_model(model, new_checkpoint_path, epoch, loss)
        write_model(model, new_ema_checkpoint_path, epoch, loss, is_ema=True)
    else:
        worst_checkpoint = max(best_checkpoints, key=lambda x: x[2])
        if loss < worst_checkpoint[2]:
            best_checkpoints.remove(worst_checkpoint)
            if os.path.exists(worst_checkpoint[0]):
                os.remove(worst_checkpoint[0])
            if os.path.exists(worst_checkpoint[1]):
                os.remove(worst_checkpoint[1])

            new_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}.pth")
            new_ema_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}_EMA.pth")
            best_checkpoints.append((new_checkpoint_path, new_ema_checkpoint_path, loss))
            
            write_model(model, new_checkpoint_path, epoch, loss)
            write_model(model, new_ema_checkpoint_path, epoch, loss, is_ema=True)
    
            print(f"{model_name} model saved at '{new_checkpoint_path}'")
            print(f"{model_name} EMA model saved at '{new_ema_checkpoint_path}'")


def load_model(model, ema_model, checkpoint_path, model_name, optimizer=None, scheduler=None, is_ema=False):
    checkpoint = torch.load(checkpoint_path)
    if is_ema:
        for name, data in checkpoint['model_state_dict'].items():
            ema_model.shadow[name].copy_(data)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    global_step = checkpoint.get('global_step', 0)
    best_checkpoints = checkpoint.get('best_checkpoints', [])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        if hasattr(scheduler, 'load_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"{model_name} {'EMA' if is_ema else ''} model loaded from '{checkpoint_path}', Epoch: {epoch}, Loss: {loss}")
    
    return epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve



def resume_training(config, model, ema_model, load_model_func, get_optimizer_and_scheduler_func):
    optimizer, scheduler = get_optimizer_and_scheduler_func(model, config)
    if hasattr(config.model, 'checkpoint') and config.model.checkpoint:
        checkpoint_path = config.model.checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_path)
        if not checkpoint_path.endswith('.pth'):
            checkpoint_path += '.pth'
        
        # Load standard model
        epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve = load_model_func(model, ema_model, checkpoint_path, "Model", optimizer, scheduler, is_ema=False)
        # Load EMA model
        ema_checkpoint_path = checkpoint_path.replace('.pth', '_EMA.pth')
        _, _, _, _, _, _ = load_model_func(model, ema_model, ema_checkpoint_path, "Model", is_ema=True)
        
        print(f"Resuming training from epoch {epoch + 1}")
        
        # Reinitialize optimizer and scheduler with the correct global_step
        optimizer, scheduler = get_optimizer_and_scheduler_func(model, config, global_step)
        
        return epoch + 1, global_step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler
    else:
        optimizer, scheduler = get_optimizer_and_scheduler_func(model, config)
        return 0, 0, [], float('inf'), 0, optimizer, scheduler
    

def get_noise_fn(sde, diffusion_model, train=True):
    return diffusion_model.get_noise_predictor_fn(sde, train)

def get_score_fn(sde, diffusion_model, train=True):
    return diffusion_model.get_score_fn(sde, train)

def eval_callback(score_fn, sde, val_dataloader, num_datapoints, device, save_path, name=None, return_svd=False):
    os.makedirs(save_path, exist_ok=True)

    singular_values = []
    idx = 0
    sampling_eps = sde.sampling_eps

    with tqdm(total=num_datapoints) as pbar:
        for x in val_dataloader:
            print(len(x))
            orig_batch = x[0]
            orig_batch = orig_batch.to(device)
            batchsize = orig_batch.size(0)

            if idx >= num_datapoints:
                break

            for x in orig_batch:
                if idx >= num_datapoints:
                    break

                ambient_dim = np.prod(x.shape[1:])
                x = x.repeat([batchsize] + [1 for _ in range(len(x.shape))])
                print("Repeated x shape:", x.shape)

                num_batches = int(np.floor(ambient_dim / batchsize)) + 1
                num_batches *= 2

                t = sampling_eps
                vec_t = torch.ones(x.size(0), device=device) * t

                scores = []
                for i in range(1, num_batches + 1):
                    batch = x.clone()

                    mean, std = sde.marginal_prob(batch, vec_t)
                    z = torch.randn_like(batch)
                    batch = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z
                    print(batch.size(), vec_t.size())
                    score = score_fn(batch, vec_t).detach().cpu()
                    print("Score shape:", score.shape)
                    scores.append(score)

                scores = torch.cat(scores, dim=0)
                print(score.size())
                scores = torch.flatten(scores, start_dim=1)
                print("Flattened scores shape:", scores.shape)

                means = scores.mean(dim=0, keepdim=True)
                normalized_scores = scores - means

                u, s, v = torch.linalg.svd(normalized_scores)
                singular_values.append(s.tolist())
                print("Singular values shape:", s.shape)

                idx += 1
                pbar.update(1)

    info = {'singular_values': singular_values}
    if return_svd:
        return info
    else:
        if name is None:
            name = 'svd'
        with open(os.path.join(save_path, f'{name}.pkl'), 'wb') as f:
            pickle.dump(info, f)
