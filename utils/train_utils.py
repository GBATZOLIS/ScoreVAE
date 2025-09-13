# train_utils.py – universal helper utilities for training
# -----------------------------------------------------------------------------
# This file keeps **exactly** the same public API/behaviour as the original
# version that worked with PyTorch 2.3, while adding a fully-backwards-compatible
# shim that makes it load checkpoints seamlessly under the new "secure pickling"
# defaults introduced in PyTorch ≥ 2.4/nightly. Nothing else about the training
# loop changes – so you can drop-in replace the old module without touching any
# of your calling code.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import pickle
from contextlib import contextmanager
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt  # noqa: F401 (kept for existing external uses)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------------------------------------------------------
# 🛠 Version helpers – detect whether the running PyTorch supports the new
#    `weights_only` argument, and register safe globals only when they are
#    AVAILABLE (& necessary).
# -----------------------------------------------------------------------------

try:
    from packaging.version import parse as _parse_version  # type: ignore
except ImportError:  # extremely old env – fall back to a very lax parser
    def _parse_version(v: str):  # type: ignore
        major, minor, *_ = v.split(".")
        return tuple(map(int, (major, minor)))

def _torch_supports_weights_only() -> bool:
    """Return *True* if the current torch version understands
    `torch.load(..., weights_only=...)`. We consider 2.4.0 the cut-off because
    that is when the secure-by-default unpickler landed (nightly builds are
    "2.x.devYYYYMMDD" which also parse > 2.4).
    """
    version_str = torch.__version__.split("+")[0]  # strip any git hash/build meta
    try:
        return _parse_version(version_str) >= _parse_version("2.4.0")
    except Exception:
        # If packaging is unavailable or parsing fails we conservatively assume
        # the old behaviour (no weights_only support) to avoid breaking.
        return False

_SUPPORTS_WEIGHTS_ONLY = _torch_supports_weights_only()

# Register *AdamW* so that a nightly build running with `weights_only=True` does
# not choke when it encounters an optim state dict referencing the optimiser.
if _SUPPORTS_WEIGHTS_ONLY:
    try:
        from torch.serialization import add_safe_globals  # type: ignore[attr-defined]

        add_safe_globals([optim.AdamW])
    except Exception:
        # If the helper is missing (future API churn) we silently ignore – worst
        # case we fall back to `weights_only=False` later.
        pass

# -----------------------------------------------------------------------------
# Directory helpers & misc. small utilities (unchanged).
# -----------------------------------------------------------------------------

def prepare_training_dirs(config):
    """Ensure tensorboard/checkpoint/eval directories exist and return them."""
    tensorboard_dir = config.tensorboard_dir
    checkpoint_dir = config.checkpoint_dir
    eval_dir = config.eval_dir
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return tensorboard_dir, checkpoint_dir, eval_dir

def prepare_batch(data, device, *, channels_last: bool = True, non_blocking: bool = True, dtype=None):
    import torch

    def _move(t):
        if not isinstance(t, torch.Tensor):
            return t
        t = t.to(device, non_blocking=non_blocking)
        if dtype is not None:
            t = t.to(dtype)
        # Only 4D image tensors should use channels_last for speed
        if channels_last and t.ndim == 4:
            t = t.contiguous(memory_format=torch.channels_last)
        return t

    if isinstance(data, torch.Tensor):
        x = _move(data)
        return [x, None]

    elif isinstance(data, (list, tuple)):
        moved = [_move(item) for item in data]
        if len(moved) == 1:
            return [moved[0], None]
        return moved

    else:
        raise ValueError("Unsupported data type.")


def print_model_summary(model: nn.Module):
    total_trainable_params = 0
    for module in model.modules():
        total_trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_trainable_params}")

# -----------------------------------------------------------------------------
# Exponential Moving Average helper (unchanged).
# -----------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
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
                self.backup[name] = param.data  # type: ignore[assignment]
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# -----------------------------------------------------------------------------
# Saving utilities – logic untouched (no need for version shims on save).
# -----------------------------------------------------------------------------

def save_model(
    model: nn.Module,
    ema_model: EMA,
    epoch: int,
    loss: float,
    model_name: str,
    checkpoint_dir: str,
    best_checkpoints: List[Tuple[str, str, float]],
    global_step: int,
    best_val_loss: float,
    epochs_no_improve: int,
    optimizer: optim.Optimizer,
    scheduler: Any,
):
    """Save *last* checkpoints + rolling top-3 best (by loss) for both
    plain and EMA weights. Exactly the same behaviour as before.
    """

    def _write(model_: nn.Module, path: str, is_ema: bool = False):
        state_dict = model_.state_dict() if not is_ema else ema_model.shadow
        ckpt = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "loss": loss,
            "global_step": global_step,
            "best_checkpoints": best_checkpoints,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if hasattr(scheduler, "state_dict"):
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(ckpt, path)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Always write the *last* checkpoint.
    last_ckpt = os.path.join(checkpoint_dir, f"{model_name}_last.pth")
    _write(model, last_ckpt)

    last_ckpt_ema = os.path.join(checkpoint_dir, f"{model_name}_last_EMA.pth")
    _write(model, last_ckpt_ema, is_ema=True)

    # Maintain top-3 best checkpoints (lowest loss).
    if len(best_checkpoints) < 3:
        new_ckpt = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}.pth")
        new_ckpt_ema = new_ckpt.replace(".pth", "_EMA.pth")
        best_checkpoints.append((new_ckpt, new_ckpt_ema, loss))
        _write(model, new_ckpt)
        _write(model, new_ckpt_ema, is_ema=True)
    else:
        worst = max(best_checkpoints, key=lambda x: x[2])
        if loss < worst[2]:
            best_checkpoints.remove(worst)
            for path in worst[:2]:
                if os.path.exists(path):
                    os.remove(path)
            new_ckpt = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}.pth")
            new_ckpt_ema = new_ckpt.replace(".pth", "_EMA.pth")
            best_checkpoints.append((new_ckpt, new_ckpt_ema, loss))
            _write(model, new_ckpt)
            _write(model, new_ckpt_ema, is_ema=True)
            print(f"{model_name} model saved at '{new_ckpt}'")
            print(f"{model_name} EMA model saved at '{new_ckpt_ema}'")

# -----------------------------------------------------------------------------
# Loading utilities – **💡new secure-pickle compatibility layer lives here**.
# -----------------------------------------------------------------------------

def _strip_orig_mod(state: Dict[str, Any], prefix: str = "_orig_mod.") -> Dict[str, Any]:
    """Remove Inductor’s `_orig_mod.` prefix from compiled checkpoints."""
    if not state:
        return state
    if not next(iter(state)).startswith(prefix):
        return state
    return {k[len(prefix):]: v for k, v in state.items()}


def _safe_torch_load(path: str, *, map_location: torch.device):
    """Attempt to load *path* using the safest API available on the current
    runtime:

    1. If the runtime supports `weights_only`, first try with that flag set to
       *True* (this is the new default in nightly builds). This avoids executing
       any code from the pickle stream.
    2. If that fails (e.g. the checkpoint contains objects that *cannot* be
       read in weights-only mode) we **fall back** to a full unpickle
       **iff** the user is presumably OK with that – we print a warning so the
       decision is explicit in stdout.
    3. On older runtimes we directly call plain `torch.load`, preserving the
       original behaviour.
    """
    if _SUPPORTS_WEIGHTS_ONLY:
        # We want to mirror nightly’s behaviour but retain compatibility with
        # old checkpoints containing optimiser objects such as `AdamW` by adding
        # them to the allow-list. (This was attempted at module import; doing it
        # again is cheap/safe.)
        try:
            from torch.serialization import add_safe_globals  # type: ignore[attr-defined]
            add_safe_globals([optim.AdamW])
        except Exception:
            pass
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception as exc:
            print(
                f"[train_utils] weights-only load failed for '{path}': {exc}\n"
                "               Falling back to *full* torch.load (weights_only=False)."
                " Make sure you trust the checkpoint source!"
            )
            return torch.load(path, map_location=map_location, weights_only=False)
    else:
        # Historical behaviour (< 2.4).
        return torch.load(path, map_location=map_location)


def load_model(
    model: nn.Module,
    ema_model: EMA,
    checkpoint_path: str,
    model_name: str,
    device: torch.device | str = torch.device("cpu"),
    optimizer: optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    is_ema: bool = False,
):
    """Load *checkpoint_path* onto *device*, filling the given `model` and
    optional optimiser/scheduler. Handles both pre-2.4 and >=2.4 checkpoints
    transparently.
    """
    checkpoint = _safe_torch_load(checkpoint_path, map_location=torch.device(device))

    # Always clean Inductor prefixes so that *compiled* checkpoints can still be
    # loaded in an *uncompiled* environment.
    state = _strip_orig_mod(checkpoint["model_state_dict"])

    if is_ema:
        for name, tensor in state.items():
            if name in ema_model.shadow:
                ema_model.shadow[name].copy_(tensor)
            else:
                # If the EMA object was created from a model with different
                # param names (rare), we skip silently rather than crashing.
                pass
    else:
        model.load_state_dict(state, strict=True)

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    global_step = checkpoint.get("global_step", 0)
    best_checkpoints = checkpoint.get("best_checkpoints", [])
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    epochs_no_improve = checkpoint.get("epochs_no_improve", 0)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(
        f"{model_name} {'EMA ' if is_ema else ''}model loaded from '{checkpoint_path}', "
        f"Epoch: {epoch}, Loss: {loss}"
    )

    return epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve

# -----------------------------------------------------------------------------
# Resume-training helper (logic unchanged – uses load_model above).
# -----------------------------------------------------------------------------

def resume_training(config, model: nn.Module, ema_model: EMA, load_model_func, get_optimizer_and_scheduler_func):
    optimizer, scheduler = get_optimizer_and_scheduler_func(model, config)
    if getattr(config.model, "checkpoint", None):
        checkpoint_path = config.model.checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_path)
        if not checkpoint_path.endswith(".pth"):
            checkpoint_path += ".pth"

        epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve = load_model_func(
            model,
            ema_model,
            checkpoint_path,
            "Model",
            device=torch.device(config.training.device),
            optimizer=optimizer,
            scheduler=scheduler,
            is_ema=False,
        )
        # Load corresponding EMA checkpoint.
        ema_path = checkpoint_path.replace(".pth", "_EMA.pth")
        load_model_func(
            model,
            ema_model,
            ema_path,
            "Model",
            device=torch.device(config.training.device),
            is_ema=True,
        )

        print(f"Resuming training from epoch {epoch + 1}")
        # Reset optimiser/scheduler with the recovered global step.
        optimizer, scheduler = get_optimizer_and_scheduler_func(model, config, global_step)
        return epoch + 1, global_step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler
    else:
        # Fresh-start training.
        optimizer, scheduler = get_optimizer_and_scheduler_func(model, config)
        return 0, 0, [], float("inf"), 0, optimizer, scheduler

# -----------------------------------------------------------------------------
# Misc. wrappers (unchanged).
# -----------------------------------------------------------------------------

def get_noise_fn(sde, diffusion_model, train: bool = True):
    return diffusion_model.get_noise_predictor_fn(sde, train)


def get_score_fn(sde, diffusion_model, train: bool = True):
    return diffusion_model.get_score_fn(sde, train)

# -----------------------------------------------------------------------------
# Evaluation callback (unchanged).
# -----------------------------------------------------------------------------

def eval_callback(
    score_fn,
    sde,
    val_dataloader,
    num_datapoints: int,
    device: torch.device | str,
    save_path: str,
    name: str | None = None,
    *,
    return_svd: bool = False,
):
    os.makedirs(save_path, exist_ok=True)

    singular_values: List[List[float]] = []
    idx = 0
    sampling_eps = sde.sampling_eps  # type: ignore[attr-defined]

    with tqdm(total=num_datapoints) as pbar:
        for batch in val_dataloader:
            orig_batch = batch[0].to(device)
            batch_size = orig_batch.size(0)
            if idx >= num_datapoints:
                break
            for x in orig_batch:
                if idx >= num_datapoints:
                    break
                ambient_dim = int(np.prod(x.shape[1:]))
                x_rep = x.repeat([batch_size] + [1] * (len(x.shape)))
                num_batches = (ambient_dim // batch_size + 1) * 2
                t = sampling_eps
                vec_t = torch.ones(x_rep.size(0), device=device) * t
                scores = []
                for _ in range(1, num_batches + 1):
                    batch_noise = x_rep.clone()
                    mean, std = sde.marginal_prob(batch_noise, vec_t)
                    z = torch.randn_like(batch_noise)
                    batch_noise = mean + std[(...,) + (None,) * (len(batch_noise.shape) - 1)] * z
                    score = score_fn(batch_noise, vec_t).detach().cpu()
                    scores.append(score)
                scores = torch.cat(scores, dim=0)
                scores = torch.flatten(scores, start_dim=1)
                means = scores.mean(dim=0, keepdim=True)
                normalized_scores = scores - means
                _, s, _ = torch.linalg.svd(normalized_scores)
                singular_values.append(s.tolist())
                idx += 1
                pbar.update(1)

    info = {"singular_values": singular_values}
    if return_svd:
        return info
    else:
        name = name or "svd"
        with open(os.path.join(save_path, f"{name}.pkl"), "wb") as f:
            pickle.dump(info, f)
