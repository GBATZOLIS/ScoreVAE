#!/usr/bin/env python3
# eval.py

import os
import argparse
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

from configs import load_config
from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import (
    prepare_training_dirs,
    EMA,
    load_model,
)
from utils.sampling_utils import get_generation_callback
from evaluation.fid import fid_evaluation_callback


def eval_model(config):
    """
    Load EMA weights, run the generation callback, and optionally FID eval.
    """
    # Setup directories and writer
    _, checkpoint_dir, eval_dir = prepare_training_dirs(config)
    writer = SummaryWriter(log_dir=eval_dir)

    # Device
    device = torch.device(config.training.device)

    # Dataloaders (needed if conditional eval or FID evaluation is desired)
    train_loader, val_loader, test_loader = get_dataloaders(
        config.data, seed=config.random_seed
    )

    # Model, SDE, EMA
    model = get_model(config.model).to(device)
    sde = configure_sde(config)
    ema = EMA(model=model, decay=config.model.ema_decay)

    # Load checkpoints
    ckpt_path = config.model.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_path)
    if not ckpt_path.endswith(".pth"):
        ckpt_path += ".pth"

    load_model(model, ema, ckpt_path, "Model", device=device, is_ema=False)
    load_model(model, ema, ckpt_path.replace(".pth", "_EMA.pth"), "Model", device=device, is_ema=True)
    ema.apply_shadow()
    model.eval()

    # Generation callback
    gen_cb = get_generation_callback(config.training.vis_callback)

    steps = config.training.steps
    shape = (config.training.num_samples, *config.data.shape)
    dummy_batch = (None, None)  # pass (None, None) if unconditional

    gen_cb(
        dummy_batch,
        writer,
        sde,
        model,
        steps,
        shape,
        device,
        epoch=0,
    )

    # ──────────────────────────────────────────────────────────────────────
    # ⬇️ Previous generation callback (kept for reference)
    #
    # from utils.sampling_utils import generation_callback
    # generation_callback(writer=writer,
    #                     sde=sde,
    #                     model=model,
    #                     steps=steps,
    #                     shape=shape,
    #                     device=device,
    #                     epoch=0)
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # ⬇️ FID Evaluation (commented out, restore if needed)
    #
    # if config.training.fid_eval_frequency > 0:
    #     fid_evaluation_callback(writer, sde, model, steps, shape, device, 0,
    #                             [train_loader, val_loader], train=True)
    #     fid_evaluation_callback(writer, sde, model, steps, shape, device, 0,
    #                             [test_loader], train=False)
    # ──────────────────────────────────────────────────────────────────────

    ema.restore()
    writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser("Evaluation script")
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to the configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Save a copy of the config
    cfg_dir = os.path.join(config.base_log_dir, config.experiment)
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "config_eval.pkl")
    with open(cfg_path, "wb") as f:
        pickle.dump(config.to_dict(), f)

    eval_model(config)
