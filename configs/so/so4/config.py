# ~/ScoreVAE/configs/so/so4/config.py
from configs.so.base import get_base_config


def get_config():
    cfg = get_base_config(n_group=4)

    # Heavier model & slightly smaller batch for memory
    cfg.training.epochs = 1000
    cfg.data.batch_size = 192
    cfg.data.data_samples = 1_000_000
    cfg.model.hidden_dim = 1024
    cfg.model.checkpoint = "Model_epoch_24_loss_0.102.pth"

    return cfg
