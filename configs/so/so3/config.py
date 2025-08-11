# ~/ScoreVAE/configs/so/so3/config.py
from configs.so.base import get_base_config


def get_config():
    cfg = get_base_config(n_group=3)

    # Optionally override a few fields suited to SO(3)
    cfg.training.epochs = 600
    cfg.data.batch_size = 256
    cfg.model.hidden_dim = 768
    cfg.model.checkpoint = "Model_epoch_99_loss_0.093.pth"

    return cfg
