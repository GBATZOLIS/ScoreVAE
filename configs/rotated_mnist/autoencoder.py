# configs/rotated_mnist/autoencoder.py
import ml_collections
from math import ceil

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed = 42

    # logging / folders
    cfg.base_log_dir    = "./results"
    cfg.experiment      = "rotmnist_ae_iso_new_loss"
    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/training_logs"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/checkpoints"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"

    # training
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device               = "cuda:0"
    tr.epochs               = 150
    tr.checkpoint_frequency = 10
    tr.patience_epochs      = 50
    tr.vis_frequency        = 5

    # data
    cfg.data = data = ml_collections.ConfigDict()
    data.device        = "cuda"
    data.dataset       = "rotated_mnist_dataset"
    data.image_size    = 32
    data.channels      = 1
    data.data_samples  = 100_000
    data.ambient_dim   = data.channels * data.image_size * data.image_size
    data.dataset_path  = "datasets/rotated_mnist_9_32_pad.pt"
    data.overwrite_cache = False
    data.batch_size    = 256
    data.n_workers     = 8
    data.digit         = 9
    data.split         = "train"
    data.sample_index  = None
    data.angle_step_deg = None
    data.pad_to_32     = True

    # model
    cfg.model = model = ml_collections.ConfigDict()
    model.network       = "AutoEncoder"
    model.in_channels   = data.channels
    model.out_channels  = data.channels
    model.image_size    = data.image_size
    model.latent_dim    = 16
    model.base_channels = 32
    model.num_down_levels = 3     # 32->16->8
    model.ema_decay     = 0.999
    model.compile       = True
    model.checkpoint    = None

    # loss/regularization
    cfg.loss = loss = ml_collections.ConfigDict()
    loss.reconstruction       = "mse"
    loss.enc_iso_weight       = 1e-3
    loss.dec_iso_weight       = 1e-3
    loss.num_v                = 2
    loss.dec_iso_detach_encoder = True

    # optim (reuse your factory)
    cfg.optim = opt = ml_collections.ConfigDict()
    opt.scheduler     = "cosine_decay"
    steps_per_epoch   = ceil(data.data_samples * 0.9 / data.batch_size)  # ~ train split only
    opt.total_steps   = steps_per_epoch * tr.epochs
    opt.optimizer     = "AdamW"
    opt.lr            = 2e-3
    opt.weight_decay  = 1e-4
    opt.beta1         = 0.9
    opt.beta2         = 0.99
    opt.eps           = 1e-8
    opt.warmup        = 2_000
    opt.grad_clip     = 1.0

    return cfg
