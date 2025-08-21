import ml_collections
from math import ceil

def get_config():
    """
    Diffusion config for Rotated-MNIST (single digit rotated on SO(2)), 32×32.
    Uses padding to 32×32 before rotation to minimize artifacts.
    """
    cfg = ml_collections.ConfigDict()
    cfg.random_seed = 42

    # logging / folders
    cfg.base_log_dir    = "./results"
    cfg.experiment      = "rotated_mnist"
    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/training_logs"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/checkpoints"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"

    # training
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device               = "cuda:0"
    tr.gpus                 = 1
    tr.epochs               = 400                # extended runway
    tr.checkpoint_frequency = 5
    tr.patience_epochs      = 250               # avoid premature early stop

    tr.vis_callback         = "base"
    tr.vis_frequency        = 10
    tr.steps                = 128
    tr.num_samples          = 64

    tr.sde                  = "vpsde"
    tr.loss                 = "simple_DSM_loss"
    tr.likelihood_weighting = False
    tr.fid_eval_frequency   = 10**9

    # data
    cfg.data = data = ml_collections.ConfigDict()
    data.device        = "cuda"
    data.dataset       = "rotated_mnist_dataset"
    data.image_size    = 32                    # 32×32
    data.channels      = 1
    data.data_samples  = 100_000
    data.ambient_dim   = data.channels * data.image_size * data.image_size  # 1*32*32=1024
    data.dataset_path  = "datasets/rotated_mnist_9_32_pad.pt"
    data.overwrite_cache = False
    data.batch_size    = 256
    data.manifold_dim  = 1
    data.shape         = [data.channels, data.image_size, data.image_size]
    data.n_workers     = 8
    # dataset-specific options (forwarded by your data loader)
    data.digit         = 9
    data.split         = "train"
    data.sample_index  = None
    data.angle_step_deg = None
    data.pad_to_32     = True                  # ← key: pad before rotate

    # model
    cfg.model = model = ml_collections.ConfigDict()
    model.network              = "DDPM"
    model.in_channels          = data.channels
    model.out_channels         = data.channels
    model.base_channels        = 32
    model.num_blocks           = 3                # 32→16→8
    model.res_blocks_per_level = 1
    model.time_embed_dim       = 256
    model.attn_resolutions     = ()               # bottleneck-only attention
    # NOTE: With L=3, attention triggers when 2**level ∈ set {1,2,4} ↔ {32,16,8}px.
    # If you want attention at 16×16, set model.attn_resolutions = (2,)
    model.num_heads            = 2
    model.head_dim             = 16
    model.dropout_conv         = 0.0
    model.dropout_attn         = 0.0
    model.ema_decay            = 0.9995
    model.compile              = True
    model.checkpoint           = "Model_last.pth"

    # optim
    cfg.optim = opt = ml_collections.ConfigDict()
    opt.scheduler     = "cosine_decay"
    # steps/epoch = ceil(100_000 / 256) = 391 → total_steps = 391 × 400 = 156,400
    opt.total_steps   = 156_400
    opt.optimizer     = "AdamW"
    opt.lr            = 3e-4
    opt.weight_decay  = 5e-4
    opt.beta1         = 0.9
    opt.beta2         = 0.99
    opt.eps           = 1e-8
    opt.warmup        = 3_000
    opt.grad_clip     = 0.5

    # evaluation
    cfg.evaluation = ev = ml_collections.ConfigDict()
    ev.eval_callback_epochs = 25
    ev.num_eval_points      = 10
    ev.eval_save_path       = "./eval"

    return cfg
