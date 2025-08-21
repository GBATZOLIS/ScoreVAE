# configs/rendered_so_dataset/so3_64x64_gray_unet.py
import math
import ml_collections

def get_config():
    """
    Light/fast config for RenderedSO3Dataset (64×64 grayscale).
    Goal: close-to-baseline quality with significant speedup.

    Key changes vs baseline:
      - Wider training window: epochs=350 (patience=150)
      - Scheduler matches run length: total_steps = 364_700
      - Slimmer UNet: base=48, 4 levels, 1 ResBlock/level
      - Bottleneck-only attention (attn_resolutions = ())
      - Larger batch if VRAM allows (192), more dataloader workers (8)
      - AdamW tuned for long cosine run: lr=3e-4, wd=5e-4, beta2=0.99
      - Slightly stronger EMA for stability: 0.9995
    """
    config = ml_collections.ConfigDict()
    config.random_seed = 42

    # logging / folders
    config.base_log_dir    = "./results"
    config.experiment      = "teapot_so3_rendered_64_light_long"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir  = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir        = f"{config.base_log_dir}/{config.experiment}/eval"

    # training
    config.training = training = ml_collections.ConfigDict()
    training.device               = "cuda:1"
    training.gpus                 = 1
    training.epochs               = 350
    training.checkpoint_frequency = 5
    training.patience_epochs      = 150

    training.vis_callback         = "base"
    training.vis_frequency        = 10
    training.steps                = 256
    training.num_samples          = 64

    training.sde                  = "vpsde"
    training.loss                 = "simple_DSM_loss"
    training.likelihood_weighting = False
    training.fid_eval_frequency   = 10**9

    # data
    config.data = data = ml_collections.ConfigDict()
    data.device        = "cuda"
    data.dataset       = "rendered_so_dataset"
    data.mesh_path     = "datasets/meshes/teapot.obj"
    data.image_size    = 64
    data.channels      = 1
    data.azim_step     = None
    data.elev_step     = None
    data.roll_step     = None          # ⇒ Haar sample over SO(3)
    data.data_samples  = 200_000
    data.ambient_dim   = data.channels * data.image_size * data.image_size  # 4096
    data.dataset_path  = "datasets/teapot_so3_gray64.pt"
    data.overwrite_cache = False
    data.batch_size    = 192           # reduce if OOM
    data.manifold_dim  = 3
    data.shape         = [data.channels, data.image_size, data.image_size]
    data.n_workers     = 8

    # model
    config.model = model = ml_collections.ConfigDict()
    model.network              = "DDPM"
    model.in_channels          = data.channels
    model.out_channels         = data.channels
    model.base_channels        = 48
    model.num_blocks           = 3                 # 4 levels: 64→32→16→8
    model.res_blocks_per_level = 1
    model.time_embed_dim       = 256
    model.attn_resolutions     = ()                # bottleneck-only (fastest)
    # NOTE: In this UNet, attention is inserted when 2**level ∈ attn_resolutions.
    # For 64×64 with L=4, those values are {1,2,4,8} ↔ spatial {64,32,16,8} px.
    # Use (4,) to place a single 16×16 attention block if you want a bit more capacity.
    model.num_heads            = 4
    model.head_dim             = 32                # kept for completeness
    model.dropout_conv         = 0.0
    model.dropout_attn         = 0.0
    model.ema_decay            = 0.9995
    model.checkpoint           = ""                # path to resume if needed

    # optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.scheduler     = "cosine_decay"
    # total_steps = ceil(200_000 / 192) * 350 = 1,042 * 350 = 364,700
    optim.total_steps   = 364_700
    optim.optimizer     = "AdamW"
    optim.lr            = 3e-4
    optim.weight_decay  = 5e-4
    optim.beta1         = 0.9
    optim.beta2         = 0.99
    optim.eps           = 1e-8
    optim.warmup        = 5_000
    optim.grad_clip     = 0.5

    # evaluation
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.eval_callback_epochs = 25
    evaluation.num_eval_points      = 10
    evaluation.eval_save_path       = "./eval"

    return config
