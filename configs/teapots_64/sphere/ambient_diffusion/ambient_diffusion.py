# configs/rendered_so_dataset/s2_axisangle_64x64_gray_unet.py
import ml_collections

def get_config():
    """
    Training config for a score-based diffusion model on RenderedSO3Dataset
    (64×64 grayscale images), using the S^2 axis–angle submanifold:
        R(n) = Exp(alpha * [n]_x), n ∈ S^2, alpha ∈ (0, π)
    """
    config = ml_collections.ConfigDict()
    config.random_seed = 42

    # logging / folders
    config.base_log_dir    = "./results/teapot_sphere"
    config.experiment      = "ambient_diffusion"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir  = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir        = f"{config.base_log_dir}/{config.experiment}/eval"

    # training (unchanged)
    config.training = training = ml_collections.ConfigDict()
    training.device               = "cuda:0"
    training.gpus                 = 1
    training.epochs               = 200
    training.checkpoint_frequency = 5
    training.patience_epochs      = 100

    training.vis_callback         = "base"
    training.vis_frequency        = 1
    training.steps                = 256
    training.num_samples          = 64

    training.sde                  = "vpsde"
    training.loss                 = "simple_DSM_loss"
    training.likelihood_weighting = False
    training.fid_eval_frequency   = 10**9

    # data (switched to S^2 axis-angle submanifold)
    config.data = data = ml_collections.ConfigDict()
    data.device        = "cuda"
    data.dataset       = "rendered_so_dataset"
    data.mesh_path     = "datasets/meshes/teapot.obj"
    data.image_size    = 64
    data.channels      = 1
    data.azim_step     = None
    data.elev_step     = None
    data.roll_step     = None            # not used for s2_axisangle
    data.data_samples  = 100_000
    data.ambient_dim   = data.channels * data.image_size * data.image_size  # 4096
    data.dataset_path  = "datasets/teapot_s2axis_gray64.pt"
    data.overwrite_cache = False
    data.batch_size    = 128
    data.manifold_dim  = 2               # << use 2D S^2 submanifold
    data.submanifold   = "s2_axisangle"  # << new submanifold
    data.axis_angle_deg = 90.0           # any value in (0, 180); 90° is a good default
    data.shape         = [data.channels, data.image_size, data.image_size]
    data.n_workers     = 4

    # model (unchanged)
    config.model = model = ml_collections.ConfigDict()
    model.network              = "DDPM"
    model.in_channels          = data.channels
    model.out_channels         = data.channels
    model.base_channels        = 64
    model.num_blocks           = 3
    model.res_blocks_per_level = 2
    model.time_embed_dim       = 512
    model.attn_resolutions     = (16,)     # 32 optional if VRAM allows
    model.num_heads            = 4
    model.head_dim             = 32
    model.dropout_conv         = 0.0
    model.dropout_attn         = 0.1
    model.ema_decay            = 0.999
    model.checkpoint           = 'Model_last.pth'

    # optim (unchanged)
    config.optim = optim = ml_collections.ConfigDict()
    optim.scheduler     = 'cosine_decay'
    optim.total_steps   = 500_000
    optim.optimizer     = "AdamW"
    optim.lr            = 2e-4
    optim.weight_decay  = 1e-4
    optim.beta1         = 0.9
    optim.beta2         = 0.999
    optim.eps           = 1e-8
    optim.warmup        = 5_000
    optim.grad_clip     = 1.0

    # evaluation (unchanged)
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.eval_callback_epochs = 25
    evaluation.num_eval_points      = 10
    evaluation.eval_save_path       = "./eval"

    return config
