# configs/rendered_so_dataset/so3_32x32_gray_unet.py
import ml_collections


def get_config():
    """
    Full training configuration for a score-based diffusion model on the
    RenderedSO3Dataset (32 × 32 grayscale images, full SO(3) manifold).

    Folder layout assumed:
        datasets/
            meshes/teapot.obj             # 3-D mesh to render
            teapot_so3_gray32.pt          # cached dataset (created automatically)
        results/teapot_so3_rendered/   # logs, checkpoints, evaluation outputs
    """
    # ────────────────────────────────────────────────────────── ROOT CONFIG
    config = ml_collections.ConfigDict()
    config.random_seed = 42

    # ───────── Logging / Output folders
    config.base_log_dir   = "./results"
    config.experiment     = "teapot_so3_rendered"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir  = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir        = f"{config.base_log_dir}/{config.experiment}/eval"

    # ────────────────────────────────────────────────────────── TRAINING
    config.training = training = ml_collections.ConfigDict()
    training.device                 = "cuda:1"      # "cpu" or "cuda[:idx]"
    training.gpus                   = 1           # number of GPUs to use
    training.epochs                 = 178
    training.checkpoint_frequency   = 25
    training.patience_epochs        = 100

    #   visualisation / sampling during training
    training.vis_callback           = "base"
    training.vis_frequency          = 1          # generate samples every N epochs
    training.steps                  = 256         # SDE integration steps for vis
    training.num_samples            = 64         # number of samples to draw

    #   diffusion & loss
    training.sde                    = "vpsde"
    training.loss                   = "simple_DSM_loss"
    training.likelihood_weighting   = False
    training.fid_eval_frequency     = 10**9       # effectively disabled for this task

    # ────────────────────────────────────────────────────────── DATA
    config.data = data = ml_collections.ConfigDict()
    data.device                = "cuda"
    data.dataset               = "rendered_so_dataset"                # dataset class selector
    data.mesh_path             = "datasets/meshes/teapot.obj"
    data.image_size            = 128                          
    data.channels              = 1                           # grayscale images
    data.azim_step             = None
    data.elev_step             = None
    data.roll_step             = None            # ⇒ dataset will use Haar sampling
    data.data_samples          = 200_000      # any integer ≥ 1
    data.ambient_dim           = data.channels * data.image_size * data.image_size     # e.g. 1 * 128 * 128 = 16384
    data.dataset_path          = "datasets/teapot_so3_gray32.pt"
    data.overwrite_cache       = False                       # regenerate dataset?
    data.batch_size            = 128
    data.manifold_dim          = 3                           # intrinsic manifold dim
    data.shape                 = [data.channels, data.image_size, data.image_size]          # flatten shape for model
    data.n_workers             = 4                           # number of workers for DataLoader

    # ────────────────────────────────────────────────────────── MODEL
    config.model = model = ml_collections.ConfigDict()

    model.network           = "DDPM"   # dynamic import
    model.in_channels       = data.channels
    model.out_channels      = data.channels
    model.base_channels     = 64
    model.num_blocks        = 3           # encoder levels
    model.res_blocks_per_level = 2
    model.time_embed_dim    = 512
    model.attn_resolutions  = (16,)       # add 32 if VRAM permits
    model.num_heads         = 4
    model.head_dim          = 32
    model.dropout_conv      = 0.0
    model.dropout_attn      = 0.1

    # Other standard options (not used by the model directly, but useful globally)
    model.ema_decay      = 0.999
    model.checkpoint     = "Model_last.pth"                  # load path if resuming

    # ────────────────────────────────────────────────────────── OPTIMISER
    config.optim = optim = ml_collections.ConfigDict()
    optim.scheduler      = 'cosine_decay'
    optim.total_steps    = 500_000
    optim.optimizer      = "AdamW"
    optim.lr             = 2e-4
    optim.weight_decay   = 1e-4
    optim.beta1          = 0.9
    optim.beta2          = 0.999
    optim.eps            = 1e-8
    optim.warmup         = 5000
    optim.grad_clip      = 1.0
    

    # ────────────────────────────────────────────────────────── EVALUATION
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.eval_callback_epochs = 25
    evaluation.num_eval_points      = 10
    evaluation.eval_save_path       = "./eval"

    return config
