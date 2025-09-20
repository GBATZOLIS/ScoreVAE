# configs/rotated_mnist/autoencoder.py
import ml_collections
from math import ceil

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed = 42

    # logging / folders
    cfg.base_log_dir    = "./results"
    cfg.experiment      = "rotmnist/iso_ae"
    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/training_logs"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/checkpoints"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"

    # training
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device               = "cuda:1"
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
    data.shape         = [data.channels, data.image_size, data.image_size]

    # model (AutoEncoder/VAE)
    cfg.model = model = ml_collections.ConfigDict()
    model.network         = "AutoEncoder"
    model.in_channels     = data.channels
    model.out_channels    = data.channels
    model.image_size      = data.image_size
    model.latent_dim      = 3
    model.base_channels   = 32
    model.num_down_levels = 3     # 32 -> 16 -> 8
    model.ema_decay       = 0.999
    model.compile         = True
    model.checkpoint      = None

    # VAE options (if your model supports it)
    model.vae = ml_collections.ConfigDict()
    model.vae.enabled              = False            # keep True if you use μ(x), logσ^2(x)
    model.vae.learnable_prior_diag = False            # learn diag prior covariance
    model.vae.prior_logvar_init    = 0.0             # log(σ_p^2) init
    model.vae.whiten_for_latent_diffusion = False

    # ------------------------------ losses --------------------------------
    cfg.loss = loss = ml_collections.ConfigDict()
    loss.reconstruction  = "mse"
    loss.beta_kl         = 1e-3                      # β-VAE style (set to 1.0 for standard ELBO)

    # Local isometry (simple, Euclidean by default)
    loss.enc_iso_weight         = 1.               # encoder local isometry (Euclidean)
    loss.dec_iso_weight         = 1.               # decoder local isometry (Euclidean)
    loss.num_v                  = 1                  # # of directions for JVP/VJP-based regs
    loss.dec_iso_detach_encoder = True               # typically keep True

    # ------------- geometry handles (kept for later; NOT used now) -------------
    loss.geom = geom = ml_collections.ConfigDict()
    geom.project_first = True                        # placeholder; unused while metrics are None

    # Observation-space metric — keep the block but DISABLE it
    geom.data = data_geom = ml_collections.ConfigDict()
    data_geom.enabled          = False               # <- DO NOT load / use data metric now
    data_geom.diffusion_config = "configs/rotated_mnist/config.py"
    data_geom.metric_type      = "jacobian"          # {"jacobian","stein"}
    data_geom.lam_metric       = 1e-3
    data_geom.t_value          = 0.03
    data_geom.use_denoiser_retraction = True
    data_geom.cg = ml_collections.ConfigDict()
    data_geom.cg.max_iter              = 5
    data_geom.cg.tol                   = 1e-5
    data_geom.cg.preconditioner        = None
    data_geom.cg.precond_diag_samples  = 8

    # Latent-space metric block — keep it, but we WON'T pass it to the AE loss.
    # We DO train the latent diffusion model online using this config.
    geom.latent = lat_geom = ml_collections.ConfigDict()
    lat_geom.enabled          = True                 # <- train diffusion model in latent space
    lat_geom.metric_type      = "jacobian"           # kept for later evaluation
    lat_geom.lam_metric       = 1e-3
    lat_geom.t_value          = 0.05
    lat_geom.diffusion_config = "configs/rotated_mnist/latent_config.py"
    lat_geom.cg = ml_collections.ConfigDict()
    lat_geom.cg.max_iter              = 5
    lat_geom.cg.tol                   = 1e-6
    lat_geom.cg.preconditioner        = "diagonal"
    lat_geom.cg.precond_diag_samples  = 5

    # AE optim/schedule
    cfg.optim = opt = ml_collections.ConfigDict()
    opt.scheduler     = "cosine_decay"
    steps_per_epoch   = ceil(data.data_samples * 0.9 / data.batch_size)
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
