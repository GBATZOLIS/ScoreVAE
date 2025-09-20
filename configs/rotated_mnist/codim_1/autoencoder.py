# configs/rotated_mnist/autoencoder_curv_iso.py
import ml_collections
from math import ceil

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed   = 42
    cfg.base_log_dir  = "./results"
    cfg.experiment    = "rotmnist/codim_1/standard_autoencoder"
    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/training_logs"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/checkpoints"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"

    # ---------------- training ----------------
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device               = "cuda:0"
    tr.epochs               = 150
    tr.checkpoint_frequency = 10
    tr.patience_epochs      = 50
    tr.vis_frequency        = 5

    # ---------------- data ----------------
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

    # ---------------- model ----------------
    cfg.model = model = ml_collections.ConfigDict()
    model.network         = "AutoEncoder"
    model.in_channels     = 1
    model.out_channels    = 1
    model.image_size      = 32
    model.latent_dim      = 2
    model.base_channels   = 32
    model.num_down_levels = 3
    model.ema_decay       = 0.999
    model.compile         = True
    model.checkpoint      = 'AE_last.pth'

    # (optional) VAE block — off by default
    model.vae = ml_collections.ConfigDict()
    model.vae.enabled = False

    # ---------------- losses ----------------
    cfg.loss = loss = ml_collections.ConfigDict()
    loss.reconstruction   = "mse"
    loss.beta_kl          = 1e-3

    # local isometry regs
    loss.enc_iso_weight   = 0.0
    loss.dec_iso_weight   = 0.0
    loss.num_v            = 1

    # --- MECAE (extrinsic) ---
    loss.curvature_weight = 0.0
    loss.curvature = curv = ml_collections.ConfigDict()
    curv.mode              = "mecae"       # kept for compatibility
    curv.target            = "both"        # "encoder" | "decoder" | "both"
    curv.reg_lambda        = 1e-4
    curv.K_v               = 1
    curv.K_w               = 2             # = latent dim m
    curv.use_rademacher    = True
    curv.estimator         = "square"
    curv.use_exact_hessian = True
    curv.fd_eps            = 1e-3
    curv.B_curv            = 128
    curv.every_n_steps     = 3

    # --- MICAE (intrinsic via Gauss) ---
    loss.intrinsic_weight  = 0.0
    loss.intrinsic = intr = ml_collections.ConfigDict()
    intr.reg_lambda        = 1e-4
    intr.R_a               = 2
    intr.use_rademacher    = True
    intr.use_exact_hessian = True
    intr.normalize_codim   = True
    intr.fd_eps            = 1e-3
    intr.B_curv            = curv.B_curv
    intr.every_n_steps     = curv.every_n_steps    

    # ---------------- geometry (observation & latent) ----------------
    # Keep latent diffusion training enabled (used elsewhere),
    # but curvature will NOT use it.
    loss.geom = geom = ml_collections.ConfigDict()

    geom.data = data_geom = ml_collections.ConfigDict()
    data_geom.enabled                = False
    data_geom.diffusion_config       = "configs/rotated_mnist/config.py"
    data_geom.metric_type            = "jacobian"
    data_geom.lam_metric             = 1e-3
    data_geom.t_value                = 0.03
    data_geom.use_denoiser_retraction= True
    data_geom.cg = ml_collections.ConfigDict()
    data_geom.cg.max_iter            = 5
    data_geom.cg.tol                 = 1e-5
    data_geom.cg.preconditioner      = None
    data_geom.cg.precond_diag_samples= 8

    # Latent diffusion still trained online (for other uses), not used for curvature.
    geom.latent = lat = ml_collections.ConfigDict()
    lat.enabled          = True
    lat.metric_type      = "jacobian"
    lat.lam_metric       = 1e-3
    lat.t_value          = 0.05
    lat.diffusion_config = "configs/rotated_mnist/latent_config.py"
    lat.cg = ml_collections.ConfigDict()
    lat.cg.max_iter              = 5
    lat.cg.tol                   = 1e-6
    lat.cg.preconditioner        = "diagonal"
    lat.cg.precond_diag_samples  = 5

    # ---------------- optim ----------------
    cfg.optim = opt = ml_collections.ConfigDict()
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
