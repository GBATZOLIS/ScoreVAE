import ml_collections
from math import ceil

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed   = 42
    cfg.base_log_dir  = "./results"
    cfg.experiment    = "teapots_64/so2/iso_autoencoder_0.001"
    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/training_logs"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/checkpoints"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"

    # ---------- training ----------
    cfg.training = tr = ml_collections.ConfigDict()
    # Single-GPU runs still use this:
    tr.device               = "cuda:0"
    tr.devices              = [0, 1]         # list of GPU ids for DataParallel
    # NEW: Multi-GPU device selection for torchrun; "auto" = all visible GPUs
    # You can also set "0,1" or [0,1,2,3]
    tr.epochs               = 220
    tr.checkpoint_frequency = 10
    tr.patience_epochs      = 90
    tr.vis_frequency        = 3

    # ---------- data ----------
    cfg.data = data = ml_collections.ConfigDict()
    data.device        = "cuda"
    data.dataset       = "rendered_so_dataset"
    data.mesh_path     = "datasets/meshes/teapot.obj"
    data.image_size    = 64
    data.channels      = 1
    data.manifold_dim  = 2
    data.submanifold   = "s2_zeroroll"         # or "torus_azim_roll"
    data.azim_step     = None
    data.elev_step     = None
    data.roll_step     = None
    data.render_batch  = 4                     # rendering micro-batch
    data.data_samples  = 100_000
    data.ambient_dim   = data.channels * data.image_size * data.image_size
    data.dataset_path  = "datasets/teapot_s2_gray64.pt"
    data.overwrite_cache = False
    data.batch_size    = 128
    data.n_workers     = 8
    data.shape         = [data.channels, data.image_size, data.image_size]

    # ---------- model ----------
    cfg.model = model = ml_collections.ConfigDict()
    model.network         = "AutoEncoder"
    model.in_channels     = 1
    model.out_channels    = 1
    model.image_size      = 64
    model.latent_dim      = 3                 # co-dim = 1 (2-D manifold)
    model.base_channels   = 64
    model.num_down_levels = 3                 # 64→32→16→8 bottleneck
    model.ema_decay       = 0.9995
    model.compile         = True
    model.checkpoint      = None
    model.vae = ml_collections.ConfigDict()
    model.vae.enabled = False

    # ---------- losses ----------
    cfg.loss = loss = ml_collections.ConfigDict()
    loss.reconstruction   = "mse"
    loss.beta_kl          = 1e-3

    loss.enc_iso_weight   = 0.001
    loss.dec_iso_weight   = 0.001
    loss.num_v            = 1

    loss.curvature_weight = 0.0 #0.001
    loss.curvature = curv = ml_collections.ConfigDict()
    curv.mode              = "mecae"
    curv.target            = "both"
    curv.reg_lambda        = 1e-4
    curv.K_v               = 1
    curv.K_w               = 2
    curv.use_rademacher    = True
    curv.estimator         = "square"
    curv.use_exact_hessian = True
    curv.fd_eps            = 1e-3
    curv.B_curv            = 36
    curv.every_n_steps     = 3

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

    # ---------- latent diffusion (ON) ----------
    loss.geom = geom = ml_collections.ConfigDict()
    geom.data = data_geom = ml_collections.ConfigDict()
    data_geom.enabled = False
    geom.latent = lat = ml_collections.ConfigDict()
    lat.enabled          = True
    lat.diffusion_config = "configs/teapots_64/so2/latent_config.py"

    # ---------- optim ----------
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
