# configs/analytic/autoencoder_curv_iso.py
import ml_collections
from math import ceil

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed   = 42
    cfg.base_log_dir  = "./results"
    cfg.experiment    = "analytic_manifold/sphere/vanilla_architecture/autoencoder"
    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/training_logs"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/checkpoints"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"

    # ---------------- training ----------------
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device               = "cuda:0"
    tr.devices              = "0"
    tr.epochs               = 150
    tr.checkpoint_frequency = 10
    tr.patience_epochs      = 50
    tr.vis_frequency        = 5
    tr.update_norm_frequency = 2

    # ---------------- data ----------------
    cfg.data = data = ml_collections.ConfigDict()
    data.device        = "cuda"
    data.dataset       = "analytic_manifold_dataset"  
    data.image_size    = 32
    data.channels      = 3                              # RGB by default (set 1 for gray)
    data.data_samples  = 100_000
    data.gen_batch     = 4096
    data.ambient_dim   = data.channels * data.image_size * data.image_size
    data.dataset_path  = "datasets/cache/analytic_S2_32x32_rgb.pt"  # or T^2 path
    data.overwrite_cache = False
    data.batch_size    = 256
    data.n_workers     = 8
    data.shape         = [data.channels, data.image_size, data.image_size]

    # Manifold selector and (optional) grid
    #   manifold: "s2" or "torus"
    data.manifold      = "s2"        # change to "torus" for the flat torus
    # S^2 grid (set both to enable grid; otherwise random sampling)
    data.azim_step     = None        # degrees, e.g. 6.0
    data.elev_step     = None        # degrees, e.g. 6.0
    # T^2 grid (set both to enable grid; otherwise random sampling)
    data.alpha_step    = None        # degrees, e.g. 6.0
    data.beta_step     = None        # degrees, e.g. 6.0

    # ---------------- model ----------------
    cfg.model = model = ml_collections.ConfigDict()
    model.network         = "AutoEncoder"
    model.in_channels     = data.channels
    model.out_channels    = data.channels
    model.image_size      = data.image_size
    model.latent_dim      = 3          # keep 2-D latent to match intrinsic dim
    model.base_channels   = 32
    model.num_down_levels = 3
    model.ema_decay       = 0.999
    model.compile         = True
    model.checkpoint      = None

    # VAE block — off by default (kept for compatibility)
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
    curv.mode              = "mecae"
    curv.target            = "both"        # "encoder" | "decoder" | "both"
    curv.reg_lambda        = 1e-4
    curv.K_v               = 1
    curv.K_w               = 2             # latent dim m
    curv.use_rademacher    = True
    curv.estimator         = "square"
    curv.use_exact_hessian = True
    curv.fd_eps            = 1e-3
    curv.B_curv            = 128
    curv.every_n_steps     = 3

    # --- Metric Smoothness (decoder pullback metric invariants) ---
    loss.metric_smooth_weight = 0.0   # start small; 1e-4–3e-3 typical
    loss.metric_smoothness = ms = ml_collections.ConfigDict()
    ms.K_w               = 1           # 1–2; raise to 2 if logs look noisy
    ms.use_rademacher    = True
    ms.use_exact_hessian = True        # nested JVPs; set False to use FD fallback
    ms.fd_eps            = 1e-3
    ms.normalize_by_dim  = True        # scale-free across latent dims
    ms.target            = "encoder"   # "encoder" | "decoder" | "both"
    ms.B_curv            = curv.B_curv # reuse same sub-batch size
    ms.every_n_steps     = curv.every_n_steps  # same cadence as MECAE

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

    geom.latent = lat = ml_collections.ConfigDict()
    lat.enabled          = True
    lat.metric_type      = "jacobian"
    lat.lam_metric       = 1e-3
    lat.t_value          = 0.05
    lat.diffusion_config = "configs/analytic_manifold/sphere/vanilla_architecture/latent_config.py"
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
