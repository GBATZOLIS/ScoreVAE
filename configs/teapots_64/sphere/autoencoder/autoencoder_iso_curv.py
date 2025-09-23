# configs/rendered_so_dataset/autoencoder_s2_axisangle_64x64_gray.py
import ml_collections
from math import ceil

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed   = 42
    cfg.base_log_dir  = "./results/teapot_sphere"
    cfg.experiment    = "autoencoder_iso_enc_1e-3_dec_2e-4_curv_1e-5"
    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/training_logs"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/checkpoints"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"

    # ---------------- training ----------------
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device                = "cuda:0"
    tr.devices               = "0"
    tr.epochs                = 85            # training budget
    tr.checkpoint_frequency  = 5             # denser checkpoints
    tr.patience_epochs       = 8             # early stop if no val improvement for ~8 epochs
    tr.vis_frequency         = 5             # more frequent visuals
    tr.update_norm_frequency = 2             # update latent normalizer each epoch

    # ---------------- data ----------------
    cfg.data = data = ml_collections.ConfigDict()
    data.device         = "cuda"
    data.dataset        = "rendered_so_dataset"
    data.mesh_path      = "datasets/meshes/teapot.obj"
    data.image_size     = 64
    data.channels       = 1                  # grayscale
    data.data_samples   = 100_000
    data.ambient_dim    = data.channels * data.image_size * data.image_size  # 4096
    data.dataset_path   = "datasets/teapot_s2axis_gray64.pt"
    data.overwrite_cache= False
    data.batch_size     = 128
    data.n_workers      = 4
    data.shape          = [data.channels, data.image_size, data.image_size]

    # Submanifold (homeomorphic to S^2)
    data.manifold_dim   = 3
    data.submanifold    = "s2_axisangle"
    data.axis_angle_deg = 90.0

    # No grid sampling
    data.azim_step      = None
    data.elev_step      = None
    data.roll_step      = None

    # ---------------- model ----------------
    cfg.model = model = ml_collections.ConfigDict()
    model.network          = "AutoEncoderCoordConv"
    model.in_channels      = data.channels
    model.out_channels     = data.channels
    model.image_size       = data.image_size
    model.latent_dim       = 3
    model.base_channels    = 32
    model.num_down_levels  = 3                 # 64 -> 32 -> 16 -> 8 spatial
    model.groups_gn        = 8
    model.ema_decay        = 0.999
    model.compile          = True
    model.checkpoint       = None  # 'AE_last_EMA.pth'

    # CoordConv toggles
    model.use_coordconv_encoder             = False
    model.use_coordconv_decoder_bottleneck  = True
    model.use_coordconv_decoder_all_levels  = False
    model.decoder_upsample_mode             = "resize_conv"
    model.output_activation                 = "linear"
    model.deconv_bilinear_init              = True

    # Fourier features
    model.use_fourier_features   = False
    model.fourier_num_freqs      = 6
    model.fourier_max_freq_log2  = 5
    model.fourier_include_self   = True

    # Geometry-friendly stem
    model.use_orthogonal_stem = False
    model.stem_reflections    = 4
    model.stem_init_alpha     = 0.1

    # Spectral norm
    model.use_spectral_norm               = False
    model.use_spectral_norm_when_compiled = False

    # Encoder anti-aliasing
    model.encoder_downsample_mode = "avg"
    model.encoder_blur_filt_size  = 5

    # VAE block — off (compatibility only)
    model.vae = ml_collections.ConfigDict()
    model.vae.enabled = False

    # ---------------- losses ----------------
    cfg.loss = loss = ml_collections.ConfigDict()
    loss.reconstruction = "mse"
    loss.beta_kl        = 1e-3

    # Local isometry regs
    loss.enc_iso_weight = 1e-3
    loss.dec_iso_weight = 2e-4
    loss.num_v          = 1

    # --- MECAE (extrinsic curvature) ---
    loss.curvature_weight = 1e-5
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
    curv.B_curv            = 40
    curv.every_n_steps     = 3

    # --- Metric Smoothness (off) ---
    loss.metric_smooth_weight = 0.0
    loss.metric_smoothness = ms = ml_collections.ConfigDict()
    ms.K_w               = 1
    ms.use_rademacher    = True
    ms.use_exact_hessian = True
    ms.fd_eps            = 1e-3
    ms.normalize_by_dim  = True
    ms.target            = "encoder"
    ms.B_curv            = curv.B_curv
    ms.every_n_steps     = curv.every_n_steps

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

    # ---------------- geometry ----------------
    cfg.loss.geom = geom = ml_collections.ConfigDict()

    # Observation-space geometry (disabled)
    geom.data = data_geom = ml_collections.ConfigDict()
    data_geom.enabled                 = False
    data_geom.diffusion_config        = "configs/rendered_so_dataset/s2_axisangle_64x64_gray_unet.py"
    data_geom.metric_type             = "jacobian"
    data_geom.lam_metric              = 1e-3
    data_geom.t_value                 = 0.03
    data_geom.use_denoiser_retraction = True
    data_geom.cg = ml_collections.ConfigDict()
    data_geom.cg.max_iter             = 5
    data_geom.cg.tol                  = 1e-5
    data_geom.cg.preconditioner       = None
    data_geom.cg.precond_diag_samples = 8

    # Latent-space geometry (enabled)
    geom.latent = lat = ml_collections.ConfigDict()
    lat.enabled           = True
    lat.metric_type       = "jacobian"
    lat.lam_metric        = 1e-3
    lat.t_value           = 0.05
    lat.diffusion_config  = "configs/teapots_64/sphere/autoencoder/latent_config.py"
    lat.cg = ml_collections.ConfigDict()
    lat.cg.max_iter              = 5
    lat.cg.tol                   = 1e-6
    lat.cg.preconditioner        = "diagonal"
    lat.cg.precond_diag_samples  = 5

    # ---------------- optim ----------------
    cfg.optim = opt = ml_collections.ConfigDict()
    steps_per_epoch = ceil(data.data_samples * 0.9 / data.batch_size)
    opt.total_steps = steps_per_epoch * tr.epochs
    opt.optimizer   = "AdamW"
    opt.lr          = 2e-3
    opt.weight_decay= 1e-4
    opt.beta1       = 0.9
    opt.beta2       = 0.99
    opt.eps         = 1e-8
    # Warmup scaled to ~5% of total steps, min 500
    opt.warmup      = max(500, int(0.05 * opt.total_steps))
    opt.grad_clip   = 1.0

    return cfg
