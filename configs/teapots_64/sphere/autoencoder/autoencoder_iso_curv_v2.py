import ml_collections
from math import ceil


def get_config():
    cfg = ml_collections.ConfigDict()

    # ─────────────────────────── Experiment Setup ─────────────────────────── #
    cfg.random_seed     = 42
    cfg.base_log_dir    = "./results/teapot_sphere"
    cfg.experiment      = "ae_decoderonly_stylefilm_iso1e-3_dec2e-4_curv1e-5"

    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/training_logs"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/checkpoints"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"


    # ─────────────────────────── Training ─────────────────────────── #
    tr = cfg.training = ml_collections.ConfigDict()
    tr.device                = "cuda:0"
    tr.devices               = "0"
    tr.epochs                = 85
    tr.checkpoint_frequency  = 5
    tr.patience_epochs       = 8
    tr.vis_frequency         = 5
    tr.update_norm_frequency = 2


    # ─────────────────────────── Data ─────────────────────────── #
    data = cfg.data = ml_collections.ConfigDict()
    data.device        = "cuda"
    data.dataset       = "rendered_so_dataset"
    data.mesh_path     = "datasets/meshes/teapot.obj"

    data.image_size    = 64
    data.channels      = 1
    data.shape         = [data.channels, data.image_size, data.image_size]

    data.data_samples  = 100_000
    data.ambient_dim   = data.channels * data.image_size * data.image_size
    data.dataset_path  = "datasets/teapot_s2axis_gray64.pt"
    data.overwrite_cache = False

    data.batch_size    = 128
    data.n_workers     = 4

    data.manifold_dim  = 3
    data.submanifold   = "s2_axisangle"
    data.axis_angle_deg = 90.0

    # (optional sampling step granularity)
    data.azim_step = None
    data.elev_step = None
    data.roll_step = None


    # ─────────────────────────── Model ─────────────────────────── #
    model = cfg.model = ml_collections.ConfigDict()
    model.network         = "AutoEncoderStyleFiLM"   # OLD encoder + NEW StyleGAN-like decoder

    # I/O
    model.in_channels     = data.channels
    model.out_channels    = data.channels
    model.image_size      = data.image_size

    # Latent space
    model.latent_dim      = 3

    # Channels & resolution schedule
    model.base_channels   = 32
    model.num_down_levels = 3
    model.groups_gn       = 8

    # Misc
    model.ema_decay       = 0.9995
    model.compile         = True
    model.checkpoint      = None #"AE_last_EMA.pth"


    # Positional encodings (disabled by default)
    model.use_coordconv_encoder            = False
    model.use_coordconv_decoder_bottleneck = False
    model.use_coordconv_decoder_all_levels = False

    model.use_fourier_features             = False
    model.fourier_num_freqs                = 4
    model.fourier_max_freq_log2            = 5
    model.fourier_include_self             = True


    # Encoder specifics
    model.encoder_downsample_mode = "blur"   # "avg" also supported
    model.encoder_blur_filt_size  = 5


    # Decoder specifics
    model.decoder_upsample_mode   = "resize_conv"   # or "deconv"
    model.output_activation       = "linear"
    model.deconv_bilinear_init    = True
    model.use_spectral_norm       = False
    model.use_spectral_norm_when_compiled = False
    model.output_head_scale       = 0.1

    # Style mapping & FiLM
    model.style_w_dim             = 128
    model.style_film_bias_init    = 0.0


    # ─────────────────────────── Loss Weights ─────────────────────────── #
    loss = cfg.loss = ml_collections.ConfigDict()

    loss.reconstruction   = "mse"
    loss.beta_kl          = 1e-3

    loss.enc_iso_weight   = 1e-3
    loss.dec_iso_weight   = 2e-4
    loss.num_v            = 1

    loss.curvature_weight = 1e-5
    loss.metric_smooth_weight = 0.0
    loss.intrinsic_weight = 0.0


    # Curvature regularization
    curv = loss.curvature = ml_collections.ConfigDict()
    curv.mode             = "mecae"
    curv.target           = "both"
    curv.reg_lambda       = 1e-4
    curv.K_v              = 1
    curv.K_w              = 2
    curv.use_rademacher   = True
    curv.estimator        = "square"
    curv.use_exact_hessian = True
    curv.fd_eps           = 1e-3
    curv.B_curv           = 40
    curv.every_n_steps    = 3


    # Metric smoothness
    ms = loss.metric_smoothness = ml_collections.ConfigDict()
    ms.K_w               = 1
    ms.use_rademacher    = True
    ms.use_exact_hessian = True
    ms.fd_eps            = 1e-3
    ms.normalize_by_dim  = True
    ms.target            = "encoder"
    ms.B_curv            = curv.B_curv
    ms.every_n_steps     = curv.every_n_steps


    # Intrinsic penalty
    intr = loss.intrinsic = ml_collections.ConfigDict()
    intr.reg_lambda       = 1e-4
    intr.R_a              = 2
    intr.use_rademacher   = True
    intr.use_exact_hessian = True
    intr.normalize_codim  = True
    intr.fd_eps           = 1e-3
    intr.B_curv           = curv.B_curv
    intr.every_n_steps    = curv.every_n_steps


    # ─────────────────────────── Latent Geometry ─────────────────────────── #
    geom = cfg.loss.geom = ml_collections.ConfigDict()

    data_geom = geom.data = ml_collections.ConfigDict()
    data_geom.enabled = False

    lat = geom.latent = ml_collections.ConfigDict()
    lat.enabled       = True
    lat.metric_type   = "jacobian"
    lat.lam_metric    = 1e-3
    lat.t_value       = 0.05
    lat.diffusion_config = "configs/teapots_64/sphere/autoencoder/latent_config.py"

    lat.cg = ml_collections.ConfigDict()
    lat.cg.max_iter    = 5
    lat.cg.tol         = 1e-6
    lat.cg.preconditioner       = "diagonal"
    lat.cg.precond_diag_samples = 5


    # ─────────────────────────── Optimizer ─────────────────────────── #
    opt = cfg.optim = ml_collections.ConfigDict()

    steps_per_epoch = ceil(data.data_samples * 0.9 / data.batch_size)
    opt.total_steps = steps_per_epoch * tr.epochs

    opt.optimizer    = "AdamW"
    opt.lr           = 1.5e-3    # slightly lower for FiLM stability
    opt.weight_decay = 1e-4
    opt.beta1        = 0.9
    opt.beta2        = 0.99
    opt.eps          = 1e-8
    opt.warmup       = max(500, int(0.05 * opt.total_steps))
    opt.grad_clip    = 1.0

    return cfg
