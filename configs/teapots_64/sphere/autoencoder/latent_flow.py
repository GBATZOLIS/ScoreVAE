import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed     = 123
    cfg.base_log_dir    = "./results/teapot_sphere"
    cfg.experiment      = "flow_step2_scoregeometry"
    cfg.tensorboard_dir = f"{cfg.base_log_dir}/{cfg.experiment}/tb"
    cfg.checkpoint_dir  = f"{cfg.base_log_dir}/{cfg.experiment}/ckpts"
    cfg.eval_dir        = f"{cfg.base_log_dir}/{cfg.experiment}/eval"

    # ---------------- paths ----------------
    cfg.paths = paths = ml_collections.ConfigDict()
    paths.ae_config     = "configs/teapots_64/sphere/autoencoder/autoencoder_iso_curv.py"
    paths.latent_config = "configs/teapots_64/sphere/autoencoder/latent_config.py"

    # ---------------- latent score ----------------
    cfg.latent = lat = ml_collections.ConfigDict()
    lat.score_time        = 0.8
    #lat.score_time_final  = 0.02   # can ramp later if you want

    # ---------------- training ----------------
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device            = "cuda:0"
    tr.epochs            = 60
    tr.batch_size        = 128
    tr.grad_clip         = 5.0
    tr.precision         = "fp32"     # "fp32" | "fp16" | "bf16"
    tr.geom_warmup_steps = 1000
    tr.log_every         = 50
    tr.ckpt_every        = 5
    tr.viz_every         = 5

    #should be settings of the FLOW model
    tr.init_alpha          = 0.05     # start at identity
    tr.final_alpha         = 1.0     # end at fully learned
    tr.alpha_warmup_steps  = 2000    # ramp length

    # ---------------- flow model --------------
    cfg.flow = fl = ml_collections.ConfigDict()
    fl.n_blocks  = 6
    fl.hidden    = 256
    fl.layers    = 2
    fl.transform = "rq"
    fl.bins      = 8
    fl.range     = 3.0

    # ---------------- optimizer ---------------
    cfg.optim = opt = ml_collections.ConfigDict()
    opt.optimizer    = "AdamW"
    opt.lr           = 2e-4
    opt.weight_decay = 0.0
    opt.beta1        = 0.9
    opt.beta2        = 0.999
    opt.eps          = 1e-8

    # ---------------- losses ------------------
    loss = ml_collections.ConfigDict()

    # weights used in the top-level loss
    loss.w_iso     = 1.0
    loss.iso_num_v = 8

    # --- MSM ---
    loss.msm = ml_collections.ConfigDict()
    loss.w_msm              = 0.7     # keep the weight here or at top-level if you prefer
    loss.msm.K_w            = 2
    loss.msm.use_exact_hessian = True
    loss.msm.fd_eps         = 1e-3
    loss.msm.normalize_by_dim = True
    # jitter to stabilize Cholesky
    loss.msm.chol_reg_abs   = 1e-6
    loss.msm.chol_reg_rel   = 1e-3

    # --- CONN ---
    loss.conn = ml_collections.ConfigDict()
    loss.w_conn            = 0.3
    loss.conn.K_pairs      = 2
    loss.conn.use_rademacher = True
    loss.conn.normalize_by_dim = True
    # jitter
    loss.conn.chol_reg_abs = 1e-6
    loss.conn.chol_reg_rel = 1e-3

    cfg.loss = loss

    return cfg
