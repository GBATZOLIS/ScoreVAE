# configs/rotated_mnist/autoencoder/latent_config.py
import ml_collections


def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed = 48

    # ---------------- Training (device/total_steps are overwritten by the driver) ----------------
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device               = "cuda:0"
    tr.gpus                 = 1
    tr.sde                  = "vpsde"                 # "vpsde" or "vesde"
    tr.loss                 = "simple_DSM_loss"
    tr.likelihood_weighting = False
    tr.steps_per_ae         = 4                       # latent steps per AE batch
    tr.batch_frac           = 1.0
    tr.use_amp              = False                   # set True if you enable AMP for latent model

    # ---------------- Latent data (overwritten by driver from AE cfg) ----------------
    cfg.data = data = ml_collections.ConfigDict()
    data.latent_dim = 3
    data.shape      = [data.latent_dim]

    # ---------------- Model (tiny MLP) ----------------
    cfg.model = model = ml_collections.ConfigDict()
    model.network    = "mlp"                          # must be supported by your get_model()
    model.state_size = data.latent_dim
    model.hidden_dim = 256
    model.depth      = 2
    model.dropout    = 0.0
    model.compile    = False
    model.ema_decay  = 0.999
    model.checkpoint = None #'LatentDiffPost_last_EMA.pth'

    # ---------------- Optimizer / Scheduler (total_steps is injected by the driver) ----------------
    cfg.optim = opt = ml_collections.ConfigDict()
    opt.scheduler     = "cosine_decay"
    opt.total_steps   = 0                             # 0 → derived by driver
    opt.optimizer     = "AdamW"
    opt.lr            = 1e-3
    opt.weight_decay  = 0.0
    opt.beta1         = 0.9
    opt.beta2         = 0.99
    opt.eps           = 1e-8
    opt.warmup        = 2_000
    opt.grad_clip     = 1.0

    # ---------------- SDE params (used by configure_sde) ----------------
    # Driver calls: latent_sde = configure_sde(cfg)
    cfg.sde = sde = ml_collections.ConfigDict()
    sde.type = tr.sde.lower()                         # mirror training.sde
    sde.sampling_eps = 1e-5
    # VPSDE defaults
    sde.beta_min = 0.1
    sde.beta_max = 20.0
    # VESDE defaults
    sde.sigma_min = 0.01
    sde.sigma_max = 50.0
    sde.T = 1.0

    # ---------------- Sampling (for generation callback) ----------------
    cfg.sampling = smp = ml_collections.ConfigDict()
    smp.steps        = 250
    smp.num_samples  = 36
    smp.grid_nrow    = 6

    # ---------------- Extra latent-only training AFTER AE finishes ----------------
    cfg.post_train = pt = ml_collections.ConfigDict()
    pt.steps        = 20_000                          # continue training latent model
    pt.lr           = None                             # None → reuse opt.lr
    pt.warmup       = 2_000
    pt.grad_clip    = 1.0
    # Extras used by the driver:
    pt.epoch_seed   = 10_000                           # reseed sampler each post-epoch
    pt.val_every    = 1                                # run validation every N post-epochs
    pt.val_batches  = None                             # limit val to N batches (None = full)
    pt.gen_every    = 1                                # run gen_cb every N post-epochs
    pt.overfit_patience = 2                            # warn after N worsening val epochs
    pt.overfit_delta    = 0.0

    return cfg
