import ml_collections

def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.random_seed = 42

    # training (overwritten by trainer for device & total_steps)
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device               = "cuda:0"
    tr.gpus                 = 1
    tr.sde                  = "vpsde"
    tr.loss                 = "simple_DSM_loss"
    tr.likelihood_weighting = False

    # AE coupling
    tr.steps_per_ae         = 4          # latent steps per AE batch
    tr.batch_frac           = 1.0

    # (latent) data — overwritten by trainer from AE config
    cfg.data = data = ml_collections.ConfigDict()
    data.latent_dim = 3
    data.shape      = [data.latent_dim]

    # model (tiny MLP)
    cfg.model = model = ml_collections.ConfigDict()
    model.network    = "mlp"
    model.state_size = data.latent_dim
    model.hidden_dim = 256
    model.depth      = 2
    model.dropout    = 0.0
    model.compile    = False
    model.ema_decay  = 0.999
    model.checkpoint = None

    # optim (total_steps is injected by trainer during joint phase)
    cfg.optim = opt = ml_collections.ConfigDict()
    opt.scheduler     = "cosine_decay"
    opt.total_steps   = 0                # 0 → derived by trainer
    opt.optimizer     = "AdamW"
    opt.lr            = 1e-3
    opt.weight_decay  = 0.0
    opt.beta1         = 0.9
    opt.beta2         = 0.99
    opt.eps           = 1e-8
    opt.warmup        = 2_000
    opt.grad_clip     = 1.0

    # sampling (for the gen callback)
    cfg.sampling = smp = ml_collections.ConfigDict()
    smp.steps        = 250
    smp.num_samples  = 36
    smp.grid_nrow    = 6

    # NEW: extra latent-only training AFTER AE finishes
    cfg.post_train = ml_collections.ConfigDict()
    cfg.post_train.steps       = 20_000
    cfg.post_train.lr          = None          # reuse main latent lr
    cfg.post_train.warmup      = 2_000
    cfg.post_train.grad_clip   = 1.0
    cfg.post_train.epoch_seed  = 10_000
    cfg.post_train.val_every   = 1
    cfg.post_train.val_batches = None
    cfg.post_train.gen_every   = 5

    return cfg
