import ml_collections

def get_config():
    """
    Latent-diffusion config used *inside* AE training.
    The trainer will:
      • override model.state_size with AE latent_dim,
      • set device to AE device,
      • derive optim.total_steps from the AE loop if left as 0.
    """
    cfg = ml_collections.ConfigDict()
    cfg.random_seed = 42

    # training
    cfg.training = tr = ml_collections.ConfigDict()
    tr.device               = "cuda:0"
    tr.gpus                 = 1
    tr.sde                  = "vpsde"
    tr.loss                 = "simple_DSM_loss"
    tr.likelihood_weighting = False

    # latent loop coupling (used by AE trainer)
    tr.steps_per_ae         = 2        # k latent steps per AE batch
    tr.batch_frac           = 1.0      # use full AE batch; (<1.0) subsamples

    # (latent) data
    cfg.data = data = ml_collections.ConfigDict()
    data.latent_dim = 3               # will be overwritten by AE cfg.model.latent_dim
    data.shape      = [data.latent_dim]

    # model (tiny MLP score net)
    cfg.model = model = ml_collections.ConfigDict()
    model.network    = "mlp"
    model.state_size = data.latent_dim
    model.hidden_dim = 256
    model.depth      = 2
    model.dropout    = 0.0
    model.compile    = False           # tiny MLP; compilation not needed
    model.ema_decay  = 0.999           # ✅ enable EMA smoothing for latent model
    model.checkpoint = 'LatentDiff_last.pth' #'LatentDiff_epoch_149_loss_0.000.pth'  # default save name (relative to AE ckpt_dir unless absolute)

    # optim
    cfg.optim = opt = ml_collections.ConfigDict()
    opt.scheduler     = "cosine_decay"
    opt.total_steps   = 0              # 0 → trainer derives from AE loop
    opt.optimizer     = "AdamW"
    opt.lr            = 1e-3
    opt.weight_decay  = 0.0
    opt.beta1         = 0.9
    opt.beta2         = 0.99
    opt.eps           = 1e-8
    opt.warmup        = 2_000
    opt.grad_clip     = 1.0

    # (optional) sampling defaults used by eval only; harmless here
    cfg.sampling = smp = ml_collections.ConfigDict()
    smp.steps        = 250
    smp.num_samples  = 36
    smp.grid_nrow    = 6

    return cfg
