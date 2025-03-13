import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results"
    config.experiment = "earth-deep"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir = f"{config.base_log_dir}/{config.experiment}/eval"

    # Training settings
    config.training = training = ml_collections.ConfigDict()
    training.device = "cpu"  # change to "cuda" if available
    training.gpus = 1  # Number of GPUs to use
    training.epochs = 1000
    training.checkpoint_frequency = 20
    training.patience_epochs = 100

    ## Settings for the generation callback during training
    training.vis_callback = 'base'
    training.vis_frequency = 50  # generate data every vis_frequency epochs
    training.steps = 256        # number of integration steps
    training.num_samples = 500   # number of samples to generate

    training.sde = 'vpsde'
    training.loss = "simple_DSM_loss"
    training.likelihood_weighting = False
    training.fid_eval_frequency = 10**9  # Disable FID evaluation for sphere data

    # Data settings
    config.data = data = ml_collections.ConfigDict()
    data.data_path = '/Users/gbatz97/Desktop/landseamask_water-global.nc'
    data.batch_size = 128
    data.dataset = 'earth'
    data.data_samples = 50000
    data.ambient_dim = 3        # ambient dimension: 2 for a 1d sphere in 2D
    data.manifold_dim = 2
    data.shape = [data.ambient_dim]

    # Model settings
    config.model = model = ml_collections.ConfigDict()
    model.network = 'mlp'
    model.checkpoint = 'Model_epoch_179_loss_0.135.pth'
    model.state_size = data.ambient_dim  # same as ambient_dim
    model.hidden_dim = 512
    model.depth = 4
    model.dropout = 0.0
    model.ema_decay = 0.9999

    # Optimization settings
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 1e-5
    optim.optimizer = 'Adam'
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.0

    # Evaluation settings
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.eval_callback_epochs = 20
    evaluation.num_eval_points = 10
    evaluation.eval_save_path = "./eval"

    return config
