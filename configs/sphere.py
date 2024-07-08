import ml_collections
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results"
    config.experiment = "check_deep_sphere_2"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir = f"{config.base_log_dir}/{config.experiment}/eval"

    # Training settings
    config.training = training = ml_collections.ConfigDict()
    training.device = "cpu"
    training.gpus = 1  # Number of GPUs to use
    training.epochs = 1000
    training.checkpoint_frequency = 20
    training.patience_epochs = 100

    ##settings for the generation callback during training
    training.vis_frequency = 50 #generate data every vis_frequency epochs
    training.steps = 1024 #number of integration steps
    training.num_samples = 500 #number of samples to generate

    training.sde = 'vesde'
    training.loss = "DSM_loss"
    training.likelihood_weighting = False

    # Data settings
    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 64
    data.dataset = 'sphere'
    data.data_samples = 10000
    data.n_spheres = 1
    data.ambient_dim = 2
    data.manifold_dim = 1
    data.noise_std = 0.0
    data.embedding_type = 'random_isometry'
    data.radii = []
    data.angle_std = -1
    data.shape = [data.ambient_dim]

    # Model settings
    config.model = model = ml_collections.ConfigDict()
    model.network = 'MLP'
    model.state_size = data.ambient_dim
    model.hidden_dim = 512
    model.depth = 5
    model.dropout = 0.0
    model.ema_decay = 0.999
    
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
