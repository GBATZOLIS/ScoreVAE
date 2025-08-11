# ~/ScoreVAE/configs/so/base.py
import ml_collections

def get_base_config(n_group: int) -> ml_collections.ConfigDict:
    """Return a ConfigDict with sensible defaults for any SO(n)."""
    config = ml_collections.ConfigDict()
    config.random_seed = 42

    # Training settings
    config.training = training = ml_collections.ConfigDict()
    training.device = "cuda:1"
    training.gpus = 1
    training.epochs = 800
    training.checkpoint_frequency = 25
    training.patience_epochs = 120
    training.vis_callback = "base"
    training.vis_frequency = 50
    training.steps = 256
    training.num_samples = 500
    training.sde = "vpsde"
    training.loss = "simple_DSM_loss"
    training.likelihood_weighting = False
    training.fid_eval_frequency = 10 ** 9

    # Data settings
    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 256
    data.dataset = "so"
    data.n_group = n_group
    data.data_samples = 200_000
    data.ambient_dim = n_group ** 2
    data.dataset_path = f"./data/so{n_group}.pt"
    data.overwrite_cache = False
    data.shape = [data.ambient_dim]

    # Model settings
    config.model = model = ml_collections.ConfigDict()
    model.network = "mlp"
    model.state_size = data.ambient_dim
    model.hidden_dim = 768
    model.depth = 4
    model.dropout = 0.1
    model.ema_decay = 0.999
    model.checkpoint = ''

    # optim settings
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "AdamW"
    optim.lr = 2e-4
    optim.weight_decay = 1e-4
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.warmup = 5_000
    optim.grad_clip = 1.0

    # Evaluation settings
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.eval_callback_epochs = 25
    evaluation.num_eval_points = 10
    evaluation.eval_save_path = "./eval"

    # Logging settings
    config.base_log_dir = "./results"
    config.experiment = f"so{n_group}"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir = f"{config.base_log_dir}/{config.experiment}/eval"

    return config
