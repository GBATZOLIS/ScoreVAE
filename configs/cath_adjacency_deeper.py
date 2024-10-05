import ml_collections
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results"
    config.experiment = "deep_cath_adjacency_self_attention_diff_implementation"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir = f"{config.base_log_dir}/{config.experiment}/eval"

    # Training settings
    config.training = training = ml_collections.ConfigDict()
    ## general training settings
    training.device = "cuda:3"
    training.gpus = 1  # Number of GPUs to use
    training.epochs = 1000
    training.checkpoint_frequency = 1
    training.patience_epochs = 300
    ## settings for the generation callback during training
    training.vis_frequency = 20  # Generate data every vis_frequency epochs
    training.fid_eval_frequency = 2000  # FID evaluation frequency
    training.steps = 128  # Number of integration steps
    training.num_samples = 32  # Number of samples to generate
    ## settings for forward SDE + loss function
    training.sde = 'vpsde'
    training.loss = "simple_DSM_loss"  # Denoising score matching loss. Simple weighting used
    training.likelihood_weighting = False

    # Data settings for CATH dataset
    config.data = data = ml_collections.ConfigDict()

    # Batch size
    data.batch_size = 64  # Adjusted based on typical sizes for 3D geometric data
    # Dataset name
    data.dataset = 'cath-preprocessed'
    # Sequence length (number of residues)
    data.max_seq_length = 256  # Max sequence length for each protein structure
    # Number of backbone atoms (N, CA, C, O)
    data.num_backbone_atoms = 4  # We are using only backbone atoms
    # Atom dimensionality (3D coordinates: x, y, z)
    data.num_coordinates = 3
    # Data shape (number of residues * number of atoms per residue, 3D coordinates)
    data.shape = [data.max_seq_length * data.num_backbone_atoms, data.num_coordinates]

    # Model settings
    config.model = model = ml_collections.ConfigDict()
    model.ema_decay = 0.999  # Exponential moving average (EMA) decay

    # Updated model parameters
    model.network = 'SE3TransformerWadjacency'  # Updated to match the class name in the transformer code
    model.checkpoint = None  # Checkpoint to load, if available
    model.num_channels = data.num_coordinates
    model.time_embedding_dim = 64  # Time embedding dimension for diffusion process
    model.time_embedding_channels = 1
    model.num_layers = 6  # Number of transformer layers
    model.num_heads = 8  # Number of attention heads in each transformer layer
    model.dim_head = 16  # Dimension of each attention head
    model.num_degrees = 2  # Number of degrees (0, 1, 2)

    #following settings are used in the construction of the adjacency matrix 
    model.num_residues = data.max_seq_length
    model.num_backbone_atoms = data.num_backbone_atoms

    # Optimization settings
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.01  # Updated weight decay
    optim.optimizer = 'AdamW'  # Use AdamW optimizer
    optim.lr = 2e-4  # Updated learning rate
    optim.beta1 = 0.9
    optim.beta2 = 0.99  # Updated beta2
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    # Evaluation settings
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.num_samples = 1000
    evaluation.devices = [2]
    evaluation.eval_callback_epochs = 20
    evaluation.num_eval_points = 10
    evaluation.eval_save_path = "./eval"

    return config
