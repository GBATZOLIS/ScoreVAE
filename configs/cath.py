import ml_collections
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results"
    config.experiment = "cath"
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
    training.vis_frequency = 50 #generate data every vis_frequency epochs
    training.fid_eval_frequency = 2000 #FID evaluation frequency
    training.steps = 128 #number of integration steps
    training.num_samples = 128 #number of samples to generate
    ## settings for forward SDE + loss function
    training.sde = 'vpsde'
    training.loss = "simple_DSM_loss" #denoising score matching loss. Simple weighting used
    training.likelihood_weighting = False

    # Data settings for CATH dataset
    config.data = data = ml_collections.ConfigDict()

    # Batch size
    data.batch_size = 64  # Adjusted based on typical sizes for 3D geometric data
    # Dataset name
    data.dataset = 'CATH'
    # Sequence length (number of residues)
    data.max_seq_length = 256  # Max sequence length for each protein structure
    # Number of backbone atoms (N, CA, C, O)
    data.num_backbone_atoms = 4  # We are using only backbone atoms
    # Atom dimensionality (3D coordinates: x, y, z)
    data.num_coordinates = 3
    # Data shape (number of residues, number of atoms per residue, 3D coordinates)
    data.shape = [data.max_seq_length, data.num_backbone_atoms, data.num_coordinates]

    # Model settings
    config.model = model = ml_collections.ConfigDict()
    model.ema_decay = 0.9999 # Exponential moving average (EMA) decay
   
    
    model.network = 'SE3_Transformer'
    model.checkpoint = None  # Checkpoint to load, if available
    model.time_embedding_dim = 128  # Time embedding dim for diffusion process
    model.num_layers = 6  # Number of transformer layers
    model.num_channels = 128  # Embedding size for each atom (node)
    model.num_heads = 8  # Attention heads in each transformer layer
    model.num_edge_channels = 128  # Embedding size for edges
    model.num_degrees = 3  # Number of irreducible representations (degrees of spherical harmonics)
    model.degree_channels_out = [32, 64, 128]  # Output channels per degree after each layer
    model.degree_channels_in = [32 + model.time_embedding_dim, 64, 128]  # Input channels per degree
    model.activation = 'relu'  # Activation function
    model.dropout = 0.1  # Dropout rate for regularization
    model.norm = 'layernorm'  # Normalization technique
    # Edge connections
    model.use_edges = False  # Whether to incorporate edge information
    model.num_edge_types = data.num_edge_types  # Number of edge types
    # Output settings
    model.output_channels = 3  # Each point (atom) is represented by 3D coordinates
    model.residual = True  # Use residual connections in transformer layers
    


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
    evaluation.devices = [0,1,2]
    evaluation.eval_callback_epochs = 20
    evaluation.num_eval_points = 10
    evaluation.eval_save_path = "./eval"

    return config
