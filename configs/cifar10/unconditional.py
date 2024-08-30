import ml_collections
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/cifar10"
    config.experiment = "unconditional"
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
    training.vis_callback = 'base'
    training.vis_frequency = 50 #generate data every vis_frequency epochs
    training.fid_eval_frequency = 2000 #FID evaluation frequency
    training.steps = 128 #number of integration steps
    training.num_samples = 128 #number of samples to generate
    ## settings for forward SDE + loss function
    training.sde = 'vpsde'
    training.loss = "simple_DSM_loss"
    training.likelihood_weighting = False

    # Data settings
    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 128
    data.dataset = 'CIFAR10'
    data.image_size = 32
    data.num_channels = 3
    data.shape = [data.num_channels, data.image_size, data.image_size]

    # Model settings
    config.model = model = ml_collections.ConfigDict()
    model.ema_decay = 0.9999
    model.network = 'BeatGANsUNet'
    model.checkpoint = 'Model_epoch_646_loss_0.025'
    model.model_channels = 128
    model.out_channels = data.num_channels
    model.num_res_blocks = 4
    model.embed_channels = 512
    model.attention_resolutions = (16,)
    model.dropout = 0.1
    model.channel_mult = (1, 2, 2, 2)
    model.input_channel_mult = None
    model.conv_resample = True
    model.dims = 2
    model.use_checkpoint = False
    model.num_heads = 1
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.resblock_updown = True
    model.use_new_attention_order = False
    model.resnet_two_cond = False
    model.resnet_cond_channels = None
    model.resnet_use_zero_module = True
    model.attn_checkpoint = False
    model.time_embed_channels = None
    model.num_input_res_blocks = None
    model.image_size = data.image_size
    model.in_channels = data.num_channels

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
