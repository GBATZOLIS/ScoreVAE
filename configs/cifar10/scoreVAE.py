import ml_collections
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/cifar10"
    config.experiment = "scoreVAE_noise"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir = f"{config.base_log_dir}/{config.experiment}/eval"

    # Training settings
    config.training = training = ml_collections.ConfigDict()
    ## general training settings
    training.device = "cuda:0"
    training.gpus = 1  # Number of GPUs to use
    training.epochs = 1000
    training.checkpoint_frequency = 1
    training.patience_epochs = 300
    ## settings for the generation callback during training
    training.vis_callback = 'scoreVAE'
    training.vis_frequency = 20 #generate data every vis_frequency epochs
    training.fid_eval_frequency = 2000 #FID evaluation frequency
    training.steps = 128 #number of integration steps
    training.num_samples = 128 #number of samples to generate
    ## settings for forward SDE + loss function
    training.sde = 'vpsde'
    training.loss = "scoreVAE_loss" #ScoreVAE setting
    training.likelihood_weighting = False
    training.kl_weight = 1e-3 #ScoreVAE setting

    # Data settings
    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 128
    data.dataset = 'CIFAR10'
    data.image_size = 32
    data.num_channels = 3
    data.shape = [data.num_channels, data.image_size, data.image_size]
    data.latent_dim = 384 #scoreVAE setting

    # Model settings
    config.model = model = ml_collections.ConfigDict()
    model.ema_decay = 0.9999
    model.network = 'CombinedDiffusionEncoder'

    # Pretrained Diffusion Model settings
    config.model.diffusion_model = diffusion_model = ml_collections.ConfigDict()
    diffusion_model.network = 'BeatGANsUNet'
    diffusion_model.checkpoint = '/home/gb511/disentanglement/results/cifar10/unconditional/checkpoints/Model_epoch_646_loss_0.025_EMA.pth'
    diffusion_model.model_channels = 128
    diffusion_model.out_channels = data.num_channels
    diffusion_model.num_res_blocks = 4
    diffusion_model.embed_channels = 512
    diffusion_model.attention_resolutions = (16,)
    diffusion_model.dropout = 0.1
    diffusion_model.channel_mult = (1, 2, 2, 2)
    diffusion_model.input_channel_mult = None
    diffusion_model.conv_resample = True
    diffusion_model.dims = 2
    diffusion_model.use_checkpoint = False
    diffusion_model.num_heads = 1
    diffusion_model.num_head_channels = -1
    diffusion_model.num_heads_upsample = -1
    diffusion_model.resblock_updown = True
    diffusion_model.use_new_attention_order = False
    diffusion_model.resnet_two_cond = False
    diffusion_model.resnet_cond_channels = None
    diffusion_model.resnet_use_zero_module = True
    diffusion_model.attn_checkpoint = False
    diffusion_model.time_embed_channels = None
    diffusion_model.num_input_res_blocks = None
    diffusion_model.image_size = data.image_size
    diffusion_model.in_channels = data.num_channels

    # Encoder settings
    config.model.encoder = encoder = ml_collections.ConfigDict()
    encoder.network = 'BeatGANsEncoderModel'
    encoder.model_channels = 64
    encoder.enc_num_res_blocks = 2
    encoder.latent_dim = data.latent_dim
    encoder.enc_attn_resolutions = ()
    encoder.enc_use_time_condition = True
    encoder.enc_channel_mult = (1, 1, 2, 4)
    encoder.enc_pool = 'flatten-linear'
    encoder.resolution_before_flattening = data.image_size // 2**(len(encoder.enc_channel_mult)-1)
    encoder.resblock_updown = False
    encoder.encoder_input_channels = data.num_channels
    encoder.enc_out_channels = 2 * data.latent_dim
    encoder.encoder_split_output = False
    encoder.dropout = 0
    encoder.dims = 2
    encoder.image_size = data.image_size
    encoder.in_channels = data.num_channels
    encoder.conv_resample = True
    encoder.num_heads = 1
    encoder.num_head_channels = -1
    encoder.num_heads_upsample = -1
    encoder.use_new_attention_order = False
    encoder.resnet_two_cond = False
    encoder.resnet_cond_channels = None
    encoder.resnet_use_zero_module = True
    encoder.attn_checkpoint = False
    encoder.use_checkpoint = False

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
