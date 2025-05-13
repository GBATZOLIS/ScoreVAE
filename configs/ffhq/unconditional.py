import ml_collections
from datetime import timedelta
import os
def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/ffhq"
    config.experiment = "unconditional_resume"
    config.tensorboard_dir = f"{config.base_log_dir}/{config.experiment}/training_logs"
    config.checkpoint_dir = f"{config.base_log_dir}/{config.experiment}/checkpoints"
    config.eval_dir = f"{config.base_log_dir}/{config.experiment}/eval"

    # Training settings
    config.training = training = ml_collections.ConfigDict()
    ## general training settings
    training.device = "cuda:0"
    training.gpus = 1  # Number of GPUs to use
    training.epochs = 10000
    training.checkpoint_frequency = 1
    training.patience_epochs = 300
    ## settings for the generation callback during training
    training.vis_callback = 'base'
    training.vis_frequency = 1 #generate data every vis_frequency epochs
    training.fid_eval_frequency = 2500 #FID evaluation frequency
    training.steps = 128 #number of integration steps
    training.num_samples = 16 #number of samples to generate
    ## settings for forward SDE + loss function
    training.sde = 'snrsde'
    training.loss = "simple_DSM_loss"
    training.likelihood_weighting = False
    training.continuous = True
    training.t_dependent = True
    training.t_batch_size = 1
    training.variational = True
    training.beta_schedule = 'linear'

    # Data settings
    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 16
    data.dataset = 'FFHQ'
    data.image_size = 128
    data.num_channels = 3
    data.shape = [data.num_channels, data.image_size, data.image_size]
    data.base_dir = os.path.expanduser("~/datasets/ffhq")
    data.centered = True
    data.class_cond = False
    data.create_dataset = False
    data.datamodule = "guided_diffusion_dataset"
    data.percentage_use = 100
    data.random_crop = False
    data.random_flip = False
    data.return_labels = False
    data.split = [0.9, 0.05, 0.05]
    data.use_data_mean = False

    # Model settings
    config.model = model = ml_collections.ConfigDict()
    model.ema_decay = 0.999
    model.network = 'BeatGANsUNet'
    model.model_channels = 128
    model.out_channels = data.num_channels
    model.num_res_blocks = 2
    model.embed_channels = 512
    model.attention_resolutions = (16,)
    model.dropout = 0.0
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
    model.checkpoint = "/home/rg625/mnt/ScoreVAE/ffhq_checkpoints/ffhq/prior/cheackpoints/epoch=141--eval_loss_epoch=0.014.ckpt"

    # Optimization settings
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = 'Adam'
    optim.lr = 5e-5
    optim.beta1 = 0.9
    optim.beta2 = 0.99
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.0
    optim.slowing_factor = 1
    optim.accumulation_steps = 1

    # Evaluation settings
    config.evaluation = evaluation = ml_collections.ConfigDict()
    evaluation.devices = [0]
    evaluation.eval_callback_epochs = 20
    evaluation.num_eval_points = 10
    evaluation.eval_save_path = "./eval"
    evaluation.batch_size = 16
    evaluation.workers = 4
    evaluation.enable_bpd = False
    evaluation.enable_loss = True
    evaluation.enable_sampling = True
    evaluation.num_samples = 50000

    # Sampling settings
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.corrector = "conditional_none"
    sampling.method = "pc"
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.predictor = "conditional_ddim"
    sampling.probability_flow = False
    sampling.snr = 0.15

    return config 