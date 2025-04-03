import ml_collections
from datetime import timedelta
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/ffhq"
    config.experiment = "scoreVAE_noise"
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
    training.vis_callback = 'scoreVAE'
    training.vis_frequency = 1 #generate data every vis_frequency epochs
    training.fid_eval_frequency = 2500 #FID evaluation frequency
    training.steps = 128 #number of integration steps
    training.num_samples = 16 #number of samples to generate
    ## settings for forward SDE + loss function
    training.sde = 'snrsde'
    training.loss = "scoreVAE_loss" #ScoreVAE setting
    training.likelihood_weighting = False
    training.kl_weight = 1e-4 #ScoreVAE setting
    training.continuous = True
    training.t_dependent = True
    training.t_batch_size = 1
    training.variational = True
    training.use_pretrained = True
    training.prior_checkpoint_path = "/home/rg625/mnt/ScoreVAE/scoreVAE checkpoints/ffhq/prior/cheackpoints/epoch=141--eval_loss_epoch=0.014.ckpt"
    training.prior_config_path = "/home/rg625/mnt/ScoreVAE/scoreVAE checkpoints/ffhq/prior/config.pkl"
    training.beta_schedule = 'linear'

    # Data settings
    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 16
    data.dataset = 'FFHQ'
    data.image_size = 128
    data.num_channels = 3
    data.shape = [data.num_channels, data.image_size, data.image_size]
    data.latent_dim = 512 #scoreVAE setting
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
    model.network = 'CombinedDiffusionEncoder'

    # Pretrained Diffusion Model settings
    config.model.diffusion_model = diffusion_model = ml_collections.ConfigDict()
    diffusion_model.network = 'BeatGANsUNet'
    diffusion_model.checkpoint = 'ffhq_checkpoints/ffhq/prior/cheackpoints/epoch=141--eval_loss_epoch=0.014.ckpt'
    diffusion_model.model_channels = 128
    diffusion_model.out_channels = data.num_channels
    diffusion_model.num_res_blocks = 2
    diffusion_model.embed_channels = 512
    diffusion_model.attention_resolutions = (16,)
    diffusion_model.dropout = 0.0
    diffusion_model.channel_mult = (1, 1, 2, 3, 4)
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
    encoder.model_channels = 128
    encoder.enc_num_res_blocks = 2
    encoder.latent_dim = data.latent_dim
    encoder.enc_attn_resolutions = ()
    encoder.enc_use_time_condition = True
    encoder.enc_channel_mult = (1, 1, 2, 3, 4, 4)
    encoder.enc_pool = 'flatten-linear'
    encoder.resolution_before_flattening = 4
    encoder.resblock_updown = False
    encoder.encoder_input_channels = data.num_channels
    encoder.enc_out_channels = 1024
    encoder.encoder_split_output = False
    encoder.dropout = 0.0
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
