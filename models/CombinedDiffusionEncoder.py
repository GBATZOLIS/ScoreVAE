import torch.nn as nn
import torch

# Import the get_model function from the __init__.py file in the models directory
from . import get_model

class CombinedDiffusionEncoder(nn.Module):
    def __init__(self, config):
        super(CombinedDiffusionEncoder, self).__init__()

        # Load the diffusion model using the provided configuration
        self.diffusion_model = get_model(config.diffusion_model)
        
        # Load the encoder model using the provided configuration
        self.encoder = get_model(config.encoder)
        
        # Prepare the diffusion model (load weights, freeze, and set to eval mode)
        self.prepare_diffusion_model(config)

    def prepare_diffusion_model(self, config):
        """
        Prepares the diffusion model by loading pretrained weights (if available),
        freezing its parameters, and setting it to evaluation mode.
        This ensures the diffusion model is not trained and behaves consistently during the encoder training.
        """
        # Check and load pretrained weights for the diffusion model
        checkpoint_path = config.diffusion_model.get('checkpoint')
        if not checkpoint_path:
            raise ValueError("Pretrained weights for the diffusion model are required. Please provide a valid checkpoint path.")

        checkpoint = torch.load(checkpoint_path)
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {checkpoint_path}")

        # Freeze the diffusion model to prevent any training
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

        # Set the diffusion model to evaluation mode to ensure consistent behavior
        self.diffusion_model.eval()

    def train(self, mode=True):
        """
        Override the default train() method to ensure that the diffusion model 
        remains in evaluation mode even when the rest of the model is set to training mode.
        """
        # Apply train/eval mode to the encoder only
        self.encoder.train(mode)
        
        # Ensure that the diffusion model remains in evaluation mode
        self.diffusion_model.eval()

    def encode(self, x):
        encoder = self.encoder
        t0 = torch.zeros(x.shape[0]).type_as(x)
        latent_distribution_parameters = encoder(x, t0)
        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]
        latent = mean_z + (0.5 * log_var_z).exp() * torch.randn_like(mean_z)
        return latent

    def get_noise_predictor_fn(self, sde, train=False):
        """
        Returns a function that predicts the noise by combining the pretrained diffusion model
        and the latent correction noise from the encoder.
        
        Args:
            sde: The SDE object that provides the marginal probability.
            train: Boolean flag indicating whether in training mode (affects gradient computation).
            
        Returns:
            noise_predictor_fn: A function that predicts the noise.
        """
        def noise_predictor_fn(x, cond, t):
            y, z = cond
            
            # Get the pretrained noise predictor function for the diffusion model
            pretrained_noise_fn = self.get_pretrained_noise_fn(sde)
            
            # Get the latent correction noise function
            latent_correction_noise_fn = self.get_latent_correction_noise_fn(sde, train)
            
            # Calculate the pretrained noise prediction
            pretrained_noise_prediction = pretrained_noise_fn(x, y, t)
            
            # Calculate the latent correction noise
            latent_correction_noise = latent_correction_noise_fn(x, z, t)
            
            # Combine the noise predictions
            noise_prediction = pretrained_noise_prediction + latent_correction_noise
            
            return noise_prediction
        
        return noise_predictor_fn


    def get_score_fn(self, sde, train=False):
        """
        Returns a function that computes the conditional score by combining
        the pretrained diffusion model score and the latent correction score from the encoder.
        
        Args:
            sde: The SDE object that provides the marginal probability.
            train: Boolean flag indicating whether in training mode (affects gradient computation).
            
        Returns:
            score_fn: A function that computes the combined conditional score.
        """
        def score_fn(x, cond, t):
            y, z = cond
            
            # Get the pretrained score function for the diffusion model
            pretrained_score_fn = self.get_pretrained_score_fn(sde)
            
            # Get the latent correction function for the encoder
            latent_correction_fn = self.get_latent_correction_score_fn(train)
            
            # Calculate the pretrained score
            pretrained_score = pretrained_score_fn(x, y, t)
            
            # Calculate the latent correction score
            latent_correction_score = latent_correction_fn(x, z, t)
            
            # Combine the scores
            conditional_score = pretrained_score + latent_correction_score
            
            return conditional_score
        
        return score_fn
    
    def get_pretrained_noise_fn(self, sde):
        """
        Returns a function that predicts noise based on the pretrained diffusion model.
        
        Args:
            sde: The SDE object that provides the marginal probability.
            
        Returns:
            noise_fn: A function that predicts noise.
        """
        def noise_fn(x, y, t):
            noise_prediction = self.diffusion_model(x, y, t)
            return noise_prediction
        
        return noise_fn
    
    def get_pretrained_score_fn(self, sde):
        """
        Returns a function that computes the score based on the pretrained diffusion model.
        
        Args:
            sde: The SDE object that provides the marginal probability.
            
        Returns:
            score_fn: A function that computes the score.
        """
        def score_fn(x, y, t):
            noise_prediction = self.diffusion_model(x, y, t)
            _, std = sde.marginal_prob(x, t)
            std = std.view(std.shape[0], *[1 for _ in range(len(x.shape) - 1)])  # Expand std to match the shape of noise_prediction
            score = -noise_prediction / std
            return score
        
        return score_fn
    
    def get_latent_correction_noise_fn(self, sde, train=False):
        """
        Returns a function that computes the latent correction noise from the encoder,
        scaled by the standard deviation from the SDE.
        
        Args:
            sde: The SDE object that provides the marginal probability.
            train: Boolean flag indicating whether in training mode (affects gradient computation).
            
        Returns:
            noise_fn: A function that computes the latent correction noise.
        """
        latent_correction_score_fn = self.get_latent_correction_score_fn(train)
        def noise_fn(x, z, t):
            latent_score = latent_correction_score_fn(x, z, t)
            # Scaling by standard deviation from the forward SDE
            _, std = sde.marginal_prob(x, t)
            std = std.view(std.shape[0], *[1 for _ in range(len(x.shape) - 1)])  # shape broadcasting
            latent_correction_noise = -std * latent_score
            return latent_correction_noise

        return noise_fn

    def get_latent_correction_score_fn(self, train):
        """
        Returns a function that computes the latent correction score from the encoder.
        
        Returns:
            latent_correction_fn: A function that computes the latent correction score.
        """
        def latent_correction_fn(x, z, t):
            def log_density_fn(x, z, t):
                latent_distribution_parameters = self.encoder(x, t)
                channels = latent_distribution_parameters.size(1) // 2
                mean_z = latent_distribution_parameters[:, :channels]
                log_var_z = latent_distribution_parameters[:, channels:]

                # Flatten mean_z and log_var_z for consistent shape handling
                mean_z_flat = mean_z.view(mean_z.size(0), -1)
                log_var_z_flat = log_var_z.view(log_var_z.size(0), -1)
                z_flat = z.view(z.size(0), -1)

                logdensity = -0.5 * torch.sum(torch.square(z_flat - mean_z_flat) / log_var_z_flat.exp(), dim=1)
                return logdensity

            if not train: 
                torch.set_grad_enabled(True)

            device = x.device
            x.requires_grad = True
            ftx = log_density_fn(x, z, t)
            grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                                   grad_outputs=torch.ones(ftx.size()).to(device),
                                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
            assert grad_log_density.size() == x.size()

            if not train:
                torch.set_grad_enabled(False)

            return grad_log_density
        
        return latent_correction_fn

    def print_model_summary(self):
        """
        Prints the number of trainable and non-trainable parameters for both the encoder 
        and the diffusion model.
        """
        def count_params(model):
            total_params = sum(p.numel() for p in model.parameters())
            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_non_trainable_params = total_params - total_trainable_params
            return total_trainable_params, total_non_trainable_params

        encoder_trainable_params, encoder_non_trainable_params = count_params(self.encoder)
        diffusion_trainable_params, diffusion_non_trainable_params = count_params(self.diffusion_model)

        print(f"Encoder - Trainable parameters: {encoder_trainable_params}, Non-trainable parameters: {encoder_non_trainable_params}")
        print(f"Diffusion Model - Trainable parameters: {diffusion_trainable_params}, Non-trainable parameters: {diffusion_non_trainable_params}")
        print(f"Total Trainable parameters: {encoder_trainable_params + diffusion_trainable_params}")
        print(f"Total Non-trainable parameters: {encoder_non_trainable_params + diffusion_non_trainable_params}")
