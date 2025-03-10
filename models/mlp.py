import torch.nn as nn
import torch

class mlp(nn.Module):
    def __init__(self, model_config): 
        super(mlp, self).__init__()
        state_size = model_config.state_size  # Use ambient_dim directly
        hidden_layers = model_config.depth
        hidden_nodes = model_config.hidden_dim
        dropout = model_config.dropout if hasattr(model_config, 'dropout') else 0.0

        input_size = state_size + 1  # +1 because of the time dimension.
        output_size = state_size

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_size, hidden_nodes))
        self.mlp.append(nn.Dropout(dropout))  # addition
        self.mlp.append(nn.ELU())

        for _ in range(hidden_layers):
            self.mlp.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.mlp.append(nn.Dropout(dropout))  # addition
            self.mlp.append(nn.ELU())
        
        self.mlp.append(nn.Linear(hidden_nodes, output_size))
        self.mlp = nn.Sequential(*self.mlp)
             
    def forward(self, x, y, t):
        # Ensure t has shape (x.size(0), 1)
        if t.dim() == 0:
            # t is a scalar -> create a batch of t's
            t = t.unsqueeze(0).expand(x.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        # Now, t is (batch, 1) and can be concatenated with x and y.
        x_t = torch.cat([x, y, t], dim=-1) if y is not None else torch.cat([x, t], dim=-1)
        return self.mlp(x_t)

    def print_model_summary(self):
        """
        Prints the number of trainable and non-trainable parameters in the diffusion model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_non_trainable_params = total_params - total_trainable_params

        print(f"Total number of parameters: {total_params}")
        print(f"Total number of trainable parameters: {total_trainable_params}")
        print(f"Total number of non-trainable parameters: {total_non_trainable_params}")
    
    def get_score_fn(self, sde):
        # Get the sigma function from the SDE
        sigma_fn = sde.get_sigma_fn()
        
        def score_fn(x_t, y, t):
            sigma_t = sigma_fn(t)
            sigma_t = sigma_t.view(sigma_t.shape[0], *[1 for _ in range(len(x_t.shape) - 1)])  # Expand dimensions
            noise_pred = self.forward(x_t, y, t)
            score = -noise_pred / sigma_t
            return score
        
        return score_fn

    def get_denoiser_fn(self, sde):
        # Infer the alpha and sigma functions from the SDE
        alpha_fn = sde.get_alpha_fn()
        sigma_fn = sde.get_sigma_fn()
        def denoiser_fn(x_t, y, t):
            sigma_t, alpha_t = sigma_fn(t), alpha_fn(t)
            noise_pred = self.forward(x_t, y, t)
            x_denoised = (x_t - sigma_t * noise_pred) / alpha_t
            return x_denoised
        return denoiser_fn