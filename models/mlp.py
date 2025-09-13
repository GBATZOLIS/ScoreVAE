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
        sigma_fn = sde.get_sigma_fn()

        def _expand_time_like(x_like: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
            if t_vec.dim() == 0:
                t_vec = t_vec.unsqueeze(0)
            if t_vec.dim() == 2 and t_vec.size(-1) == 1:
                t_vec = t_vec.squeeze(-1)
            view_shape = (t_vec.shape[0],) + (1,) * (x_like.dim() - 1)
            return t_vec.view(view_shape)

        def score_fn(x_t, y, t):
            sigma_t = _expand_time_like(x_t, sigma_fn(t))
            noise_pred = self.forward(x_t, y, t)
            return -noise_pred / sigma_t

        return score_fn


    def get_denoiser_fn(self, sde):
        alpha_fn = sde.get_alpha_fn()
        sigma_fn = sde.get_sigma_fn()

        def _expand_time_like(x_like: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
            # t_vec: (B,) or (B,1) or scalar; return (B, 1, ..., 1) to match x_like dims
            if t_vec.dim() == 0:
                t_vec = t_vec.unsqueeze(0)
            if t_vec.dim() == 2 and t_vec.size(-1) == 1:
                t_vec = t_vec.squeeze(-1)  # (B,)
            view_shape = (t_vec.shape[0],) + (1,) * (x_like.dim() - 1)
            return t_vec.view(view_shape)

        def denoiser_fn(x_t, y, t):
            # Predict noise with the network
            noise_pred = self.forward(x_t, y, t)  # shape (B, state_size, ...)

            # Expand alpha/sigma over feature axes to match x_t / noise_pred
            sigma_t = _expand_time_like(x_t, sigma_fn(t))
            alpha_t = _expand_time_like(x_t, alpha_fn(t))

            # EDM-style x0 estimator when the net predicts noise ε
            x_denoised = (x_t - sigma_t * noise_pred) / alpha_t
            return x_denoised

        return denoiser_fn