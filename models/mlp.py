import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, model_config): 
        super(MLP, self).__init__()
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
        # Ensuring t has the same batch dimension as x
        t = t.unsqueeze(-1)
        # Concatenate x, y (if not None), and t along the last dimension
        x_t = torch.cat([x, y, t], dim=-1) if y is not None else torch.cat([x, t], dim=-1)
        return self.mlp(x_t)
