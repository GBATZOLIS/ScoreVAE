import torch
import torch.nn as nn
from se3_transformer_pytorch import SE3Transformer
import math

# Simple sinusoidal time embedding function
def sinusoidal_time_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal embeddings for the given timesteps.
    """
    half_dim = embedding_dim // 2
    device = timesteps.device
    timesteps = timesteps.unsqueeze(1)  # Shape: [batch_size, 1]
    emb = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=device).float() / half_dim
    )  # Shape: [half_dim]
    emb = timesteps * emb  # Shape: [batch_size, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # Shape: [batch_size, embedding_dim]
    return emb

class SE3DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super(SE3DiffusionTransformer, self).__init__()
        self.num_layers = config.num_layers
        self.num_channels = config.num_channels
        self.time_embedding_dim = config.time_embedding_dim
        self.time_embedding_channels = config.time_embedding_channels

        # MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),  # Intermediate hidden size remains the same
            nn.SiLU(),  # Smooth activation function
            nn.Linear(self.time_embedding_dim, self.time_embedding_channels),  # Project to time embedding channels
        )

        # Node embedding (positions)
        #self.node_embedding = nn.Linear(3, self.num_channels)

        # SE3Transformer using degree-0 and degree-1 features (scalars and vectors)
        self.transformer = SE3Transformer(
            dim=(self.num_channels+self.time_embedding_channels,),
            depth=self.num_layers,
            num_degrees=2,  # Using scalars (degree-0) and vectors (degree-1)
            input_degrees=2,  # Degree-1 for positions (as vectors)
            output_degrees=1,
            dim_head=config.dim_head,  # Dimension of each attention head
            heads=config.num_heads,  # Number of attention heads
            num_neighbors=8,
            reduce_dim_out=False,  # Do not reduce dimension prematurely
        )

        # Output layer for score function
        self.output_layer = nn.Linear(self.num_channels+self.time_embedding_channels, 3)

    def forward(self, x, y, t):
        """
        Forward pass of the SE3DiffusionTransformer.
        :param x: Tensor of shape [batch_size, num_nodes, 3], node positions.
        :param t: Tensor of shape [batch_size], timesteps.
        :return: Tensor of shape [batch_size, num_nodes, 3], score function.
        """
        batch_size, num_nodes, _ = x.shape

        # Compute time embeddings
        t_embed = sinusoidal_time_embedding(t, self.time_embedding_dim)  # Shape: [batch_size, time_embedding_dim]
        t_embed = self.time_mlp(t_embed)  # Shape: [batch_size, num_channels]

        # Node features (degree-1 vectors)
        #x_embed = self.node_embedding(x)  # Shape: [batch_size, num_nodes, num_channels]
        x_embed = x

        # Time embedding influences transformer weights (equivariant way)
        # Concatenate x_embed with t_embed for degree-0 features
        t_embed_expanded = t_embed.unsqueeze(1).expand(-1, num_nodes, -1)  # Expand time embedding for all nodes
        degree_0_features = torch.cat([t_embed_expanded, x_embed], dim=-1)  # Concatenate along feature dimension
        degree_0_features = degree_0_features.unsqueeze(-1)  # Add an extra dimension for SE(3) transformer input
        features = {'0': degree_0_features}  # Scalar time embedding
        features['1'] = x_embed.unsqueeze(-1)  # Vector node embedding

        # Pass through the SE3Transformer
        out = self.transformer(feats=features, coors=x, return_type=1)  # Output degree-1 (vectors)

        # Output shape: [batch_size, num_nodes, num_channels, 3]
        out = out.squeeze(-1)  # Remove last dimension

        # Map to score function
        score = self.output_layer(out)  # Shape: [batch_size, num_nodes, 3]

        return score