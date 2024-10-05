import torch
import torch.nn as nn
from se3_transformer_pytorch import SE3Transformer
import math

# Simple sinusoidal time embedding function
def sinusoidal_time_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    device = timesteps.device
    timesteps = timesteps.unsqueeze(1)  # Shape: [batch_size, 1]
    emb = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=device).float() / half_dim
    )  # Shape: [half_dim]
    emb = timesteps * emb  # Shape: [batch_size, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # Shape: [batch_size, embedding_dim]
    return emb

class SE3TransformerWadjacency(nn.Module):
    def __init__(self, config):
        super(SE3TransformerWadjacency, self).__init__()
        self.num_layers = config.num_layers
        self.num_channels = config.num_channels
        self.time_embedding_dim = config.time_embedding_dim
        self.time_embedding_channels = config.time_embedding_channels

        self.num_residues = config.num_residues
        self.num_backbone_atoms = config.num_backbone_atoms
        self.num_nodes = self.num_residues * self.num_backbone_atoms

        # MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_channels),
        )

        # SE3Transformer using degree-0 and degree-1 features
        self.transformer = SE3Transformer(
            dim=(self.num_channels + self.time_embedding_channels,),
            depth=self.num_layers,
            num_degrees=2,
            input_degrees=2,
            output_degrees=1,
            dim_head=config.dim_head,
            heads=config.num_heads,
            attend_sparse_neighbors=True,  # Respect adjacency matrix for meaningful neighbors
            attend_self=True,              # Enable self-attention explicitly
            num_neighbors=0,               # Only use adjacency matrix
            reduce_dim_out=False,
        )

        # Output layer for score function
        self.output_layer = nn.Linear(self.num_channels + self.time_embedding_channels, 3)

        # Compute adjacency matrix once and register as buffer
        self.register_buffer('adj_mat', self.create_adjacency_matrix())

    def forward(self, x, y, t):
        """
        Forward pass of the SE3TransformerWadjacency.
        :param x: Tensor of shape [batch_size, num_nodes, 3], node positions.
        :param t: Tensor of shape [batch_size], timesteps.
        :return: Tensor of shape [batch_size, num_nodes, 3], score function.
        """
        batch_size, num_nodes, _ = x.shape

        # Compute time embeddings
        t_embed = sinusoidal_time_embedding(t, self.time_embedding_dim)
        t_embed = self.time_mlp(t_embed)

        # Node features (positions)
        x_embed = x  # Assuming positions as features

        # Time embedding influences transformer weights
        t_embed_expanded = t_embed.unsqueeze(1).expand(-1, num_nodes, -1)
        degree_0_features = torch.cat([t_embed_expanded, x_embed], dim=-1)
        degree_0_features = degree_0_features.unsqueeze(-1)
        features = {'0': degree_0_features}
        features['1'] = x_embed.unsqueeze(-1)

        # Expand adjacency matrix to match batch size
        adj_mat = self.adj_mat.unsqueeze(0).expand(batch_size, -1, -1)

        # Pass through the SE3Transformer with adjacency matrix
        out = self.transformer(
            feats=features,
            coors=x,
            adj_mat=adj_mat,
            return_type=1
        )

        # Output layer
        out = out.squeeze(-1)
        score = self.output_layer(out)

        return score

    def create_adjacency_matrix(self):
        num_nodes = self.num_nodes
        num_backbone_atoms = self.num_backbone_atoms
        num_residues = self.num_residues

        adj_mat = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)

        for res_idx in range(num_residues):
            base = res_idx * num_backbone_atoms
            # Within residue connections
            adj_mat[base, base + 1] = True  # N to CA
            adj_mat[base + 1, base + 2] = True  # CA to C
            adj_mat[base + 2, base + 3] = True  # C to O

            # Between residues connections (except last residue)
            if res_idx < num_residues - 1:
                next_base = (res_idx + 1) * self.num_backbone_atoms
                adj_mat[base + 2, next_base] = True  # C to N of next residue

        # Make the adjacency matrix symmetric (without self-loops)
        adj_mat = adj_mat | adj_mat.transpose(0, 1)

        return adj_mat
