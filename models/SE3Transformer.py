import torch
import torch.nn as nn
import math
from se3_transformer_pytorch import SE3TransformerLayer
# Import equivariant versions of normalization and dropout
from se3_transformer_pytorch import EquivariantLayerNorm, EquivariantDropout
from se3_transformer_pytorch import EquivariantLinear

def sinusoidal_time_embedding(t, embedding_dim):
    """
    Generates sinusoidal positional embeddings for time steps.

    :param t: Tensor of time steps [batch_size]
    :param embedding_dim: Dimensionality of the embedding.
    :return: Sinusoidal embeddings of shape [batch_size, embedding_dim]
    """
    half_dim = embedding_dim // 2
    device = t.device
    freqs = torch.exp(
        -torch.arange(0, half_dim, dtype=torch.float32, device=device) * (math.log(10000.0) / half_dim)
    )
    args = t[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # If embedding_dim is odd, we may need to pad to maintain the correct size
    if embedding_dim % 2 != 0:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding  # Shape: [batch_size, embedding_dim]

def create_edges_and_attr(use_edges, max_seq_length, num_backbone_atoms):
    """
    Creates edge connections and edge attributes for the SE(3) transformer.

    :param use_edges: Boolean flag to determine if edges should be created.
    :param max_seq_length: The maximum sequence length (number of residues).
    :param num_backbone_atoms: Number of backbone atoms (typically 4 for N, CA, C, O).
    :return: edges, edge_attr tensors
    """
    if not use_edges:
        return None, None

    num_residues = max_seq_length
    num_nodes = num_residues * num_backbone_atoms

    # Initialize edge list and edge attributes
    edges = []
    edge_attr = []

    # Type-1 connections: Connect all backbone atoms within each residue
    for residue_idx in range(num_residues):
        start_idx = residue_idx * num_backbone_atoms
        for i in range(num_backbone_atoms):
            for j in range(i + 1, num_backbone_atoms):
                edges.append([start_idx + i, start_idx + j])
                edges.append([start_idx + j, start_idx + i])

                # Type-1 connection attribute
                edge_attr.append([1])
                edge_attr.append([1])

    # Type-2 connections: Connect all atoms of neighboring residues
    for residue_idx in range(num_residues - 1):
        curr_start = residue_idx * num_backbone_atoms
        next_start = (residue_idx + 1) * num_backbone_atoms

        for i in range(num_backbone_atoms):
            for j in range(num_backbone_atoms):
                edges.append([curr_start + i, next_start + j])
                edges.append([next_start + j, curr_start + i])

                # Type-2 connection attribute
                edge_attr.append([2])
                edge_attr.append([2])

    # Convert edge list and attributes to tensors
    edges = torch.tensor(edges, dtype=torch.long).t()  # Shape: (2, num_edges)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return edges, edge_attr

class SE3Transformer(nn.Module):
    def __init__(self, config):
        super(SE3Transformer, self).__init__()

        self.num_layers = config.model.num_layers
        self.num_heads = config.model.num_heads
        self.num_edge_channels = config.model.num_edge_channels
        self.num_degrees = config.model.num_degrees  # Degrees 0, 1, 2
        self.time_embedding_dim = config.model.time_embedding_dim
        self.dropout_rate = config.model.dropout

        # Channels per degree
        self.degree_channels_out = config.model.degree_channels_out  # Output channels per degree after each layer
        self.degree_channels_in = config.model.degree_channels_in    # Input channels per degree

        # Equivariant normalization and dropout
        self.norm = EquivariantLayerNorm(self.degree_channels_out)
        self.dropout = EquivariantDropout(self.dropout_rate)

        # Precompute edges and edge attributes
        self.edges, self.edge_attr = create_edges_and_attr(
            config.model.use_edges,
            config.data.max_seq_length,
            config.data.num_backbone_atoms
        )

        # Input projection layers
        # For scalar features (degree-0), we initialize to zeros (no initial scalar features)
        # For vector features (degree-1), project from 3D to desired dimension using an equivariant linear layer
        self.vector_proj = EquivariantLinear(1, self.degree_channels_out[1])  # From degree-1 (3D) to degree-1 (desired dimension)

        # For degree-2 features, initialize to zeros (no initial features)

        # Build SE(3)-equivariant layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = self.build_layer(
                channels_in=self.degree_channels_in,
                channels_out=self.degree_channels_out
            )
            self.layers.append(layer)
            # No need to update in_channels, as they remain constant
            # Concatenation of time embedding is handled separately

        # Output layer
        self.output_layer = self.build_output_layer(channels_in=self.degree_channels_in)

        # Output projection to map degree-1 features back to 3D coordinates
        self.output_proj = EquivariantLinear(self.degree_channels_out[1], 1)  # From degree-1 to degree-1 (3D)

    def build_layer(self, channels_in, channels_out):
        return SE3TransformerLayer(
            num_degrees=self.num_degrees,
            channels_in=channels_in,
            channels_out=channels_out,
            num_heads=self.num_heads,
            edge_dim=self.num_edge_channels,
            use_layer_norm=False,  # We'll handle normalization separately
            dropout=self.dropout_rate,
            use_gating=True  # Gating allows scalar features to modulate vector features
        )

    def build_output_layer(self, channels_in):
        return SE3TransformerLayer(
            num_degrees=self.num_degrees,
            channels_in=channels_in,
            channels_out=self.degree_channels_out,  # Output channels per degree
            num_heads=self.num_heads,
            edge_dim=self.num_edge_channels,
            use_layer_norm=False,
            dropout=0.0,
            use_gating=False
        )

    def handle_edges(self, batch_size, num_nodes):
        """
        Handle the replication of edges and edge attributes for each graph in the batch.
        If no edges are provided, return None for both batch_edges and batch_edge_attr.
        """
        if self.edges is not None and self.edge_attr is not None:
            device = self.edges.device

            # Number of edges per graph
            num_edges = self.edges.shape[1]

            # Create batch offsets
            batch_offsets = torch.arange(batch_size, device=device) * num_nodes  # Shape: [batch_size]
            batch_offsets = batch_offsets.view(batch_size, 1, 1)  # Shape: [batch_size, 1, 1]

            # Expand and offset edges
            batch_edges = self.edges.unsqueeze(0).expand(batch_size, -1, -1) + batch_offsets  # [batch_size, 2, num_edges]
            batch_edges = batch_edges.permute(1, 0, 2).reshape(2, -1)  # [2, batch_size * num_edges]

            # Repeat edge attributes
            batch_edge_attr = self.edge_attr.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.edge_attr.shape[-1])
        else:
            # If no edges or edge attributes are provided, return None
            batch_edges = None
            batch_edge_attr = None

        return batch_edges, batch_edge_attr

    def forward(self, node_features, t):
        batch_size, num_nodes, _ = node_features.shape

        # Compute time embedding
        time_emb = sinusoidal_time_embedding(t, self.time_embedding_dim)  # [batch_size, time_embedding_dim]
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch_size, num_nodes, time_embedding_dim]

        # Prepare degree-0 features (scalar features)
        scalar_features = torch.zeros(batch_size, num_nodes, 32, device=node_features.device)  # [batch_size, num_nodes, 32]
        # Concatenate time embedding
        scalar_features = torch.cat([scalar_features, time_emb_expanded], dim=-1)  # [batch_size, num_nodes, 32 + time_embedding_dim]

        # Prepare degree-1 features (vector features)
        vector_features = self.vector_proj(node_features.unsqueeze(-1))  # [batch_size, num_nodes, degree_channels_out[1], 3]

        # Prepare degree-2 features (initialize to zeros)
        degree_2_features = torch.zeros(batch_size, num_nodes, self.degree_channels_out[2], device=node_features.device)  # [batch_size, num_nodes, 128]

        # Initialize node features dictionary
        node_features_dict = {
            '0': scalar_features,
            '1': vector_features,
            '2': degree_2_features
        }

        # Handle edges and edge attributes
        batch_edges, batch_edge_attr = self.handle_edges(batch_size, num_nodes)

        # Pass through SE(3)-equivariant layers
        for layer in self.layers:
            # Process with the SE(3)-equivariant layer
            node_features_dict = layer(node_features_dict, batch_edges, batch_edge_attr)

            # After layer, scalar features have dimension [batch_size, num_nodes, 32]
            # Concatenate time embedding for next layer
            scalar_features = torch.cat([node_features_dict['0'], time_emb_expanded], dim=-1)  # [batch_size, num_nodes, 32 + time_embedding_dim]
            node_features_dict['0'] = scalar_features

            # Apply equivariant dropout and normalization
            node_features_dict = self.dropout(node_features_dict)
            node_features_dict = self.norm(node_features_dict)

        # SE(3)-equivariant output
        output = self.output_layer(node_features_dict, batch_edges, batch_edge_attr)

        # Extract the degree-1 features (positions)
        output_positions = output['1']  # [batch_size, num_nodes, degree_channels_out[1], 3]

        # Project output_positions back to 3D coordinates
        output_positions = self.output_proj(output_positions)  # [batch_size, num_nodes, 1, 3]
        output_positions = output_positions.squeeze(-2)  # [batch_size, num_nodes, 3]

        return output_positions
