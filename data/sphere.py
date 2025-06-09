import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset

class KSphereDataset(Dataset):
    def __init__(self, args, seed=0):
        self.seed = seed
        data_samples = getattr(args, 'data_samples', 10000)
        n_spheres = getattr(args, 'n_spheres', 1)
        ambient_dim = getattr(args, 'ambient_dim', 10)
        manifold_dim = getattr(args, 'manifold_dim', 3)
        noise_std = getattr(args, 'noise_std', 0.05)
        embedding_type = getattr(args, 'embedding_type', 'random_isometry')
        radii = getattr(args, 'radii', [])
        angle_std = getattr(args, 'angle_std', -1)

        self.embedding_matrix = None
        self.manifold_dim = manifold_dim

        self.data = self.generate_data(data_samples,
                                       n_spheres,
                                       ambient_dim,
                                       manifold_dim,
                                       noise_std,
                                       embedding_type,
                                       radii,
                                       angle_std)

        if not isinstance(self.data, torch.Tensor):
            raise TypeError("Generated data must be a PyTorch tensor")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def generate_data(self, n_samples, n_spheres, ambient_dim,
                      manifold_dim, noise_std, embedding_type,
                      radii, angle_std):
        if not radii:
            radii = [1] * n_spheres

        if isinstance(manifold_dim, int):
            manifold_dims = [manifold_dim] * n_spheres
        else:
            manifold_dims = manifold_dim

        data = []
        for i in range(n_spheres):
            manifold_dim = manifold_dims[i]
            new_data = self.sample_sphere(n_samples, manifold_dim, angle_std)
            new_data = new_data * radii[i]

            if embedding_type == 'random_isometry':
                g = torch.Generator().manual_seed(self.seed)
                A = torch.randn(ambient_dim, manifold_dim + 1, generator=g)
                q, _ = torch.linalg.qr(A)
                self.embedding_matrix = q  # store for geodesic mapping
                new_data = (q @ new_data.T).T

            elif embedding_type == 'first':
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)

            elif embedding_type == 'separating':
                if n_spheres * (manifold_dim + 1) > ambient_dim:
                    raise RuntimeError('Too many spheres for the given ambient dimension.')
                prefix_zeros = torch.zeros((n_samples, i * (manifold_dim + 1)))
                new_data = torch.cat([prefix_zeros, new_data], dim=1)
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)

            elif embedding_type == 'along_axis':
                if (n_spheres - 1) + (manifold_dim + 1) > ambient_dim:
                    raise RuntimeError('Too many spheres for the given ambient dimension.')
                prefix_zeros = torch.zeros((n_samples, i))
                new_data = torch.cat([prefix_zeros, new_data], dim=1)
                suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
                new_data = torch.cat([new_data, suffix_zeros], dim=1)

            else:
                raise RuntimeError('Unknown embedding type.')

            new_data = new_data + noise_std * torch.randn_like(new_data)
            data.append(new_data)

        return torch.cat(data, dim=0)

    def sample_sphere(self, n_samples, manifold_dim, std=-1):
        if std == -1:
            x = torch.randn((n_samples, manifold_dim + 1))
            x = x / torch.norm(x, dim=1, keepdim=True)
            return x
        else:
            raise NotImplementedError("Non-uniform sampling not implemented yet.")

    def great_arc(self, p: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Batched great arc interpolation on the unit sphere.

        Args:
            p: Tensor of shape [B, K+1] – start points on the sphere
            q: Tensor of shape [B, K+1] – end points on the sphere
            t: Tensor of shape [T] – interpolation times in [0, 1]

        Returns:
            Tensor of shape [B, T, K+1] – interpolated points on the sphere
        """
        p = p / p.norm(dim=1, keepdim=True)
        q = q / q.norm(dim=1, keepdim=True)
        dot = (p * q).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)  # [B]
        theta = torch.acos(dot)                              # [B]
        sin_theta = torch.sin(theta).unsqueeze(1)            # [B, 1]

        t = t.to(p.device)
        t_expand = t[None, :, None]                          # [1, T, 1]
        theta_expand = theta[:, None, None]                  # [B, 1, 1]

        p_expand = p[:, None, :]                             # [B, 1, K+1]
        q_expand = q[:, None, :]                             # [B, 1, K+1]

        term1 = torch.sin((1 - t_expand) * theta_expand) / sin_theta[:, None, :] * p_expand
        term2 = torch.sin(t_expand * theta_expand) / sin_theta[:, None, :] * q_expand
        return term1 + term2                                 # [B, T, K+1]


    def compute_geodesic(self, p_data, q_data, t):
        """
        Compute great arcs between pairs of points on the sphere, mapped into ambient space.

        Args:
            p_data: Tensor of shape [B, D]
            q_data: Tensor of shape [B, D]
            t: Tensor of shape [T] with values in [0, 1]

        Returns:
            Tensor of shape [B, T, D]
        """
        if self.embedding_matrix is None:
            raise ValueError("Embedding matrix is not defined. Use 'random_isometry' embedding.")

        # Step 1: Project back to sphere coordinates
        pinv = torch.linalg.pinv(self.embedding_matrix)  # [K+1, D]
        p_sphere = (pinv @ p_data.T).T  # [B, K+1]
        q_sphere = (pinv @ q_data.T).T

        # Step 2: Compute intrinsic geodesic on sphere
        geodesics = self.great_arc(p_sphere, q_sphere, t)  # [B, T, K+1]

        # Step 3: Map geodesics back to ambient space
        B, T, K1 = geodesics.shape
        geodesics_flat = geodesics.reshape(-1, K1)  # [B*T, K+1]
        arc_mapped = (self.embedding_matrix @ geodesics_flat.T).T  # [B*T, D]
        return arc_mapped.view(B, T, -1)  # [B, T, D]
