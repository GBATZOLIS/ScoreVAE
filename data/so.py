# data/so_dataset.py
# ─────────────────────────────────────────────────────────────────────────────
"""
Uniform sampling on SO(n) and closed-form geodesic evaluation.

Fast paths
----------
* PyTorch ≥ 2.4   → uses ``torch.linalg.matrix_log / exp`` on GPU (any n)
* PyTorch < 2.4
      n = 3     → analytic SO(3) log/exp (Rodrigues) on GPU  O(1)
      n = 4     → GPU eigen-log (torch.linalg.eig)            O(n³)
      else     → SciPy ``logm/expm`` on CPU (slow)

External dependency
-------------------
    pip install scipy          # (NumPy already comes with SciPy wheels)
"""
from __future__ import annotations

import os, pathlib
import numpy as np
import torch
import scipy.linalg as spla
from torch.utils.data import Dataset


# ────────────────────────────────────────────────────────────────── #
class SOdataset(Dataset):
    """
    Uniform samples on SO(n)  (stored flattened as [N, n²]).

    Expected ``args`` fields
    ------------------------
    n_group          : int       # n ≥ 2
    data_samples     : int
    ambient_dim      : int       # ≥ n²; > n² ⇒ random isometry embedding
    dataset_path     : str | None
    overwrite_cache  : bool
    """

    # ────────────────────────────────────────────────────────────── #
    def __init__(self, args, seed: int = 0):
        g = torch.Generator().manual_seed(seed)

        self.n      = getattr(args, "n_group", 3)
        self.N      = getattr(args, "data_samples", 10_000)
        self.d_emb  = getattr(args, "ambient_dim", self.n ** 2)
        self.path   = getattr(args, "dataset_path", None)
        self.ovr    = getattr(args, "overwrite_cache", False)

        # ---------- load from disk if possible --------------------
        if self.path and os.path.isfile(self.path) and not self.ovr:
            blob       = torch.load(self.path, map_location="cpu")
            self.data  = blob["data"]
            self.P     = blob["proj"]
            print(f"[SOdataset] loaded {self.data.shape[0]} samples from {self.path}")
            return

        # ---------- otherwise generate ----------------------------
        R    = self._sample_so(self.N, self.n, g)           # [N,n,n]
        flat = R.reshape(self.N, -1).float()                # [N, n²]

        if self.d_emb > self.n ** 2:
            A, _   = torch.linalg.qr(torch.randn(self.d_emb, self.n ** 2, generator=g))
            self.P = A
            flat   = (A @ flat.T).T
        else:
            self.P = torch.eye(self.n ** 2)

        self.data = flat

        # ---------- save to disk ---------------------------------
        if self.path:
            pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"data": self.data, "proj": self.P}, self.path)
            print(f"[SOdataset] cached dataset at {self.path}")

    # ────────────────────────────────────────────────────────────── #
    @staticmethod
    def _sample_so(N: int, n: int, g: torch.Generator) -> torch.Tensor:
        """Haar-uniform SO(n) via Mezzadri QR."""
        G     = torch.randn(N, n, n, generator=g)
        Q, _  = torch.linalg.qr(G)               # batched QR
        det   = torch.linalg.det(Q)
        Q[det < 0, :, 0] *= -1                  # make det = +1
        return Q

    # torch-Dataset boiler-plate ----------------------------------
    def __len__(self):          # type: ignore[override]
        return self.data.shape[0]

    def __getitem__(self, idx):  # type: ignore[override]
        return self.data[idx]

    # ────────────────────── fast analytic SO(3) ops ────────────── #
    @staticmethod
    def _so3_log(R: torch.Tensor) -> torch.Tensor:
        """Analytic logarithm for SO(3) – GPU friendly."""
        tr   = R.diagonal(dim1=-2, dim2=-1).sum(-1)         # […]
        cosθ = ((tr - 1) * 0.5).clamp(-1.0, 1.0)
        θ    = torch.acos(cosθ)
        sinθ = torch.sin(θ)

        # θ / (2 sin θ)  with a series for small θ
        small = θ.abs() < 1e-5
        coef  = torch.where(
            small,
            0.5 - θ**2 / 12 + θ**4 / 720,
            θ / (2.0 * sinθ)
        ).unsqueeze(-1).unsqueeze(-1)

        return coef * (R - R.transpose(-2, -1))

    @staticmethod
    def _so3_exp(A: torch.Tensor) -> torch.Tensor:
        """Analytic exponential for so(3) – GPU friendly."""
        w = torch.stack((A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]), dim=-1)
        θ = torch.linalg.vector_norm(w, dim=-1)
        θ2 = θ * θ

        small = θ.abs() < 1e-5
        sθ_by_θ = torch.where(
            small,
            1 - θ2 / 6 + θ2 * θ2 / 120,
            torch.sin(θ) / θ
        )
        one_minus_c_by_θ2 = torch.where(
            small,
            0.5 - θ2 / 12 + θ2 * θ2 / 720,
            (1 - torch.cos(θ)) / θ2
        )
        sθ_by_θ     = sθ_by_θ.unsqueeze(-1).unsqueeze(-1)
        one_minus_c = one_minus_c_by_θ2.unsqueeze(-1).unsqueeze(-1)

        A2   = A @ A
        I    = torch.eye(3, dtype=A.dtype, device=A.device).expand(A.shape)
        return I + sθ_by_θ * A + one_minus_c * A2

    # ───────────────────── generic GPU eig-log (n ≤ 4) ─────────── #
    @staticmethod
    def _eig_log(M: torch.Tensor) -> torch.Tensor:
        """Eigen-decomposition-based logarithm (complex) – GPU."""
        M_c  = M.to(torch.complex128)
        w, V = torch.linalg.eig(M_c)                   # [...]
        logD = torch.diag_embed(torch.log(w))
        Vinv = torch.linalg.inv(V)
        return (V @ logD @ Vinv).real.to(M.dtype)

    # ────────────────────── dispatcher: log + exp ──────────────── #
    @staticmethod
    def _matrix_log(M: torch.Tensor) -> torch.Tensor:
        """Batched matrix log with specialised fast paths."""
        if hasattr(torch.linalg, "matrix_log"):
            return torch.linalg.matrix_log(M)

        n = M.shape[-1]
        if n == 3:
            return SOdataset._so3_log(M)
        if n == 4:
            return SOdataset._eig_log(M)

        # ---------- SciPy CPU fallback ---------------------------
        M_cpu  = M.detach().to("cpu").numpy()
        logcpu = np.stack([spla.logm(x) for x in M_cpu], axis=0)
        return torch.from_numpy(logcpu).to(M.device, dtype=M.dtype)

    @staticmethod
    def _matrix_exp(M: torch.Tensor) -> torch.Tensor:
        """Batched matrix exp with specialised fast paths."""
        if hasattr(torch.linalg, "matrix_exp"):
            return torch.linalg.matrix_exp(M)

        n = M.shape[-1]
        if n == 3:
            return SOdataset._so3_exp(M)

        # ---------- SciPy CPU fallback ---------------------------
        M_cpu  = M.detach().to("cpu").numpy()
        expcpu = np.stack([spla.expm(x) for x in M_cpu], axis=0)
        return torch.from_numpy(expcpu).to(M.device, dtype=M.dtype)

    # ───────────────────────── geodesic (evaluation) ───────────── #
    def compute_geodesic(self,
                         p_data: torch.Tensor,
                         q_data: torch.Tensor,
                         t:      torch.Tensor) -> torch.Tensor:
        """
        Compute the shortest-path geodesic between rotations P and Q.

        Parameters
        ----------
        p_data, q_data : [B, d_emb] tensors (flattened rotations)
        t              : [T]  values in [0, 1]  (time samples)

        Returns
        -------
        path : [B, T, d_emb]  points in ambient space along the geodesic
        """
        B, T = p_data.shape[0], t.numel()
        pinv = self.P.T                                   # orthonormal

        p_mat = (pinv @ p_data.T).T.view(B, self.n, self.n)
        q_mat = (pinv @ q_data.T).T.view(B, self.n, self.n)

        R   = torch.matmul(p_mat.transpose(1, 2), q_mat)
        A   = self._matrix_log(R)                         # fast path aware
        exp_tA = self._matrix_exp(A[:, None] * t[None, :, None, None])
        path   = torch.matmul(p_mat[:, None], exp_tA)     # [B,T,n,n]

        flat = path.view(B * T, -1)
        emb  = (self.P @ flat.T).T
        return emb.view(B, T, -1)
