#!/usr/bin/env python3
# datasets/rotated_mnist_dataset.py
# SO(2) manifold from a single MNIST digit rotated by θ ~ Uniform(0, 2π).
# - Pads 28×28 to 32×32 (default) to reduce resampling artifacts.
# - Caches tensor dataset and supports exact geodesic rendering on SO(2).

from __future__ import annotations
import argparse, math, os, pathlib
from contextlib import suppress
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

# ───────────────────────── helpers ───────────────────────── #

def _angle_to_R2(theta: torch.Tensor) -> torch.Tensor:
    """theta: (...,) radians → R: (..., 2, 2) rotation matrices"""
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s], dim=-1),
                     torch.stack([s,  c], dim=-1)], dim=-2)
    return R

def _R2_to_angle(R: torch.Tensor) -> torch.Tensor:
    """R: (..., 2, 2) → theta: (...,) radians"""
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])

def _rotate_img(imgCHW: torch.Tensor, deg: float) -> torch.Tensor:
    """
    Rotate a single image tensor (C,H,W) by 'deg' degrees, bilinear, center-preserving.
    Background fill=0 (black), matches MNIST style.
    """
    return transforms.functional.rotate(
        imgCHW, angle=deg,
        interpolation=InterpolationMode.BILINEAR,
        expand=False, fill=0
    )

def _resize_img(imgCHW: torch.Tensor, size: int) -> torch.Tensor:
    return transforms.functional.resize(
        imgCHW, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True
    )

def _pad_to_32(imgCHW: torch.Tensor) -> torch.Tensor:
    """Pad (1,28,28) → (1,32,32) with 2-pixel zeros on each side."""
    return transforms.functional.pad(imgCHW, padding=[2, 2, 2, 2], fill=0)

# ───────────────────────── dataset ───────────────────────── #

class RotatedMNIST(Dataset):
    """
    Cached rotated-MNIST dataset using a single fixed digit instance.
    __getitem__ returns (image[C,H,W], R[2,2]) where R is the SO(2) rotation used.

    Ground-truth geodesics:
        compute_geodesic(P, Q, t) -> (B, T, C, H, W)
        where P,Q are (B,2,2) rotations and t ∈ [0,1]^T.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        digit: int = 9,
        split: str = "train",                  # "train" or "test" for source digit
        sample_index: Optional[int] = None,    # which instance of 'digit' (None → first occurrence)
        image_size: int = 32,
        channels: int = 1,
        data_samples: int = 100_000,
        ambient_dim: Optional[int] = None,     # default C*H*W
        overwrite_cache: bool = False,
        device: str = "cpu",
        angle_step_deg: Optional[float] = None,# if set, build a grid over [0,360)
        pad_to_32: bool = True,                # ← minimize artifacts by padding before rotation
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.dataset_path = dataset_path
        self.digit = int(digit)
        self.split = split
        self.sample_index = sample_index
        self.H = self.W = int(image_size)
        self.channels = int(channels)
        assert self.channels == 1, "RotatedMNIST is grayscale; set channels=1."
        self.N = int(data_samples)
        self.device = torch.device(device)
        self.overwrite = bool(overwrite_cache)
        self.ambient_dim = ambient_dim or (self.channels * self.H * self.W)
        self.angle_step_deg = angle_step_deg
        self.pad_to_32 = bool(pad_to_32)

        if os.path.isfile(self.dataset_path) and not self.overwrite:
            blob = torch.load(self.dataset_path, map_location="cpu")
            self.data     = blob["data"].float()        # (N,1,H,W) in [0,1]
            self.angles   = blob["angles"].float()      # (N,)
            self.Rs       = blob["rot"].float()         # (N,2,2)
            self.P        = blob.get("proj", torch.eye(self.ambient_dim))
            self.base_img = blob["base_img"].float()    # (1,H,W)
            print(f"[RotatedMNIST] Loaded {len(self)} cached samples from {self.dataset_path}")
        else:
            self._build_cache()
            pathlib.Path(self.dataset_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"data": self.data, "angles": self.angles, "rot": self.Rs,
                 "proj": self.P, "base_img": self.base_img},
                self.dataset_path,
            )
            print(f"[RotatedMNIST] Cached → {self.dataset_path}")

    # ---------- dataset protocol ----------

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.Rs[idx]

    # ---------- geodesic renderer (ground truth on SO(2)) ----------

    @torch.no_grad()
    def compute_geodesic(self, P: torch.Tensor, Q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        P,Q: (B,2,2) rotation matrices; t: (T,) in [0,1]
        Returns: (B, T, 1, H, W) images in [0,1]
        """
        dev = self.base_img.device
        if P.dim() != 3 or Q.dim() != 3 or P.shape[-2:] != (2, 2) or Q.shape[-2:] != (2, 2):
            raise ValueError("P and Q must be (B,2,2) rotation matrices.")

        B = P.size(0)
        T = t.numel()

        thetaP = _R2_to_angle(P.to(dev))
        thetaQ = _R2_to_angle(Q.to(dev))
        # shortest angular difference (wrap to [-pi, pi))
        d = (thetaQ - thetaP + math.pi) % (2 * math.pi) - math.pi
        thetas = thetaP[:, None] + t.to(dev)[None, :] * d[:, None]    # (B,T)

        out = []
        for b in range(B):
            frames = []
            for k in range(T):
                deg = float(thetas[b, k].item() * 180.0 / math.pi)
                frames.append(_rotate_img(self.base_img, deg))   # (1,H,W)
            out.append(torch.stack(frames, dim=0))  # (T,1,H,W)
        return torch.stack(out, dim=0).clamp(0, 1)   # (B,T,1,H,W)

    # ---------- internals ----------

    def _select_base_digit(self) -> torch.Tensor:
        """
        Load MNIST, pick the requested digit instance, return (1,H,W) in [0,1].
        If pad_to_32=True and target size is 32, pad 28→32 (no resize) before rotations.
        """
        root = os.environ.get("TORCH_HOME", "datasets/torchvision")
        split_train = (self.split.lower() == "train")
        mnist = datasets.MNIST(root=root, train=split_train, download=True)
        x = mnist.data   # (N,28,28), uint8
        y = mnist.targets

        idxs = (y == self.digit).nonzero(as_tuple=False).view(-1)
        if idxs.numel() == 0:
            raise RuntimeError(f"Digit {self.digit} not found in MNIST {self.split} split.")

        idx = int(self.sample_index) if self.sample_index is not None else int(idxs[0].item())
        img = x[idx].float() / 255.0  # (28,28) in [0,1]
        img = img.unsqueeze(0)        # (1,28,28)

        if self.pad_to_32:
            # 28→32 padding first
            img = _pad_to_32(img)     # (1,32,32)
            if self.H != 32 or self.W != 32:
                img = _resize_img(img, self.H)  # only if requested size != 32
        else:
            # no padding: resize directly if needed
            if self.H != 28 or self.W != 28:
                img = _resize_img(img, self.H)
        return img

    def _build_cache(self):
        self.base_img = self._select_base_digit()  # (1,H,W)
        N = self.N

        # angle sampling
        if self.angle_step_deg is not None:
            n_steps = max(1, int(round(360.0 / float(self.angle_step_deg))))
            thetas = torch.linspace(0.0, 2 * math.pi, n_steps, endpoint=False)
            if n_steps < N:
                reps = (N + n_steps - 1) // n_steps
                thetas = thetas.repeat(reps)[:N]
            else:
                thetas = thetas[:N]
        else:
            thetas = torch.rand(N) * (2 * math.pi)  # Uniform(0, 2π)

        imgs = []
        print(f"[RotatedMNIST] Generating N={N} rotations (size={self.H}×{self.W}, pad_to_32={self.pad_to_32})")
        for th in tqdm(thetas.tolist(), desc="Rotating"):
            deg = th * 180.0 / math.pi
            imgs.append(_rotate_img(self.base_img, deg))

        data = torch.stack(imgs, dim=0).float()      # (N,1,H,W)
        Rs   = _angle_to_R2(thetas).float()          # (N,2,2)

        flat = data.view(N, -1)
        d_img = flat.size(1)
        d_emb = self.ambient_dim or d_img

        # identity projection by default
        if d_emb == d_img:
            P = torch.eye(d_img, dtype=torch.float32)
            data_emb = flat
        else:
            A, _ = torch.linalg.qr(torch.randn(d_emb, d_img))
            P = A.float()
            data_emb = (A @ flat.T).T

        self.data   = data_emb.view(N, self.channels, self.H, self.W).contiguous()
        self.angles = thetas.float()
        self.Rs     = Rs
        self.P      = P

        # quick preview
        with suppress(Exception):
            prev_path = (os.path.splitext(self.dataset_path)[0] if self.dataset_path else "rot_mnist") + "_preview.png"
            pathlib.Path(prev_path).parent.mkdir(parents=True, exist_ok=True)
            save_image(self.data[:25], prev_path, nrow=5, normalize=True)
            print("[RotatedMNIST] Preview saved →", prev_path)

# ──────────────────────────────── CLI ───────────────────────────────── #

def _cli():
    P = argparse.ArgumentParser("Build Rotated-MNIST dataset")
    P.add_argument("--dataset_path", required=True)
    P.add_argument("--digit", type=int, default=9)
    P.add_argument("--split", type=str, default="train", choices=["train", "test"])
    P.add_argument("--sample_index", type=int, default=None)
    P.add_argument("--image_size", type=int, default=32)   # ← 32×32
    P.add_argument("--channels", type=int, default=1)
    P.add_argument("--data_samples", type=int, default=100_000)
    P.add_argument("--ambient_dim", type=int, default=None)
    P.add_argument("--overwrite_cache", action="store_true")
    P.add_argument("--device", default="cpu")
    P.add_argument("--angle_step_deg", type=float, default=None)
    P.add_argument("--pad_to_32", action="store_true")     # ← enable padding via flag
    P.add_argument("--seed", type=int, default=42)
    args = P.parse_args()

    ds = RotatedMNIST(
        dataset_path=args.dataset_path,
        digit=args.digit,
        split=args.split,
        sample_index=args.sample_index,
        image_size=args.image_size,
        channels=args.channels,
        data_samples=args.data_samples,
        ambient_dim=args.ambient_dim,
        overwrite_cache=args.overwrite_cache,
        device=args.device,
        angle_step_deg=args.angle_step_deg,
        pad_to_32=args.pad_to_32,
        seed=args.seed,
    )
    print(f"Dataset built: {len(ds)} samples → {args.dataset_path}")

if __name__ == "__main__":
    _cli()
