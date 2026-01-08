# data/data_utils_ddp.py
"""
High-throughput DataLoader helper.

DDP-ready via DistributedSampler, but BACKWARD-COMPATIBLE:
- By default returns (train_loader, val_loader, test_loader) like before.
- If return_samplers=True, returns (train_loader, val_loader, test_loader, samplers_dict).
- If return_latent_loader=True, returns (train_loader, train_loader_lat, val_loader, test_loader, ...)
Each loader also gets an attribute ._dist_sampler with its sampler (or None).
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Union

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from .sphere import KSphereDataset
from .earth import EarthDataset
from .so import SOdataset

# Legacy dataset (optional import – requires pytorch3d)
try:
    from .rendered_so_dataset import RenderedSO3Dataset
except Exception:
    RenderedSO3Dataset = None

# New unified teapot renderer dataset (optional import)
try:
    from .rendered_teapots import RenderedTeapots
except Exception:
    RenderedTeapots = None  # still works if the new file isn't present

# NEW: Analytic S^2 / T^2 (no rendering)
try:
    from .analytic_manifold_dataset import AnalyticManifoldDataset
except Exception:
    AnalyticManifoldDataset = None

# NEW: Rotated MNIST (pad→rotate SO(2))
try:
    from .rotated_mnist_dataset import RotatedMNIST
except Exception:
    RotatedMNIST = None


def _split_2way(dataset, train_frac: float = 0.9, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    t = int(train_frac * n)
    v = n - t
    return random_split(dataset, [t, v], generator=g)


def _split_3way(dataset, train_frac: float = 0.9, val_frac: float = 0.05, seed: int = 42):
    """Train/val/test with (train_frac, val_frac, 1-train_frac-val_frac)."""
    assert 0.0 < train_frac < 1.0
    assert 0.0 <= val_frac < 1.0
    assert train_frac + val_frac < 1.0
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    train_len = int(train_frac * n)
    val_len = int(val_frac * n)
    test_len = n - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len], generator=g)


def _make_loader(
    dataset,
    *,
    shuffle: bool,
    drop_last: bool,
    batch_size: int,
    n_workers: int,
    pin_mem: bool,
    sampler=None,
):
    """Build a DataLoader, only setting worker-related kwargs when n_workers>0."""
    kwargs = dict(
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )
    if sampler is not None:
        kwargs["shuffle"] = False
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle

    if n_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
        kwargs["multiprocessing_context"] = "spawn"

    loader = DataLoader(dataset, **kwargs)
    setattr(loader, "_dist_sampler", sampler)
    return loader


def _maybe_sampler(
    dataset,
    *,
    shuffle: bool,
    drop_last: bool,
    batch_size: int,
    n_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    pin_mem: bool,
):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    loader = _make_loader(
        dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        batch_size=batch_size,
        n_workers=n_workers,
        pin_mem=pin_mem,
        sampler=sampler,
    )
    return loader, sampler


def _make_latent_train_loader(
    train_ds,
    *,
    batch_size: int,
    n_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    pin_mem: bool,
):
    """
    Independent train stream for latent diffusion updates.
    Uses a separate DistributedSampler so successive k-steps see fresh batches.
    """
    lat_sampler = None
    if distributed:
        lat_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    lat_loader = _make_loader(
        train_ds,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size,
        n_workers=n_workers,
        pin_mem=pin_mem,
        sampler=lat_sampler,
    )
    return lat_loader, lat_sampler


def _build_all_loaders(
    *,
    train_ds,
    val_ds,
    test_ds,
    bs: int,
    workers: int,
    pin_mem: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    return_latent_loader: bool,
    latent_bs: Optional[int],
):
    train_loader, train_sampler = _maybe_sampler(
        train_ds,
        shuffle=True,
        drop_last=True,
        batch_size=bs,
        n_workers=workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        pin_mem=pin_mem,
    )
    val_loader, val_sampler = _maybe_sampler(
        val_ds,
        shuffle=False,
        drop_last=False,
        batch_size=bs,
        n_workers=workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        pin_mem=pin_mem,
    )
    test_loader, test_sampler = _maybe_sampler(
        test_ds,
        shuffle=False,
        drop_last=False,
        batch_size=bs,
        n_workers=workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        pin_mem=pin_mem,
    )

    train_lat_loader = None
    lat_sampler = None
    if return_latent_loader:
        lat_bs_eff = int(latent_bs) if latent_bs is not None else bs
        train_lat_loader, lat_sampler = _make_latent_train_loader(
            train_ds,
            batch_size=lat_bs_eff,
            n_workers=workers,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            pin_mem=pin_mem,
        )

    samplers: Dict[str, Optional[DistributedSampler]] = {
        "train": train_sampler,
        "val": val_sampler,
        "test": test_sampler,
        "latent": lat_sampler,
    }
    return train_loader, train_lat_loader, val_loader, test_loader, samplers


def get_dataloaders(
    args,
    seed: int = 42,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    return_samplers: bool = False,
    return_latent_loader: bool = False,
) -> Union[
    Tuple[DataLoader, DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Optional[DistributedSampler]]],
    Tuple[DataLoader, DataLoader, DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader, DataLoader, DataLoader, Dict[str, Optional[DistributedSampler]]],
]:
    """
    Returns:
      - default: (train_loader, val_loader, test_loader)
      - if return_samplers: (..., samplers_dict)
      - if return_latent_loader: (train_loader, train_loader_lat, val_loader, test_loader)
      - if both: (train_loader, train_loader_lat, val_loader, test_loader, samplers_dict)
    """
    name = args.dataset
    bs = int(args.batch_size)

    workers = int(getattr(args, "n_workers", 4))
    pin_mem = bool(getattr(args, "pin_memory", torch.cuda.is_available()))
    latent_bs = getattr(args, "latent_batch_size", None)
    latent_bs = int(latent_bs) if latent_bs is not None else None

    # ───────────────── rendered datasets (old + new) ─────────────────
    if name in {"rendered_so_dataset", "rendered_teapots"}:
        if name == "rendered_so_dataset":
            if RenderedSO3Dataset is None:
                raise ImportError("RenderedSO3Dataset requires pytorch3d, which is not installed.")
            dataset_cls = RenderedSO3Dataset
        else:
            if RenderedTeapots is None:
                raise ImportError("RenderedTeapots dataset not available.")
            dataset_cls = RenderedTeapots

        full = dataset_cls(args, seed=seed)
        train_ds, val_ds, test_ds = _split_3way(full, train_frac=0.9, val_frac=0.05, seed=seed)

        train_loader, train_loader_lat, val_loader, test_loader, samplers = _build_all_loaders(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            bs=bs,
            workers=workers,
            pin_mem=pin_mem,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            return_latent_loader=return_latent_loader,
            latent_bs=latent_bs,
        )

    # ───────────────── Rotated MNIST (SO(2) pad→rotate) ─────────────────
    elif name in {"rotated_mnist_dataset", "rotated_mnist"}:
        if RotatedMNIST is None:
            raise ImportError("datasets/rotated_mnist_dataset.py not found or failed to import.")

        full = RotatedMNIST(
            dataset_path=args.dataset_path,
            digit=getattr(args, "digit", 9),
            split=getattr(args, "split", "train"),
            sample_index=getattr(args, "sample_index", None),
            image_size=args.image_size,
            channels=args.channels,
            data_samples=args.data_samples,
            ambient_dim=args.ambient_dim,
            overwrite_cache=args.overwrite_cache,
            device=args.device,
            angle_step_deg=getattr(args, "angle_step_deg", None),
            pad_to_32=getattr(args, "pad_to_32", True),
            seed=seed,
        )
        train_ds, val_ds, test_ds = _split_3way(full, train_frac=0.9, val_frac=0.05, seed=seed)

        train_loader, train_loader_lat, val_loader, test_loader, samplers = _build_all_loaders(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            bs=bs,
            workers=workers,
            pin_mem=pin_mem,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            return_latent_loader=return_latent_loader,
            latent_bs=latent_bs,
        )

    # ───────────────── Analytic S^2 / T^2 (no 3D; exact geodesics) ─────────────────
    elif name in {"analytic_manifold_dataset", "analytic_manifold"}:
        if AnalyticManifoldDataset is None:
            raise ImportError("datasets/analytic_manifold_dataset.py not found or failed to import.")

        full = AnalyticManifoldDataset(args, seed=seed)
        train_ds, val_ds, test_ds = _split_3way(full, train_frac=0.9, val_frac=0.05, seed=seed)

        train_loader, train_loader_lat, val_loader, test_loader, samplers = _build_all_loaders(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            bs=bs,
            workers=workers,
            pin_mem=pin_mem,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            return_latent_loader=return_latent_loader,
            latent_bs=latent_bs,
        )

    # ───────────────── sphere / earth / so / MNIST / CIFAR10 ─────────────────
    elif name == "sphere":
        full = KSphereDataset(args, seed=seed)
        train_ds, val_ds, test_ds = _split_3way(full, train_frac=0.9, val_frac=0.05, seed=seed)

        train_loader, train_loader_lat, val_loader, test_loader, samplers = _build_all_loaders(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            bs=bs,
            workers=workers,
            pin_mem=pin_mem,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            return_latent_loader=return_latent_loader,
            latent_bs=latent_bs,
        )

    elif name == "earth":
        full = EarthDataset(args)
        train_ds, val_ds, test_ds = _split_3way(full, train_frac=0.9, val_frac=0.05, seed=seed)

        train_loader, train_loader_lat, val_loader, test_loader, samplers = _build_all_loaders(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            bs=bs,
            workers=workers,
            pin_mem=pin_mem,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            return_latent_loader=return_latent_loader,
            latent_bs=latent_bs,
        )

    elif name == "so":
        full = SOdataset(args, seed=seed)
        train_ds, val_ds, test_ds = _split_3way(full, train_frac=0.9, val_frac=0.05, seed=seed)

        train_loader, train_loader_lat, val_loader, test_loader, samplers = _build_all_loaders(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            bs=bs,
            workers=workers,
            pin_mem=pin_mem,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            return_latent_loader=return_latent_loader,
            latent_bs=latent_bs,
        )

    elif name == "MNIST":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full_train = datasets.MNIST("./datasets", train=True, transform=tfm, download=True)
        test_ds = datasets.MNIST("./datasets", train=False, transform=tfm, download=True)
        train_ds, val_ds = _split_2way(full_train, train_frac=0.9, seed=seed)

        train_loader, train_loader_lat, val_loader, test_loader, samplers = _build_all_loaders(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            bs=bs,
            workers=workers,
            pin_mem=pin_mem,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            return_latent_loader=return_latent_loader,
            latent_bs=latent_bs,
        )

    elif name == "CIFAR10":
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        full_train = datasets.CIFAR10("./datasets", train=True, transform=tfm, download=True)
        test_ds = datasets.CIFAR10("./datasets", train=False, transform=tfm, download=True)
        train_ds, val_ds = _split_2way(full_train, train_frac=0.9, seed=seed)

        train_loader, train_loader_lat, val_loader, test_loader, samplers = _build_all_loaders(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            bs=bs,
            workers=workers,
            pin_mem=pin_mem,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            return_latent_loader=return_latent_loader,
            latent_bs=latent_bs,
        )

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # ---------------- return policy (backward compatible) ----------------
    if return_latent_loader and return_samplers:
        return train_loader, train_loader_lat, val_loader, test_loader, samplers
    if return_latent_loader and (not return_samplers):
        return train_loader, train_loader_lat, val_loader, test_loader
    if return_samplers:
        return train_loader, val_loader, test_loader, samplers
    return train_loader, val_loader, test_loader
