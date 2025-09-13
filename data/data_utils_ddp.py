"""
High-throughput DataLoader helper.
DDP-ready via DistributedSampler, but BACKWARD-COMPATIBLE:
- By default returns (train_loader, val_loader, test_loader) like before.
- If return_samplers=True, returns (train_loader, val_loader, test_loader, samplers_dict).
Each loader also gets an attribute ._dist_sampler with its sampler (or None).
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from typing import Optional, Tuple, Dict, Union

from .sphere import KSphereDataset
from .earth  import EarthDataset
from .so     import SOdataset

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

# NEW: Rotated MNIST (pad→rotate SO(2))
try:
    from .rotated_mnist_dataset import RotatedMNIST
except Exception:
    RotatedMNIST = None


def _split(dataset, train_frac=0.9, seed=42):
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    t = int(train_frac * n)
    v = n - t
    return random_split(dataset, [t, v], generator=g)


def _make_loader(dataset,
                 *,
                 shuffle: bool,
                 drop_last: bool,
                 batch_size: int,
                 n_workers: int,
                 pin_mem: bool,
                 sampler=None):
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
    # Attach for convenience
    setattr(loader, "_dist_sampler", sampler)
    return loader


def _maybe_sampler(dataset,
                   *,
                   shuffle: bool,
                   drop_last: bool,
                   batch_size: int,
                   n_workers: int,
                   distributed: bool,
                   rank: int,
                   world_size: int,
                   pin_mem: bool):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, drop_last=drop_last)
    loader = _make_loader(dataset,
                          shuffle=shuffle, drop_last=drop_last, batch_size=batch_size,
                          n_workers=n_workers, pin_mem=pin_mem, sampler=sampler)
    return loader, sampler


def get_dataloaders(
    args,
    seed: int = 42,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    return_samplers: bool = False,  # NEW: default False keeps old 3-return behavior
) -> Union[
    Tuple[DataLoader, DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Optional[DistributedSampler]]]
]:
    name, bs = args.dataset, args.batch_size

    samplers: Dict[str, Optional[DistributedSampler]] = {"train": None, "val": None, "test": None}

    # ───────────────── rendered datasets (old + new) ─────────────────
    if name in {"rendered_so_dataset", "rendered_teapots"}:
        if name == "rendered_so_dataset":
            if RenderedSO3Dataset is None:
                raise ImportError("RenderedSO3Dataset requires pytorch3d, which is not installed.")
            dataset_cls = RenderedSO3Dataset
        elif name == "rendered_teapots":
            if RenderedTeapots is None:
                raise ImportError("RenderedTeapots dataset not available.")
            dataset_cls = RenderedTeapots
        dataset = dataset_cls(args, seed=seed)

        g = torch.Generator().manual_seed(seed)
        n = len(dataset)
        train_len = int(0.9 * n)
        val_len   = int(0.05 * n)
        test_len  = n - train_len - val_len
        train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=g)

        workers = getattr(args, "n_workers", 4)
        pin_mem = torch.cuda.is_available()

        train_loader, train_sampler = _maybe_sampler(
            train_ds, shuffle=True, drop_last=True, batch_size=bs, n_workers=workers,
            distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
        )
        val_loader, val_sampler = _maybe_sampler(
            val_ds, shuffle=False, drop_last=False, batch_size=bs, n_workers=workers,
            distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
        )
        test_loader, test_sampler = _maybe_sampler(
            test_ds, shuffle=False, drop_last=False, batch_size=bs, n_workers=workers,
            distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
        )

        samplers.update({"train": train_sampler, "val": val_sampler, "test": test_sampler})
        return (train_loader, val_loader, test_loader, samplers) if return_samplers \
            else (train_loader, val_loader, test_loader)

    # ───────────────── Rotated MNIST (SO(2) pad→rotate) ─────────────────
    elif name in {"rotated_mnist_dataset", "rotated_mnist"}:
        if RotatedMNIST is None:
            raise ImportError("datasets/rotated_mnist_dataset.py not found or failed to import.")
        ds = RotatedMNIST(
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

        g = torch.Generator().manual_seed(seed)
        n = len(ds)
        train_len = int(0.9 * n)
        val_len   = int(0.05 * n)
        test_len  = n - train_len - val_len
        train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, test_len], generator=g)

        workers = getattr(args, "n_workers", 8)
        pin_mem = torch.cuda.is_available()

        train_loader, train_sampler = _maybe_sampler(
            train_ds, shuffle=True, drop_last=True, batch_size=bs, n_workers=workers,
            distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
        )
        val_loader, val_sampler = _maybe_sampler(
            val_ds, shuffle=False, drop_last=False, batch_size=bs, n_workers=workers,
            distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
        )
        test_loader, test_sampler = _maybe_sampler(
            test_ds, shuffle=False, drop_last=False, batch_size=bs, n_workers=workers,
            distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
        )

        samplers.update({"train": train_sampler, "val": val_sampler, "test": test_sampler})
        return (train_loader, val_loader, test_loader, samplers) if return_samplers \
            else (train_loader, val_loader, test_loader)

    # ───────────────────── lightweight branches (unchanged) ───────────────────
    elif name == "sphere":
        train_ds, val_ds, test_ds = random_split(
            KSphereDataset(args, seed=seed),
            [int(0.9*len(args)), int(0.05*len(args)), int(0.05*len(args))],
            generator=torch.Generator().manual_seed(seed),
        )
    elif name == "earth":
        train_ds, val_ds, test_ds = random_split(
            EarthDataset(args),
            [int(0.9*len(args)), int(0.05*len(args)), int(0.05*len(args))],
            generator=torch.Generator().manual_seed(seed),
        )
    elif name == "so":
        full = SOdataset(args, seed=seed)
        g = torch.Generator().manual_seed(seed)
        n = len(full)
        train_len = int(0.9 * n)
        val_len   = int(0.05 * n)
        test_len  = n - train_len - val_len
        train_ds, val_ds, test_ds = random_split(full, [train_len, val_len, test_len], generator=g)
    elif name == "MNIST":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full    = datasets.MNIST("./datasets", train=True,  transform=tfm, download=True)
        test_ds = datasets.MNIST("./datasets", train=False, transform=tfm, download=True)
        train_ds, val_ds = _split(full, seed=seed)
    elif name == "CIFAR10":
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        full    = datasets.CIFAR10("./datasets", train=True,  transform=tfm, download=True)
        test_ds = datasets.CIFAR10("./datasets", train=False, transform=tfm, download=True)
        train_ds, val_ds = _split(full, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    workers = getattr(args, "n_workers", 4)
    pin_mem = torch.cuda.is_available()

    train_loader, train_sampler = _maybe_sampler(
        train_ds, shuffle=True, drop_last=True, batch_size=bs, n_workers=workers,
        distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
    )
    val_loader, val_sampler = _maybe_sampler(
        val_ds, shuffle=False, drop_last=False, batch_size=bs, n_workers=workers,
        distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
    )
    test_loader, test_sampler = _maybe_sampler(
        test_ds, shuffle=False, drop_last=False, batch_size=bs, n_workers=workers,
        distributed=distributed, rank=rank, world_size=world_size, pin_mem=pin_mem
    )

    samplers.update({"train": train_sampler, "val": val_sampler, "test": test_sampler})
    return (train_loader, val_loader, test_loader, samplers) if return_samplers \
        else (train_loader, val_loader, test_loader)

