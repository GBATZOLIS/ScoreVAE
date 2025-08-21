"""
High-throughput DataLoader helper.
Only `RenderedSO3Dataset`/`RenderedTeapots` are heavy enough to need all tweaks.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .sphere import KSphereDataset
from .earth  import EarthDataset
from .so     import SOdataset

# Legacy dataset (kept for backwards compat)
from .rendered_so_dataset import RenderedSO3Dataset

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

# ─────────────────────────────────────────────────────────────────────────
def _split(dataset, train_frac=0.9, seed=42):
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    t = int(train_frac * n)
    v = n - t
    return random_split(dataset, [t, v], generator=g)


def get_dataloaders(args, seed: int = 42):
    name, bs = args.dataset, args.batch_size

    # ───────────────── rendered datasets (old + new) ─────────────────
    if name in {"rendered_so_dataset", "rendered_teapots"}:
        dataset_cls = RenderedSO3Dataset
        if name == "rendered_teapots" and RenderedTeapots is not None:
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
        loader_kwargs = dict(batch_size=bs, num_workers=workers, pin_memory=pin_mem)
        if workers > 0:
            loader_kwargs.update(dict(persistent_workers=False, prefetch_factor=2, multiprocessing_context="spawn"))

        return (
            DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs),
            DataLoader(val_ds,   shuffle=False, **loader_kwargs),
            DataLoader(test_ds,  shuffle=False, **loader_kwargs),
        )

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
        loader_kwargs = dict(batch_size=bs, num_workers=workers, pin_memory=pin_mem)
        if workers > 0:
            loader_kwargs.update(dict(persistent_workers=False, prefetch_factor=2, multiprocessing_context="spawn"))

        return (
            DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs),
            DataLoader(val_ds,   shuffle=False, **loader_kwargs),
            DataLoader(test_ds,  shuffle=False, **loader_kwargs),
        )

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

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds,   batch_size=bs),
        DataLoader(test_ds,  batch_size=bs),
    )
