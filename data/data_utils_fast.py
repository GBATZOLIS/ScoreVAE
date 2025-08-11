"""
High-throughput DataLoader helper.
Only `RenderedSO3Dataset` is heavy enough to need all tweaks.
"""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .sphere           import KSphereDataset
from .earth            import EarthDataset
from .so              import SOdataset
from .rendered_so_dataset import RenderedSO3Dataset


# ─────────────────────────────────────────────────────────────────────────
def _split(dataset, train_frac=0.9, seed=42):
    """[LEGACY] Kept for other datasets that use a simple train/val split."""
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    t = int(train_frac * n)
    v = n - t
    return random_split(dataset, [t, v], generator=g)


def get_dataloaders(args, seed: int = 42):
    name, bs = args.dataset, args.batch_size

    if name == "rendered_so_dataset":
        dataset = RenderedSO3Dataset(args, seed=seed)
        
        # Create a 90/5/5 train/val/test split
        g = torch.Generator().manual_seed(seed)
        n = len(dataset)
        train_len = int(0.9 * n)
        val_len = int(0.05 * n)
        test_len = n - train_len - val_len # Ensure all samples are used
        
        train_ds, val_ds, test_ds = random_split(
            dataset, [train_len, val_len, test_len], generator=g
        )

        workers = getattr(args, "n_workers", 4)
        pin_mem = torch.cuda.is_available()

        loader_kwargs = dict(
            batch_size     = bs,
            num_workers    = workers,
            pin_memory     = pin_mem,
            persistent_workers = True,
            prefetch_factor= 4,          # 4 × batch on each worker
        )
        return (
            DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs),
            DataLoader(val_ds,   shuffle=False, **loader_kwargs),
            DataLoader(test_ds,  shuffle=False, **loader_kwargs),
        )

    # —— lightweight branches below identical to original ——
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
        # [MODIFIED] Apply the 90/5/5 split to the SOdataset as well.
        full = SOdataset(args, seed=seed)
        g = torch.Generator().manual_seed(seed)
        n = len(full)
        train_len = int(0.9 * n)
        val_len = int(0.05 * n)
        test_len = n - train_len - val_len
        train_ds, val_ds, test_ds = random_split(
            full, [train_len, val_len, test_len], generator=g
        )
    elif name == "MNIST":
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])
        full = datasets.MNIST("./datasets", train=True, transform=tfm, download=True)
        test_ds = datasets.MNIST("./datasets", train=False, transform=tfm, download=True)
        train_ds, val_ds = _split(full, seed=seed)
    elif name == "CIFAR10":
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        full = datasets.CIFAR10("./datasets", train=True, transform=tfm, download=True)
        test_ds = datasets.CIFAR10("./datasets", train=False, transform=tfm, download=True)
        train_ds, val_ds = _split(full, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds,   batch_size=bs),
        DataLoader(test_ds,  batch_size=bs),
    )
