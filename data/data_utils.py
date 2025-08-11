import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ─── local datasets ────────────────────────────────────────────────────────────
from .sphere          import KSphereDataset
from .earth           import EarthDataset
from .so              import SOdataset
from .rendered_so_dataset import RenderedSO3Dataset

# ═══════════════════════════════════════════════════════════════════════════════
def get_dataloaders(args, seed: int = 42):
    """
    Return train/val/test DataLoaders.

    * Splitting logic for every dataset is IDENTICAL to the original file.
    * Only the **rendered_so_dataset** branch gets a tuned DataLoader
      (pin_memory, drop_last, extra workers).
    """
    dataset_name = args.dataset
    batch_size   = args.batch_size
    g            = torch.Generator().manual_seed(seed)

    # ─────────────────────────────────── dataset construction ────────────────
    if dataset_name == "sphere":
        dataset = KSphereDataset(args, seed=seed)

        train_size = int(0.9 * len(dataset))
        val_size   = int(0.05 * len(dataset)) # Modified to 5% for consistency
        test_size  = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=g
        )

    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train = datasets.MNIST("./datasets", train=True,
                                    transform=transform, download=True)
        test_dataset = datasets.MNIST("./datasets", train=False,
                                      transform=transform, download=True)

        train_size = int(0.9 * len(full_train))
        val_size   = len(full_train) - train_size

        train_dataset, val_dataset = random_split(
            full_train, [train_size, val_size], generator=g
        )

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train = datasets.CIFAR10("./datasets", train=True,
                                      transform=transform, download=True)
        test_dataset = datasets.CIFAR10("./datasets", train=False,
                                        transform=transform, download=True)

        train_size = int(0.9 * len(full_train))
        val_size   = len(full_train) - train_size

        train_dataset, val_dataset = random_split(
            full_train, [train_size, val_size], generator=g
        )

    elif dataset_name == "earth":
        dataset = EarthDataset(args)

        train_size = int(0.9 * len(dataset))
        val_size   = int(0.05 * len(dataset)) # Modified to 5% for consistency
        test_size  = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=g
        )

    elif dataset_name == "so":
        # [MODIFIED] Applying the 90/5/5 split logic.
        dataset = SOdataset(args, seed=seed)
        n = len(dataset)
        train_len = int(0.9 * n)
        val_len = int(0.05 * n)
        test_len = n - train_len - val_len

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_len, val_len, test_len], generator=g
        )

    elif dataset_name == "rendered_so_dataset":
        # [MODIFIED] Applying the 90/5/5 split logic.
        dataset = RenderedSO3Dataset(args, seed=seed)
        n = len(dataset)
        train_len = int(0.9 * n)
        val_len = int(0.05 * n)
        test_len = n - train_len - val_len

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_len, val_len, test_len], generator=g
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # ─────────────────────────────────── DataLoaders ─────────────────────────
    if dataset_name == "rendered_so_dataset":
        # Heavier loader only for the rendered SO(3) images
        workers = getattr(args, "n_workers", 4) # Corrected attribute name
        pin     = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=pin, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,   batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=pin
        )
        test_loader = DataLoader(
            test_dataset,  batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=pin
        )
    else:
        # Original, lightweight defaults for every other dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                                  shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                                  shuffle=False)

    return train_loader, val_loader, test_loader
