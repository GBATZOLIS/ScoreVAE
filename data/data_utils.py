import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from .sphere import KSphereDataset
from .earth import EarthDataset

def get_dataloaders(args, seed=42):
    dataset_name = args.dataset
    batch_size = args.batch_size
    generator = torch.Generator().manual_seed(seed)

    if dataset_name == 'sphere':
        dataset = KSphereDataset(args)
        
        train_size = int(0.9 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train_dataset = datasets.MNIST(
            root='./datasets', train=True, transform=transform, download=True
        )
        test_dataset = datasets.MNIST(
            root='./datasets', train=False, transform=transform, download=True
        )

        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )

    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train_dataset = datasets.CIFAR10(
            root='./datasets', train=True, transform=transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root='./datasets', train=False, transform=transform, download=True
        )

        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )

    elif dataset_name == 'earth':
        dataset = EarthDataset(args)
        
        train_size = int(0.9 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
