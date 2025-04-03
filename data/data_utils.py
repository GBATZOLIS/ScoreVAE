from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from .sphere import KSphereDataset
import os
from PIL import Image
import glob

class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as label since FFHQ doesn't have labels

def get_dataloaders(args):
    dataset_name = args.dataset
    batch_size = args.batch_size
    
    if dataset_name == 'sphere':
        dataset = KSphereDataset(args)
        
        # Determine sizes for train, val, and test sets
        train_size = int(0.9 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        # Split dataset into train, val, and test sets
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train_dataset = datasets.MNIST(root='~/datasets', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='~/datasets', train=False, transform=transform, download=True)

        # Split train dataset into train and val sets (90/10 split)
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train_dataset = datasets.CIFAR10(root='~/datasets', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='~/datasets', train=False, transform=transform, download=True)

        # Split train dataset into train and val sets (90/10 split)
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
    elif dataset_name == 'FFHQ':
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load FFHQ dataset from local directory
        home_dir = os.path.expanduser('~')
        dataset_path = os.path.join(home_dir, 'datasets', 'ffhq')
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"FFHQ dataset not found at {dataset_path}. Please download it first.")
            
        full_dataset = FFHQDataset(dataset_path, transform=transform)
        
        # Split into train, val, and test sets according to args.split
        total_size = len(full_dataset)
        train_size = int(args.split[0] * total_size)
        val_size = int(args.split[1] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
