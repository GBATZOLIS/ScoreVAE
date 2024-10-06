from torch.utils.data import DataLoader, random_split, DistributedSampler
from .cath import CATHOriginalDataset, CATHPreprocessedDataset
import pandas as pd
import torch 

def get_dataloaders_ddp(args, distributed=False):
    dataset_name = args.dataset
    batch_size = args.batch_size
    seed = 42  # Use a fixed seed for reproducibility

    if dataset_name == 'cath-original':
        df = pd.read_json('datasets/cath/chain_set.jsonl', lines=True)
        cath_splits = pd.read_json('datasets/cath/chain_set_splits.json', lines=True)
        
        def get_split(pdb_name):
            if pdb_name in cath_splits.train[0]:
                return 'train'
            elif pdb_name in cath_splits.test[0]:
                return 'test'
            return 'None'

        df['split'] = df.name.apply(get_split)
        df['seq_len'] = df.seq.apply(len)

        # Use only the train split for further train/validation split
        train_val_df = df[df.split == 'train']
        
        dataset = CATHOriginalDataset(train_val_df, max_seq_length=args.max_seq_length)
        
        # Split into train and validation sets (90/10 split)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
        
        # Load test dataset separately
        test_df = df[df.split == 'test']
        test_dataset = CATHOriginalDataset(test_df, max_seq_length=args.max_seq_length)

    elif dataset_name == 'cath-preprocessed':
        train_dataset = CATHPreprocessedDataset('datasets/cath/processed/train.pt')
        val_dataset = CATHPreprocessedDataset('datasets/cath/processed/validation.pt')
        test_dataset = CATHPreprocessedDataset('datasets/cath/processed/test.pt')

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # If distributed, we need to use DistributedSampler
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler, val_sampler = None, None

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_dataloaders(args):
    dataset_name = args.dataset
    batch_size = args.batch_size
    seed = 42  # Use a fixed seed for reproducibility
    
    if dataset_name == 'cath-original':
        df = pd.read_json('datasets/cath/chain_set.jsonl', lines=True)
        cath_splits = pd.read_json('datasets/cath/chain_set_splits.json', lines=True)
        
        def get_split(pdb_name):
            if pdb_name in cath_splits.train[0]:
                return 'train'
            elif pdb_name in cath_splits.test[0]:
                return 'test'
            return 'None'

        df['split'] = df.name.apply(get_split)
        df['seq_len'] = df.seq.apply(len)

        # Use only the train split for further train/validation split
        train_val_df = df[df.split == 'train']
        
        dataset = CATHOriginalDataset(train_val_df, max_seq_length=args.max_seq_length)
        
        # Split into train and validation sets (90/10 split)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
        
        # Load test dataset separately
        test_df = df[df.split == 'test']
        test_dataset = CATHOriginalDataset(test_df, max_seq_length=args.max_seq_length)

    elif dataset_name == 'cath-preprocessed':
        train_dataset = CATHPreprocessedDataset('datasets/cath/processed/train.pt')
        val_dataset = CATHPreprocessedDataset('datasets/cath/processed/validation.pt')
        test_dataset = CATHPreprocessedDataset('datasets/cath/processed/test.pt')

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
