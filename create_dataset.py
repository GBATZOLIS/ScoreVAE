import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# Backbone atom indices for 'N', 'CA', 'C', 'O'
ATOM_TYPES = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
BACKBONE_IDX = [0, 1, 2, 4]  # Indices for 'N', 'CA', 'C', 'O'

class CATHOriginalDataset(Dataset):
    def __init__(self, df, max_seq_length=512):
        self.data = df
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords_dict = self.data.iloc[idx].coords
        np_example = self.make_np_example(coords_dict)
        self.make_fixed_size(np_example)
        self.center_positions(np_example)
        backbone_positions, _ = self.extract_backbone(np_example)
        return torch.tensor(backbone_positions, dtype=torch.float32)

    def make_np_example(self, coords_dict):
        """Create a numpy example from a dictionary of atomic coordinates."""
        num_res = len(coords_dict['N'])
        atom_positions = np.zeros([num_res, len(ATOM_TYPES), 3], dtype=float)

        for i, atom_type in enumerate(ATOM_TYPES):
            if atom_type in ['N', 'CA', 'C', 'O']:
                atom_positions[:, i, :] = np.array(coords_dict[atom_type])

        nan_pos = np.isnan(atom_positions)[..., 0]
        atom_positions[nan_pos] = 0.

        return {
            'atom_positions': atom_positions,
            'atom_mask': np.ones([num_res, len(ATOM_TYPES)])  # Dummy mask
        }

    def make_fixed_size(self, np_example):
        """Ensure fixed size by padding or truncating."""
        pad = self.max_seq_length - np_example['atom_positions'].shape[0]
        if pad > 0:
            np_example['atom_positions'] = np.pad(np_example['atom_positions'], ((0, pad), (0, 0), (0, 0)))
        elif pad < 0:
            np_example['atom_positions'] = np_example['atom_positions'][:self.max_seq_length]

    def center_positions(self, np_example):
        """Center atom positions around CA atoms."""
        atom_positions = np_example['atom_positions']
        ca_positions = atom_positions[:, 1, :]  # CA positions
        ca_center = np.mean(ca_positions, axis=0)
        np_example['atom_positions'] -= ca_center

    def extract_backbone(self, np_example):
        atom_positions = np_example['atom_positions']
        backbone_positions = atom_positions[:, BACKBONE_IDX, :]  # Shape: (seq_len, 4, 3)
        return backbone_positions, None  # Returning None for the mask, as it's not needed here


class CATHPreprocessedDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def save_preprocessed_data(train_loader, val_loader, test_loader, save_train_percentage=0.2, folder_path='processed'):
    """Save preprocessed datasets into .pt files for fast loading."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    def save_loader(loader, filename):
        processed_data = []
        for batch in loader:
            processed_data.append(batch)
        torch.save(processed_data, os.path.join(folder_path, filename))

    # Save only 20% of the train_loader data for debugging
    reduced_train_data = []
    for i, batch in enumerate(train_loader):
        if i < int(len(train_loader) * save_train_percentage):
            reduced_train_data.append(batch)
    torch.save(reduced_train_data, os.path.join(folder_path, 'train.pt'))

    save_loader(val_loader, 'validation.pt')
    save_loader(test_loader, 'test.pt')

    print(f'Processed data saved in {folder_path}/')


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


def preprocess_and_save_datasets(args):
    """Preprocess and save datasets as .pt files."""
    train_loader, val_loader, test_loader = get_dataloaders(args)
    save_preprocessed_data(train_loader, val_loader, test_loader)


# Example Config for Testing
class Config:
    def __init__(self):
        self.dataset = 'cath-original'
        self.batch_size = 64
        self.max_seq_length = 256

# Main function to preprocess and save
if __name__ == "__main__":
    args = Config()
    preprocess_and_save_datasets(args)
