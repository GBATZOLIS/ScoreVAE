import os
import dataclasses
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import py3Dmol
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global constants
ATOM_TYPES = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

RESTYPES = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
RESTYPE_ORDER = {restype: i for i, restype in enumerate(RESTYPES)}
RESTYPE_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 
    'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 
    'Y': 'TYR', 'V': 'VAL'
}

# Backbone atom indices for 'N', 'CA', 'C', 'O'
BACKBONE_IDX = [0, 1, 2, 4]

# Data processing helper functions
def make_np_example(coords_dict):
    """Create a numpy example from a dictionary of atomic coordinates."""
    bb_atom_types = ['N', 'CA', 'C', 'O']
    bb_idx = [i for i, atom_type in enumerate(ATOM_TYPES) if atom_type in bb_atom_types]

    num_res = np.array(coords_dict['N']).shape[0]
    atom_positions = np.zeros([num_res, len(ATOM_TYPES), 3], dtype=float)

    for i, atom_type in enumerate(ATOM_TYPES):
        if atom_type in bb_atom_types:
            atom_positions[:, i, :] = np.array(coords_dict[atom_type])

    nan_pos = np.isnan(atom_positions)[..., 0]
    atom_positions[nan_pos] = 0.
    atom_mask = np.zeros([num_res, len(ATOM_TYPES)])
    atom_mask[..., bb_idx] = 1
    atom_mask[nan_pos] = 0

    return {
        'atom_positions': atom_positions,
        'atom_mask': atom_mask,
        'residue_index': np.arange(num_res)
    }

def make_fixed_size(np_example, max_seq_length=500):
    """Ensure that the example has a fixed size by padding or truncating."""
    for k, v in np_example.items():
        pad = max_seq_length - v.shape[0]
        if pad > 0:
            np_example[k] = np.pad(v, ((0, pad),) + ((0, 0),) * (len(v.shape) - 1))
        elif pad < 0:
            np_example[k] = v[:max_seq_length]

def center_positions(np_example):
    """Center the positions of atoms around the center of CA atoms."""
    atom_positions = np_example['atom_positions']
    atom_mask = np_example['atom_mask']
    ca_positions = atom_positions[:, 1, :]  # CA positions
    ca_mask = atom_mask[:, 1]

    ca_center = (np.sum(ca_mask[..., None] * ca_positions, axis=0) /
                 (np.sum(ca_mask, axis=0) + 1e-9))
    atom_positions = ((atom_positions - ca_center[None, ...]) * atom_mask[..., None])
    np_example['atom_positions'] = atom_positions

# Dataset class
class DatasetFromDataframe(torch.utils.data.Dataset):
    def __init__(self, data_frame, max_seq_length=512):
        self.data = data_frame
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords_dict = self.data.iloc[idx].coords
        np_example = make_np_example(coords_dict)
        make_fixed_size(np_example, self.max_seq_length)
        center_positions(np_example)
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in np_example.items()}

# Protein data structure
@dataclasses.dataclass(frozen=True)
class Protein:
    atom_positions: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    chain_index: np.ndarray
    b_factors: np.ndarray

# Visualization
def plot_protein(np_batch):
    atom_positions = np_batch['atom_positions'][0]  # First protein in batch
    atom_mask = np_batch['atom_mask'][0]

    backbone_positions = atom_positions[:, BACKBONE_IDX, :]  # (seq_len, 4, 3)
    backbone_mask = atom_mask[:, BACKBONE_IDX]  # (seq_len, 4)

    # Filter non-zero atoms
    non_zero_mask = np.any(backbone_mask > 0, axis=1)

    backbone_positions = backbone_positions[non_zero_mask]
    backbone_mask = backbone_mask[non_zero_mask]

    # Compute mean positions (center of mass) for each residue
    residue_means = np.sum(backbone_positions * backbone_mask[..., None], axis=1) / \
                    (np.sum(backbone_mask, axis=1, keepdims=True) + 1e-9)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot backbone atom positions (blue points)
    for i in range(backbone_positions.shape[0]):
        for j in range(backbone_positions.shape[1]):
            if backbone_mask[i, j] > 0:
                ax.scatter(backbone_positions[i, j, 0], backbone_positions[i, j, 1], backbone_positions[i, j, 2], c='b')

    # Plot residue mean positions (red points)
    ax.scatter(residue_means[:, 0], residue_means[:, 1], residue_means[:, 2], c='r')

    # Draw lines between residue means (red lines)
    ax.plot(residue_means[:, 0], residue_means[:, 1], residue_means[:, 2], 'r-')

    # Draw green lines connecting atom positions to their mean
    for i in range(backbone_positions.shape[0]):
        for j in range(backbone_positions.shape[1]):
            if backbone_mask[i, j] > 0:
                ax.plot([backbone_positions[i, j, 0], residue_means[i, 0]],
                        [backbone_positions[i, j, 1], residue_means[i, 1]],
                        [backbone_positions[i, j, 2], residue_means[i, 2]], 'g-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Configuration class
class Config:
    def __init__(self):
        self.max_seq_length = 256
        self.batch_size = 64
        
# Main function for testing
def main():
    print('Loading dataset...')
    df = pd.read_json('datasets/cath/chain_set.jsonl', lines=True)
    cath_splits = pd.read_json('datasets/cath/chain_set_splits.json', lines=True)

    def get_split(pdb_name):
        if pdb_name in cath_splits.train[0]:
            return 'train'
        elif pdb_name in cath_splits.validation[0]:
            return 'validation'
        elif pdb_name in cath_splits.test[0]:
            return 'test'
        return 'None'

    df['split'] = df.name.apply(get_split)
    df['seq_len'] = df.seq.apply(len)

    cfg = Config()
    train_set = DatasetFromDataframe(df[df.split == 'train'], max_seq_length=cfg.max_seq_length)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size)
    train_iter = iter(train_loader)

    batch = next(train_iter)
    np_batch = {k: v.detach().numpy() for k, v in batch.items()}

    plot_protein(np_batch)  # Call the new plotting function

if __name__ == "__main__":
    main()
