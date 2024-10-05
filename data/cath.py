import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .utils import flatten_structure, unflatten_structure

class CATHPreprocessedDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Ensure that each tensor in the item is of type float32
        if isinstance(item, torch.Tensor):
            return item.float()
        elif isinstance(item, dict):
            # If the item is a dictionary, convert all tensors in the dict
            return {key: value.float() if isinstance(value, torch.Tensor) else value for key, value in item.items()}
        elif isinstance(item, list):
            # If the item is a list, convert all tensors in the list
            return [element.float() if isinstance(element, torch.Tensor) else element for element in item]
        else:
            return item

    
class CATHOriginalDataset(Dataset):
    ATOM_TYPES = [
        'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        'CZ3', 'NZ', 'OXT'
    ]
    BACKBONE_IDX = [0, 1, 2, 4]  # Indices for 'N', 'CA', 'C', 'O'

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
        backbone_positions, backbone_mask = self.extract_backbone(np_example)
        
        # Divide the backbone positions by 100
        backbone_positions /= 100.0
        
        item = flatten_structure(torch.tensor(backbone_positions, dtype=torch.float32))
        return item
        
        #return {
        #    'backbone_positions': torch.tensor(backbone_positions, dtype=torch.float32),
        #    'backbone_mask': torch.tensor(backbone_mask, dtype=torch.float32)
        #}

    def make_np_example(self, coords_dict):
        """Create a numpy example from a dictionary of atomic coordinates."""
        num_res = len(coords_dict['N'])
        atom_positions = np.zeros([num_res, len(self.ATOM_TYPES), 3], dtype=float)

        for i, atom_type in enumerate(self.ATOM_TYPES):
            if atom_type in ['N', 'CA', 'C', 'O']:
                atom_positions[:, i, :] = np.array(coords_dict[atom_type])

        nan_pos = np.isnan(atom_positions)[..., 0]
        atom_positions[nan_pos] = 0.
        atom_mask = np.zeros([num_res, len(self.ATOM_TYPES)])
        atom_mask[:, self.BACKBONE_IDX] = 1
        atom_mask[nan_pos] = 0

        return {
            'atom_positions': atom_positions,
            'atom_mask': atom_mask,
            'residue_index': np.arange(num_res)
        }

    def make_fixed_size(self, np_example):
        """Ensure fixed size by padding or truncating."""
        for k, v in np_example.items():
            pad = self.max_seq_length - v.shape[0]
            if pad > 0:
                np_example[k] = np.pad(v, ((0, pad),) + ((0, 0),) * (len(v.shape) - 1))
            elif pad < 0:
                np_example[k] = v[:self.max_seq_length]

    def center_positions(self, np_example):
        """Center atom positions around CA atoms."""
        atom_positions = np_example['atom_positions']
        atom_mask = np_example['atom_mask']
        ca_positions = atom_positions[:, 1, :]  # CA positions
        ca_mask = atom_mask[:, 1]

        ca_center = (np.sum(ca_mask[..., None] * ca_positions, axis=0) /
                     (np.sum(ca_mask, axis=0) + 1e-9))
        atom_positions -= ca_center
        np_example['atom_positions'] = atom_positions

    def extract_backbone(self, np_example):
        atom_positions = np_example['atom_positions']
        atom_mask = np_example['atom_mask']

        # Extract backbone atoms
        backbone_positions = atom_positions[:, self.BACKBONE_IDX, :]  # Shape: (seq_len, 4, 3)
        backbone_mask = atom_mask[:, self.BACKBONE_IDX]  # Shape: (seq_len, 4)

        return backbone_positions, backbone_mask
