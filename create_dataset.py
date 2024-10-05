import os
import torch
from torch.utils.data import Dataset, DataLoader
from data import get_dataloaders
from tqdm import tqdm  # For progress tracking

def save_preprocessed_data(train_loader, val_loader, test_loader, save_train_percentage=1, folder_path='datasets/cath/processed'):
    """Save preprocessed datasets into .pt files for fast loading and report sizes."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    def save_loader(loader, filename):
        processed_data = []
        for batch in tqdm(loader, desc=f"Processing {filename}"):
            processed_data.append(batch)  # Assuming the data is in the first element of each batch

        concatenated_data = torch.cat(processed_data, dim=0)  # Concatenate along the batch dimension (dim=0)
        print(f"Size of {filename}: {concatenated_data.size()}")  # Report size
        torch.save(concatenated_data, os.path.join(folder_path, filename))  # Save tensor
        return concatenated_data.size(0)  # Return the number of samples

    # Save only 20% of the train_loader data for debugging
    reduced_train_data = []
    print("Processing train.pt (20% of training data)")
    for i, batch in enumerate(tqdm(train_loader, desc="Processing train.pt")):
        if i < int(len(train_loader) * save_train_percentage):
            reduced_train_data.append(batch)  # Assuming data is in batch

    concatenated_train_data = torch.cat(reduced_train_data, dim=0)
    print(f"Size of train.pt: {concatenated_train_data.size()}")  # Report size
    torch.save(concatenated_train_data, os.path.join(folder_path, 'train.pt'))

    # Save and report the sizes of validation and test sets
    val_size = save_loader(val_loader, 'validation.pt')
    print(f"Saved {val_size} samples to validation.pt")

    test_size = save_loader(test_loader, 'test.pt')
    print(f"Saved {test_size} samples to test.pt")

    print(f'Processed data saved in {folder_path}/')

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
