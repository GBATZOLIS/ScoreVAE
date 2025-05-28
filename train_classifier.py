import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pickle

from models import get_model
from configs import load_config

class LatentMultiAttributeClassifier(nn.Module):
    def __init__(self, latent_dim, num_attrs):
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_attrs)

    def forward(self, z):
        return self.linear(z)  # shape: (batch, num_attrs)

def encode_dataset(encoder, dataloader, device):
    encoder.eval()
    all_z, all_y = [], []

    with torch.no_grad():
        for images, attributes in tqdm(dataloader, desc="Encoding dataset"):
            images = images.to(device)
            z = encoder(images)
            all_z.append(z.cpu())
            all_y.append((attributes > 0).float())  # CelebA: -1/1 → 0/1

    return torch.cat(all_z), torch.cat(all_y)

def train_classifier(config):
    writer = SummaryWriter(log_dir=os.path.join(config.base_log_dir, config.experiment, "classifier_logs"))
    device = torch.device(config.training.device)

    # Load encoder
    encoder = get_model(config.model)
    encoder.load_state_dict(torch.load(config.model.pretrained_path))
    encoder.to(device)
    encoder.eval()

    # Load CelebA
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = CelebA(root=config.data.data_dir, split='train', transform=transform, target_type='attr', download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Encode images into latent vectors
    latent_vectors, attributes = encode_dataset(encoder, dataloader, device)

    latent_dim = latent_vectors.shape[1]
    num_attrs = attributes.shape[1]

    # Define classifier
    classifier = LatentMultiAttributeClassifier(latent_dim, num_attrs).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.training.epochs):
        classifier.train()
        optimizer.zero_grad()

        logits = classifier(latent_vectors.to(device))  # (N, 40)
        loss = criterion(logits, attributes.to(device))  # broadcasted per attribute
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/Train', loss.item(), epoch)
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save the full classifier
    os.makedirs(config.model.classifier_dir, exist_ok=True)
    torch.save(classifier.state_dict(), os.path.join(config.model.classifier_dir, 'joint_classifier.pt'))
    print("Joint classifier saved.")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train joint latent space classifier for attributes.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train_classifier(config)
