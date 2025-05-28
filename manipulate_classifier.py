import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import argparse

from models import get_model
from configs import load_config

class LatentMultiAttributeClassifier(torch.nn.Module):
    def __init__(self, latent_dim, num_attrs):
        super().__init__()
        self.linear = torch.nn.Linear(latent_dim, num_attrs)

    def forward(self, z):
        return self.linear(z)

def load_image(path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)  # shape: (1, C, H, W)

def save_images(images, out_dir, name_prefix):
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        save_path = os.path.join(out_dir, f"{name_prefix}_{i}.png")
        save_image(img, save_path)

def manipulate(config, image_path, attr_index, step_size, steps):
    device = torch.device(config.training.device)

    # Load models
    encoder = get_model(config.model)
    decoder = get_model(config.model.decoder)
    classifier = LatentMultiAttributeClassifier(config.model.latent_dim, 40)

    encoder.load_state_dict(torch.load(config.model.pretrained_path))
    decoder.load_state_dict(torch.load(config.model.decoder_path))
    classifier.load_state_dict(torch.load(config.model.classifier_path))

    encoder, decoder, classifier = encoder.to(device), decoder.to(device), classifier.to(device)
    encoder.eval(); decoder.eval(); classifier.eval()

    # Load image
    img = load_image(image_path).to(device)

    with torch.no_grad():
        # Encode to latent space
        z = encoder(img)  # shape: (1, latent_dim)
        z = z.squeeze(0)  # shape: (latent_dim,)

        # Get direction vector for attribute
        w = classifier.linear.weight[attr_index]  # shape: (latent_dim,)
        w = w / w.norm()  # normalize for consistent step size

        manipulated_images = []

        # Traverse from -N to +N steps along attribute direction
        for alpha in torch.linspace(-step_size, step_size, steps):
            z_mod = z + alpha * w
            x_mod = decoder(z_mod.unsqueeze(0))  # (1, C, H, W)
            manipulated_images.append(x_mod.squeeze(0).cpu())

    save_images(manipulated_images, config.output_dir, os.path.splitext(os.path.basename(image_path))[0])
    print(f"Saved {steps} manipulated images to {config.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Attribute Manipulation Script")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--attribute_index", type=int, required=True)
    parser.add_argument("--step_size", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=7)
    args = parser.parse_args()

    config = load_config(args.config)
    manipulate(config, args.image, args.attribute_index, args.step_size, args.steps)
    