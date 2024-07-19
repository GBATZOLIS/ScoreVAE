from torchmetrics.image.fid import FrechetInceptionDistance as _FID
from utils.train_utils import prepare_batch
import torch
from utils.sampling_utils import generate_specified_num_samples, generate_specified_num_samples_parallel
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision.utils as vutils
import math

def update_metric_with_real_activations(metric, dataloaders, device):
    print("Updating metric with real data activations...")
    num_real_datapoints = 0
    first_batch_real = None
    with torch.no_grad():
        for dataloader in dataloaders:
            for data in tqdm(dataloader, desc="Processing real data"):
                batch = prepare_batch(data, device)
                x, _ = batch
                if first_batch_real is None:
                    first_batch_real = x
                num_real_datapoints += x.size(0)

                # Assert that the real data is in the range [-1, 1]
                assert torch.min(x) >= -1.0 and torch.max(x) <= 1.0, "Real data is not in the range [-1, 1]"
                x_normalized = (x + 1) / 2  # Rescale to [0, 1]
                metric.update(x_normalized, real=True)
    print(f"Processed {num_real_datapoints} real data points.")
    return num_real_datapoints, first_batch_real

def get_generated_activations(model, sde, num_samples, steps, shape, device_ids):
    generated_samples = generate_specified_num_samples_parallel(num_samples, sde, model, steps, shape, device_ids)
    
    # Clip and normalize generated samples
    generated_samples = torch.clamp(generated_samples, -1, 1)
    generated_samples = (generated_samples + 1) / 2  # Rescale to [0, 1]

    return generated_samples

def update_metric_with_fake_activations(metric, generated_samples, dataloaders, device):
    print("Updating metric with fake data activations...")
    generated_dataset = TensorDataset(generated_samples)
    batch_size = dataloaders[0].batch_size  # Assuming all dataloaders have the same batch size
    generated_dataloader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)

    first_batch_fake = None
    with torch.no_grad():
        for i, batch in enumerate(tqdm(generated_dataloader, desc="Processing generated data")):
            batch = batch[0].to(device)  # Move to device and unpack the tensor from the list
            if i == 0:
                first_batch_fake = batch
            metric.update(batch, real=False)
    print("Updated metric with generated data activations.")
    return first_batch_fake

def fid_evaluation_callback(writer, sde, model, steps, shape, device_ids, epoch, dataloaders, train=True):
    primary_device = torch.device(f'cuda:{device_ids[0]}')
    
    # If argument normalize is True images are expected to be dtype float and have values in the [0,1] range
    fid = _FID(normalize=True).to(primary_device)

    # Get activations for real data
    num_real_datapoints, first_batch_real = update_metric_with_real_activations(fid, dataloaders, primary_device)

    # Generate samples and get activations for generated data
    num_samples = num_real_datapoints
    print(f'Generating {num_samples} samples for FID evaluation.')
    generated_samples = get_generated_activations(model, sde, num_samples, steps, shape, device_ids)
    
    # Update metric with fake activations
    first_batch_fake = update_metric_with_fake_activations(fid, generated_samples, dataloaders, primary_device)

    # Calculate FID
    fid_score = fid.compute().item()

    # Log FID score
    dataset = 'Train' if train else 'Test'
    writer.add_scalar(f'FID/{dataset}', fid_score, epoch)
    print(f'Epoch {epoch + 1}: {dataset} FID: {fid_score:.4f}')
    
    
    # Plot the first batch of real data
    if first_batch_real is not None:
        num_rows = int(math.sqrt(first_batch_real.size(0)))
        real_grid = vutils.make_grid(first_batch_real, nrow=num_rows, normalize=True, scale_each=True)
        writer.add_image(f'Real Images/{dataset}', real_grid, epoch)

    # Plot the first batch of generated data
    if first_batch_fake is not None:
        num_rows = int(math.sqrt(first_batch_fake.size(0)))
        fake_grid = vutils.make_grid(first_batch_fake, nrow=num_rows, normalize=True, scale_each=True)
        writer.add_image(f'Generated Images/{dataset}', fake_grid, epoch)
    
