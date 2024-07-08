from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
from utils.train_utils import prepare_batch
import torch
from utils.sampling_utils import generate_specified_num_samples

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    mean_diff = mu1 - mu2
    mean_diff_sq = mean_diff.dot(mean_diff)
    trace_covmean = np.trace(covmean)
    return mean_diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean

def get_activations(model, dataloaders, device):
    model.eval()
    activations = []
    with torch.no_grad():
        for dataloader in dataloaders:
            for data in dataloader:
                batch = prepare_batch(data, device)
                x, _ = batch
                pred = model(x)[0]
                pred = adaptive_avg_pool2d(pred, (1, 1)).squeeze(3).squeeze(2)
                activations.append(pred.cpu().numpy())
    return np.concatenate(activations, axis=0)

def fid_evaluation_callback(writer, sde, model, steps, shape, device, epoch, dataloaders, train=True):
    # Load InceptionV3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # Get activations for real data (combining dataloaders)
    real_activations = get_activations(inception_model, dataloaders, device)

    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)

    # Generate samples
    num_samples = len(real_activations)
    print(f'Generating {num_samples} samples for FID evaluation.')
    generated_samples = generate_specified_num_samples(num_samples, sde, model, steps, shape, device)

    # Convert generated samples to tensor and move to device for activations calculation
    generated_samples = torch.tensor(generated_samples).to(device)
    generated_activations = get_activations(inception_model, [generated_samples], device)

    mu_generated = np.mean(generated_activations, axis=0)
    sigma_generated = np.cov(generated_activations, rowvar=False)

    # Calculate FID
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)

    # Log FID score
    dataset = 'Train' if train else 'Test'
    writer.add_scalar(f'FID/{dataset}', fid_score, epoch)
    print(f'Epoch {epoch + 1}: {dataset} FID: {fid_score:.4f}')
