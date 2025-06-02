import os
import torch
import numpy as np
import json
import seaborn as sns
import pandas as pd
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from configs import load_config
from utils.sampling_utils import generate_samples

def load_model(model, checkpoint_path, device=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_generation_callback(vis_callback, G=None, guidance_interval=None):
    if vis_callback == 'base':
        def generation_callback(batch, sde, diffusion_model, steps, shape, device):
            _, y = batch
            samples = generate_samples(
                y, sde, diffusion_model, steps, shape, device, G=G, guidance_interval=guidance_interval
            )
            return samples
    elif vis_callback == 'scoreVAE':
        def generation_callback(batch, sde, diffusion_model, steps, shape, device):
            x, y = batch
            z = diffusion_model.encode(x)
            cond = [y, z]
            reconstruction = generate_samples(
                cond, sde, diffusion_model, steps, shape, device, G=G, guidance_interval=guidance_interval
            )
            return reconstruction
    else:
        raise ValueError(f"Unknown vis_callback '{vis_callback}'. Expected 'base' or 'scoreVAE'.")
    return generation_callback

def sample_with_guidance(
    model, sde, dataloader, steps, shape, device,
    get_generation_callback,
    vis_callback,
    guidance_weight,
    guidance_interval,
    num_batches=1,
    plot_grid=True,
    grid_save_path=None
):
    model.eval()
    generation_callback = get_generation_callback(
        vis_callback,
        G=guidance_weight,
        guidance_interval=guidance_interval
    )

    l2_losses = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x, y = data if isinstance(data, (tuple, list)) else (data, None)
            x = x.to(device)
            if y is not None:
                y = y.to(device)
            recon = generation_callback(
                batch=(x, y),
                sde=sde,
                diffusion_model=model,
                steps=steps,
                shape=shape,
                device=device,
            )
            l2 = torch.mean((recon - x) ** 2, dim=tuple(range(1, recon.ndim)))
            l2_losses.extend(l2.cpu().numpy())
            has_plotted = False  # Ensure only first batch is plotted
            if plot_grid and not has_plotted:
                orig = x.clone().detach().cpu()
                rec = recon.clone().detach().cpu()
                nrow = min(16, orig.shape[0])
                grid_orig = vutils.make_grid(orig, nrow=nrow, normalize=True, scale_each=True)
                grid_recon = vutils.make_grid(rec, nrow=nrow, normalize=True, scale_each=True)
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(np.transpose(grid_orig.numpy(), (1, 2, 0)))
                axs[0].set_title("Originals")
                axs[0].axis("off")
                axs[1].imshow(np.transpose(grid_recon.numpy(), (1, 2, 0)))
                axs[1].set_title("Reconstructions")
                axs[1].axis("off")
                title = f"Guidance Weight: {guidance_weight}, Interval: ({guidance_interval[0]}, {guidance_interval[1]})"
                fig.suptitle(title, fontsize=14)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                if grid_save_path is None:
                    os.makedirs("ablation", exist_ok=True)
                    grid_save_path = f"ablation/weight_{guidance_weight:.3f}_int{guidance_interval[0]:.3f}-{guidance_interval[1]:.3f}.png"

                plt.savefig(grid_save_path, bbox_inches='tight')
                print(f"Grid plot saved to {grid_save_path}")
                plt.close(fig)
                has_plotted = True  # Ensure only first batch is plotted


            if i + 1 >= num_batches:
                break
    if np.isnan(l2_losses).any():
        print(f"Warning: {np.isnan(l2_losses).sum()} NaN losses detected! They will be ignored in mean.")
    mean_l2 = float(np.nanmean(l2_losses))
    return mean_l2, l2_losses

def grid_guided_sampling(
    model, sde, dataloader, steps, shape, device,
    get_generation_callback,
    vis_callback,
    guidance_weights,
    interval_limits,
    num_batches=1,
):
    results = {}
    best_loss = float('inf')
    best_params = None
    for G in guidance_weights:
        for interval in interval_limits:
            print(f"Sampling with weight={G}, interval={interval} ...")
            mean_l2, _ = sample_with_guidance(
                model, sde, dataloader, steps, shape, device,
                get_generation_callback,
                vis_callback,
                G, interval, num_batches=num_batches, plot_grid=True
            )
            results[(G, interval)] = mean_l2
            print(f"=> Mean L2 loss: {mean_l2:.6f}")
            if mean_l2 < best_loss:
                best_loss = mean_l2
                best_params = (G, interval)
    print(f"\nBest params: weight={best_params[0]}, interval={best_params[1]}, mean L2 loss={best_loss:.6f}")
    return results, best_params

def save_results_and_plot_heatmap(results):
    os.makedirs("ablation", exist_ok=True)

    # Save to JSON
    json_path = "ablation/grid_results.json"
    with open(json_path, "w") as f:
        json.dump({f"{k[0]:.3f},{k[1][0]:.3f}-{k[1][1]:.3f}": v for k, v in results.items()}, f, indent=4)
    print(f"Saved grid search results to {json_path}")

    # Convert to DataFrame
    heatmap_data = []
    for (G, interval), loss in results.items():
        interval_str = f"{interval[0]:.2f}-{interval[1]:.2f}"
        heatmap_data.append((G, interval_str, loss))
    df = pd.DataFrame(heatmap_data, columns=["GuidanceWeight", "Interval", "MeanL2Loss"])
    pivot = df.pivot(index="GuidanceWeight", columns="Interval", values="MeanL2Loss")

    # Plot heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Mean L2 Loss"})
    plt.title("Heatmap of Mean L2 Loss across Guidance Weights and Intervals", fontsize=16)
    plt.xlabel("Guidance Interval", fontsize=12)
    plt.ylabel("Guidance Weight", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    heatmap_path = "ablation/grid_heatmap.png"
    plt.savefig(heatmap_path)
    print(f"Saved heatmap to {heatmap_path}")
    plt.show()

def main():
    # --- Configurable grid search parameters ---
    config_path = 'configs/cifar10/scoreVAE.py'
    ckpt_path = 'results/cifar10/scoreVAE_noise/checkpoints/Model_last.pth'
    guidance_weights = np.linspace(0.1, 4.0, 15)
    interval_lows  = np.linspace(0.005, 4.0, 25)
    interval_highs = np.linspace(5.0, 90.0, 25)
    interval_limits = [(float(low), float(high)) for low in interval_lows for high in interval_highs if low < high]

    # --- Setup ---
    config = load_config(config_path)
    device = config.training.device
    train_loader, val_loader, test_loader = get_dataloaders(config.data)
    num_samples = config.training.num_samples
    shape = (num_samples, *config.data.shape)

    model = get_model(config.model).to(device)
    loaded_model = load_model(model=model, checkpoint_path=ckpt_path, device=device)
    sde = configure_sde(config)

    # --- Test on a single batch ---
    mean_l2, l2_losses = sample_with_guidance(
        model=loaded_model,
        sde=sde,
        dataloader=test_loader,
        steps=config.training.steps,
        shape=shape,
        device=device,
        get_generation_callback=get_generation_callback,
        vis_callback=config.training.vis_callback,
        guidance_weight=0,
        guidance_interval=(5.0, 90.),
        num_batches=1,
        plot_grid=True,
        grid_save_path="ablation/recon_grid.png"
    )
    print(f"Mean L2 loss: {mean_l2}")
    print(f"L2 losses: {l2_losses}")

    # --- Grid search ---
    results, best_params = grid_guided_sampling(
        model=loaded_model,
        sde=sde,
        dataloader=test_loader,
        steps=config.training.steps,
        shape=shape,
        device=device,
        get_generation_callback=get_generation_callback,
        vis_callback=config.training.vis_callback,
        guidance_weights=guidance_weights,
        interval_limits=interval_limits,
        num_batches=5,
    )

    save_results_and_plot_heatmap(results)
    print(f"Best params: {best_params} with mean L2 loss {results[best_params]:.6f}")

if __name__ == '__main__':
    main()
    