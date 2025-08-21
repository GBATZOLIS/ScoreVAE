# utils/ae_utils.py
from __future__ import annotations
import math
import torch
import torchvision.utils as vutils
from contextlib import contextmanager
import matplotlib.pyplot as plt

@contextmanager
def evaluation_mode(model):
    was_train = model.training
    model.eval()
    try:
        yield
    finally:
        if was_train:
            model.train()

def get_reconstruction_callback():
    def recon_callback(batch, writer, model, device, epoch, tag_prefix="AE"):
        x = batch[0].to(device)
        n = min(x.size(0), 36)
        x = x[:n]
        with evaluation_mode(model), torch.no_grad():
            x_hat, _ = model(x)

        nrow = int(math.sqrt(n))
        grid_in  = vutils.make_grid(x, nrow=nrow, normalize=True, scale_each=True)
        grid_out = vutils.make_grid(x_hat.clamp(0,1), nrow=nrow, normalize=True, scale_each=True)

        writer.add_image(f"{tag_prefix}/original", grid_in, epoch)
        writer.add_image(f"{tag_prefix}/reconstruction", grid_out, epoch)
    return recon_callback

def get_latent_scatter_callback(num_batches: int = 2, max_points: int = 2000):
    """
    Returns a callback that plots the first latent dims of a few batches.
    If latent_dim >= 3: plots (z0,z1), (z1,z2), (z0,z2) as separate figures.
    If latent_dim == 2: plots (z0,z1) only.
    """
    def _plot_and_log(writer, tag_prefix, epoch, z_all, i, j, title_suffix=""):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(z_all[:, i], z_all[:, j], s=5, alpha=0.6)
        ax.set_xlabel(f"z[{i}]")
        ax.set_ylabel(f"z[{j}]")
        ttl = f"Latent scatter (z[{i}] vs z[{j}])"
        if title_suffix:
            ttl += f" — {title_suffix}"
        ax.set_title(ttl)
        writer.add_figure(f"{tag_prefix}/latent_scatter/z{i}{j}", fig, epoch)
        plt.close(fig)

    def latent_callback(val_loader, writer, model, device, epoch, tag_prefix="AE"):
        zs = []
        with evaluation_mode(model), torch.no_grad():
            for b_idx, (x, *_) in enumerate(val_loader):
                if b_idx >= num_batches:
                    break
                x = x.to(device)
                z = model.encode(x)
                zs.append(z.detach().cpu())

        if not zs:
            print("[Warning] No batches collected for latent scatter.")
            return

        z_all = torch.cat(zs, dim=0)
        if z_all.size(1) < 2:
            print("[Warning] Latent dim < 2, cannot plot scatter.")
            return

        z_all = z_all[:max_points]

        # Always plot z0 vs z1
        _plot_and_log(writer, tag_prefix, epoch, z_all, 0, 1)

        # If we have at least 3 dims, also plot z1 vs z2 and z0 vs z2
        if z_all.size(1) >= 3:
            _plot_and_log(writer, tag_prefix, epoch, z_all, 1, 2)
            _plot_and_log(writer, tag_prefix, epoch, z_all, 0, 2)

    return latent_callback
