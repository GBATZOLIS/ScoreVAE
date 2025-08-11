import torch


def get_loss_fn(config, sde, t_dist):
    """
    Simple denoising-score-matching loss with on-GPU timestep sampling.
    """
    def loss_fn(model, batch, train: bool):
        x, y = batch
        t = torch.empty(x.size(0), device=x.device).uniform_(t_dist.low, t_dist.high)
        noise = torch.randn_like(x)

        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * (x.ndim - 1)] * noise
        pred = model(perturbed_x, y, t)
        return torch.mean((noise - pred) ** 2)

    return loss_fn
