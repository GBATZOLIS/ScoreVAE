import torch

def get_loss_fn(config, sde, t_dist):
    def loss_fn(model, batch, train):
        x, y = batch
        t = t_dist.sample((x.shape[0],)).type_as(x)
        n = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n

        noise_pred = model(perturbed_x, y, t)
        loss = torch.mean((n - noise_pred) ** 2)
        return loss
    return loss_fn
