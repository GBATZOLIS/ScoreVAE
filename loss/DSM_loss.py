import torch
from utils.train_utils import get_score_fn

def get_loss_fn(sde, t_dist, likelihood_weighting=False):
    def loss_fn(model, batch):
        score_fn = get_score_fn(sde, model)
        x, y = batch
        t = t_dist.sample((x.shape[0],)).type_as(x)
        n = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n

        cond = y
        score = score_fn(perturbed_x, cond, t)
        grad_log_pert_kernel = -1 * n / std[(...,) + (None,) * len(x.shape[1:])]
        losses = torch.square(score - grad_log_pert_kernel)

        if likelihood_weighting:
            _, g = sde.sde(torch.zeros_like(x), t, True)
            w2 = g ** 2
        else:
            w2 = std ** 2

        importance_weight = torch.exp(-1 * t_dist.log_prob(t).type_as(t))
        losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2 * importance_weight
        losses *= 1 / 2
        loss = torch.mean(losses)
        return loss
    return loss_fn