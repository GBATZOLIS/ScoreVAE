import torch

def modify_labels_for_classifier_free_guidance(y):
    """
    Randomly choose half of the indices in `y` and set their value to -1.
    Args:
    - y (Tensor): The labels tensor.

    Returns:
    - Modified labels tensor with half of the values set to -1.
    """
    indices = torch.randperm(len(y))[:len(y) // 2]  # Randomly select half of the indices
    y[indices] = -1  # Set selected labels to -1
    return y

def get_loss_fn(sde, t_dist, likelihood_weighting=True):
    def loss_fn(score_fn, batch):
        x, y = batch
        t = t_dist.sample((x.shape[0],)).type_as(x)
        n = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n
        
        cond = modify_labels_for_classifier_free_guidance(y)
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
