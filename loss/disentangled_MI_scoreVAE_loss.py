import torch
from .MI_loss import get_loss_fn as get_MI_loss_fn

def get_loss_fn(sde, likelihood_weighting=True, kl_weight=1, disentanglement_factor=1):
    def loss_fn(score_fn, MI_diffusion_model_score_fn, encoder, batch, t_dist):
        x, y = batch

        # Get the encoded latent and its associated parameters (mean_z, log_var_z)
        t0 = torch.zeros(x.shape[0]).type_as(x)
        latent_distribution_parameters = encoder(x, t0)
        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]
        latent = mean_z + (0.5 * log_var_z).exp() * torch.randn_like(mean_z)

        # Compute VAE loss
        kl_loss = -0.5 * torch.sum(
                      1 + log_var_z.view(log_var_z.size(0), -1) - mean_z.view(mean_z.size(0), -1).pow(2) - log_var_z.view(log_var_z.size(0), -1).exp(), dim=1).mean()

        t = t_dist.sample((x.shape[0],)).type_as(x)
        n = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n

        cond = [y, latent]
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
        rec_loss = torch.mean(losses)
        vae_loss = rec_loss + kl_weight * kl_loss

        # Compute mutual information loss
        MI_loss_fn = get_MI_loss_fn(sde)
        MI_batch = [y, latent]
        mutual_information = MI_loss_fn(MI_diffusion_model_score_fn, MI_batch, t_dist)

        loss = vae_loss + disentanglement_factor * mutual_information
        return loss
    
    return loss_fn
