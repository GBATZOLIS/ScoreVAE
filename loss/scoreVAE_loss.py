import torch
from utils.train_utils import get_score_fn, get_noise_fn

def get_loss_fn(config, sde, t_dist):
    kl_weight = config.training.get('kl_weight', 1e-3)
    def loss_fn(model, batch, train):
        encoder = model.encoder
        noise_predictor_fn = get_noise_fn(sde, model, train)
        x, y = batch

        # Get the encoded latent and its associated parameters (mean_z, log_var_z)
        t0 = torch.zeros(x.shape[0]).type_as(x)
        latent_distribution_parameters = encoder(x, t0)
        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]
        latent = mean_z + (0.5 * log_var_z).exp() * torch.randn_like(mean_z)

        # Compute KL penalty term
        kl_loss = -0.5 * torch.sum(
                      1 + log_var_z.view(log_var_z.size(0), -1) - mean_z.view(mean_z.size(0), -1).pow(2) - log_var_z.view(log_var_z.size(0), -1).exp(), dim=1).mean()

        # Compute the likelihood term (known as the reconstruction loss)
        t = t_dist.sample((x.shape[0],)).type_as(x)
        n = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n

        cond = [y, latent]
        n_prediction = noise_predictor_fn(perturbed_x, cond, t)
        losses = torch.square(n_prediction - n)
        losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
        losses *= 1 / 2
        rec_loss = torch.mean(losses) #likelihood loss / reconstruction loss
        
        loss = rec_loss + kl_weight * kl_loss #kl_weight is beta in beta-VAEs.

        return loss
    
    return loss_fn


def get_old_loss_fn(config, sde, t_dist):
    likelihood_weighting = config.training.get('likelihood_weighting', False)
    kl_weight = config.training.get('kl_weight', 1e-3)
    def loss_fn(model, batch, train):
        encoder = model.encoder
        score_fn = get_score_fn(sde, model, train)
        x, y = batch

        # Get the encoded latent and its associated parameters (mean_z, log_var_z)
        t0 = torch.zeros(x.shape[0]).type_as(x)
        latent_distribution_parameters = encoder(x, t0)
        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]
        latent = mean_z + (0.5 * log_var_z).exp() * torch.randn_like(mean_z)

        # Compute KL penalty term
        kl_loss = -0.5 * torch.sum(
                      1 + log_var_z.view(log_var_z.size(0), -1) - mean_z.view(mean_z.size(0), -1).pow(2) - log_var_z.view(log_var_z.size(0), -1).exp(), dim=1).mean()

        # Compute the likelihood term (known as the reconstruction loss)
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
        rec_loss = torch.mean(losses) #likelihood loss / reconstruction loss
        
        loss = rec_loss + kl_weight * kl_loss #kl_weight is beta in beta-VAEs.

        return loss
    
    return loss_fn
