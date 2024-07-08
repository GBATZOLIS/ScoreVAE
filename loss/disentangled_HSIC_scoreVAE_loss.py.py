import torch
from .hsic_utils import median_heuristic, HSIC, rbf_kernel, dot_product_kernel, convert_to_one_hot, gram_matrix_condition_number_svd

def scoreVAE_loss_fn(score_fn, x, y, mean_z, log_var_z, latent, t_dist, likelihood_weighting, kl_weight, sde):
  # Compute KL loss using directly flattened tensors
  kl_loss = -0.5 * torch.sum(
                  1 + log_var_z.view(log_var_z.size(0), -1) - mean_z.view(mean_z.size(0), -1).pow(2) - log_var_z.view(log_var_z.size(0), -1).exp(),dim=1).mean()
      
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
            
  importance_weight = torch.exp(-1*t_dist.log_prob(t).type_as(t))
  losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2 * importance_weight
  losses *= 1/2
  rec_loss = torch.mean(losses)
  loss = rec_loss + kl_weight * kl_loss
  return loss


def get_loss_fn(sde, likelihood_weighting=True, kl_weight=1, disentanglement_factor=1):
    def loss_fn(score_fn, encoder, batch, t_dist, sigma, sigma_decay, step_idx):
        x, y = batch

        t0 = torch.zeros(x.shape[0]).type_as(x)
        latent_distribution_parameters = encoder(x, t0)
        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]
        latent = mean_z + (0.5 * log_var_z).exp() * torch.randn_like(mean_z)

        scoreVAE_loss = scoreVAE_loss_fn(score_fn, x, y, mean_z, log_var_z, latent, t_dist, likelihood_weighting, kl_weight, sde)

        # HSIC part
        new_sigma = median_heuristic(latent)
        if sigma < 0:
            sigma = new_sigma
        else:
            sigma = sigma_decay * sigma + (1 - sigma_decay) * new_sigma

        condition_numbers = {'latent': None, 'observation': None}
        if step_idx % 10 == 0:
            kernel_x_fn = lambda X: rbf_kernel(X, sigma=sigma)
            kernel_x = kernel_x_fn(latent)
            condition_number_latent = gram_matrix_condition_number_svd(kernel_x)
            kernel_y_fn = lambda Y: dot_product_kernel(convert_to_one_hot(Y))
            kernel_y = kernel_y_fn(y)
            condition_number_observation = gram_matrix_condition_number_svd(kernel_y)
            condition_numbers = {'latent': condition_number_latent, 'observation': condition_number_observation}

        hsic_instance = HSIC(
            kernel_x=lambda X: rbf_kernel(X, sigma=sigma),
            kernel_y=lambda Y: dot_product_kernel(convert_to_one_hot(Y)),
            algorithm='biased'
        )
        hsic_value = hsic_instance(latent, y)

        loss = scoreVAE_loss + disentanglement_factor * hsic_value
        return loss, hsic_value, condition_numbers, sigma
    
    return loss_fn
