import torch

def get_loss_fn(sde):
    def mutual_information_fn(score_fn, batch, t_dist):
        x, y = batch
        t = t_dist.sample((x.shape[0],)).type_as(x)
        n = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * n
        
        cond = y
        conditional_score = score_fn(perturbed_x, cond, t)
        
        cond = torch.ones_like(y) * -1  # this signifies no side information
        unconditional_score = score_fn(perturbed_x, cond, t)
        
        losses = torch.square(unconditional_score - conditional_score)
        
        # we must use likelihood weighting for entropy estimation  
        _, g = sde.sde(torch.zeros_like(x), t, True)
        w2 = g ** 2
                
        importance_weight = torch.exp(-1 * t_dist.log_prob(t).type_as(t))
        losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * w2 * importance_weight
        losses *= 1 / 2
        loss = torch.mean(losses)
        return loss
    
    return mutual_information_fn
