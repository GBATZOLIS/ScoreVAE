import torch.optim as optim

class WarmUpScheduler:
    def __init__(self, optimizer, target_lr, warmup_steps):
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.target_lr * min(1.0, self.step_num / self.warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def get_optimizer_and_scheduler(model, config):
    """
    Sets up the optimizer and scheduler based on the configuration provided.
    
    Args:
        model: The neural network model.
        config: Configuration dictionary containing optimization settings.
    
    Returns:
        optimizer: The initialized optimizer.
        scheduler: The initialized scheduler.
    """
    if config.optim.optimizer == 'Adam':
        # Best default values for Adam in training diffusion models
        # lr: 1e-4, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 1e-5
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.optim.get('lr', 1e-4),
            betas=(config.optim.get('beta1', 0.9), config.optim.get('beta2', 0.999)),
            eps=config.optim.get('eps', 1e-8),
            weight_decay=config.optim.get('weight_decay', 1e-5)
        )
    elif config.optim.optimizer == 'RMSprop':
        # Best default values for RMSprop in training diffusion models
        # lr: 1e-4 to 1e-3, alpha: 0.99, eps: 1e-8, weight_decay: 1e-5
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config.optim.get('lr', 1e-4),
            alpha=config.optim.get('alpha', 0.99),
            eps=config.optim.get('eps', 1e-8),
            weight_decay=config.optim.get('weight_decay', 1e-5)
        )
    elif config.optim.optimizer == 'AdamW':
        # Best default values for AdamW in training diffusion models
        # lr: 1e-4 to 1e-3, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 1e-2 to 1e-3
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.optim.get('lr', 1e-4),
            betas=(config.optim.get('beta1', 0.9), config.optim.get('beta2', 0.999)),
            eps=config.optim.get('eps', 1e-8),
            weight_decay=config.optim.get('weight_decay', 1e-2)
        )
    elif config.optim.optimizer == 'SGD':
        # Best default values for SGD with Momentum in training diffusion models
        # lr: 1e-3 to 1e-2, momentum: 0.9, weight_decay: 1e-5
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.optim.get('lr', 1e-2),
            momentum=config.optim.get('momentum', 0.9),
            weight_decay=config.optim.get('weight_decay', 1e-5)
        )
    else:
        raise ValueError(f"Optimizer {config.optim.optimizer} is not supported.")
    
    # Setup the scheduler
    scheduler = WarmUpScheduler(optimizer, config.optim.get('lr', 1e-4), warmup_steps=config.optim.get('warmup', 5000))
    
    return optimizer, scheduler
