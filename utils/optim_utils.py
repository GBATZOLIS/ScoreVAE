import torch.optim as optim

class WarmUpScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr, global_step=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.global_step = global_step

    def step(self):
        self.global_step += 1
        if self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step / self.warmup_steps)
        else:
            lr = self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return self.__dict__
    
def get_optimizer_and_scheduler(model, config, global_step=0):
    """
    Sets up the optimizer and scheduler based on the configuration provided.
    
    Args:
        model: The neural network model.
        config: Configuration dictionary containing optimization settings.
        global_step: The global step counter.
    
    Returns:
        optimizer: The initialized optimizer.
        scheduler: The initialized scheduler.
    """
    if config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.optim.get('lr', 2e-4),
            betas=(config.optim.get('beta1', 0.9), config.optim.get('beta2', 0.99)),
            eps=config.optim.get('eps', 1e-8),
            weight_decay=config.optim.get('weight_decay', 0.01)
        )
    elif config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.optim.get('lr', 2e-4),
            betas=(config.optim.get('beta1', 0.9), config.optim.get('beta2', 0.99)),
            eps=config.optim.get('eps', 1e-8),
            weight_decay=config.optim.get('weight_decay', 1e-5)
        )
    elif config.optim.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config.optim.get('lr', 1e-4),
            alpha=config.optim.get('alpha', 0.99),
            eps=config.optim.get('eps', 1e-8),
            weight_decay=config.optim.get('weight_decay', 1e-5)
        )
    elif config.optim.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.optim.get('lr', 1e-2),
            momentum=config.optim.get('momentum', 0.9),
            weight_decay=config.optim.get('weight_decay', 1e-5)
        )
    else:
        raise ValueError(f"Optimizer {config.optim.optimizer} is not supported.")
    
    warmup_steps = max(0, config.optim.get('warmup', 1000) - global_step)
    scheduler = WarmUpScheduler(optimizer, warmup_steps, config.optim.get('lr', 2e-4), global_step)
    
    return optimizer, scheduler
