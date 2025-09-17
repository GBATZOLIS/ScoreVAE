import math
import torch.optim as optim

# ────────────────────────── Schedulers ───────────────────────────

class WarmUpCosineDecayScheduler:
    """
    [NEW] Learning rate scheduler that combines a linear warmup phase with a
    cosine decay phase. Activated by setting `config.optim.scheduler = 'cosine_decay'`.
    """
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 base_lr: float,
                 min_lr: float = 1e-6,
                 global_step: int = 0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.global_step = global_step

    def step(self):
        """Advance the scheduler by one step."""
        self.global_step += 1
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step / self.warmup_steps)
        else:
            progress = (self.global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = max(0.0, min(1.0, progress)) # Clamp progress
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return self.__dict__


class WarmUpScheduler:
    """
    [ORIGINAL] Scheduler with a linear warmup followed by a constant learning rate.
    This is the default behavior if `config.optim.scheduler` is not specified.
    """
    def __init__(self, optimizer, warmup_steps, base_lr, global_step=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.global_step = global_step

    def step(self):
        self.global_step += 1
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step / self.warmup_steps)
        else:
            lr = self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return self.__dict__


# ───────────────── Factory Function (Main Entry Point) ──────────────────

def get_optimizer_and_scheduler(model, config, global_step=0):
    """
    Sets up the optimizer and scheduler based on the configuration.
    This version is fully backwards compatible. If `config.optim.scheduler`
    is not defined, it defaults to the original WarmUpScheduler.

    Args:
        model: The neural network model to be optimized.
        config: The ml_collections.ConfigDict containing optimization settings.
        global_step (int): The current global step for resuming training.

    Returns:
        A tuple containing the initialized optimizer and scheduler.
    """
    # --- Optimizer Setup (retains original logic) ---
    if config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.optim.get('lr', 2e-4),
            betas=(config.optim.get('beta1', 0.9), config.optim.get('beta2', 0.99)),
            eps=config.optim.get('eps', 1e-8),
            weight_decay=config.optim.get('weight_decay', 0.01),
            fused=False
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

    # --- Scheduler Selection (with backwards compatibility) ---
    # Default to 'constant' (the original behavior) if the field doesn't exist.
    scheduler_type = config.optim.get('scheduler', 'constant')

    if scheduler_type == 'cosine_decay':
        total_steps = config.optim.get('total_steps', 500_000) # Default for safety
        scheduler = WarmUpCosineDecayScheduler(
            optimizer,
            warmup_steps=config.optim.get('warmup', 1000),
            total_steps=total_steps,
            base_lr=config.optim.get('lr', 2e-4),
            global_step=global_step
        )
    elif scheduler_type == 'constant':
        # This branch ensures old configs work without modification.
        warmup_steps = max(0, config.optim.get('warmup', 1000) - global_step)
        scheduler = WarmUpScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            base_lr=config.optim.get('lr', 2e-4),
            global_step=global_step
        )
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' not supported.")

    return optimizer, scheduler
