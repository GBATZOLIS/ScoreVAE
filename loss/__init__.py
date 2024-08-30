from importlib import import_module

def get_loss_fn(config, *args, **kwargs):
    loss_name = config.training.loss
    loss_module = import_module(f'loss.{loss_name}')
    return loss_module.get_loss_fn(config, *args, **kwargs)