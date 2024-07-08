from importlib import import_module

def get_loss_fn(loss_name, *args, **kwargs):
    loss_module = import_module(f'loss.{loss_name}')
    return loss_module.get_loss_fn(*args, **kwargs)