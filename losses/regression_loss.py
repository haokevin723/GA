import torch.nn as nn

def get_loss(name='mse'):
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    elif name == 'smooth_l1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f'Unknown loss: {name}')
