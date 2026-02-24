from .resnet_reg import ResNetRegressor
from .densenet_reg import DenseNetRegressor

def get_model(name, **kwargs):
    if name == 'resnet':
        return ResNetRegressor(**kwargs)
    elif name == 'densenet':
        return DenseNetRegressor(**kwargs)
    else:
        raise ValueError(f'Unknown model: {name}')
