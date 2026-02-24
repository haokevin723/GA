import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetRegressor(nn.Module):
    def __init__(self, backbone='densenet121', pretrained=True):
        super().__init__()
        densenet = getattr(models, backbone)(pretrained=pretrained)
        self.features = densenet.features
        in_features = densenet.classifier.in_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reg_head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.reg_head(x)
        return x.squeeze(1)
