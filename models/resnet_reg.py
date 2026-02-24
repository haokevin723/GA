import torch
import torch.nn as nn
import torchvision.models as models

class ResNetRegressor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()
        resnet = getattr(models, backbone)(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features
        self.reg_head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.reg_head(x)
        return x.squeeze(1)
