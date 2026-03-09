import torch
import torch.nn as nn


def get_loss(name='mse'):
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    elif name == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif name == 'custom_weighted_mse':
        return CustomWeightedMSE()
    else:
        raise ValueError(f'Unknown loss: {name}')

class CustomWeightedMSE(nn.Module):
    def __init__(self):
        super(CustomWeightedMSE, self).__init__()

    def forward(self, preds, targets):
        # 1. 强制对齐维度，防止广播错误 [batch, 1] vs [batch]
        preds = preds.squeeze()
        targets = targets.squeeze()
        
        # 2. 计算基础平方误差
        mse = (preds - targets) ** 2
        
        # 3. 定义权重向量
        weights = torch.ones_like(targets)
        
        # 4. 根据你的混淆矩阵，重点加权中间 5 类 (25-36周，约 175-252天)
        # 31-34周 (210-238天) 是最差的，给最高权重
        mask_heavy = (targets >= 210) & (targets < 238)
        weights[mask_heavy] = 3.0
        
        # 其他中间周次 (25-30, 35-36) 给次高权重
        mask_mid = ((targets >= 175) & (targets < 210)) | ((targets >= 238) & (targets <= 252))
        weights[mask_mid] = 2.0
        
        # 5. 返回加权平均值
        return (mse * weights).mean()