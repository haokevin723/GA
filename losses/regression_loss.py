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
        """
        preds: 模型输出的天数 [BatchSize]
        targets: 标签真实天数 [BatchSize]
        """
        # 1. 维度安全对齐：强制转为一维向量，防止 BatchSize=1 时 squeeze 报错
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # 2. 计算基础平方误差 (MSE 不求平均)
        mse = (preds - targets) ** 2
        
        # 3. 初始化权重向量，默认权重为 1.0
        weights = torch.ones_like(targets)
        
        # 4. 根据你的混淆矩阵分布进行“阶梯式”加权
        # 核心痛点 A：31-34周 (210-238天)，F1 分数最低，给 3.5 倍最高权重
        mask_heavy = (targets >= 210) & (targets < 238)
        weights[mask_heavy] = 3.5
        
        # 核心痛点 B：其他中间周次 25-30周 (175-210天) 和 35-36周 (238-252天)
        # 这些区域目前在 60%+，给 2.0 倍权重冲刺 75%
        mask_mid = ((targets >= 175) & (targets < 210)) | ((targets >= 238) & (targets <= 252))
        weights[mask_mid] = 2.0
        
        # 21-24周 和 37-40周 保持权重 1.0 (因为已经达标)

        # 5. 计算加权 MSE：先点乘权重，再求平均
        weighted_mse = (mse * weights).mean()
        
        return weighted_mse