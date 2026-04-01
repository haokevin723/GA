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

class RegressionContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=5.0, pos_thresh=3, neg_thresh=10):
        """
        margin: 负样本对在特征空间中需要被推开的最小距离
        pos_thresh: 天数差 <= 该值，视为极度相似的正样本 (拉近)
        neg_thresh: 天数差 >= 该值，视为绝对不同的负样本 (推开)
        """
        super().__init__()
        self.margin = margin
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

    def forward(self, features, labels):
        # features: [Batch, Feature_Dim], labels: [Batch]
        B = features.size(0)
        
        # 1. 算出 Batch 内所有图片特征两两之间的距离矩阵
        dist_matrix = torch.cdist(features, features, p=2)
        
        # 2. 算出 Batch 内所有真实天数两两之间的绝对差值
        label_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))

        # 3. 划定正负样本掩码 (剔除对角线自己跟自己的配对)
        eye_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask = (label_diff <= self.pos_thresh) & ~eye_mask
        neg_mask = (label_diff >= self.neg_thresh)

        # 4. 正样本 Loss：天数相近的，特征距离越小越好 (Pull)
        if pos_mask.any():
            pos_loss = dist_matrix[pos_mask].pow(2).mean()
        else:
            pos_loss = torch.tensor(0.0, device=features.device)

        # 5. 负样本 Loss：天数差得远的，距离如果小于 margin，就狠狠惩罚 (Push)
        if neg_mask.any():
            neg_loss = F.relu(self.margin - dist_matrix[neg_mask]).pow(2).mean()
        else:
            neg_loss = torch.tensor(0.0, device=features.device)

        return pos_loss + neg_loss