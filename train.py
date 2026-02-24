import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.regression_dataset import RegressionDataset
from models.model_factory import get_model
from losses.regression_loss import get_loss
from utils.seed import set_seed
from utils.misc import ensure_dir

# 读取配置
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=None, help='GPU id to use, e.g. 0 or 1')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using GPU: {args.gpu}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {device}")

    cfg = load_config('configs/default.yaml')
    set_seed(cfg['seed'])

    ensure_dir('checkpoints')

    #trasform的作用是对图像进行预处理和数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)), #双线性插值调整图像大小
        transforms.ToTensor(),
    ])

    train_dataset = RegressionDataset(cfg['data_dir'], cfg['label_csv'], cfg['train_split'], transform)
    val_dataset = RegressionDataset(cfg['data_dir'], cfg['label_csv'], cfg['val_split'], transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # 获取模型、损失函数和优化器，pretrained表示是否使用预训练权重
    model = get_model(cfg['model'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    model = model.to(device) 

    criterion = get_loss(cfg['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    best_mae = float('inf')
    for epoch in range(cfg['epochs']):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.float().to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 验证
        model.eval()
        val_losses = []
        val_maes = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.float().to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                val_losses.append(loss.item())
                val_maes.append(torch.mean(torch.abs(preds - targets)).item())
        mean_val_mae = sum(val_maes) / len(val_maes)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | Val MAE: {mean_val_mae:.3f}")
        # 保存最优模型
        if mean_val_mae < best_mae:
            best_mae = mean_val_mae
            torch.save(model.state_dict(), os.path.join('checkpoints', 'best_model.pth'))
            print("[INFO] Saved best model.")

if __name__ == '__main__':
    main()
