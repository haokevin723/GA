import os
import yaml
import torch
import torch.nn as nn
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

# 完美引用你刚加进去的两个类
from datasets.regression_dataset import RegressionDataset, RefineDataset 
from models.model_factory import get_model
from utils.seed import set_seed
from utils.misc import ensure_dir

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

    # 读取 Model B 专属配置 (记得新建 refine.yaml，把 lr 改为 1e-5, batch_size 改为 16)
    cfg = load_config('configs/refine.yaml')
    set_seed(cfg['seed'])

    ensure_dir('checkpoints')
    ensure_dir('logs')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 1. 加载基础数据集
    base_train_dataset = RegressionDataset(cfg['data_dir'], cfg['label_csv'], cfg['train_split'], transform)
    base_val_dataset = RegressionDataset(cfg['data_dir'], cfg['label_csv'], cfg['val_split'], transform)

    # 2. 套上过滤器，专供 Model B
    train_dataset_B = RefineDataset(base_train_dataset, min_days=217, max_days=258)
    val_dataset_B = RefineDataset(base_val_dataset, min_days=217, max_days=258)

    train_loader = DataLoader(train_dataset_B, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset_B, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # 3. 初始化模型，并加载 Model A 的强力基座权重
    model = get_model(cfg['model'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    
    model_A_path = os.path.join('checkpoints', 'best_model_A.pth')
    if not os.path.exists(model_A_path):
        raise FileNotFoundError(f"找不到基座模型！请确保 {model_A_path} 存在。")
        
    model.load_state_dict(torch.load(model_A_path))
    print(f"[INFO] 成功继承全科医生 (Model A) 权重: {model_A_path}")
    model = model.to(device) 

    # 4. 使用平滑的 Huber Loss 和极小的学习率微调
    #criterion = nn.HuberLoss(delta=5.0)
    criterion = nn.get_loss(cfg['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses = []
    val_losses_curve = []
    best_mae = float('inf')
    
    print("\n" + "="*40)
    print(" 🚀 开始训练：晚孕期专科医生 (Model B)")
    print("="*40 + "\n")
    
    for epoch in range(cfg['epochs']):
        # --- 训练阶段 ---
        model.train()
        epoch_train_losses = []
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.float().to(device)
            
            preds = model(images).view(-1)
            targets = targets.view(-1)
            
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 保护基座参数
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
            
        mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(mean_train_loss)
        
        # --- 验证阶段 ---
        model.eval()
        val_losses = []
        val_maes = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.float().to(device)
                
                preds = model(images).view(-1)
                targets = targets.view(-1)
                
                loss = criterion(preds, targets)
                val_losses.append(loss.item())
                val_maes.append(torch.mean(torch.abs(preds - targets)).item())
                
        mean_val_loss = sum(val_losses) / len(val_losses)
        mean_val_mae = sum(val_maes) / len(val_maes)
        val_losses_curve.append(mean_val_loss)
        
        print(f"Epoch {epoch+1}/{cfg['epochs']} | MSE Loss: {mean_train_loss:.4f} | Val MAE: {mean_val_mae:.3f} 天")
        
        # 保存最优的 Model B
        if mean_val_mae < best_mae:
            best_mae = mean_val_mae
            torch.save(model.state_dict(), os.path.join('checkpoints', 'best_model_B.pth'))
            print(f"  🌟 [保存] 发现更强专科医生，MAE: {best_mae:.3f} 天 -> best_model_B.pth")

        scheduler.step(mean_val_mae)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 40)

        pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses_curve}).to_csv('logs/loss_curve_B.csv', index=False)

if __name__ == '__main__':
    main()