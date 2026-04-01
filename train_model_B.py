import os
import yaml
import torch
import torch.nn as nn
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from losses.regression_loss import get_loss
from losses.regression_loss import CustomWeightedMSE, RegressionContrastiveLoss

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

    """
    # 数据预处理重新处理
    # 专科医生专属的“高倍镜与防作弊”预处理
    # 1. 训练集专属（带高倍镜 + 破坏性增强）
    train_transform = transforms.Compose([
        transforms.Resize((400, 400)), 
        transforms.RandomCrop(384),    # 产生最大 16 像素的微小位移，防死记硬背
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
    ])

    # 2. 验证集/测试集专属（只做高倍镜放大，绝对公平对齐）
    val_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.CenterCrop(384),    # 严格取正中心，保证尺寸一致但不引入随机性
        transforms.ToTensor(),
    ])
    """
    # 加载基础数据集时，分别传入不同的 transform
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
        
    #model.load_state_dict(torch.load(model_A_path))
    # 明确告诉 PyTorch 把权重映射到 cpu 上，同时消除 weights_only 的警告
    model.load_state_dict(torch.load(model_A_path, map_location=torch.device('cpu'), weights_only=True))
    print(f"[INFO] 成功继承全科医生 (Model A) 权重: {model_A_path}")
    model = model.to(device) 

    #获取最后一层的特征
    features_dict = {}
    def get_features_hook(module, input, output):
        # input[0] 就是进入最后一层分类器之前的深层特征矩阵
        features_dict['features'] = input[0] 

    # 1. 防御性脱壳 (防止你用了多卡 DataParallel)
    actual_model = model.module if hasattr(model, 'module') else model

    # 2. 直接将窃听器拍在你自定义的 reg_head 上！
    actual_model.reg_head.register_forward_hook(get_features_hook)
    print("🌟 [INFO] 窃听器已精准狙击并挂载到自建层: reg_head")
    # 3. 初始化两位监工：MSE 主导，Contrastive 辅助排雷
    criterion = get_loss(cfg['loss'])
    criterion_contra = RegressionContrastiveLoss(margin=5.0, pos_thresh=3, neg_thresh=10).to(device)

    # 4. 使用平滑的 Huber Loss 和极小的学习率微调
    #criterion = nn.HuberLoss(delta=5.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses = []
    val_losses_curve = []
    best_mae = float('inf')
    
    print("\n" + "="*40)
    print(" 🚀 开始训练：晚孕期专科医生 (Model B)")
    print("="*40 + "\n")
    
    # 【重要提醒】请确保在此循环之前，你已经初始化了以下变量：
    # criterion_mse = torch.nn.MSELoss()
    # criterion_contra = RegressionContrastiveLoss(margin=5.0, pos_thresh=3, neg_thresh=10).to(device)
    # features_dict = {}  (并且已经用 register_forward_hook 挂载到了网络上)

    for epoch in range(cfg['epochs']):
            # --- 训练阶段 ---
            model.train()
            epoch_train_losses = []
            epoch_mse_losses = []    # [新增] 专门记录主线任务 Loss
            epoch_contra_losses = [] # [新增] 专门记录支线任务 Loss

            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.float().to(device)
                
                # [修改] 前向传播 (此时 Hook 已经悄悄把 1920 维特征存进了 features_dict)
                preds = model(images).view(-1)
                targets = targets.view(-1)
                
                # [新增] 获取截获的深层特征
                features = features_dict['features']
                
                # [修改] 1. 计算主线任务 Loss (MSE)
                loss_mse = criterion(preds, targets)
                
                # [新增] 2. 计算支线任务 Loss (Contrastive)
                loss_contra = criterion_contra(features, targets)
                
                # [新增] 3. 终极 Loss 合体 (0.1 是权重，保证以猜天数为主，排斥为辅)
                loss = loss_mse + 0.1 * loss_contra
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 保护基座参数
                optimizer.step()
                
                epoch_train_losses.append(loss.item())
                epoch_mse_losses.append(loss_mse.item())       
                epoch_contra_losses.append(loss_contra.item())
                
            mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            mean_mse_loss = sum(epoch_mse_losses) / len(epoch_mse_losses)          
            mean_contra_loss = sum(epoch_contra_losses) / len(epoch_contra_losses)
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
                    
                    # [修改] 验证集不需要算对比损失，老老实实算 MSE 即可
                    loss = criterion(preds, targets) 
                    val_losses.append(loss.item())
                    val_maes.append(torch.mean(torch.abs(preds - targets)).item())
                    
            mean_val_loss = sum(val_losses) / len(val_losses)
            mean_val_mae = sum(val_maes) / len(val_maes)
            val_losses_curve.append(mean_val_loss)
            
            # [修改] 打印信息升级！你能清晰看到对比损失是否在发挥作用
            print(f"Epoch {epoch+1}/{cfg['epochs']} | Total Train Loss: {mean_train_loss:.4f} (MSE: {mean_mse_loss:.4f}, Contra: {mean_contra_loss:.4f}) | Val MAE: {mean_val_mae:.3f} 天")
            
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