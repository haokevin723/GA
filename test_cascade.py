import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd  # 新增 pandas 用于保存详细预测记录
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findSystemFonts
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.regression_dataset import RegressionDataset
from models.model_factory import get_model
from utils.metrics import mae, rmse
from utils.misc import ensure_dir

# ==========================================
# 保持你原有的所有 Helper 和画图函数完全不变
# ==========================================
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def day_to_week(day):
    return int(day // 7)

def week_and_day(day):
    week = int(day // 7)
    days = int(day % 7)
    return week, days

def pred_to_interval(pred):
    intervals = [
        ('21-24w', 147, 168), ('25-28w', 175, 196),
        ('29-30w', 203, 216), ('31-32w', 217, 230),
        ('33-34w', 231, 244), ('35-36w', 245, 258),
        ('37-40w', 259, 280)
    ]
    for i, (label, start, end) in enumerate(intervals):
        if i < len(intervals) - 1:
            if start <= pred < intervals[i+1][1]: return label, start, end
        else:
            if start <= pred <= end: return label, start, end
    return 'Other', None, None

def print_class_distribution(class_labels, classes, title):
    print(f"\n[{title}]")
    for label in class_labels: print(f"{label}: {classes.count(label)}")

def print_confusion_matrix(class_labels, true_classes, pred_classes):
    print("\n[Confusion Matrix]")
    matrix = defaultdict(lambda: Counter())
    for t, p in zip(true_classes, pred_classes): matrix[t][p] += 1
    print("True Class -> Predicted Class:")
    header = '\t'.join(class_labels)
    print(f"Class\t{header}")
    for t in class_labels:
        row = [str(matrix[t][p]) for p in class_labels]
        print(f"{t}\t" + '\t'.join(row))
    return matrix

def print_classification_metrics(class_labels, matrix):
    print("\n[Classification Metrics]")
    for label in class_labels:
        TP = matrix[label][label]
        FP = sum(matrix[other][label] for other in class_labels if other != label)
        FN = sum(matrix[label][other] for other in class_labels if other != label)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{label}: Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

def week_to_class(week):
    if 21 <= week <= 24: return '21-24w'
    elif 25 <= week <= 28: return '25-28w'
    elif 29 <= week <= 30: return '29-30w'
    elif 31 <= week <= 32: return '31-32w'
    elif 33 <= week <= 34: return '33-34w'
    elif 35 <= week <= 36: return '35-36w'
    elif 37 <= week <= 40: return '37-40w'
    else: return 'other'

def save_scatter_plot(all_targets, all_preds, true_classes, class_labels, result_dir):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(7,7))
    simhei_path = None
    for font in findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'simhei' in font.lower():
            simhei_path = font
            break
    if simhei_path:
        my_font = FontProperties(fname=simhei_path)
    else:
        my_font = None
    
    color_map = {'21-24w': 'tab:blue', '25-28w': 'tab:orange', '29-30w': 'tab:green',
                 '31-32w': 'tab:red', '33-34w': 'tab:purple', '35-36w': 'tab:brown',
                 '37-40w': 'tab:pink', 'Other': 'gray'}
    for label in class_labels[:-1]:
        idxs = [i for i, c in enumerate(true_classes) if c == label]
        if idxs:
            plt.scatter(all_targets[idxs], all_preds[idxs], alpha=0.7, label=label, color=color_map[label], edgecolors='none')
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    if my_font:
        plt.xlabel('True Value', fontproperties=my_font)
        plt.ylabel('Predicted Value', fontproperties=my_font)
        plt.title('Cascade Regression Scatter Plot by Gestational Age', fontproperties=my_font)
        plt.legend(prop=my_font)
    else:
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title('Cascade Regression Scatter Plot by Gestational Age')
        plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(result_dir, 'scatter_plot.png')
    plt.savefig(scatter_path, dpi=200)

def save_tracking_plot(all_targets, all_preds, result_dir):
    idx_sorted = np.argsort(-all_targets)
    sorted_targets = all_targets[idx_sorted]
    sorted_preds = all_preds[idx_sorted]
    x = np.arange(len(sorted_targets))
    plt.figure(figsize=(10,5))
    plt.scatter(x, sorted_targets, color='blue', label='True', s=18, alpha=0.8)
    plt.scatter(x, sorted_preds, color='orange', label='Pred', s=18, alpha=0.8)
    plt.xlabel('Sample (sorted by true value)')
    plt.ylabel('Days')
    plt.title('Cascade Prediction Tracking Plot')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(result_dir, 'tracking_plot.png')
    plt.savefig(save_path, dpi=200)
    plt.close()

# ==========================================
# 核心大改：双模型级联推理逻辑
# ==========================================
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

    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.CenterCrop(384),    # 严格取正中心，保证尺寸一致但不引入随机性
        transforms.ToTensor(),
    ])

    test_dataset = RegressionDataset(cfg['data_dir'], cfg['label_csv'], cfg['test_split'], transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # --- 1. 加载双模型 ---
    print("\n[INFO] 正在加载双模型...")
    # 全科医生 (Model A)
    model_A = get_model(cfg['model'], backbone=cfg['backbone'], pretrained=False)
    ckpt_A_path = os.path.join('checkpoints', 'best_model_A.pth')
    model_A.load_state_dict(torch.load(ckpt_A_path, map_location=device, weights_only=True))
    model_A = model_A.to(device)
    model_A.eval()

    # 晚孕期专科医生 (Model B)
    model_B = get_model(cfg['model'], backbone=cfg['backbone'], pretrained=False)
    ckpt_B_path = os.path.join('checkpoints', 'best_model_B.pth')
    model_B.load_state_dict(torch.load(ckpt_B_path, map_location=device, weights_only=True))
    model_B = model_B.to(device)
    model_B.eval()
    print("[INFO] 全科医生 & 专科医生 已就位！开始会诊...\n")

    all_preds = []
    all_targets = []
    all_names = []
    
    # 记录详细的路由信息，用于后续分析
    routing_records = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.float().to(device).view(-1)
            
            # --- 2. 核心路由逻辑 (Cascade Routing) ---
            # 步骤 1：全科医生给出初步预测
            preds_A = model_A(images).view(-1)
            
            # 步骤 2：建立最终预测值的容器 (默认先放入 A 的结果)
            final_preds = preds_A.clone()
            
            # 步骤 3：圈定需要专科医生出马的样本 (预测值落在 217~258 天)
            mask_need_B = (preds_A >= 217) & (preds_A <= 258)
            
            # 步骤 4：专科医生对这些疑难杂症进行精修
            if mask_need_B.any():
                images_for_B = images[mask_need_B]
                preds_B = model_B(images_for_B).view(-1)
                # 用 B 的精修结果覆盖 A 的结果
                final_preds[mask_need_B] = preds_B
                
            # 收集结果用于评估
            all_preds.append(final_preds.cpu())
            all_targets.append(targets.cpu())
            
            batch_start = batch_idx * cfg['batch_size']
            batch_end = batch_start + images.size(0)
            batch_names = test_dataset.file_list[batch_start:batch_end]
            all_names.extend(batch_names)
            
            # 收集日志信息
            for i in range(len(targets)):
                routing_records.append({
                    'Sample_Name': batch_names[i],
                    'True_Days': targets[i].item(),
                    'Pred_A': preds_A[i].item(),
                    'Final_Pred': final_preds[i].item(),
                    'Triggered_B': mask_need_B[i].item(),
                    'Abs_Error': abs(targets[i].item() - final_preds[i].item())
                })

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # --- 3. 打印和保存结果 ---
    print("\n" + "="*40)
    print(" 🏆 级联网络 (A+B) 测试报告 🏆")
    print("="*40)
    
    # 打印全局指标
    print(f"Test MAE: {mae(all_preds, all_targets):.3f}")
    print(f"Test RMSE: {rmse(all_preds, all_targets):.3f}")
    
    # 打印触发次数统计
    df_routing = pd.DataFrame(routing_records)
    triggered_count = df_routing['Triggered_B'].sum()
    print(f"[INFO] 专科医生(Model B) 共被呼叫了 {triggered_count} 次")

    class_labels = ['21-24w','25-28w','29-30w','31-32w','33-34w','35-36w','37-40w','Other']
    true_classes = [week_to_class(day_to_week(t)) for t in all_targets]
    pred_classes = [week_to_class(day_to_week(p)) for p in all_preds]

    # 直接调用你原有的极其完善的评估体系
    print_class_distribution(class_labels, true_classes, "True Class Distribution")
    print_class_distribution(class_labels, pred_classes, "Predicted Class Distribution")
    matrix = print_confusion_matrix(class_labels, true_classes, pred_classes)
    print_classification_metrics(class_labels, matrix)

    # 创建保存目录
    result_dir = datetime.datetime.now().strftime('results/cascade_%Y%m%d_%H%M%S')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存你最喜欢的散点图和追踪图
    save_scatter_plot(all_targets, all_preds, true_classes, class_labels, result_dir)
    save_tracking_plot(all_targets, all_preds, result_dir)
    
    # 【新增】：保存极其宝贵的详细预测日志 (用于后续画 B-A图和箱线图)
    csv_path = os.path.join(result_dir, 'cascade_detailed_predictions.csv')
    df_routing.to_csv(csv_path, index=False)
    print(f"[INFO] 详细预测和路由记录已保存至: {csv_path}")
    print(f"[INFO] 所有图表已保存至: {result_dir}")

if __name__ == '__main__':
    main()