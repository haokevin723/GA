import os
import yaml
import torch
import argparse
import numpy as np
import datetime
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.regression_dataset import RegressionDataset
from models.model_factory import get_model
from utils.metrics import mae, rmse
from utils.misc import ensure_dir


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def day_to_week(day):
    return int(day // 7)

def week_to_class(week):
    if 21 <= week <= 24:
        return '21-24周'
    elif 25 <= week <= 28:
        return '25-28周'
    elif 29 <= week <= 30:
        return '29-30周'
    elif 31 <= week <= 32:
        return '31-32周'
    elif 33 <= week <= 34:
        return '33-34周'
    elif 35 <= week <= 36:
        return '35-36周'
    elif 37 <= week <= 40:
        return '37-40周'
    else:
        return '其它'
    
def main():
        # 1. 评估和打印所有统计信息（保持原有流程）
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = RegressionDataset(cfg['data_dir'], cfg['label_csv'], cfg['test_split'], transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    model = get_model(cfg['model'], backbone=cfg['backbone'], pretrained=False)
    ckpt_path = os.path.join('checkpoints', 'best_model.pth')
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(device)

    model.eval()
    all_preds = []
    all_targets = []
    all_names = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.float().to(device)
            preds = model(images)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            # 获取图片名
            batch_start = batch_idx * cfg['batch_size']
            batch_end = batch_start + images.size(0)
            all_names.extend(test_dataset.file_list[batch_start:batch_end])

    all_preds = torch.cat(all_preds).squeeze().numpy()
    all_targets = torch.cat(all_targets).squeeze().numpy()


    print(f"Test MAE: {mae(all_preds, all_targets):.3f}")
    print(f"Test RMSE: {rmse(all_preds, all_targets):.3f}")

    # 统计类别分布和混淆矩阵
    class_labels = ['21-24周','25-28周','29-30周','31-32周','33-34周','35-36周','37-40周','其它']
    true_classes = [week_to_class(day_to_week(t)) for t in all_targets]
    pred_classes = [week_to_class(day_to_week(p)) for p in all_preds]

    # 真实类别分布
    print("\n[真实类别分布]")
    for label in class_labels:
        print(f"{label}: {true_classes.count(label)}")

    # 预测类别分布
    print("\n[预测类别分布]")
    for label in class_labels:
        print(f"{label}: {pred_classes.count(label)}")


    # 混淆矩阵
    print("\n[混淆矩阵]")
    matrix = defaultdict(lambda: Counter())
    for t, p in zip(true_classes, pred_classes):
        matrix[t][p] += 1
    print("真实类别 -> 预测类别:")
    header = '\t'.join(class_labels)
    print(f"类别\t{header}")
    for t in class_labels:
        row = [str(matrix[t][p]) for p in class_labels]
        print(f"{t}\t" + '\t'.join(row))

    # 计算每个类别的准确率、召回率和F1值
    print("\n[分类指标]")
    for label in class_labels:
        TP = matrix[label][label]
        FP = sum(matrix[other][label] for other in class_labels if other != label)
        FN = sum(matrix[label][other] for other in class_labels if other != label)
        # 支持零分母
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{label}: Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

    # 辅助函数：几周几天
    def week_and_day(day):
        week = int(day // 7)
        days = int(day % 7)
        return week, days

    # 区间映射

    def pred_to_interval(pred):
        intervals = [
            ('21-24周', 147, 168),
            ('25-28周', 175, 196),
            ('29-30周', 203, 216),
            ('31-32周', 217, 230),
            ('33-34周', 231, 244),
            ('35-36周', 245, 258),
            ('37-40周', 259, 280)
        ]
        for i, (label, start, end) in enumerate(intervals):
            # 左闭右开，最后一个区间右闭
            if i < len(intervals) - 1:
                if start <= pred < intervals[i+1][1]:
                    return label, start, end
            else:
                if start <= pred <= end:
                    return label, start, end
        return '其它', None, None

    print("\n[Sample Results]")
    for name, target, pred in zip(all_names, all_targets, all_preds):
        true_week, true_day = week_and_day(target)
        pred_class, start, end = pred_to_interval(pred)
        if start is not None:
            pred_week = int(pred // 7)
            pred_day = int(pred % 7)
        else:
            pred_week = int(pred // 7)
            pred_day = int(pred % 7)
        print(f"{name}\tTrue: {target:.1f} ({true_week}周{true_day}天, {week_to_class(true_week)})\tPred: {pred:.1f} ({pred_week}周{pred_day}天, {pred_class})")

    # 2. 生成以年月日时分秒命名的结果文件夹，并保存散点图

    result_dir = datetime.datetime.now().strftime('results/%Y%m%d_%H%M%S')
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(7,7))
        # 颜色映射
    color_map = {
            '21-24周': 'tab:blue',
            '25-28周': 'tab:orange',
            '29-30周': 'tab:green',
            '31-32周': 'tab:red',
            '33-34周': 'tab:purple',
            '35-36周': 'tab:brown',
            '37-40周': 'tab:pink',
            '其它': 'gray'
    }
    # 按类别分组画点
    for label in class_labels[:-1]:  # 只画7种孕周
        idxs = [i for i, c in enumerate(true_classes) if c == label]
        if idxs:
            plt.scatter(all_targets[idxs], all_preds[idxs], alpha=0.7, label=label, color=color_map[label], edgecolors='none')
    # 画对角线
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Regression Scatter Plot by Gestational Age')
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(result_dir, 'scatter_plot.png')
    plt.savefig(scatter_path, dpi=200)
    print(f'散点图已保存为 {scatter_path}')

if __name__ == '__main__':
    main()
