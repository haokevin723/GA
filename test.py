import os
import yaml
import torch
import argparse
import numpy as np
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
        ('21-24w', 147, 168),
        ('25-28w', 175, 196),
        ('29-30w', 203, 216),
        ('31-32w', 217, 230),
        ('33-34w', 231, 244),
        ('35-36w', 245, 258),
        ('37-40w', 259, 280)
    ]
    for i, (label, start, end) in enumerate(intervals):
        # 左闭右开，最后一个区间右闭
        if i < len(intervals) - 1:
            if start <= pred < intervals[i+1][1]:
                return label, start, end
        else:
            if start <= pred <= end:
                return label, start, end
    return 'Other', None, None

def print_class_distribution(class_labels, classes, title):
    print(f"\n[{title}]")
    for label in class_labels:
        print(f"{label}: {classes.count(label)}")

def print_confusion_matrix(class_labels, true_classes, pred_classes):
    print("\n[Confusion Matrix]")
    matrix = defaultdict(lambda: Counter())
    for t, p in zip(true_classes, pred_classes):
        matrix[t][p] += 1
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

def print_sample_results(all_names, all_targets, all_preds, week_to_class, week_and_day, pred_to_interval):
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
        print(f"{name}\tTrue: {target:.1f} ({true_week}w{true_day}d, {week_to_class(true_week)})\tPred: {pred:.1f} ({pred_week}w{pred_day}d, {pred_class})")

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
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        print(f"[INFO] 使用 SimHei 字体: {simhei_path}")
    else:
        my_font = None
        print("[WARNING] 未找到 SimHei 字体，中文可能无法正常显示。请在系统中安装 simhei.ttf。")
    color_map = {
        '21-24w': 'tab:blue',
        '25-28w': 'tab:orange',
        '29-30w': 'tab:green',
        '31-32w': 'tab:red',
        '33-34w': 'tab:purple',
        '35-36w': 'tab:brown',
        '37-40w': 'tab:pink',
        'Other': 'gray'
    }
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
        plt.title('Regression Scatter Plot by Gestational Age', fontproperties=my_font)
        plt.legend(prop=my_font)
    else:
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title('Regression Scatter Plot by Gestational Age')
        plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(result_dir, 'scatter_plot.png')
    plt.savefig(scatter_path, dpi=200)
    print(f'Scatter plot saved as {scatter_path}')

def week_to_class(week):
    if 21 <= week <= 24:
        return '21-24week'
    elif 25 <= week <= 28:
        return '25-28week'
    elif 29 <= week <= 30:
        return '29-30week'
    elif 31 <= week <= 32:
        return '31-32week'
    elif 33 <= week <= 34:
        return '33-34week'
    elif 35 <= week <= 36:
        return '35-36week'
    elif 37 <= week <= 40:
        return '37-40week'
    else:
        return 'other'
    
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


    class_labels = ['21-24w','25-28w','29-30w','31-32w','33-34w','35-36w','37-40w','Other']
    true_classes = [week_to_class(day_to_week(t)) for t in all_targets]
    pred_classes = [week_to_class(day_to_week(p)) for p in all_preds]

    print_class_distribution(class_labels, true_classes, "True Class Distribution")
    print_class_distribution(class_labels, pred_classes, "Predicted Class Distribution")
    matrix = print_confusion_matrix(class_labels, true_classes, pred_classes)
    print_classification_metrics(class_labels, matrix)
    print_sample_results(all_names, all_targets, all_preds, week_to_class, week_and_day, pred_to_interval)

    result_dir = datetime.datetime.now().strftime('results/%Y%m%d_%H%M%S')
    os.makedirs(result_dir, exist_ok=True)
    save_scatter_plot(all_targets, all_preds, true_classes, class_labels, result_dir)

if __name__ == '__main__':
    main()
