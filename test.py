import os
import yaml
import torch
import argparse
import numpy as np
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

    print("\n[Sample Results]")
    for name, target, pred in zip(all_names, all_targets, all_preds):
        true_week = day_to_week(target)
        pred_week = day_to_week(pred)
        true_class = week_to_class(true_week)
        pred_class = week_to_class(pred_week)
        print(f"{name}\tTrue: {target:.1f} ({true_week}周, {true_class})\tPred: {pred:.1f} ({pred_week}周, {pred_class})")

if __name__ == '__main__':
    main()
