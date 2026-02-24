import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.regression_dataset import RegressionDataset
from models.model_factory import get_model
from utils.metrics import mae, rmse
from utils.misc import ensure_dir

# 读取配置
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

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
model = model.cuda() if torch.cuda.is_available() else model

model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.cuda() if torch.cuda.is_available() else images
        targets = targets.float().cuda() if torch.cuda.is_available() else targets.float()
        preds = model(images)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

all_preds = torch.cat(all_preds).squeeze().numpy()
all_targets = torch.cat(all_targets).squeeze().numpy()

print(f"Test MAE: {mae(all_preds, all_targets):.3f}")
print(f"Test RMSE: {rmse(all_preds, all_targets):.3f}")
