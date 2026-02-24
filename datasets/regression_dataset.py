import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, data_dir, label_csv, split_txt, transform=None):
        self.data_dir = data_dir
        self.labels = pd.read_csv(label_csv, index_col='png_name')
        with open(split_txt, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = float(self.labels.loc[img_name, 'gest_week_days'])
        if self.transform:
            image = self.transform(image)
        return image, label
