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

# 这个 RefineDataset 是为了给 Model B 筛选出更困难的样本，专注于 31-36 周（217-258 天）的样本
class RefineDataset(Dataset):
    def __init__(self, original_dataset, min_days=217, max_days=258):
        self.original_dataset = original_dataset
        self.indices = []
        print(f"[Model B] 正在筛选 {min_days}-{max_days} 天的困难样本...")
        for idx in range(len(original_dataset)):
            # 这里是获取真实天数的逻辑，可能需要根据你原来的 __getitem__ 怎么写的来调整
            _, target = original_dataset[idx] 
            if min_days <= target <= max_days:
                self.indices.append(idx)
        print(f"[Model B] 筛选完成！可用样本数: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]