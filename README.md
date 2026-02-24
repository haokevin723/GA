# image_regression

医学影像孕周回归 PyTorch 工程模板

## 目录结构

```
image_regression/
├── data/
│   ├── images/           # 原始图像
│   ├── labels.csv        # 标签文件
│   └── splits/           # 数据划分
├── datasets/             # 数据集定义
├── models/               # 回归模型
├── losses/               # 损失函数
├── utils/                # 工具函数
├── configs/              # 配置文件
├── checkpoints/          # 模型权重
├── logs/                 # 日志
├── notebooks/            # Jupyter分析
├── scripts/              # 其他脚本
├── train.py              # 训练主脚本
├── test.py               # 测试主脚本
├── requirements.txt      # 依赖
└── README.md             # 项目说明
```

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 配置数据与参数，编辑 `configs/default.yaml`
3. 训练模型：
   ```bash
   python train.py
   ```
4. 测试模型：
   ```bash
   python test.py
   ```

## 说明
- 支持自定义数据集、ResNet/DenseNet回归模型、MAE/RMSE指标。
- 数据输入：图片+csv，train/val/test划分。
- 适用于医学影像孕周回归等回归任务。
