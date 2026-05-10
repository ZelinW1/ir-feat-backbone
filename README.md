# FLIR 红外特征主干训练工程（InceptionV3）

本项目用于在 **FLIR COCO 格式红外数据集** 上进行整图多标签监督学习，微调 ImageNet 预训练 InceptionV3，并导出可复用的 2048 维特征提取主干权重。

## 1. 功能概述
- 支持 FLIR COCO 标注解析（按图像聚合多标签，生成 multi-hot）
- 红外灰度图自动转 3 通道 RGB
- Letterbox Resize 到 `299x299`（黑边填充，保持宽高比）
- 基于 `BCEWithLogitsLoss` 的多标签训练
- 验证指标：Macro-mAP（主指标）与 Macro-F1
- 差分学习率：backbone 小学习率、分类头大学习率
- 导出移除分类头后的纯特征提取主干权重

## 2. 项目结构
```text
project_root/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── default_inception_v3.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── flir_dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_factory.py
│   │   └── inception.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── utils/
│       ├── logger.py
│       └── checkpoint.py
├── train.py
└── export_feature.py
```

## 3. 数据准备
默认使用**独立 train/val COCO JSON**：
- `data.train_json`
- `data.val_json`
- `data.image_root`

请在 `configs/default_inception_v3.yaml` 中配置你的本地路径。

## 4. 安装依赖
```bash
pip install -r requirements.txt
```

## 5. 训练
```bash
python train.py --config configs/default_inception_v3.yaml
```

输出内容：
- 日志：`outputs/train.log`
- TensorBoard：`outputs/runs`
- Checkpoint：`outputs/checkpoints/best_model.pth` 与 `last_model.pth`

## 6. 导出特征主干权重
```bash
python export_feature.py \
  --config configs/default_inception_v3.yaml \
  --checkpoint outputs/checkpoints/best_model.pth \
  --out outputs/feature_backbone_inceptionv3.pth
```

导出后模型 `fc=Identity`，前向输出维度为 `2048`。

## 7. 可扩展性设计
- **Model Factory**：通过 `build_model`/注册表扩展 ResNet、ViT 或 SSL Backbone。
- **Dataset/Transform 解耦**：可替换为其他红外数据源与增强策略。
- **Trainer 模块化**：可平滑扩展到多任务训练或自监督预训练流程。
