from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models import Inception_V3_Weights, inception_v3

FEATURE_DIM: int = 2048


def build_inception_v3(
    num_classes: int,
    pretrained: str = "IMAGENET1K_V1",
    aux_logits: bool = False,
    **_: Any,
) -> nn.Module:
    """构建InceptionV3多标签分类模型。"""
    # 支持在不使用预训练权重时传入 None 或 空字符串，避免触发权重查找/下载
    weights = None
    if pretrained:
        try:
            weights = Inception_V3_Weights[pretrained]
        except Exception:
            # 如果给定字符串无法从枚举中解析，则不加载权重
            weights = None

    # 关键兼容性：当加载预训练权重时，torchvision 会强制要求 aux_logits=True
    # 因此需先按True构建，再根据配置关闭Aux分支。
    if weights is not None:
        model = inception_v3(weights=weights, aux_logits=True)
    else:
        model = inception_v3(weights=None, aux_logits=aux_logits)
    if not aux_logits and hasattr(model, "AuxLogits"):
        model.AuxLogits = None
        model.aux_logits = False
    model.fc = nn.Linear(FEATURE_DIM, num_classes)
    return model


def get_feature_dim() -> int:
    """返回InceptionV3特征维度。"""

    return FEATURE_DIM
