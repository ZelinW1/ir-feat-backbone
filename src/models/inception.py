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

    weights = Inception_V3_Weights[pretrained]
    model = inception_v3(weights=weights, aux_logits=aux_logits)
    model.fc = nn.Linear(FEATURE_DIM, num_classes)
    return model


def get_feature_dim() -> int:
    """返回InceptionV3特征维度。"""

    return FEATURE_DIM
