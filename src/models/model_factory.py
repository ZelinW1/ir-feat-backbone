from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .inception import build_inception_v3

ModelBuilder = Callable[..., nn.Module]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder) -> None:
    """注册模型构建函数。"""

    key = name.lower()
    _MODEL_REGISTRY[key] = builder


def _register_defaults() -> None:
    if "inception_v3" not in _MODEL_REGISTRY:
        register_model("inception_v3", build_inception_v3)


def build_model(model_name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    """根据模型名称构建模型实例。"""

    _register_defaults()
    key = model_name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"未注册的模型: {model_name}")
    return _MODEL_REGISTRY[key](num_classes=num_classes, **kwargs)
