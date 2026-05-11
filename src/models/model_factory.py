from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .dinov2_ssl import build_ssl_dinov2
from .inception import build_inception_v3

ModelBuilder = Callable[..., nn.Module]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder) -> None:
    key = name.lower()
    _MODEL_REGISTRY[key] = builder


def _register_defaults() -> None:
    if "inception_v3" not in _MODEL_REGISTRY:
        register_model("inception_v3", build_inception_v3)
    if "ssl_dinov2" not in _MODEL_REGISTRY:
        register_model("ssl_dinov2", build_ssl_dinov2)


def build_model(model_name: str, num_classes: int | None = None, **kwargs: Any) -> nn.Module:
    _register_defaults()
    key = model_name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"未注册的模型: {model_name}")

    if key == "inception_v3":
        if num_classes is None:
            raise ValueError("inception_v3 需要 num_classes")
        return _MODEL_REGISTRY[key](num_classes=num_classes, **kwargs)

    return _MODEL_REGISTRY[key](**kwargs)
