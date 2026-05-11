from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel

FEATURE_DIM: int = 768


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SelfSupervisedDINOv2(nn.Module):
    """SimSiam范式：transformers DINOv2 backbone + projection/prediction head。"""

    def __init__(
        self,
        projection_dim: int = 2048,
        projection_hidden_dim: int = 2048,
        prediction_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.projection_head = MLPHead(
            in_dim=FEATURE_DIM,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
        )
        self.prediction_head = PredictionHead(
            in_dim=projection_dim,
            hidden_dim=prediction_hidden_dim,
            out_dim=projection_dim,
        )

    def encode(self, x: Tensor) -> Tensor:
        out = self.backbone(pixel_values=x)
        feat: Tensor
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]

        if feat.ndim != 2 or feat.shape[1] != FEATURE_DIM:
            raise RuntimeError(f"DINOv2 backbone输出维度异常: {tuple(feat.shape)}")
        return feat

    def forward(self, x1: Tensor, x2: Tensor) -> dict[str, Tensor]:
        z1 = self.projection_head(self.encode(x1))
        z2 = self.projection_head(self.encode(x2))
        p1 = self.prediction_head(z1)
        p2 = self.prediction_head(z2)
        return {"p1": p1, "p2": p2, "z1": z1, "z2": z2}


class DINOv2BackboneWrapper(nn.Module):
    """统一导出/推理接口：输入pixel_values，输出(batch, 768)特征。"""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: Tensor) -> Tensor:
        out = self.backbone(pixel_values=x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]
        return feat


def negative_cosine_similarity(p: Tensor, z: Tensor) -> Tensor:
    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)
    return -(p * z).sum(dim=1).mean()


def build_ssl_dinov2(**kwargs: Any) -> nn.Module:
    return SelfSupervisedDINOv2(**kwargs)
