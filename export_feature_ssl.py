from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models import build_model
from src.models.dinov2_ssl import DINOv2BackboneWrapper, FEATURE_DIM
from src.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出SSL训练后的DINOv2主干")
    parser.add_argument("--config", type=str, default="configs/default_ssl_dinov2.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model = build_model(
        model_name=str(config["model"]["name"]),
        projection_dim=int(config["model"]["projection_dim"]),
        projection_hidden_dim=int(config["model"]["projection_hidden_dim"]),
        prediction_hidden_dim=int(config["model"]["prediction_hidden_dim"]),
    )
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")

    backbone_only = DINOv2BackboneWrapper(model.backbone)
    backbone_only.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        feat = backbone_only(dummy)
    assert tuple(feat.shape) == (1, FEATURE_DIM), f"特征维度错误: {tuple(feat.shape)}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": backbone_only.state_dict(),
            "feature_dim": FEATURE_DIM,
            "backbone": "facebook/dinov2-base",
        },
        out_path,
    )
    print(f"已导出SSL特征主干: {out_path}")


if __name__ == "__main__":
    main()
