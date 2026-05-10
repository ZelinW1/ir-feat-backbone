from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from src.models import build_model
from src.models.inception import get_feature_dim
from src.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出InceptionV3特征提取主干权重")
    parser.add_argument("--config", type=str, default="configs/default_inception_v3.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="最佳模型checkpoint路径")
    parser.add_argument("--out", type=str, required=True, help="导出特征权重路径(.pth)")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cpu")

    ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
    cat_map = ckpt_raw.get("config", {}).get("category_id_to_index")
    if cat_map is None:
        raise ValueError("checkpoint中未找到category_id_to_index，无法确定类别数。")

    num_classes = len(cat_map)
    model = build_model(
        model_name=str(config["model"]["name"]),
        num_classes=num_classes,
        pretrained=str(config["model"]["pretrained"]),
        aux_logits=bool(config["model"]["aux_logits"]),
    ).to(device)

    load_checkpoint(args.checkpoint, model=model, map_location=device)

    # 移除分类头，保留2048维特征输出
    model.fc = nn.Identity()
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, int(config["data"]["img_size"]), int(config["data"]["img_size"]))
        feat = model(dummy)
    if feat.ndim != 2 or feat.shape[1] != get_feature_dim():
        raise RuntimeError(f"导出失败，特征维度异常: {tuple(feat.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": get_feature_dim(),
            "backbone": str(config["model"]["name"]),
        },
        out_path,
    )
    print(f"已导出特征主干权重: {out_path}")


if __name__ == "__main__":
    main()
