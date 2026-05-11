from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import FLIRMultiLabelDataset, build_transforms
from src.engine import Trainer
from src.models import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLIR InceptionV3 多标签训练")
    parser.add_argument("--config", type=str, default="configs/default_inception_v3.yaml")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, int, dict[int, int]]:
    data_cfg = config["data"]
    train_tf = build_transforms(img_size=int(data_cfg["img_size"]), is_train=True)
    val_tf = build_transforms(img_size=int(data_cfg["img_size"]), is_train=False)

    train_ds = FLIRMultiLabelDataset(
        annotation_path=data_cfg["train_json"],
        image_root=data_cfg["image_root"],
        transform=train_tf,
    )
    val_ds = FLIRMultiLabelDataset(
        annotation_path=data_cfg["val_json"],
        image_root=data_cfg["image_root"],
        transform=val_tf,
        category_id_to_index=train_ds.category_id_to_index,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    return train_loader, val_loader, train_ds.num_classes, train_ds.category_id_to_index


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """设置backbone参数是否参与训练，分类头始终可训练。"""
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def build_optimizer(model: nn.Module, config: dict[str, Any], freeze_backbone: bool) -> AdamW:
    """按阶段构建优化器：冻结阶段仅训练head，解冻后全局微调。"""
    weight_decay = float(config["optim"]["weight_decay"])
    head_lr = float(config["optim"]["head_lr"])
    finetune_backbone_lr = float(
        config["optim"].get("finetune_backbone_lr", config["optim"]["backbone_lr"])
    )

    head_params = [
        p for n, p in model.named_parameters() if n.startswith("fc.") and p.requires_grad
    ]
    if freeze_backbone:
        return AdamW(
            [{"params": head_params, "lr": head_lr}],
            weight_decay=weight_decay,
        )

    backbone_params = [
        p for n, p in model.named_parameters() if (not n.startswith("fc.")) and p.requires_grad
    ]
    return AdamW(
        [
            {"params": backbone_params, "lr": finetune_backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "train.log")

    set_seed(int(config["project"]["seed"]))

    device_str = str(config["project"]["device"])
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    train_loader, val_loader, num_classes, cat_map = build_dataloaders(config)
    logger.info("类别数: %d", num_classes)

    model = build_model(
        model_name=str(config["model"]["name"]),
        num_classes=num_classes,
        pretrained=str(config["model"]["pretrained"]),
        aux_logits=bool(config["model"]["aux_logits"]),
        dropout_p=float(config["model"].get("dropout_p", 0.5)),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    freeze_backbone_epochs = int(config["train"].get("freeze_backbone_epochs", 0))
    freeze_backbone = freeze_backbone_epochs > 0
    set_backbone_trainable(model, trainable=not freeze_backbone)
    optimizer = build_optimizer(model=model, config=config, freeze_backbone=freeze_backbone)
    scheduler_name = str(config["scheduler"].get("name", "cosine")).lower()
    if scheduler_name != "cosine":
        raise ValueError(f"当前仅支持cosine调度器，收到: {scheduler_name}")
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["train"]["epochs"])),
        eta_min=float(config["scheduler"]["min_lr"]),
    )
    if freeze_backbone:
        logger.info("前 %d 个epoch冻结backbone，仅训练分类头。", freeze_backbone_epochs)

    writer = SummaryWriter(log_dir=str(config["logging"]["log_dir"]))
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        writer=writer,
        threshold=float(config["train"]["threshold"]),
        use_amp=bool(config["train"]["amp"]),
        print_freq=int(config["logging"]["print_freq"]),
    )

    ckpt_dir = Path(config["checkpoint"]["dir"])
    best_path = ckpt_dir / str(config["checkpoint"]["best_name"])
    last_path = ckpt_dir / str(config["checkpoint"]["last_name"])

    best_map = -1.0
    epochs = int(config["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        if freeze_backbone and epoch == freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(model=model, config=config, freeze_backbone=False)
            trainer.optimizer = optimizer
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - freeze_backbone_epochs),
                eta_min=float(config["scheduler"]["min_lr"]),
            )
            logger.info(
                "从Epoch %d开始解冻backbone，全局微调LR(backbone)=%.2e。",
                epoch,
                float(config["optim"].get("finetune_backbone_lr", config["optim"]["backbone_lr"])),
            )

        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.val_epoch(val_loader, epoch)
        scheduler.step()

        logger.info(
            "Epoch %d | Train loss=%.4f mAP=%.4f F1=%.4f | Val loss=%.4f mAP=%.4f F1=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["macro_map"],
            train_metrics["macro_f1"],
            val_metrics["loss"],
            val_metrics["macro_map"],
            val_metrics["macro_f1"],
        )
        if "per_class_ap" in val_metrics:
            per_class_ap = val_metrics["per_class_ap"]
            if isinstance(per_class_ap, list):
                index_to_cat_id = {idx: cat_id for cat_id, idx in cat_map.items()}
                for cls_idx, ap in enumerate(per_class_ap):
                    cat_id = index_to_cat_id.get(cls_idx, cls_idx)
                    if ap != ap:  # NaN
                        logger.info("Val AP | class_idx=%d coco_cat_id=%s ap=nan(样本单一)", cls_idx, cat_id)
                    else:
                        logger.info("Val AP | class_idx=%d coco_cat_id=%s ap=%.4f", cls_idx, cat_id, ap)

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_map,
            config={**config, "category_id_to_index": cat_map},
        )

        if val_metrics["macro_map"] > best_map:
            best_map = val_metrics["macro_map"]
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_map,
                config={**config, "category_id_to_index": cat_map},
            )
            logger.info("已更新最佳模型, mAP=%.4f", best_map)

    writer.close()
    logger.info("训练完成。最佳验证 mAP: %.4f", best_map)


if __name__ == "__main__":
    main()
