from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import FLIRUnlabeledDataset, build_ssl_transform
from src.engine import SSLTrainer
from src.models import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLIR DINOv2 SSL训练")
    parser.add_argument("--config", type=str, default="configs/default_ssl_dinov2.yaml")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            p.requires_grad = trainable
        else:
            p.requires_grad = True


def build_ssl_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
    freeze_backbone: bool,
) -> AdamW:
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups: list[dict[str, Any]] = []
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": float(config["optim"]["backbone_lr"])}
        )
    param_groups.append({"params": head_params, "lr": float(config["optim"]["head_lr"])})

    return AdamW(param_groups, weight_decay=float(config["optim"]["weight_decay"]))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "train_ssl.log")

    set_seed(int(config["project"]["seed"]))

    device_str = str(config["project"]["device"])
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    transform = build_ssl_transform(crop_size=int(config["data"]["crop_size"]))
    dataset = FLIRUnlabeledDataset(
        image_root=config["data"]["image_root"],
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"].get("pin_memory", True)),
        drop_last=True,
    )
    logger.info("SSL样本数: %d", len(dataset))

    model = build_model(
        model_name=str(config["model"]["name"]),
        projection_dim=int(config["model"]["projection_dim"]),
        projection_hidden_dim=int(config["model"]["projection_hidden_dim"]),
        prediction_hidden_dim=int(config["model"]["prediction_hidden_dim"]),
    ).to(device)

    freeze_backbone_epochs = int(config["train"].get("freeze_backbone_epochs", 5))
    freeze_backbone = freeze_backbone_epochs > 0
    set_backbone_trainable(model, trainable=not freeze_backbone)
    optimizer = build_ssl_optimizer(model=model, config=config, freeze_backbone=freeze_backbone)
    if freeze_backbone:
        logger.info("前 %d 个epoch冻结backbone，仅训练projection/prediction头。", freeze_backbone_epochs)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["train"]["epochs"])),
        eta_min=float(config["scheduler"]["min_lr"]),
    )

    writer = SummaryWriter(log_dir=str(config["logging"]["log_dir"]))
    trainer = SSLTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        writer=writer,
        use_amp=bool(config["train"]["amp"]),
        print_freq=int(config["logging"]["print_freq"]),
    )

    ckpt_dir = Path(config["checkpoint"]["dir"])
    best_path = ckpt_dir / str(config["checkpoint"]["best_name"])
    last_path = ckpt_dir / str(config["checkpoint"]["last_name"])

    best_loss = float("inf")
    epochs = int(config["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        if freeze_backbone and epoch == freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_ssl_optimizer(model=model, config=config, freeze_backbone=False)
            trainer.optimizer = optimizer
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - freeze_backbone_epochs),
                eta_min=float(config["scheduler"]["min_lr"]),
            )
            logger.info(
                "从Epoch %d开始解冻backbone，进入低学习率全局微调(backbone_lr=%.2e)。",
                epoch,
                float(config["optim"]["backbone_lr"]),
            )

        train_metrics = trainer.train_epoch(dataloader, epoch)
        scheduler.step()

        ssl_loss = float(train_metrics["ssl_loss"])
        feat_std = float(train_metrics["feature_std"])
        logger.info("Epoch %d | SSL_Loss=%.6f | FeatureStd=%.6f", epoch, ssl_loss, feat_std)

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=-best_loss,
            config=config,
        )

        if ssl_loss < best_loss:
            best_loss = ssl_loss
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=-best_loss,
                config=config,
            )
            logger.info("已更新最佳SSL模型, loss=%.6f", best_loss)

    writer.close()
    logger.info("SSL训练完成。最佳loss: %.6f", best_loss)


if __name__ == "__main__":
    main()
