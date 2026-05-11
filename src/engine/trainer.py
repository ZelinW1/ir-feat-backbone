from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import compute_macro_f1, compute_macro_map, compute_per_class_ap


class Trainer:
    """标准训练器，封装训练与验证流程。"""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        writer: SummaryWriter,
        threshold: float = 0.5,
        use_amp: bool = False,
        print_freq: int = 20,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
        self.threshold = threshold
        self.use_amp = use_amp and device.type == "cuda"
        self.print_freq = print_freq
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, dataloader: DataLoader[tuple[Tensor, Tensor, int]], epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        targets_all: list[np.ndarray] = []
        probs_all: list[np.ndarray] = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train {epoch}")
        for step, (images, targets, _) in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.item())

            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())

            if step % self.print_freq == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        targets_np = np.concatenate(targets_all, axis=0)
        probs_np = np.concatenate(probs_all, axis=0)

        avg_loss = running_loss / max(1, len(dataloader))
        macro_map = compute_macro_map(targets_np, probs_np)
        macro_f1 = compute_macro_f1(targets_np, probs_np, threshold=self.threshold)

        self.writer.add_scalar("train/loss", avg_loss, epoch)
        self.writer.add_scalar("train/macro_map", macro_map, epoch)
        self.writer.add_scalar("train/macro_f1", macro_f1, epoch)
        self.writer.add_scalar("train/lr_backbone", self.optimizer.param_groups[0]["lr"], epoch)
        if len(self.optimizer.param_groups) > 1:
            self.writer.add_scalar("train/lr_head", self.optimizer.param_groups[1]["lr"], epoch)

        return {"loss": avg_loss, "macro_map": macro_map, "macro_f1": macro_f1}

    @torch.no_grad()
    def val_epoch(self, dataloader: DataLoader[tuple[Tensor, Tensor, int]], epoch: int) -> dict[str, object]:
        self.model.eval()
        running_loss = 0.0
        targets_all: list[np.ndarray] = []
        probs_all: list[np.ndarray] = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Val {epoch}")
        for _, (images, targets, _) in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            running_loss += float(loss.item())
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            targets_all.append(targets.cpu().numpy())

        targets_np = np.concatenate(targets_all, axis=0)
        probs_np = np.concatenate(probs_all, axis=0)

        avg_loss = running_loss / max(1, len(dataloader))
        macro_map = compute_macro_map(targets_np, probs_np)
        macro_f1 = compute_macro_f1(targets_np, probs_np, threshold=self.threshold)
        per_class_ap = compute_per_class_ap(targets_np, probs_np)

        self.writer.add_scalar("val/loss", avg_loss, epoch)
        self.writer.add_scalar("val/macro_map", macro_map, epoch)
        self.writer.add_scalar("val/macro_f1", macro_f1, epoch)
        for cls_idx, ap in enumerate(per_class_ap):
            if not np.isnan(ap):
                self.writer.add_scalar(f"val/ap_class_{cls_idx}", float(ap), epoch)

        return {
            "loss": avg_loss,
            "macro_map": macro_map,
            "macro_f1": macro_f1,
            "per_class_ap": per_class_ap,
        }
