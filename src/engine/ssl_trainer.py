from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.dinov2_ssl import negative_cosine_similarity


class SSLTrainer:
    """自监督训练器（SimSiam-style）。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        writer: SummaryWriter,
        use_amp: bool = False,
        print_freq: int = 20,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
        self.use_amp = use_amp and device.type == "cuda"
        self.print_freq = print_freq
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _feature_collapse_std(self, feats: Tensor) -> float:
        # 监控坍缩：标准化后按维度计算std，再取均值
        normed = torch.nn.functional.normalize(feats, dim=1)
        return float(normed.std(dim=0).mean().item())

    def train_epoch(self, dataloader: DataLoader[tuple[Tensor, Tensor]], epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_collapse_std = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"SSL Train {epoch}")
        for step, (x1, x2) in pbar:
            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(x1, x2)
                # SimSiam核心：目标分支必须stop-grad，防止表示坍缩到平凡解
                loss = 0.5 * (
                    negative_cosine_similarity(out["p1"], out["z2"]) +
                    negative_cosine_similarity(out["p2"], out["z1"])
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.item())
            collapse_std = self._feature_collapse_std(out["z1"].detach())
            running_collapse_std += collapse_std

            if step % self.print_freq == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", feat_std=f"{collapse_std:.4f}")

        avg_loss = running_loss / max(1, len(dataloader))
        avg_collapse_std = running_collapse_std / max(1, len(dataloader))

        self.writer.add_scalar("ssl/loss", avg_loss, epoch)
        self.writer.add_scalar("ssl/feature_std", avg_collapse_std, epoch)
        self.writer.add_scalar("ssl/lr_backbone", self.optimizer.param_groups[0]["lr"], epoch)
        if len(self.optimizer.param_groups) > 1:
            self.writer.add_scalar("ssl/lr_head", self.optimizer.param_groups[1]["lr"], epoch)

        return {"ssl_loss": avg_loss, "feature_std": avg_collapse_std}
