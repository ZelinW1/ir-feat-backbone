from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class FLIRUnlabeledDataset(Dataset[tuple[Tensor, Tensor]]):
    """FLIR无标签数据集：递归扫描图像文件并返回同图双视角。"""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, image_root: str | Path, transform: Callable[[Image.Image], tuple[Tensor, Tensor]]) -> None:
        self.image_root = Path(image_root)
        self.transform = transform

        if not self.image_root.exists():
            raise FileNotFoundError(f"图像根目录不存在: {self.image_root}")

        self.image_paths = sorted(
            p for p in self.image_root.rglob("*") if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        )
        if not self.image_paths:
            raise RuntimeError(f"未在目录中找到图像: {self.image_root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            view_1, view_2 = self.transform(image)
        return view_1, view_2
