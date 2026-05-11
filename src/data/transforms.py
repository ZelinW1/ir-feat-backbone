from __future__ import annotations

from typing import Callable

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms


class GrayToRGB:
    """将单通道灰度图复制为三通道RGB。"""

    def __call__(self, image: Image.Image) -> Image.Image:
        # 更稳健且语义明确的做法：直接转换到RGB模式（会复制单通道到三通道）
        if image.mode != "RGB":
            return image.convert("RGB")
        return image


class LetterboxResize:
    """保持宽高比缩放并黑边填充到目标尺寸。"""

    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError("输入图像尺寸非法，宽高必须大于0。")

        scale = min(self.size / src_w, self.size / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        resized = image.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return ImageOps.expand(resized, border=(left, top, right, bottom), fill=self.fill)


def build_transforms(img_size: int, is_train: bool) -> Callable[[Image.Image], Tensor]:
    """构建训练/验证预处理。"""

    _ = is_train
    return transforms.Compose(
        [
            GrayToRGB(),
            LetterboxResize(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
