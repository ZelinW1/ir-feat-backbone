from __future__ import annotations

from typing import Callable

from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms


class GrayToRGB:
    """将灰度红外图安全复制为3通道RGB。"""

    def __call__(self, image: Image.Image) -> Image.Image:
        # 使用更稳健的转换：直接转换到RGB模式（PIL会在内部复制单通道到三通道）
        if image.mode != "RGB":
            return image.convert("RGB")
        return image


class SSLTransform:
    """同一输入图像生成两个不同增强视角。"""

    def __init__(self, crop_size: int = 224) -> None:
        common = [
            GrayToRGB(),
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.3, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=8.0,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                fill=(0, 0, 0),
            ),
        ]
        self.view1 = transforms.Compose(
            [
                *common,
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.view2 = transforms.Compose(
            [
                *common,
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 2.0))],
                    p=0.7,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> tuple[Tensor, Tensor]:
        view_1 = self.view1(image)
        view_2 = self.view2(image)
        return view_1, view_2


def build_ssl_transform(crop_size: int = 224) -> Callable[[Image.Image], tuple[Tensor, Tensor]]:
    return SSLTransform(crop_size=crop_size)
