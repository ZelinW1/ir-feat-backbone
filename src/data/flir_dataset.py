from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class FLIRMultiLabelDataset(Dataset[tuple[Tensor, Tensor, int]]):
    """FLIR COCO格式多标签整图数据集。"""

    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        transform: Callable[[Image.Image], Tensor] | None = None,
        category_id_to_index: dict[int, int] | None = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.transform = transform

        with self.annotation_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        self.image_id_to_file: dict[int, str] = {
            int(img["id"]): str(img["file_name"]) for img in images
        }

        sorted_category_ids = sorted(int(cat["id"]) for cat in categories)
        if category_id_to_index is None:
            self.category_id_to_index = {
                cat_id: idx for idx, cat_id in enumerate(sorted_category_ids)
            }
        else:
            self.category_id_to_index = category_id_to_index

        self.num_classes = len(self.category_id_to_index)

        image_to_label_ids: dict[int, set[int]] = {
            image_id: set() for image_id in self.image_id_to_file
        }
        for ann in annotations:
            image_id = int(ann["image_id"])
            cat_id = int(ann["category_id"])
            if image_id in image_to_label_ids and cat_id in self.category_id_to_index:
                image_to_label_ids[image_id].add(cat_id)

        self.samples: list[tuple[int, str, Tensor]] = []
        for image_id, file_name in self.image_id_to_file.items():
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for cat_id in image_to_label_ids.get(image_id, set()):
                cls_idx = self.category_id_to_index[cat_id]
                target[cls_idx] = 1.0
            self.samples.append((image_id, file_name, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        image_id, file_name, target = self.samples[index]
        image_path = self.image_root / file_name

        with Image.open(image_path) as image:
            image = image.convert("L")
            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                arr = np.array(image, dtype=np.float32) / 255.0
                image_tensor = torch.from_numpy(arr).unsqueeze(0)

        return image_tensor, target.clone(), image_id
