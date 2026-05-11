from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def compute_per_class_ap(targets: np.ndarray, probs: np.ndarray) -> list[float]:
    """计算每个类别的AP，若该类AP无定义则返回nan。"""

    num_classes = targets.shape[1]
    ap_values: list[float] = []
    for cls_idx in range(num_classes):
        y_true = targets[:, cls_idx]
        y_score = probs[:, cls_idx]
        if np.unique(y_true).size < 2:
            ap_values.append(float("nan"))
            continue
        ap_values.append(float(average_precision_score(y_true, y_score)))
    return ap_values


def compute_macro_map(targets: np.ndarray, probs: np.ndarray) -> float:
    """计算多标签宏平均mAP。"""
    per_class_ap = compute_per_class_ap(targets, probs)
    ap_values = [ap for ap in per_class_ap if not np.isnan(ap)]

    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))


def compute_macro_f1(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    """计算多标签宏平均F1。"""

    preds = (probs >= threshold).astype(np.int32)
    return float(f1_score(targets, preds, average="macro", zero_division=0))
