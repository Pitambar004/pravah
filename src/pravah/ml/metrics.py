"""Pixel confusion metrics for binary water labels, excluding invalid truth pixels."""
import numpy as np


def confusion(truth, prediction, ignore_index: int = -1) -> dict[str, int]:
    y = np.asarray(truth)
    p = np.asarray(prediction)
    if y.shape != p.shape:
        raise ValueError("Mismatched shapes")
    valid = (y != ignore_index)
    if not np.all(np.isin(y[valid], (0, 1))):
        raise ValueError("Ground truth contains unsupported classes")
    if not np.all(np.isin(p[valid], (0, 1))):
        raise ValueError("Prediction has unsupported classes in valid area")
    return {"tp": int(np.sum((y == 1) & (p == 1) & valid)),
            "tn": int(np.sum((y == 0) & (p == 0) & valid)),
            "fp": int(np.sum((y == 0) & (p == 1) & valid)),
            "fn": int(np.sum((y == 1) & (p == 0) & valid)),
            "valid": int(valid.sum())}


def score(counts: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    if counts.get("valid", 0) == 0:
        return {"precision": None, "recall": None, "f1": None, "iou": None}
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
