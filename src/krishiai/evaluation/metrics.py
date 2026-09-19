"""Evaluation metrics for detection and classification models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def compute_iou(
    box1: Sequence[float],
    box2: Sequence[float],
) -> float:
    """Computes Intersection over Union (IoU) for two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Computes multiclass classification metrics (accuracy, precision, recall, macro F1, confusion matrix)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if len(y_t) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "sample_count": 0,
            "per_class": {},
            "confusion_matrix": [],
        }

    accuracy = float(np.mean(y_t == y_p))

    # Distinct classes
    all_classes = sorted(list(set(y_t) | set(y_p)))
    if class_names is not None:
        num_classes = len(class_names)
        all_classes = list(range(num_classes))
    else:
        num_classes = max(all_classes) + 1 if all_classes else 0

    # Build confusion matrix: rows = true, cols = predicted
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for c in all_classes:
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        c_name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
        per_class[c_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": int(np.sum(cm[c, :])),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "sample_count": len(y_t),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
