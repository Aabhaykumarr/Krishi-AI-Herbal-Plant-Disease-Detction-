import numpy as np
from krishiai.evaluation.metrics import compute_classification_metrics, compute_iou


def test_compute_iou_perfect_overlap():
    box1 = [0.0, 0.0, 1.0, 1.0]
    box2 = [0.0, 0.0, 1.0, 1.0]
    assert compute_iou(box1, box2) == 1.0


def test_compute_iou_no_overlap():
    box1 = [0.0, 0.0, 0.5, 0.5]
    box2 = [0.6, 0.6, 1.0, 1.0]
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_partial_overlap():
    box1 = [0.0, 0.0, 2.0, 2.0]  # area = 4
    box2 = [1.0, 1.0, 3.0, 3.0]  # area = 4, intersection = 1x1 = 1, union = 4+4-1=7
    iou = compute_iou(box1, box2)
    assert round(iou, 4) == round(1.0 / 7.0, 4)


def test_classification_metrics_perfect():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    names = ["cat0", "cat1", "cat2"]
    res = compute_classification_metrics(y_true, y_pred, class_names=names)
    assert res["accuracy"] == 1.0
    assert res["macro_precision"] == 1.0
    assert res["macro_recall"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["sample_count"] == 6


def test_classification_metrics_with_errors():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    res = compute_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.75
    assert res["sample_count"] == 4
    assert len(res["confusion_matrix"]) == 2


def test_classification_metrics_empty():
    res = compute_classification_metrics([], [])
    assert res["sample_count"] == 0
    assert res["accuracy"] == 0.0
