#!/usr/bin/env python3
"""KrishiAI Model Evaluation Entry Point.

Usage:
    python scripts/evaluate.py --task detection --model models/pretrained/yolo26n.pt --split test
    python scripts/evaluate.py --task classification --model models/pretrained/yolo26n-cls.pt --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from krishiai.evaluation.evaluator import KrishiEvaluator  # noqa: E402
from krishiai.utils.logging import get_logger  # noqa: E402

logger = get_logger("krishiai.evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["detect", "detection", "classify", "classification"],
        default="detection",
        help="Evaluation task type",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml (detection) or data directory (classification)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "train"],
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cpu, 0, auto)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--out", type=str, default=None, help="Output directory for reports")
    args = parser.parse_args()

    task = "detect" if args.task in {"detect", "detection"} else "classify"
    evaluator = KrishiEvaluator(device=args.device)

    if task == "detect":
        data_yaml = args.data or str(REPO_ROOT / "data" / "final" / "detection" / "data.yaml")
        imgsz = args.imgsz or 640
        summary = evaluator.evaluate_detection(
            model_path=args.model,
            data_yaml=data_yaml,
            split=args.split,
            imgsz=imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            output_dir=args.out,
        )
        print("\n" + "=" * 60)
        print(f"KrishiAI Detection Evaluation Results ({args.split.upper()} split)")
        print(f"Model: {args.model}")
        print(f"mAP50:     {summary['metrics']['mAP50']:.4f}")
        print(f"mAP50-95:  {summary['metrics']['mAP50_95']:.4f}")
        print(f"Precision: {summary['metrics']['precision']:.4f}")
        print(f"Recall:    {summary['metrics']['recall']:.4f}")
        print("=" * 60)
    else:
        data_dir = args.data or str(REPO_ROOT / "data" / "final" / "classification")
        imgsz = args.imgsz or 224
        summary = evaluator.evaluate_classification(
            model_path=args.model,
            data_dir=data_dir,
            split=args.split,
            imgsz=imgsz,
            batch=args.batch,
            output_dir=args.out,
        )
        print("\n" + "=" * 60)
        print(f"KrishiAI Classification Evaluation Results ({args.split.upper()} split)")
        print(f"Model: {args.model}")
        print(f"Top-1 Accuracy: {summary['metrics']['top1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {summary['metrics']['top5_accuracy']:.4f}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
