"""Model evaluation engine for detection and classification."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics import YOLO

from krishiai.training.trainer import get_optimal_device
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


class KrishiEvaluator:
    """Canonical evaluator for KrishiAI YOLO26 models."""

    def __init__(self, device: str = "auto"):
        self.device = get_optimal_device(device)

    def evaluate_detection(
        self,
        model_path: Union[str, Path],
        data_yaml: Union[str, Path] = "data/final/detection/data.yaml",
        split: str = "val",
        imgsz: int = 640,
        batch: int = 16,
        conf: float = 0.25,
        iou: float = 0.6,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs evaluation on YOLO detection dataset and returns structured metrics."""
        model = YOLO(str(model_path))
        logger.info(
            f"Evaluating Detection Model: {model_path} | Split: {split} | Device: {self.device}"
        )

        results = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            device=self.device,
            plots=True,
            save_json=False,
        )

        # Extract metrics safely from Ultralytics results
        box_metrics = results.box
        map50 = float(box_metrics.map50)
        map50_95 = float(box_metrics.map)
        precision = float(box_metrics.mp)
        recall = float(box_metrics.mr)

        # Per-class metrics
        per_class: Dict[str, Dict[str, float]] = {}
        names = results.names
        if hasattr(box_metrics, "maps") and box_metrics.maps is not None:
            for cid, cls_map in enumerate(box_metrics.maps):
                cname = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
                per_class[cname] = {
                    "mAP50_95": round(float(cls_map), 4),
                }

        eval_summary = {
            "task": "detection",
            "model_path": str(model_path),
            "data_yaml": str(data_yaml),
            "split": split,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": self.device,
            "metrics": {
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "per_class": per_class,
            },
        }

        # Save report
        out_root = Path(output_dir) if output_dir else Path("experiments/eval/detection")
        out_root.mkdir(parents=True, exist_ok=True)
        report_path = out_root / f"eval_{split}_{Path(model_path).stem}.json"
        report_path.write_text(json.dumps(eval_summary, indent=2))
        logger.info(f"Detection evaluation report saved -> {report_path}")

        return eval_summary

    def evaluate_classification(
        self,
        model_path: Union[str, Path],
        data_dir: Union[str, Path] = "data/final/classification",
        split: str = "val",
        imgsz: int = 224,
        batch: int = 32,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs evaluation on YOLO classification dataset and returns structured metrics."""
        model = YOLO(str(model_path))
        logger.info(
            f"Evaluating Classification Model: {model_path} | Split: {split} | Device: {self.device}"
        )

        results = model.val(
            data=str(data_dir),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            plots=True,
        )

        top1 = float(results.top1)
        top5 = float(results.top5)

        eval_summary = {
            "task": "classification",
            "model_path": str(model_path),
            "data_dir": str(data_dir),
            "split": split,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": self.device,
            "metrics": {
                "top1_accuracy": round(top1, 4),
                "top5_accuracy": round(top5, 4),
            },
        }

        # Save report
        out_root = Path(output_dir) if output_dir else Path("experiments/eval/classification")
        out_root.mkdir(parents=True, exist_ok=True)
        report_path = out_root / f"eval_{split}_{Path(model_path).stem}.json"
        report_path.write_text(json.dumps(eval_summary, indent=2))
        logger.info(f"Classification evaluation report saved -> {report_path}")

        return eval_summary
