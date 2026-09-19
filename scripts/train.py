#!/usr/bin/env python3
"""KrishiAI Model Training Entry Point.

Usage:
    python scripts/train.py --task detection --config configs/training/detection_baseline.yaml
    python scripts/train.py --task classification --config configs/training/classification_baseline.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from krishiai.training.trainer import KrishiTrainer  # noqa: E402
from krishiai.utils.logging import get_logger  # noqa: E402

logger = get_logger("krishiai.train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["detect", "detection", "classify", "classification"],
        default="detection",
        help="Task type: detection or classification",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image resolution")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cpu, 0, auto)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint/weights path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    args = parser.parse_args()

    # Normalize task string
    task = "detect" if args.task in {"detect", "detection"} else "classify"

    # Default config path if not provided
    if not args.config:
        default_config = (
            REPO_ROOT / "configs" / "training" / (
                "detection_baseline.yaml" if task == "detect" else "classification_baseline.yaml"
            )
        )
        config_path = str(default_config)
    else:
        config_path = args.config

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.device is not None:
        overrides["device"] = args.device
    if args.name is not None:
        overrides["name"] = args.name
    if args.model is not None:
        overrides["model"] = args.model
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.patience is not None:
        overrides["patience"] = args.patience

    trainer = KrishiTrainer(config=config_path, task=task, overrides=overrides)
    summary = trainer.train()

    print("\n" + "=" * 60)
    print("KrishiAI Training Completed Successfully!")
    print(f"Project: {summary.get('project_name')}")
    print(f"Duration: {summary.get('duration_seconds'):.1f}s")
    print(f"Best Checkpoint: {summary.get('best_checkpoint')}")
    print(f"Save Directory: {summary.get('save_dir')}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
