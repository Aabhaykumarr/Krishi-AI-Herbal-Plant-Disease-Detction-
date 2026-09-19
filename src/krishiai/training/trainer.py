"""Training orchestrator for KrishiAI YOLO26 models.

Wraps Ultralytics YOLO26 training pipeline with standardized logging,
hardware configuration, experiment artifact tracking, and checkpointing.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml
from ultralytics import YOLO

from krishiai.config.settings import Settings
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


def get_optimal_device(requested: str = "auto") -> str:
    """Resolves optimal compute device without hardcoding GPU assumptions."""
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req in {"cuda", "gpu"}:
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


class KrishiTrainer:
    """Canonical training service for KrishiAI detection and classification models."""

    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        task: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self.settings = Settings.load()
        self.config_data = self._load_config(config)
        if overrides:
            self.config_data.update(overrides)

        # Infer or validate task
        self.task = task or self.config_data.get("task", "detect")
        self.config_data["task"] = self.task

        # Resolve device
        requested_device = self.config_data.get("device", self.settings.device)
        self.device = get_optimal_device(requested_device)
        self.config_data["device"] = self.device

        # Resolve model path
        model_target = self.config_data.get("model")
        if not model_target:
            model_target = "yolo26n.pt" if self.task == "detect" else "yolo26n-cls.pt"
        self.model_path = str(model_target)
        self.config_data["model"] = self.model_path

        # Model instance
        self.model = YOLO(self.model_path)

    def _load_config(self, config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return dict(config)
        p = Path(config)
        if not p.exists():
            raise FileNotFoundError(f"Training config file not found: {p}")
        return yaml.safe_load(p.read_text()) or {}

    def train(self) -> Dict[str, Any]:
        """Executes model training run and logs structured artifacts."""
        logger.info(
            f"Starting KrishiAI training | Task: {self.task} | Model: {self.model_path} | Device: {self.device}"
        )
        start_time = datetime.now(timezone.utc)

        # Prepare kwargs for YOLO.train()
        train_args = dict(self.config_data)
        # Remove keys that YOLO.train doesn't directly expect as hyperparameters
        train_args.pop("task", None)

        # Execute Ultralytics training loop
        results = self.model.train(**train_args)

        end_time = datetime.now(timezone.utc)
        duration_sec = (end_time - start_time).total_seconds()

        # Save run metadata and copy best checkpoint to models/checkpoints/
        save_dir = Path(getattr(self.model.trainer, "save_dir", "runs/train"))
        project_name = str(self.config_data.get("name", "krishiai_run"))
        checkpoint_dir = Path("models/checkpoints") / project_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"

        saved_best = None
        saved_last = None
        if best_pt.exists():
            saved_best = checkpoint_dir / "best.pt"
            shutil.copy2(best_pt, saved_best)
            logger.info(f"Copied best model checkpoint -> {saved_best}")
        if last_pt.exists():
            saved_last = checkpoint_dir / "last.pt"
            shutil.copy2(last_pt, saved_last)

        summary = {
            "task": self.task,
            "project_name": project_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_sec,
            "device": self.device,
            "save_dir": str(save_dir),
            "best_checkpoint": str(saved_best) if saved_best else None,
            "last_checkpoint": str(saved_last) if saved_last else None,
            "config": self.config_data,
        }

        summary_path = save_dir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Training completed successfully in {duration_sec:.1f}s. Summary: {summary_path}")

        return summary
