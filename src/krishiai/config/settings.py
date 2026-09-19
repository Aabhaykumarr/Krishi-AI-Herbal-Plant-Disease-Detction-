"""Central configuration loader (Section 79 — don't scatter config values
across multiple files).

Loads configs/base.yaml and lets environment variables (see .env.example)
override the handful of values that commonly change between machines
(device, data root, log level). There is exactly one Settings loader — do
not create a second configuration system for a later phase (Section 97).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Settings:
    data_root: Path
    device: str
    seed: int
    log_level: str

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        config_path = config_path or (REPO_ROOT / "configs" / "base.yaml")
        raw: dict = {}
        if yaml is not None and config_path.exists():
            raw = yaml.safe_load(config_path.read_text()) or {}

        paths = raw.get("paths", {})
        hardware = raw.get("hardware", {})
        repro = raw.get("reproducibility", {})
        logging_cfg = raw.get("logging", {})

        return cls(
            data_root=Path(os.environ.get("DATA_ROOT", paths.get("data_root", "data"))),
            device=os.environ.get("MODEL_DEVICE", hardware.get("device", "auto")),
            seed=int(os.environ.get("SEED", repro.get("seed", 42))),
            log_level=os.environ.get("LOG_LEVEL", logging_cfg.get("level", "INFO")),
        )
