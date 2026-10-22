"""Central configuration loader (Section 79 — don't scatter config values
across multiple files).

Loads configs/base.yaml and lets environment variables (see .env.example)
override the handful of values that commonly change between machines
(device, data root, log level). There is exactly one Settings loader — do
not create a second configuration system for a later phase (Section 97).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[3]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]


@dataclass
class Settings:
    data_root: Path
    device: str
    seed: int
    log_level: str
    incoming_data_root: Path
    max_upload_size_mb: int
    gemini_api_key: Optional[str] = field(default=None, repr=False)
    gemini_vision_enabled: bool = True
    gemini_model: str = "models/gemini-3.5-flash-lite"
    gemini_timeout_seconds: float = 15.0
    gemini_max_detections: int = 4
    gemini_correction_threshold: float = 0.82
    gemini_max_retries: int = 1

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        # .env is for local development; existing process/Render variables win.
        if load_dotenv is not None:
            load_dotenv(REPO_ROOT / ".env", override=False)

        config_path = config_path or (REPO_ROOT / "configs" / "base.yaml")
        raw: dict = {}
        if yaml is not None and config_path.exists():
            raw = yaml.safe_load(config_path.read_text()) or {}

        paths = raw.get("paths", {})
        hardware = raw.get("hardware", {})
        repro = raw.get("reproducibility", {})
        logging_cfg = raw.get("logging", {})

        def env_bool(name: str, default: bool) -> bool:
            value = os.environ.get(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
            try:
                return max(minimum, min(maximum, int(os.environ.get(name, default))))
            except (TypeError, ValueError):
                return default

        def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
            try:
                return max(minimum, min(maximum, float(os.environ.get(name, default))))
            except (TypeError, ValueError):
                return default

        return cls(
            data_root=Path(os.environ.get("DATA_ROOT", paths.get("data_root", "data"))),
            device=os.environ.get("MODEL_DEVICE", hardware.get("device", "auto")),
            seed=int(os.environ.get("SEED", repro.get("seed", 42))),
            log_level=os.environ.get("LOG_LEVEL", logging_cfg.get("level", "INFO")),
            incoming_data_root=Path(os.environ.get("INCOMING_DATA_ROOT", "data/incoming")),
            max_upload_size_mb=env_int("MAX_UPLOAD_SIZE_MB", 10, 1, 50),
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
            gemini_vision_enabled=env_bool("GEMINI_VISION_ENABLED", True),
            gemini_model=os.environ.get("GEMINI_MODEL", "models/gemini-3.5-flash-lite").strip() or "models/gemini-3.5-flash-lite",
            gemini_timeout_seconds=env_float("GEMINI_TIMEOUT_SECONDS", 15.0, 1.0, 60.0),
            gemini_max_detections=env_int("GEMINI_MAX_DETECTIONS", 4, 1, 10),
            gemini_correction_threshold=env_float("GEMINI_CORRECTION_THRESHOLD", 0.82, 0.0, 1.0),
            gemini_max_retries=env_int("GEMINI_MAX_RETRIES", 1, 0, 2),
        )
