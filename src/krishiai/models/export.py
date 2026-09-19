"""Model export and serialization utilities for KrishiAI (Section 52)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ultralytics import YOLO

from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


def export_model(
    model_path: Union[str, Path],
    format: str = "onnx",
    imgsz: Union[int, tuple] = 640,
    half: bool = False,
    dynamic: bool = False,
    opset: int = 17,
    simplify: bool = True,
) -> Path:
    """Exports a trained YOLO26 PyTorch checkpoint to optimized inference formats.

    Args:
        model_path: Path to PyTorch .pt checkpoint.
        format: Target format: 'onnx', 'torchscript', 'engine', 'openvino'.
        imgsz: Image size for export (e.g. 640 for detection, 224 for classification).
        half: Export with FP16 half-precision.
        dynamic: Support dynamic input shapes.
        opset: ONNX opset version.
        simplify: Run ONNX-Simplifier.

    Returns:
        Path to the exported model artifact.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    logger.info(
        f"Exporting model {model_path} -> format: {format} | imgsz: {imgsz} | half: {half} | dynamic: {dynamic}"
    )
    model = YOLO(str(model_path))

    export_args: Dict[str, Any] = {
        "format": format,
        "imgsz": imgsz,
        "half": half,
        "dynamic": dynamic,
        "simplify": simplify,
    }
    if format.lower() == "onnx":
        export_args["opset"] = opset

    exported_path_str = model.export(**export_args)

    exported_path = Path(exported_path_str)
    logger.info(f"Export successful: {exported_path} ({exported_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return exported_path
