"""CLI entrypoint for exporting KrishiAI YOLO26 models to ONNX/TorchScript (Section 52)."""
import argparse
from pathlib import Path
import sys

from krishiai.models.export import export_model
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export KrishiAI YOLO26 Models")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch .pt model")
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "openvino", "engine"],
        help="Target export format",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 for det, 224 for cls)")
    parser.add_argument("--half", action="store_true", help="FP16 half precision")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic shape support")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    args = parser.parse_args()

    try:
        exported_path = export_model(
            model_path=args.model,
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            opset=args.opset,
        )
        print(f"\nModel exported successfully to: {exported_path}")
    except Exception as exc:
        logger.error(f"Export failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
