"""Unit tests for model export pipeline (Section 52)."""
from pathlib import Path
import pytest
from krishiai.models.export import export_model


def test_export_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        export_model("models/nonexistent_model.pt")


def test_export_torchscript_dummy(tmp_path):
    # Test exporting pretrained yolo26n-cls to torchscript (lightweight & built-in)
    pt_path = Path("models/pretrained/yolo26n-cls.pt")
    if pt_path.exists():
        exported = export_model(
            model_path=pt_path,
            format="torchscript",
            imgsz=224,
        )
        assert exported.exists()
        assert exported.suffix == ".torchscript"
