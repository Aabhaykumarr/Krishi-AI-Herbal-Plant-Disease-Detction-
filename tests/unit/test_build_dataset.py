import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_dataset import get_base_scene_name, parse_and_convert_labels  # noqa: E402


def test_get_base_scene_name_with_roboflow_suffix():
    fname = "Basil-with-Mildew-133_jpg.rf.f784e36b18cae61109ac7dbb8713638f.jpg"
    assert get_base_scene_name(fname) == "Basil-with-Mildew-133_jpg"


def test_get_base_scene_name_standard():
    fname = "IMG_20220603_164938.jpg"
    assert get_base_scene_name(fname) == "IMG_20220603_164938"


def test_parse_and_convert_labels_bbox(tmp_path):
    lbl = tmp_path / "box.txt"
    lbl.write_text("0 0.5 0.5 0.2 0.3\n")
    converted = parse_and_convert_labels(lbl, class_offset=7)
    assert len(converted) == 1
    assert converted[0] == "7 0.500000 0.500000 0.200000 0.300000"


def test_parse_and_convert_labels_polygon(tmp_path):
    lbl = tmp_path / "poly.txt"
    # Triangle: (0.1, 0.2), (0.5, 0.6), (0.3, 0.4) -> xc=0.3, yc=0.4, w=0.4, h=0.4
    lbl.write_text("2 0.1 0.2 0.5 0.6 0.3 0.4\n")
    converted = parse_and_convert_labels(lbl, class_offset=0)
    assert len(converted) == 1
    assert converted[0] == "2 0.300000 0.400000 0.400000 0.400000"


def test_parse_and_convert_labels_missing_file(tmp_path):
    lbl = tmp_path / "nonexistent.txt"
    assert parse_and_convert_labels(lbl) == []
