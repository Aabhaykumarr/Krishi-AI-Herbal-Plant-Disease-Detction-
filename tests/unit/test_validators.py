from krishiai.data.validators import (
    polygon_to_bbox,
    validate_image,
    validate_yolo_label_file,
)


def test_valid_image_passes(tmp_classification_dataset):
    img = next(tmp_classification_dataset.rglob("*.png"))
    assert validate_image(img) is None


def test_corrupt_image_is_flagged(tmp_corrupt_image):
    issue = validate_image(tmp_corrupt_image)
    assert issue is not None
    assert "failed to open" in issue.reason


def test_unsupported_extension_is_flagged(tmp_path):
    f = tmp_path / "leaf.gif"
    f.write_bytes(b"not checked, extension rejected first")
    issue = validate_image(f)
    assert issue is not None
    assert "unsupported extension" in issue.reason


def test_valid_yolo_label_passes(tmp_yolo_dataset):
    label = next((tmp_yolo_dataset / "labels").glob("*.txt"))
    assert validate_yolo_label_file(label) == []


def test_out_of_range_center_is_flagged(tmp_path):
    label = tmp_path / "bad.txt"
    label.write_text("0 1.5 0.5 0.3 0.4\n")  # x_center outside [0, 1]
    issues = validate_yolo_label_file(label)
    # x_center=1.5 legitimately trips two independent checks (normalization
    # AND the box-inside-image bounds check) — assert the specific issue is
    # present rather than an exact count, since both flags are correct.
    assert any("not normalized" in i.reason for i in issues)


def test_non_positive_box_size_is_flagged(tmp_path):
    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5 0.0 0.4\n")  # zero width
    issues = validate_yolo_label_file(label)
    assert any("non-positive box size" in i.reason for i in issues)


def test_class_id_out_of_range_is_flagged(tmp_path):
    label = tmp_path / "bad.txt"
    label.write_text("7 0.5 0.5 0.3 0.4\n")
    issues = validate_yolo_label_file(label, num_classes=3)
    assert any("out of range" in i.reason for i in issues)


def test_valid_polygon_label_passes(tmp_path):
    label = tmp_path / "polygon.txt"
    label.write_text("1 0.1 0.2 0.3 0.4 0.5 0.6 0.1 0.2\n")
    issues = validate_yolo_label_file(label, num_classes=3)
    assert issues == []


def test_invalid_polygon_point_is_flagged(tmp_path):
    label = tmp_path / "bad_polygon.txt"
    label.write_text("1 0.1 0.2 1.5 0.4 0.5 0.6\n")
    issues = validate_yolo_label_file(label, num_classes=3)
    assert any("not normalized" in i.reason for i in issues)


def test_polygon_to_bbox_conversion():
    # Triangle from (0.1, 0.2) to (0.5, 0.6) and (0.3, 0.4)
    coords = [0.1, 0.2, 0.5, 0.6, 0.3, 0.4]
    xc, yc, w, h = polygon_to_bbox(coords)
    assert xc == 0.3
    assert yc == 0.4
    assert w == 0.4
    assert h == 0.4


