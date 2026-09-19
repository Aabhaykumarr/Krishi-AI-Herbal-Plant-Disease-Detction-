from krishiai.data.formats import DatasetFormat, detect_format


def test_detects_classification_folders(tmp_classification_dataset):
    assert detect_format(tmp_classification_dataset) == DatasetFormat.CLASSIFICATION_FOLDERS


def test_detects_yolo(tmp_yolo_dataset):
    assert detect_format(tmp_yolo_dataset) == DatasetFormat.YOLO


def test_unknown_for_empty_dir(tmp_path):
    empty = tmp_path / "mystery_dataset"
    empty.mkdir()
    assert detect_format(empty) == DatasetFormat.UNKNOWN
