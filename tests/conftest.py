"""Shared pytest fixtures — synthetic data only, generated on the fly.

Nothing here is checked in as static binary fixtures; every image is
generated at test time so the repo never carries fake data that could be
mistaken for real project data (Section 73: label synthetic data explicitly
and keep it separate from originals).
"""
from __future__ import annotations

import pytest

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


def _make_image(path, size=(32, 32), color=(120, 180, 90)):
    if Image is None:
        pytest.skip("Pillow not installed in this environment")
    img = Image.new("RGB", size, color)
    img.save(path)


@pytest.fixture
def tmp_classification_dataset(tmp_path):
    """A tiny synthetic folder-classification dataset: 2 classes, 2 images each."""
    root = tmp_path / "classification_sample"
    for cls in ("healthy", "diseased"):
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(2):
            _make_image(cls_dir / f"img_{i}.png")
    return root


@pytest.fixture
def tmp_yolo_dataset(tmp_path):
    """A tiny synthetic YOLO-format dataset: 2 images, 1 box label each."""
    root = tmp_path / "yolo_sample"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(2):
        _make_image(root / "images" / f"img_{i}.jpg")
        (root / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    return root


@pytest.fixture
def tmp_corrupt_image(tmp_path):
    """A file with an image extension that is not actually a valid image."""
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not a real jpeg")
    return path
