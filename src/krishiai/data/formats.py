"""Best-effort dataset format detection (Section 20).

Different sources (Kaggle, Roboflow, manual collection) ship annotations in
different shapes. This module makes a heuristic guess so the rest of the
pipeline can dispatch to the right validator/converter — it does NOT
guarantee correctness. Section 5 explicitly warns not to assume the Aloe
Vera archive contains a clean YOLO dataset; this detector's output must
still be confirmed manually (or by a future AI generation) before being
trusted for a new source, especially one that hasn't been inspected before.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class DatasetFormat(str, Enum):
    YOLO = "yolo"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    CLASSIFICATION_FOLDERS = "classification_folders"
    UNKNOWN = "unknown"


def detect_format(root: Path) -> DatasetFormat:
    """Shallow, heuristic guess at a dataset's annotation format.

    Checks (in order): COCO-style JSON, Pascal VOC XML, loose YOLO .txt
    files, then folder-per-class classification layout. Falls back to
    UNKNOWN rather than guessing wrong with false confidence.
    """
    files = list(root.rglob("*"))

    if any(
        f.suffix.lower() == ".json"
        and f.name.lower() in {"annotations.json", "_annotations.coco.json"}
        for f in files
    ):
        return DatasetFormat.COCO

    if any(f.suffix.lower() == ".xml" for f in files):
        return DatasetFormat.PASCAL_VOC

    if any(f.suffix.lower() == ".txt" and f.name != "classes.txt" for f in files):
        return DatasetFormat.YOLO

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs and all(
        any(p.suffix.lower() in IMAGE_SUFFIXES for p in d.iterdir() if p.is_file())
        for d in subdirs
    ):
        return DatasetFormat.CLASSIFICATION_FOLDERS

    return DatasetFormat.UNKNOWN
