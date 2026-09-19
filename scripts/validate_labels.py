#!/usr/bin/env python3
"""Validate annotations under a directory (Section 19).

Deep-validates YOLO-format labels and does basic structural checks on
folder-classification datasets. COCO / Pascal VOC are detected but not yet
deeply validated — that's reported explicitly rather than silently marked
"passed" (Section 101: never fake a result).

Usage:
    python scripts/validate_labels.py <directory>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from krishiai.data.formats import DatasetFormat, detect_format  # noqa: E402
from krishiai.data.validators import validate_yolo_label_file  # noqa: E402


def validate_yolo_dir(root: Path) -> list[dict]:
    # Look for label files in labels/ subdirectories or directly under root
    labels_dirs = list(root.rglob("labels"))
    label_files = []
    if labels_dirs:
        for ld in labels_dirs:
            label_files.extend(ld.glob("*.txt"))
    else:
        label_files = [
            f for f in root.rglob("*.txt")
            if not f.name.startswith("README") and f.name != "classes.txt"
        ]

    issues = []
    for label_file in label_files:
        if label_file.name == "classes.txt" or label_file.name.startswith("README"):
            continue
        for issue in validate_yolo_label_file(label_file):
            issues.append(issue.__dict__)
    return issues


def validate_classification_dir(root: Path) -> list[dict]:
    issues = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        images = [p for p in child.iterdir() if p.is_file()]
        if not images:
            issues.append({"path": str(child), "line_number": 0, "reason": "empty class directory"})
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.exists():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    fmt = detect_format(root)
    print(f"Detected format for {root}: {fmt.value} (heuristic — confirm manually)")

    if fmt == DatasetFormat.YOLO:
        issues = validate_yolo_dir(root)
    elif fmt == DatasetFormat.CLASSIFICATION_FOLDERS:
        issues = validate_classification_dir(root)
    else:
        print(
            f"No deep validator implemented yet for '{fmt.value}'. Extend "
            "krishiai/data/validators.py rather than assuming it's fine."
        )
        issues = []

    print(f"{len(issues)} issue(s) found.")
    for issue in issues[:50]:
        print(f"  - {issue}")

    out_path = root / "label_validation_report.json"
    out_path.write_text(json.dumps({"root": str(root), "format": fmt.value, "issues": issues}, indent=2))
    print(f"Full report: {out_path}")
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
