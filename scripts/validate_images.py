#!/usr/bin/env python3
"""Validate every image under a directory (Section 18).

Never deletes files — invalid images are written to a report for manual
review/quarantine, which is a separate, explicit step you control.

Usage:
    python scripts/validate_images.py <directory> [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from krishiai.data.validators import VALID_IMAGE_EXTENSIONS, validate_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.exists():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    images = [p for p in root.rglob("*") if p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    issues = []
    for img in images:
        issue = validate_image(img)
        if issue:
            issues.append(issue.__dict__)

    print(f"Checked {len(images)} image(s) under {root}. {len(issues)} issue(s) found.")
    for issue in issues[:50]:
        print(f"  - {issue['path']}: {issue['reason']}")
    if len(issues) > 50:
        print(f"  ... and {len(issues) - 50} more (see report file).")

    out_path = Path(args.out) if args.out else root / "image_validation_report.json"
    out_path.write_text(json.dumps({"root": str(root), "checked": len(images), "issues": issues}, indent=2))
    print(f"Full report: {out_path}")
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
