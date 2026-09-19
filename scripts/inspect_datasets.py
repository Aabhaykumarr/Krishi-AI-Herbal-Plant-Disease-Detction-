#!/usr/bin/env python3
"""Step 1 of the KrishiAI pipeline: audit everything under data/raw/.

Usage:
    python scripts/inspect_datasets.py [--raw-dir data/raw] [--full-scan]

Produces data/manifests/audit_report_<timestamp>.json and prints a summary.
Never modifies data/raw/ — Section 15 requires raw data to stay untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from krishiai.data.formats import detect_format  # noqa: E402
from krishiai.data.validators import VALID_IMAGE_EXTENSIONS, validate_image  # noqa: E402


def audit_source(source_dir: Path, full_scan: bool) -> dict:
    images = [p for p in source_dir.rglob("*") if p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    fmt = detect_format(source_dir)

    scan_targets = images if full_scan else images[:200]
    invalid = []
    for img in scan_targets:
        issue = validate_image(img)
        if issue:
            invalid.append(issue.__dict__)

    class_dirs = sorted(
        {
            p.relative_to(source_dir).parts[0]
            for p in images
            if len(p.relative_to(source_dir).parts) > 1
        }
    )

    return {
        "dataset_id": source_dir.name,
        "local_path": str(source_dir),
        "detected_format": fmt.value,
        "image_count": len(images),
        "images_scanned_for_validity": len(scan_targets),
        "invalid_images_found": len(invalid),
        "invalid_image_samples": invalid[:20],
        "candidate_class_folders": class_dirs,
        "note": (
            "detected_format is a heuristic guess — confirm manually before "
            "trusting it (see docs/dataset.md)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Validate every image instead of just the first 200 per source.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"No such directory: {raw_dir}", file=sys.stderr)
        return 1

    sources = [d for d in sorted(raw_dir.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    if not sources:
        print(
            f"No dataset subfolders found under {raw_dir}.\n"
            "Nothing to audit yet — this is expected on a fresh checkout.\n"
            "Add one subfolder per dataset (e.g. data/raw/mint_kaggle_v1/) "
            "and re-run this script."
        )
        return 0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "sources": [audit_source(s, args.full_scan) for s in sources],
    }

    out_dir = REPO_ROOT / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"audit_report_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Audited {len(sources)} dataset source(s). Full report: {out_path}\n")
    for s in report["sources"]:
        print(
            f"- {s['dataset_id']}: {s['image_count']} images, "
            f"format guess = {s['detected_format']}, "
            f"invalid images found = {s['invalid_images_found']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
