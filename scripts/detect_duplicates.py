#!/usr/bin/env python3
"""Duplicate / near-duplicate detection (Section 23).

Diagnostic only — reports duplicate groups, never deletes or moves files.
See docs/decisions.md (D-005) for why removal is kept as a separate,
explicit, logged step instead of living inside a "detect" script.

Usage:
    python scripts/detect_duplicates.py <directory> [directory2 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from krishiai.data.hashing import perceptual_available, perceptual_hash, sha256_file  # noqa: E402
from krishiai.data.validators import VALID_IMAGE_EXTENSIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "manifests" / "duplicate_report.json"))
    args = parser.parse_args()

    exact_groups: dict[str, list[str]] = defaultdict(list)
    perceptual_groups: dict[str, list[str]] = defaultdict(list)

    total = 0
    for d in args.directories:
        root = Path(d)
        if not root.exists():
            print(f"Skipping missing directory: {root}")
            continue
        for img in root.rglob("*"):
            if img.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            total += 1
            exact_groups[sha256_file(img)].append(str(img))
            phash = perceptual_hash(img)
            if phash is not None:
                perceptual_groups[phash].append(str(img))

    exact_dupes = {h: paths for h, paths in exact_groups.items() if len(paths) > 1}
    perceptual_dupes = {h: paths for h, paths in perceptual_groups.items() if len(paths) > 1}

    print(f"Scanned {total} image(s) across {len(args.directories)} director(y/ies).")
    print(f"Exact duplicate groups: {len(exact_dupes)}")
    if perceptual_available():
        print(f"Near-duplicate (perceptual hash) groups: {len(perceptual_dupes)}")
    else:
        print(
            "Near-duplicate detection SKIPPED: 'imagehash' is not installed. "
            "Run 'pip install imagehash' and re-run for near-duplicate "
            "coverage (the exact-duplicate results above are still valid)."
        )

    report = {
        "scanned": total,
        "perceptual_hashing_available": perceptual_available(),
        "exact_duplicate_groups": exact_dupes,
        "near_duplicate_groups": perceptual_dupes,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Full report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
