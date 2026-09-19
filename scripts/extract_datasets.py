#!/usr/bin/env python3
"""Extract archived datasets (Section 17) — e.g. the Aloe Vera RAR.

Looks for .zip and .rar files under data/raw/_archives/ and extracts each
into its own folder under data/extracted/<archive_stem>/. Never modifies or
deletes the original archive, and never assumes what's inside is a clean
YOLO dataset (Section 5) — that's what inspect_datasets.py is for, next.

Usage:
    python scripts/extract_datasets.py [--archive-dir data/raw/_archives]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import rarfile

    _RAR_AVAILABLE = True
except ImportError:
    _RAR_AVAILABLE = False


def extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def extract_rar(archive: Path, dest: Path) -> None:
    if not _RAR_AVAILABLE:
        raise RuntimeError(
            "The 'rarfile' package is not installed. Run:\n"
            "    pip install rarfile\n"
            "and make sure an 'unrar' or 'unar' binary is on PATH "
            "(e.g. 'apt install unrar' / 'brew install unar'), then retry."
        )
    with rarfile.RarFile(archive) as rf:
        rf.extractall(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", default=str(REPO_ROOT / "data" / "raw" / "_archives"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "extracted"))
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    out_dir = Path(args.out_dir)

    if not archive_dir.exists():
        print(f"No such directory: {archive_dir}\nNothing to extract yet.")
        return 0

    archives = sorted(list(archive_dir.glob("*.zip")) + list(archive_dir.glob("*.rar")))
    if not archives:
        print(f"No .zip or .rar files found under {archive_dir}. Nothing to extract yet.")
        return 0

    for archive in archives:
        dest = out_dir / archive.stem
        if dest.exists() and any(dest.iterdir()):
            print(f"Skipping {archive.name}: {dest} already has content.")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive.name} -> {dest} ...")
        try:
            if archive.suffix.lower() == ".zip":
                extract_zip(archive, dest)
            else:
                extract_rar(archive, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        print(
            f"  done. Inspect {dest} manually before assuming it's a clean "
            f"dataset (Section 5 — do not assume the archive contains a "
            f"clean YOLO dataset)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
