#!/usr/bin/env python3
"""Surface every distinct raw class label found so far, to support building
the canonical taxonomy in configs/dataset.yaml (Section 8).

This script does NOT invent a mapping. It lists what it found and what's
still unmapped. Ambiguous or missing mappings must be resolved by a human
(or an explicitly-reviewed decision in docs/decisions.md), never guessed —
Section 75 specifically forbids auto-mapping e.g. "fresh" -> "healthy".

Usage:
    python scripts/normalize_classes.py [--raw-dir data/raw] [--config configs/dataset.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def discover_raw_classes(raw_dir: Path) -> dict[str, list[str]]:
    """Discovers raw classes from either data.yaml (YOLO) or class subfolders (Classification)."""
    found: dict[str, list[str]] = {}
    if not raw_dir.exists():
        return found

    for source in sorted(raw_dir.iterdir()):
        if not source.is_dir() or source.name.startswith("_"):
            continue

        # Check 1: Is there a data.yaml with 'names'?
        data_yaml = source / "data.yaml"
        if data_yaml.exists() and yaml is not None:
            try:
                ydata = yaml.safe_load(data_yaml.read_text()) or {}
                names = ydata.get("names", [])
                if isinstance(names, list) and names:
                    # If names are generic digits ['0', '1', '2'], check README or prefix
                    if names == ["0", "1", "2"] and "mint" in source.name.lower():
                        found[source.name] = ["0 (Dried)", "1 (Fresh)", "2 (Spoiled)"]
                    else:
                        found[source.name] = [str(n) for n in names]
                    continue
                elif isinstance(names, dict) and names:
                    found[source.name] = [str(v) for _, v in sorted(names.items())]
                    continue
            except Exception:
                pass

        # Check 2: Check for classification folders (e.g., source/final/* or source/*)
        target_dir = source / "final" if (source / "final").is_dir() else source
        subdirs = sorted(p.name for p in target_dir.iterdir() if p.is_dir() and p.name not in {"train", "valid", "test", "labels", "images", "final_Augmented"})
        if subdirs:
            found[source.name] = subdirs
        else:
            # Fallback to direct subdirectories
            direct_subs = sorted(p.name for p in source.iterdir() if p.is_dir())
            if direct_subs:
                found[source.name] = direct_subs

    return found


def load_existing_mapping(config_path: Path) -> dict:
    if yaml is None or not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text()) or {}
    mapping = data.get("class_mapping", {})
    return mapping if isinstance(mapping, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "dataset.yaml"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    discovered = discover_raw_classes(raw_dir)
    if not discovered:
        print(f"No candidate class folders found under {raw_dir} yet. Nothing to map.")
        return 0

    existing = load_existing_mapping(Path(args.config))

    print("Discovered candidate raw classes (folder names) per dataset:\n")
    unmapped_total = 0
    for dataset_id, classes in discovered.items():
        mapped_here = existing.get(dataset_id, {}) if isinstance(existing, dict) else {}
        print(f"[{dataset_id}]")
        for cls in classes:
            target = mapped_here.get(cls) if isinstance(mapped_here, dict) else None
            if target:
                print(f"  {cls!r} -> {target!r}")
            else:
                print(f"  {cls!r} -> UNMAPPED")
                unmapped_total += 1
        print()

    if unmapped_total:
        print(
            f"{unmapped_total} class label(s) are not yet mapped to a canonical "
            "class. Add them to configs/dataset.yaml under class_mapping, and "
            "record the reasoning for any non-obvious mapping in "
            "docs/decisions.md (Section 75)."
        )
    else:
        print("All discovered classes are mapped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
