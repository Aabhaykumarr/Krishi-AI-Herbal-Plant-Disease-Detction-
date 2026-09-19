# KrishiAI — Dataset Pipeline & Manifest Reference

## Directory layout

```
data/
├── raw/            # exactly what a source gave us — never edited, never trained on directly
│   └── _archives/  # .zip / .rar files waiting to be extracted (e.g. the Aloe Vera RAR)
├── extracted/      # output of scripts/extract_datasets.py
├── interim/        # intermediate cleanup steps (dedup passes, format conversion)
├── processed/      # per-source, validated, canonical-format data
├── final/          # the actual train/val/test split YOLO26 trains on
└── manifests/      # audit reports + the dataset manifest (JSON)
```

Nothing is deleted at any stage — later directories are *derived from*
earlier ones, so the whole pipeline can be re-run from `raw/` if a bug is
found in a later stage (Section 15).

## Manifest schema (`data/manifests/manifest.json`)

Backed by `krishiai.data.manifest.DatasetRecord`:

| Field | Meaning |
|---|---|
| `dataset_id` | Unique slug, e.g. `mint_kaggle_v1` |
| `source` | `kaggle` / `roboflow` / `manual` / ... |
| `source_url` | Where it came from |
| `license` | As stated by the source |
| `version` | Source's own version tag, if any |
| `local_path` | Path under `data/raw/` or `data/extracted/` |
| `task_type` | `classification` / `detection` / `segmentation` / `unknown` |
| `class_names` | Original (pre-mapping) class names found |
| `annotation_format` | `yolo` / `coco` / `pascal_voc` / `classification_folders` / `unknown` |
| `image_count`, `object_count` | From the audit |
| `acquisition_date` | When it was added to this project |
| `preprocessing_status`, `validation_status` | Pipeline progress tracking |

This is currently an **empty manifest** — no dataset has been registered.

## Format detection

`krishiai.data.formats.detect_format()` makes a best-effort, shallow guess:
COCO (an `*annotations*.json` file), Pascal VOC (`.xml` files), YOLO (loose
`.txt` files that aren't `classes.txt`), or classification-folders (every
first-level subdirectory contains only images). **This is a heuristic** —
Section 20 requires converting into one canonical internal representation
rather than writing dataset-specific logic ad hoc, but the detector's
output still needs a manual sanity check before being trusted, especially
for the Aloe Vera archive, which Section 5 explicitly warns may not be a
clean YOLO export.

## Taxonomy status

See `configs/dataset.yaml` and `docs/decisions.md` (D-003, D-004). In
short: nothing is finalized. The only facts on record are the ones stated
directly in the project brief — three plant species (basil, aloe_vera,
mint) and Mint's `fresh` / `spoiled` / `dried` **condition** labels, which
are explicitly not treated as disease labels without evidence.
