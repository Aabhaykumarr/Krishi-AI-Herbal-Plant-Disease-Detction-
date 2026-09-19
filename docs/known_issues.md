# KrishiAI — Known Issues & Failure Memory

> This document records significant bugs, data hazards, edge cases, and failed approaches (Section 101/119).
> Future AI generations must inspect this file before attempting equivalent strategies to prevent rediscovering known failures.

---

## ISSUE-001: Data Leakage in Raw Roboflow Basil Export
- **Date Discovered:** 2026-09-17
- **Component:** `data/raw/basil_roboflow_v4`
- **Symptoms:** 105 exact duplicate hash groups found spanning across `train` and `valid`/`test` folders.
- **Root Cause:** Roboflow's online pre-split applies augmentation/crop variations to the same source images before partitioning into splits without patient/scene-level isolation.
- **Impact:** Evaluating directly on Roboflow's default `valid`/`test` splits creates false optimism and metric leakage.
- **Final Solution:** In Phase 1 harmonization, all images are grouped by perceptual/exact scene hash and partitioned cleanly using a strict 70/15/15 group split in `scripts/build_dataset.py`.
- **DO NOT REPEAT:** Never train or benchmark models on raw off-the-shelf Roboflow train/val splits without verifying zero cross-split image duplication.

---

## ISSUE-002: Conflicting Class Labels on Identical Images in Aloe Vera Dataset
- **Date Discovered:** 2026-09-17
- **Component:** `data/raw/aloevera_kaggle_v1/final`
- **Symptoms:** 63 exact duplicate hash groups found where the exact same image appears in mutually exclusive class directories (e.g., `Aloe Rust` vs `Healthy`, `Anthracnose` vs `Sunburn`).
- **Root Cause:** Inconsistent scraping or manual labeling errors in the source Kaggle dataset.
- **Impact:** Contaminates model loss gradients; the model cannot learn separable representations for conflicting identical inputs.
- **Final Solution:** Quarantine all conflicting cross-class duplicate images into `data/interim/quarantine/aloevera_conflicts.json` and exclude them from training/validation datasets.
- **DO NOT REPEAT:** Never train on un-audited folder-classification datasets without cross-class hash validation.

---

## ISSUE-003: Polygon Segmentation Annotations in YOLO Detection Dataset
- **Date Discovered:** 2026-09-17
- **Component:** `data/raw/basil_roboflow_v4` label files
- **Symptoms:** 39 label files in Basil contained polygon coordinates (>5 space-separated values) instead of standard 4-coordinate bounding boxes.
- **Root Cause:** Mixed annotation export in Roboflow where some objects were polygon-annotated while the majority were bounding boxes.
- **Impact:** Label validator threw parse errors when expecting exactly 5 values.
- **Final Solution:** Updated `validate_yolo_label_file()` in `src/krishiai/data/validators.py` to recognize polygon geometry, and added bounding-box enclosure conversion in the normalization pipeline.
- **DO NOT REPEAT:** Do not assume all `.txt` annotations in a YOLO directory strictly follow `[class x y w h]` format; support polygon conversion.

---

## ISSUE-004: Mixed Modality Across Target Plant Datasets
- **Date Discovered:** 2026-09-17
- **Component:** Dataset Harmonization
- **Symptoms:** Basil (5,096 images) and Mint (3,533 images) are YOLO Object Detection datasets with spatial coordinates, whereas Aloe Vera (2,500 images) is a whole-image folder classification dataset without bounding boxes.
- **Root Cause:** Heterogeneous external data sources (Roboflow vs Kaggle).
- **Impact:** Cannot naively train a single monolithic detection model without fabricating false full-image bounding boxes for Aloe Vera, nor train a pure classification model without discarding 15,834 spatial bounding boxes.
- **Final Solution:** Adopted Dual-Pipeline Architecture under a Unified `ModelManager` Gateway (Decision D-007): YOLO26 Detection model for Basil & Mint + YOLO26 Classification model for Aloe Vera.
- **DO NOT REPEAT:** Do not force-fit incompatible dataset modalities into a single loss function without scientific justification.
