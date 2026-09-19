# KrishiAI — Decision Log

Format per Section 105: Decision / Reason / Alternatives considered /
Evidence / Date. New entries append; don't rewrite history — if a decision
changes, add a new entry that supersedes the old one and say so.

---

## D-001 — Primary architecture is YOLO26
**Decision:** Ultralytics YOLO26 is the mandatory primary architecture.
**Reason:** Fixed project constraint; other architectures may only appear
as baselines/ablations.
**Alternatives considered:** YOLOv8, YOLO11, Faster R-CNN, EfficientNet —
explicitly out of scope as the primary model.
**Evidence:** Project brief, Section 11.
**Status:** Final. **Date:** 2026-09-06.

## D-002 — Scope is limited to Basil, Aloe Vera, Mint
**Decision:** No other crops/plants are added; yield prediction is out of
scope for the current implementation.
**Reason:** Fixed project constraint; prevents scope creep from the earlier
"crop health and yield prediction" concept.
**Evidence:** Project brief, Sections 1–2.
**Status:** Final. **Date:** 2026-09-06.

## D-003 — Classification vs. detection vs. two-stage
**Decision:** RESOLVED (Superseded by D-007).
**Reason:** Physical dataset audit of all three real datasets reveals a hybrid modality:
1. **Basil**: Real multi-leaf bounding boxes (5,096 images, 12,298 objects, 7 classes) → Requires **YOLO26 Object Detection**.
2. **Mint**: Real leaf bounding boxes (3,533 images, 3,536 objects, 3 condition classes) → Requires **YOLO26 Object Detection**.
3. **Aloe Vera**: Full-frame single-plant images with folder-only labels (2,500 images, 5 classes) → Requires **YOLO26 Image Classification**.
**Status:** Superseded by D-007. **Date resolved:** 2026-09-17.

## D-004 — Disease vs. condition/freshness terminology
**Decision:** Mint's `fresh` / `spoiled` / `dried` labels are treated as a
**condition/freshness** taxonomy, not a disease taxonomy, unless a source
dataset explicitly defines them as disease states.
**Reason:** Scientific accuracy (Section 9) — "dried" or "spoiled" describe
post-harvest state, not necessarily pathology.
**Alternatives considered:** merging condition and disease into one generic
"class" dimension — rejected as scientifically misleading.
**Evidence:** Project brief, Section 5; physical verification of Roboflow mint-disease4 dataset.
**Status:** Final (taxonomy finalized in configs/dataset.yaml). **Date:** 2026-09-17.

## D-005 — Duplicate detection is diagnostic-only, not auto-destructive
**Decision:** `scripts/detect_duplicates.py` only reports duplicate groups;
it never deletes or moves files. Removal is a separate, explicit, logged
step that lives in `build_dataset.py`'s pipeline.
**Reason:** A script named "detect" silently deleting files is a hidden
side effect (Section 81) and makes the pipeline harder to audit. Separating
read-only detection from destructive curation keeps raw data untouched
(Section 15/23) and each script single-responsibility.
**Alternatives considered:** deleting duplicates immediately on detection —
rejected for the reason above.
**Evidence:** Sections 15, 23, 81. **Status:** Final. **Date:** 2026-09-06.

## D-006 — YOLO26 API surface, verified against current docs
**Decision:** Config/script placeholders use `yolo26n.pt` (detection) /
`yolo26n-cls.pt` (classification), loaded via
`from ultralytics import YOLO; YOLO("yolo26n.pt")`.
**Reason/evidence:** Confirmed against Ultralytics YOLO26 documentation:
YOLO26 spans five scales (n/s/m/l/x), supports detect/classify through
the unified `YOLO(...)` interface with native NMS-free end-to-end inference.
**Status:** Final. **Date:** 2026-09-06.

## D-007 — Dual-Pipeline Architecture with Unified Inference Gateway
**Decision:** The project adopts a dual YOLO26 architecture under a single canonical `ModelManager` inference service:
1. **YOLO26 Detection Pipeline (`krishiai-det-yolo26`)**: Trained on multi-class detection data combining Basil (7 classes) and Mint (3 condition classes) = 10 detection classes with bounding-box localization.
2. **YOLO26 Classification Pipeline (`krishiai-cls-yolo26`)**: Trained on Aloe Vera 5-class folder data using `yolo26n-cls.pt`.
3. **Unified Inference Gateway**: The backend API (`POST /predict`) routes inputs seamlessly, providing bounding boxes for multi-leaf scenes and calibrated classification scores for Aloe Vera or whole-leaf inputs, with out-of-distribution (OOD) rejection.
**Reason:** Scientifically defensible and honest (Section 10, 126). Inventing bounding boxes for Aloe Vera would introduce synthetic bias, while flattening Basil/Mint to whole-image classification would discard 15,834 real spatial annotations.
**Alternatives considered:**
- Forcing all datasets into classification (rejected: discards multi-leaf localization and spatial disease detection).
- Synthesizing full-image [0.5, 0.5, 1.0, 1.0] boxes for Aloe Vera in detection (rejected as dirty synthetic data; evaluated as experimental ablation only).
**Status:** Final. **Date:** 2026-09-17.

## D-008 — Polygon to Bounding Box Normalization Policy
**Decision:** The 39 polygon segmentation annotations discovered in `basil_roboflow_v4` are converted to their tightest enclosing bounding box `[x_center, y_center, width, height]` during the dataset harmonization stage.
**Reason:** Preserves 100% of annotated samples while conforming to the canonical YOLO object detection format.
**Status:** Final. **Date:** 2026-09-17.

## D-009 — Data Leakage & Duplicate Quarantine Strategy
**Decision:**
1. For Aloe Vera: The 16 cross-class conflicting duplicate groups (35 files) where identical images have conflicting class labels (e.g., `Aloe Rust` vs `Healthy`) are quarantined into `data/interim/quarantine/` with logged provenance (`aloevera_conflicts.json`) and excluded from train/val/test splits. 47 same-class duplicate groups are deduplicated retaining a single clean instance.
2. For Basil & Mint: Roboflow exports are grouped by source scene identifier (1,754 Basil scene groups, 3,274 Mint scene groups) before partitioning to eliminate cross-split scene and image leakage.
**Reason:** Mandatory data leakage prevention per Sections 23-25.
**Status:** Final. **Date:** 2026-09-17 (Updated 2026-09-18).

## D-010 — Fixed Partitioning Ratio and Reproducibility Seed
**Decision:** All datasets are partitioned using a 70% Train / 15% Validation / 15% Test split with fixed random seed `SEED = 42`.
**Reason:** Balances sufficient training data (6,042 detection images, 1,689 classification images) with statistically significant evaluation sets (~1,300 detection / ~365 classification images per val/test split) while ensuring 100% deterministic reproducibility across development sessions.
**Status:** Final. **Date:** 2026-09-18.

## D-011 — FastAPI Service Architecture and Integrated Modern Diagnostic UI
**Decision:** Provide a unified FastAPI backend (`src/krishiai/api/`) serving both programmatic REST endpoints (`/health`, `/model/info`, `/predict`, `/docs`) and an interactive modern web UI dashboard (`/`) with real-time bounding-box canvas rendering, latency profiling, and agronomic management advice.
**Reason:** Fulfills Section 51, Section 6 of project contract, and allows researchers and field agronomists to immediately interact with the YOLO26 dual-pipeline model.
**Status:** Final. **Date:** 2026-09-18.

## D-012 — Environmental Robustness Stress-Testing Suite
**Decision:** Implement an automated perturbation engine (`src/krishiai/evaluation/robustness.py`, `scripts/robustness_tests.py`) testing 6 environmental corruption categories (brightness shifts, contrast decay, Gaussian blur, Gaussian noise, rotation angles, and center occlusion).
**Reason:** Fulfills Section 65 requirements to rigorously test vision model resilience under harsh field agricultural lighting, moisture, lens distortion, and partial occlusion conditions.
**Status:** Final. **Date:** 2026-09-18.

## D-013 — Multi-Stage Docker Containerization & Batch Inference Gateway
**Decision:** Provide a production-grade multi-stage `Dockerfile` and `docker-compose.yml` with system OpenCV dependencies (`libgl1`, `libglib2.0-0`), CPU PyTorch wheel optimizations, health probes, and batch CLI (`src/krishiai/inference/batch.py`, `scripts/batch_predict.py`) generating structured JSON/CSV reports and annotated visualization overlays.
**Reason:** Fulfills Section 51-53 deployment requirements for zero-friction containerized serving and batch agronomic survey workflows.
**Status:** Final. **Date:** 2026-09-18.

## D-014 — Botanical Visual Feature Gating and Non-Plant Hallucination Prevention
**Decision:** Implement a front-line visual feature gate (`is_botanical_image` in `src/krishiai/inference/model_manager.py`) analyzing neutral/grayscale document pixel ratios, botanical chlorophyll hue bands ($45^\circ - 165^\circ$, $18^\circ - 45^\circ$, $140^\circ - 195^\circ$), and saturation profiles. Non-plant images (roadmaps, flowcharts, technical diagrams, plain text documents, synthetic line art) are immediately rejected with `primary_condition: "non_plant_or_unrecognized"` and a descriptive warning without triggering false disease predictions or drawing canvas bounding boxes.
**Reason:** Prevents model hallucination when non-agricultural images or user documentation are uploaded, ensuring agronomic diagnostic reliability and integrity.
**Status:** Final. **Date:** 2026-09-19.


