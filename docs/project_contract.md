# KrishiAI — Project Contract & Consistency Register

> This document defines the canonical invariants, approved names, schemas, paths,
> and conventions for the KrishiAI project (Section 98). All code, configuration,
> and documentation across all AI generations must strictly adhere to these standards.

---

## 1. Project Identity & Scope
- **Project Name:** `krishiai` (Python package / repo name)
- **Display Name:** `KrishiAI`
- **Approved Project Title:** "Multi-Class Herbal Plant Leaf (Basil, Aloe Vera, Mint) Disease Detection and Classification using YOLO26 Architecture"
- **Target Plants (3 Primary Species):**
  1. `basil` (Ocimum basilicum)
  2. `aloe_vera` (Aloe barbadensis miller)
  3. `mint` (Mentha spicata / Mentha piperita)
- **Scope Boundary:** No yield prediction. No unrelated crops. Focus is image-based plant identification and leaf disease/condition classification.

---

## 2. Canonical Class Names & Taxonomy (Version 1.0.0)

### 2.1 Canonical Plant Species
- `basil`
- `aloe_vera`
- `mint`

### 2.2 Canonical Disease Classes (Pathological)
- `basil_bacterial_spot` (Pseudomonas cichorii / Xanthomonas)
- `basil_bug_spot` (Insect / pest feeding puncture spots)
- `basil_downy_mildew` (Peronospora belbahrii)
- `basil_powdery_mildew` (Golovinomyces biocellatus / Erysiphe)
- `aloe_vera_rust` (Uromyces aloes)
- `aloe_vera_anthracnose` (Colletotrichum gloeosporioides)
- `aloe_vera_leaf_spot` (Alternaria / Cercospora aloes)

### 2.3 Canonical Deficiency Classes (Physiological)
- `basil_magnesium_deficiency` (Interveinal chlorosis)
- `basil_nitrogen_deficiency` (General chlorosis / yellowing)

### 2.4 Canonical Condition & Freshness Classes (Post-Harvest / Environmental)
- `aloe_vera_sunburn` (Photo-oxidative stress / abiotic condition)
- `mint_dried` (Post-harvest desiccation condition)
- `mint_fresh` (Optimal post-harvest freshness condition)
- `mint_spoiled` (Post-harvest microbial decomposition / spoilage)

### 2.5 Canonical Healthy Reference Classes
- `basil_healthy`
- `aloe_vera_healthy`
- `mint_fresh`

---

## 3. Canonical Configuration Variables (Section 79, 107)

| Responsibility | Canonical Variable Name | FORBIDDEN Aliases (DO NOT CREATE) |
|---|---|---|
| Model checkpoint path | `MODEL_PATH` | `MODEL_FILE`, `MODEL_CHECKPOINT`, `MODEL_WEIGHTS`, `MODEL_WEIGHTS_PATH` |
| Primary CV Architecture | `YOLO26` | `YOLOv8`, `YOLO11`, `YOLO_V26` |
| Data directory root | `DATA_ROOT` | `DATA_DIR`, `DATASET_PATH`, `DATA_FOLDER` |
| Hardware compute device | `MODEL_DEVICE` | `DEVICE_TYPE`, `CUDA_DEVICE`, `GPU_ID` |
| Global random seed | `SEED` | `RANDOM_SEED`, `SEED_VALUE` |
| Log level | `LOG_LEVEL` | `LOGGING_LEVEL`, `DEBUG_MODE` |
| Confidence threshold | `CONFIDENCE_THRESHOLD` | `CONF_THRESH`, `MIN_CONFIDENCE` |
| IoU threshold | `IOU_THRESHOLD` | `NMS_THRESH`, `IOU_THRESH` |
| API Host | `API_HOST` | `SERVER_HOST`, `HOST_IP` |
| API Port | `API_PORT` | `SERVER_PORT`, `PORT_NUM` |

---

## 4. Canonical Directory Structure & Paths

| Path | Purpose | State Policy |
|---|---|---|
| `data/raw/` | Pristine source downloads (Kaggle, Roboflow, RAR) | READ-ONLY. Never modify or delete files. |
| `data/extracted/` | Unpacked archives | Intermediary. |
| `data/interim/` | Harmonized, cleaned, and quarantined images | Workspace. |
| `data/processed/` | Deduplicated, verified, normalized dataset | Staged for splitting. |
| `data/final/` | Canonical Train / Val / Test splits for YOLO26 | Fixed versioned artifact. |
| `data/manifests/` | Central `manifest.json` and audit logs | Single source of dataset truth. |
| `configs/` | `base.yaml`, `dataset.yaml`, training configs | Version-controlled configuration. |
| `models/` | Trained model weights and metadata | Version-controlled checkpoint store. |
| `experiments/` | Structured experiment logs, metrics, plots | Reproducibility archive. |
| `src/krishiai/` | Production Python library | Single canonical implementation per domain. |
| `scripts/` | Standalone CLI entrypoints | Invokable operational scripts. |
| `tests/` | Unit, integration, and data tests | Synthetic fixtures only. |
| `docs/` | Persistent memory & scientific documentation | Updated every development session. |

---

## 5. Canonical Services & Class Names

- **Model Management Service:** `ModelManager` (in `src/krishiai/inference/model_manager.py`)
  - *Do NOT create: `YOLOModelManager`, `InferenceModelManager`, `ModelService`*
- **Dataset Manifest:** `DatasetManifest` & `DatasetRecord` (in `src/krishiai/data/manifest.py`)
  - *Do NOT create: `DatasetTracker`, `ManifestManager`*
- **Settings Loader:** `Settings.load()` (in `src/krishiai/config/settings.py`)
  - *Do NOT create: `ConfigManager`, `AppConfig`*

---

## 6. Canonical API Routes & Schemas

### 6.1 Endpoints
- `GET /health` — Health check & model status
- `POST /predict` — Image prediction (detection + classification)
- `GET /model/info` — Active model versions and taxonomy
- `GET /docs` — Swagger / OpenAPI documentation

### 6.2 Prediction Response Schema
```json
{
  "request_id": "req_uuid4",
  "timestamp": "2026-09-17T14:00:00Z",
  "model_version": "krishiai-yolo26-v1.0",
  "task_type": "detection | classification",
  "plant_species": "basil | aloe_vera | mint | unknown",
  "primary_condition": "string",
  "confidence": 0.92,
  "uncertainty_warning": null,
  "detections": [
    {
      "class_id": 0,
      "class_name": "basil_bacterial_spot",
      "category": "disease",
      "confidence": 0.94,
      "bbox": {
        "x_center": 0.52,
        "y_center": 0.41,
        "width": 0.18,
        "height": 0.22
      }
    }
  ],
  "latency_ms": {
    "preprocessing": 1.2,
    "inference": 12.4,
    "postprocessing": 0.8,
    "total": 14.4
  }
}
```

---

## 7. Versioning Invariants
- Models are tagged as: `krishiai-yolo26-<variant>-v<major>.<minor>` (e.g. `krishiai-yolo26-det-v1.0`, `krishiai-yolo26-cls-v1.0`)
- Datasets are tagged as: `krishiai-dataset-v<major>.<minor>` (e.g. `krishiai-dataset-v1.0`)
- Manifest records link each model artifact to its training dataset version, random seed, and Git commit hash.
