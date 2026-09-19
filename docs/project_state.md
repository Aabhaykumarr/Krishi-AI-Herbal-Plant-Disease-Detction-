# KrishiAI — Project State

> This file is the single source of truth for "where the project actually
> is." Per Sections 96/113, every future work session — human or AI — must
> read this file (and docs/decisions.md) before making changes, and must
> update it after any meaningful change. Do not restart the project or
> re-derive this from scratch; inspect the repo and continue from here.
>
> Last updated: 2026-09-19 (Phase 2 & Phase 3 Full Infrastructure Operational)

## 1. Current phase
**Phase 2 Model Development & Phase 3 REST API / Diagnostic Web Service Fully Operational with Hierarchical Dual-Pipeline Routing.**

## 2. What exists right now
- **Canonical Harmonized Datasets in `data/final/`**:
  - **Detection (`data/final/detection/`)**:
    - 8,629 total images, 15,834 bounding-box annotations across 10 canonical classes (7 Basil + 3 Mint).
    - Leakage-free scene-grouped split: 6,042 train (70.02%), 1,293 val (14.98%), 1,294 test (14.99%).
    - 39 polygon segmentation annotations cleanly normalized to enclosing bounding boxes (Decision D-008).
    - Canonical `data/final/detection/data.yaml` YOLO26 dataset configuration.
  - **Classification (`data/final/classification/`)**:
    - 2,416 clean unique images across 5 Aloe Vera classes (`aloe_vera_anthracnose`, `aloe_vera_healthy`, `aloe_vera_leaf_spot`, `aloe_vera_rust`, `aloe_vera_sunburn`).
    - Stratified split: 1,689 train (69.91%), 360 val (14.90%), 367 test (15.19%).
  - **Unified 15-Class Leaf Crop Dataset (`data/final/leaf_crops/` & `data/final/leaf_crops_balanced/`)**:
    - 7,097 images across all 15 botanical classes (7 Basil + 3 Mint + 5 Aloe Vera) extracted directly from bounding boxes and classification datasets.
    - Balanced distribution: 3,689 train, 798 val, 2,610 test.
- **Quarantine and Integrity Layer**:
  - `data/interim/quarantine/aloevera_conflicts/` storing 35 conflicting cross-class duplicate images across 16 groups (documented in `aloevera_conflicts.json`).
  - 47 same-class duplicate groups deduplicated with single clean representative instances retained.
- **Trained Model Checkpoints & Export Pipelines (`models/`)**:
  - `models/checkpoints/krishiai_yolo26_det_baseline/best.pt` — Fine-tuned YOLO26n 10-class object detection checkpoint for Basil & Mint (5.38 MB).
  - `models/checkpoints/krishiai_leaf_classifier_balanced/best.pt` — Balanced 15-class YOLO26n-cls fine-grained leaf classifier (**71.99% Top-1, 97.70% Top-5 accuracy on holdout test set**, 12.0 MB).
  - `models/checkpoints/krishiai_leaf_classifier_balanced/best.torchscript` — Optimized TorchScript export for 15-class classifier (6.08 MB).
  - `models/checkpoints/baseline_cls_run/best.pt` — Fine-tuned YOLO26n-cls 5-class checkpoint for Aloe Vera (3.19 MB).
  - `models/checkpoints/baseline_cls_run/best.torchscript` — Optimized TorchScript export (6.03 MB).
  - `src/krishiai/models/export.py` & `scripts/export_models.py` — Multi-format export engine (ONNX, TorchScript, OpenVINO, TensorRT).
- **Unified Hierarchical Dual-Pipeline Inference Gateway (`src/krishiai/inference/model_manager.py`)**:
  - Seamlessly orchestrates object detection (for localized spot/lesion bounding boxes) and 15-class fine-grained leaf classification (for holistic whole-leaf condition diagnosis).
  - Evaluates localized bounding boxes when symptoms are present; seamlessly falls back to 15-class whole-leaf classification on macro/single-leaf photos or subtle symptoms.
  - Implements botanical visual feature gating (`is_botanical_image`) to detect and reject non-plant documents, roadmaps, flowcharts, and diagrams without false diagnosis or hallucination.
  - Implements species-conditioned probability filtering and normalization when crop species (`basil`, `mint`, `aloe_vera`) is selected.
  - Generates visual bounding box annotations for canvas rendering on the web dashboard across all crops.
- **FastAPI Backend & Interactive Diagnostic Web Application (`src/krishiai/api/` & `static/`)**:
  - FastAPI application factory in `src/krishiai/api/app.py` with CORS, health monitoring (`/health`), taxonomy inspection (`/model/info`), and high-performance prediction (`POST /predict`).
  - Interactive Web Dashboard in `static/index.html` (TailwindCSS, Lucide icons) with live drag-and-drop, bounding-box canvas rendering, latency profile metrics, non-plant alert banners, real botanical leaf sample buttons (`/static/samples/`), and agronomic treatment recommendations.
  - Server currently live on `http://localhost:8000` (PID managed in background).
  - Verified with 10-point automated integration suite `scripts/test_live_server.py`.
  - Evaluated on authentic unseen internet botanical images (`scripts/test_unseen_images.py`) and OOD rejection suite (`scripts/test_ood_and_live.py`).
- **Testing, Benchmarking, & Robustness Assurance**:
  - **Comprehensive Unit & Integration Test Suite**: 50 passing tests (`pytest tests/`) covering data validation, hashing, manifests, formats, dataset construction, training, evaluation, inference gateway, export, batch processing, benchmarking, robustness, OOD non-plant rejection, and API endpoints (100% pass rate).
  - **Live Server Integration Suite**: 10 passing end-to-end integration tests (`scripts/test_live_server.py`).
  - **Latency & Throughput Profiler (`src/krishiai/evaluation/benchmark.py`, `scripts/benchmark_latency.py`)**: High-precision statistical percentile profiling (mean, std, P50, P90, P95, P99) and FPS throughput across all execution stages.
  - **Environmental Robustness Perturbation Suite (`src/krishiai/evaluation/robustness.py`, `scripts/robustness_tests.py`)**: Automated stress-testing across 6 corruption dimensions (brightness shifts, contrast decay, Gaussian blur, Gaussian noise, rotation, center occlusion).
  - **Batch Prediction CLI (`src/krishiai/inference/batch.py`, `scripts/batch_predict.py`)**: Bulk image folder processing with structured JSON/CSV reports and annotated visualization overlays.
- **Production Containerization**:
  - Production multi-stage `Dockerfile` with optimized runtime OS libraries (`libgl1`, `libglib2.0-0`), CPU PyTorch wheel optimizations, healthcheck probe (`/health`), and `docker-compose.yml` service specification.

## 3. What does NOT exist yet
- Long multi-epoch GPU fine-tuning runs (ready for cloud/cluster execution with `scripts/train.py`).
- Mobile deployment wrapper / Flutter app (planned Month 4).

## 4. Next concrete action
- Continue or monitor model training runs on dedicated compute if higher epoch counts are desired:
  - `python scripts/train.py --task detection --config configs/training/detection_baseline.yaml --epochs 50`
  - `python scripts/train.py --task classification --config configs/training/classification_baseline.yaml --epochs 50`
- Launch and serve the full API and web diagnostic interface:
  - `python scripts/run_server.py --port 8000 --reload`
- Access the web interface at `http://localhost:8000` and Swagger docs at `http://localhost:8000/docs`.

## 5. Key interfaces
- `krishiai.api.app.create_app()` — FastAPI application factory.
- `krishiai.inference.model_manager.ModelManager` — Canonical dual-pipeline inference gateway.
- `krishiai.training.trainer.KrishiTrainer` — Training orchestrator.
- `krishiai.evaluation.evaluator.KrishiEvaluator` — Evaluation engine.
- `krishiai.models.export.export_model` — Model export engine.
- `scripts/run_server.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/export_models.py`, `scripts/build_dataset.py`.

## 6. Dataset / model / taxonomy versions
- Dataset version: `krishiai-dataset-v1.0` (in `data/final/`)
- Class mapping version: `v1.0.0` (in `configs/dataset.yaml`)
- Canonical Taxonomy: `v1.0.0`
- Model version: `krishiai-yolo26-v1.0`
