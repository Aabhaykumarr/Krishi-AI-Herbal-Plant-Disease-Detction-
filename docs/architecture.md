# KrishiAI — System Architecture

Comprehensive system architecture for the KrishiAI Herbal Crop Vision & Agronomic Diagnostic Platform.

## End-to-end Pipeline

```mermaid
flowchart TD
    subgraph Sources["Data Sources"]
        K[Kaggle Aloe Vera Dataset]
        R1[Roboflow Basil v4 Dataset]
        R2[Roboflow Mint v1 Dataset]
    end

    Sources --> ING["Data Extraction & Ingestion<br/>extract_datasets.py"]
    ING --> AUD["Physical Dataset Audit<br/>inspect_datasets.py"]
    AUD --> TAX["Taxonomy Harmonization<br/>configs/dataset.yaml"]
    TAX --> NORM["Polygon-to-BBox Normalization<br/>validate_labels.py"]
    NORM --> DUP["Deduplication & Quarantine<br/>build_dataset.py"]
    DUP --> FINAL[("Final Zero-Leakage Dataset<br/>data/final/")]

    FINAL --> DET["YOLO26 Object Detection<br/>krishiai_yolo26_det_baseline"]
    FINAL --> CLS["15-Class Leaf Classifier<br/>krishiai_leaf_classifier_balanced"]
    
    DET --> EVAL["Evaluation Engine<br/>scripts/evaluate.py"]
    CLS --> EVAL
    
    DET --> EXPORT["Model Serialization<br/>TorchScript / ONNX"]
    CLS --> EXPORT
    
    EXPORT --> GATE["Botanical Visual Feature Gate<br/>ExG + Chlorophyll + Entropy"]
    GATE --> SVC["ModelManager Gateway<br/>src/krishiai/inference/"]
    SVC --> API["FastAPI REST Backend<br/>src/krishiai/api/"]
    API --> UI["Web Diagnostic Dashboard<br/>static/index.html"]
    UI --> USER(("Agronomist / User"))

    classDef done fill:#d7f5dd,stroke:#2e7d32;
    class ING,AUD,TAX,NORM,DUP,FINAL,DET,CLS,EVAL,EXPORT,GATE,SVC,API,UI,USER done
```

## Component Responsibilities

| Component | Responsibility | Location | Status |
| :--- | :--- | :--- | :--- |
| **Dataset Manifest** | Source of truth for datasets, provenance, and hashes | `src/krishiai/data/manifest.py` | Complete & Verified |
| **Format Detection** | YOLO / COCO / VOC / Classification folder heuristics | `src/krishiai/data/formats.py` | Complete & Verified |
| **Validators** | Image, YOLO label, and polygon coordinate validation | `src/krishiai/data/validators.py` | Complete & Verified |
| **Hashing Engine** | Exact SHA256 & perceptual hash collision prevention | `src/krishiai/data/hashing.py` | Complete & Verified |
| **Settings** | Central configuration and environment overrides | `src/krishiai/config/settings.py` | Complete & Verified |
| **Training Engine** | YOLO26 Detection and Classification training | `src/krishiai/training/` | Complete & Verified |
| **Evaluation Suite** | Precision, recall, mAP, Top-1/Top-5 accuracy metrics | `src/krishiai/evaluation/` | Complete & Verified |
| **Model Serialization** | Multi-format export (TorchScript, ONNX, OpenVINO) | `src/krishiai/models/export.py` | Complete & Verified |
| **Botanical Gating** | $ExG$, chlorophyll spectrum, and entropy OOD filter | `src/krishiai/inference/model_manager.py` | Complete & Verified |
| **Unified Gateway** | Hierarchical dual-pipeline dispatch (`ModelManager`) | `src/krishiai/inference/model_manager.py` | Complete & Verified |
| **API Layer** | FastAPI HTTP server, OpenAPI v3 contracts | `src/krishiai/api/` | Complete & Verified |
| **Web Dashboard** | Responsive single-page UI with HTML5 canvas | `static/index.html` | Complete & Verified |
| **Benchmarking Engine**| Statistical latency profiling (P50/P90/P95/P99, FPS) | `src/krishiai/evaluation/benchmark.py` | Complete & Verified |
| **Robustness Suite** | 6-factor environmental perturbation stress-testing | `src/krishiai/evaluation/robustness.py` | Complete & Verified |
| **Batch Predictor** | Directory batch inference and annotated overlays | `src/krishiai/inference/batch.py` | Complete & Verified |
| **Containerization** | Production multi-stage Docker & compose config | `Dockerfile`, `docker-compose.yml` | Complete & Verified |
