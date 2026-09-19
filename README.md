# 🌱 KrishiAI: Intelligent Herbal Crop Vision & Agronomic Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO26](https://img.shields.io/badge/YOLO26-Object%20Detection%20%26%20Classification-00FFFF.svg)](https://docs.ultralytics.com/)
[![Tests](https://img.shields.io/badge/Tests-50%2F50%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**KrishiAI** is an end-to-end computer vision and agronomic intelligence platform designed for localized disease detection, nutrient deficiency diagnosis, and condition assessment across high-value herbal crops: **Basil**, **Aloe Vera**, and **Mint**.

Built on a unified **Hierarchical Dual-Pipeline Architecture**, KrishiAI combines spatial multi-leaf object detection with fine-grained 15-class leaf classification, front-line botanical visual feature gating, out-of-distribution (OOD) non-plant rejection, hardware benchmarking, and a responsive web dashboard.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Herbal Crop Taxonomy](#-herbal-crop-taxonomy)
- [Botanical Feature Gating & OOD Defense](#-botanical-feature-gating--ood-defense)
- [Quick Start & Installation](#-quick-start--installation)
- [Running the Live Service](#-running-the-live-service)
- [REST API Reference](#-rest-api-reference)
- [Training & Model Evaluation](#-training--model-evaluation)
- [Benchmarking & Robustness Suite](#-benchmarking--robustness-suite)
- [Docker Containerization](#-docker-containerization)
- [Test Suite](#-test-suite)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🌟 Key Features

- **Hierarchical Dual-Pipeline Gateway**: Seamlessly routes inputs between localized object detection (YOLO26n for lesion/spot bounding boxes) and holistic leaf classification (15-class balanced YOLO26n-cls), achieving 72.0% Top-1 / 97.7% Top-5 accuracy.
- **Botanical Visual Gating & OOD Rejection**: Multi-spectral Excess Green Index ($ExG$), chlorophyll hue band gating ($45^\circ - 180^\circ$), and normalized Shannon entropy ($H > 0.75$) prevent model hallucinations on non-plant images (IoT packaging boxes, book pages, diagrams, documents).
- **Zero-Leakage Harmonized Datasets**: Scene-grouped deduplication and hash collision audits across 11,045+ images and 15,834 bounding-box annotations.
- **Interactive Diagnostic Web Dashboard**: Built with TailwindCSS and HTML5 Canvas for real-time bounding box visualization, drag-and-drop uploads, latency breakdowns, and actionable agronomic treatments.
- **Production REST API**: High-performance FastAPI server with OpenAPI v3 docs, health checks, and JSON response contracts.
- **Hardware Benchmarking & Environmental Robustness Engine**: Latency percentile profiling (P50/P90/P95/P99) and 6-factor perturbation stress-testing (brightness, contrast, blur, noise, rotation, occlusion).
- **Multi-Format Export & Containerization**: TorchScript, ONNX, and OpenVINO serialization pipelines + production multi-stage Docker & Docker Compose configs.

---

## 🏛 System Architecture

```
                                  [ User Input / API Upload ]
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Botanical Visual Feature Gate   │
                              │     • Excess Green Index (ExG)   │
                              │     • Chlorophyll Hue Spectrum   │
                              │     • Paper & Packaging Filter   │
                              └────────────────┬─────────────────┘
                                               │
                                 ┌─────────────┴─────────────┐
                         [Pass Botanical]             [Fail / Non-Plant]
                                 │                           │
                                 ▼                           ▼
                   ┌───────────────────────────┐   ┌───────────────────────────┐
                   │   Hierarchical Router     │   │   OOD Rejection Response  │
                   └─────────────┬─────────────┘   │   (No False Hallucinations│
                                 │                 │    or Canvas Bounding Box)│
                   ┌─────────────┴─────────────┐   └───────────────────────────┘
                   │                           │
                   ▼                           ▼
     ┌───────────────────────────┐   ┌───────────────────────────┐
     │   YOLO26 Object Detection │   │ 15-Class Leaf Classifier  │
     │   (Localized BBoxes for   │   │ (Holistic Diagnosis &     │
     │    Multi-Leaf Lesions)    │   │  Species-Conditioned Mass)│
     └─────────────┬─────────────┘   └─────────────┬─────────────┘
                   │                               │
                   └─────────────┬─────────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │ Unified Prediction Output │
                   │  • Disease Diagnosis      │
                   │  • Bounding Box Canvas    │
                   │  • Agronomic Treatment    │
                   │  • Sub-20ms Latency Stats │
                   └───────────────────────────┘
```

---

## 🌿 Herbal Crop Taxonomy

KrishiAI supports **15 canonical conditions** across three primary crops:

| Crop Species | Canonical Condition / Class Name | Category | Modality |
| :--- | :--- | :--- | :--- |
| **Basil** (*Ocimum basilicum*) | `basil_bacterial_spot` | Disease | Spatial BBox / Classifier |
| | `basil_downy_mildew` | Disease | Spatial BBox / Classifier |
| | `basil_powdery_mildew` | Disease | Spatial BBox / Classifier |
| | `basil_fusarium_wilt` | Disease | Spatial BBox / Classifier |
| | `basil_bug_spot` | Condition | Spatial BBox / Classifier |
| | `basil_magnesium_deficiency` | Nutrient Deficiency | Spatial BBox / Classifier |
| | `basil_healthy` | Healthy Foliage | Spatial BBox / Classifier |
| **Mint** (*Mentha*) | `mint_fresh` | Condition & Freshness | Spatial BBox / Classifier |
| | `mint_dried` | Condition & Freshness | Spatial BBox / Classifier |
| | `mint_spoiled` | Condition & Freshness | Spatial BBox / Classifier |
| **Aloe Vera** (*Aloe barbadensis*) | `aloe_vera_anthracnose` | Disease | Full-Frame Classifier |
| | `aloe_vera_rust` | Disease | Full-Frame Classifier |
| | `aloe_vera_leaf_spot` | Disease | Full-Frame Classifier |
| | `aloe_vera_sunburn` | Condition | Full-Frame Classifier |
| | `aloe_vera_healthy` | Healthy Foliage | Full-Frame Classifier |

---

## 🛡 Botanical Feature Gating & OOD Defense

To prevent vision models from hallucinating botanical diseases on non-plant inputs, KrishiAI executes a two-tier defense:

1. **Spectral Feature Extraction**:
   - **Excess Green Index ($ExG$)**: $ExG = 2G - R - B$. Foliage requires $ExG > 0.03$ with green channel dominance.
   - **Chlorophyll Hue Band**: $45^\circ \le \text{Hue} \le 180^\circ$ with saturation $\ge 0.06$.
   - **Document & Cardboard Segmentation**: Detects white paper ($|R-G| < 0.045$, $|G-B| < 0.045$), warm book pages ($R \ge G \ge B$, Sat $< 0.28$), and cardboard packaging ($Hue \in [18^\circ, 42^\circ]$). If non-plant ratios exceed thresholds, the input is immediately rejected.
2. **Normalized Shannon Entropy Gating**:
   $$H(p) = -\frac{\sum_{i=1}^{K} p_i \log_2(p_i)}{\log_2(K)}$$
   Closed-set classifier distributions with high entropy ($H > 0.75$) or diffuse confidence ($p_{\text{top1}} < 0.40$) are intercepted before output.

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.13)
- PyTorch 2.0+

### Setup
```bash
# Clone the repository
git clone https://github.com/Aabhaykumarr/krishi-ai.git
cd krishi-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dependencies
pip install -e ".[dev]"
```

---

## 🖥 Running the Live Service

### Start the FastAPI Server & Web Dashboard
```bash
python scripts/run_server.py --port 8000 --reload
```

- **Interactive Web UI**: Navigate to [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Docs (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 REST API Reference

### Health Check
`GET /health`
```json
{
  "status": "healthy",
  "service": "krishiai-api",
  "version": "1.0.0",
  "models": {
    "detection": true,
    "classification": true,
    "device": "cpu"
  }
}
```

### Run Prediction
`POST /predict` (multipart/form-data)
- **Parameters**:
  - `file`: Image file (`image/jpeg`, `image/png`, `image/webp`)
  - `plant_species`: *(Optional)* `"basil"`, `"mint"`, `"aloe_vera"`, or `"all"`
  - `conf`: *(Optional)* Confidence threshold (default `0.25`)
  - `iou`: *(Optional)* NMS IoU threshold (default `0.60`)

**Sample Response**:
```json
{
  "task_type": "detection",
  "plant_species": "basil",
  "primary_condition": "basil_powdery_mildew",
  "confidence": 0.8942,
  "uncertainty_warning": null,
  "detections": [
    {
      "class_id": 2,
      "class_name": "basil_powdery_mildew",
      "category": "disease",
      "confidence": 0.8942,
      "bbox": {
        "x_center": 0.4821,
        "y_center": 0.3915,
        "width": 0.2841,
        "height": 0.3129
      }
    }
  ],
  "latency_ms": {
    "preprocessing": 1.45,
    "inference": 11.23,
    "postprocessing": 0.82,
    "total": 13.50
  }
}
```

---

## 🏋️ Training & Model Evaluation

### 1. Build Harmonized Datasets
```bash
python scripts/build_dataset.py --clean
python scripts/build_leaf_crops_dataset.py
python scripts/build_balanced_leaf_dataset.py
```

### 2. Train Models
```bash
# Train YOLO26 Object Detection (Basil & Mint)
python scripts/train.py --task detection --config configs/training/detection_baseline.yaml --epochs 50

# Train 15-Class Leaf Classifier
python scripts/train.py --task classification --config configs/training/leaf_classifier.yaml --epochs 30
```

### 3. Evaluate Models on Holdout Test Split
```bash
python scripts/evaluate.py --task classification --model models/checkpoints/krishiai_leaf_classifier_balanced/best.pt --data data/final/leaf_crops_balanced --split test
```

### 4. Export Checkpoints
```bash
# Export to TorchScript and ONNX
python scripts/export_models.py --model models/checkpoints/krishiai_leaf_classifier_balanced/best.pt --format torchscript
python scripts/export_models.py --model models/checkpoints/krishiai_leaf_classifier_balanced/best.pt --format onnx
```

---

## 📊 Benchmarking & Robustness Suite

### Latency & Throughput Profiling
Profiles statistical latency percentiles (P50, P90, P95, P99) and FPS throughput:
```bash
python scripts/benchmark_latency.py --runs 50 --warmup 10
```

### Environmental Robustness Stress-Testing
Evaluates resilience across 6 real-world corruptions:
```bash
python scripts/robustness_tests.py --max-images 20
```

---

## 🐳 Docker Containerization

Run the production stack in an isolated container:
```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Check health
curl http://localhost:8000/health
```

---

## 🧪 Test Suite

Run the full unit, integration, and assurance test suite:
```bash
# Run all 50 pytest tests
pytest tests/ -v

# Run live server integration suite
python scripts/test_live_server.py

# Run Out-of-Distribution (OOD) rejection assurance suite
python scripts/test_ood_and_live.py
```

---

## 📂 Project Structure

```
krishi-ai/
├── configs/                  # Dataset taxonomies and training configs
│   ├── dataset.yaml          # Canonical 15-class taxonomy & color definitions
│   └── training/             # Hyperparameter configurations for detection & classification
├── data/
│   ├── final/                # Harmonized, zero-leakage datasets (detection, classification, leaf_crops)
│   └── manifests/            # Central dataset manifest & build reports
├── docs/                     # Architecture specs, decision logs, and contract documentation
│   ├── architecture.md       # Full end-to-end system architecture
│   ├── decisions.md          # Architectural decision log (D-001 through D-014)
│   ├── known_issues.md       # Leakage and annotation audit log
│   ├── project_contract.md   # System schema & REST contracts
│   └── project_history.md    # Chronological engineering record
├── models/
│   ├── checkpoints/          # Trained weights (YOLO26 detection & 15-class classifier)
│   └── pretrained/           # Backbone weights (yolo26n.pt, yolo26n-cls.pt)
├── scripts/                  # Executable CLI tools
│   ├── build_dataset.py      # Dataset harmonization & deduplication pipeline
│   ├── run_server.py         # FastAPI service launcher
│   ├── train.py              # YOLO26 training engine
│   ├── evaluate.py           # Evaluation & metrics generator
│   ├── export_models.py      # Multi-format serialization (TorchScript, ONNX)
│   ├── benchmark_latency.py  # Latency & FPS profiler
│   ├── robustness_tests.py   # Environmental perturbation engine
│   ├── batch_predict.py      # Directory batch inference CLI
│   └── test_ood_and_live.py  # OOD non-plant rejection & botanical assurance suite
├── src/krishiai/             # Core Python package
│   ├── api/                  # FastAPI endpoints & app factory
│   ├── config/               # Settings management
│   ├── data/                 # Validators, hashing, manifest schemas
│   ├── evaluation/           # Metrics, benchmark, and robustness logic
│   ├── inference/            # Unified ModelManager & batch processing
│   ├── models/               # Serialization and export engine
│   └── training/             # Training orchestrator
├── static/                   # Web dashboard frontend
│   ├── index.html            # Diagnostic UI with HTML5 canvas
│   └── samples/              # Authentic leaf test samples
├── tests/                    # Comprehensive unit & integration tests
│   ├── conftest.py           # Synthetic fixtures
│   └── unit/                 # 50 unit tests across all subsystems
├── Dockerfile                # Multi-stage production container definition
├── docker-compose.yml        # Docker compose specification
├── pyproject.toml            # Project metadata & build dependencies
└── README.md                 # Project documentation
```

---

## 👤 Author

**Aabhay Kumar**
- **GitHub**: [@Aabhaykumarr](https://github.com/Aabhaykumarr)
- **Project**: KrishiAI — Herbal Crop Vision & Agronomic Diagnostic System

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
