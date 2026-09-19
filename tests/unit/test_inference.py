from pathlib import Path

import numpy as np
from PIL import Image
from krishiai.inference.model_manager import (
    ModelManager,
    infer_category,
    infer_species,
)
from krishiai.inference.types import (
    BBox,
    DetectionItem,
    LatencyInfo,
    PredictionResponse,
)


def test_infer_category():
    assert infer_category("basil_bacterial_spot") == "disease"
    assert infer_category("basil_magnesium_deficiency") == "deficiency"
    assert infer_category("basil_healthy") == "healthy"
    assert infer_category("mint_dried") == "condition_and_freshness"
    assert infer_category("aloe_vera_sunburn") == "condition_and_freshness"
    assert infer_category("aloe_vera_rust") == "disease"


def test_infer_species():
    assert infer_species("basil_powdery_mildew") == "basil"
    assert infer_species("mint_fresh") == "mint"
    assert infer_species("aloe_vera_healthy") == "aloe_vera"
    assert infer_species("random_unrelated") == "unknown"


def test_prediction_response_schema():
    bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.3)
    det = DetectionItem(
        class_id=0,
        class_name="basil_bacterial_spot",
        category="disease",
        confidence=0.92,
        bbox=bbox,
    )
    latency = LatencyInfo(
        preprocessing=1.2,
        inference=10.5,
        postprocessing=0.5,
        total=12.2,
    )
    resp = PredictionResponse(
        task_type="detection",
        plant_species="basil",
        primary_condition="basil_bacterial_spot",
        confidence=0.92,
        detections=[det],
        latency_ms=latency,
    )

    d = resp.to_dict()
    assert d["task_type"] == "detection"
    assert d["plant_species"] == "basil"
    assert len(d["detections"]) == 1
    assert d["detections"][0]["bbox"]["x_center"] == 0.5
    assert d["latency_ms"]["total"] == 12.2


def test_model_manager_initialization_empty():
    mm = ModelManager()
    assert mm.det_model is None
    assert mm.cls_model is None


def test_model_manager_classification_inference(tmp_path):
    # Test inference using balanced classifier if available, or pretrained
    checkpoint = Path("models/checkpoints/krishiai_leaf_classifier_balanced/best.pt")
    if checkpoint.exists():
        weights = str(checkpoint)
    else:
        weights = "models/pretrained/yolo26n-cls.pt"
    mm = ModelManager(classification_model_path=weights, device="cpu")

    # Create synthetic botanical green foliage image
    img = Image.new("RGB", (224, 224), (40, 160, 40))
    resp = mm.predict(img)

    assert resp.task_type in ("classification", "unrecognized")
    assert resp.latency_ms is not None
    assert resp.latency_ms.total > 0.0


def test_model_manager_non_plant_rejection():
    mm = ModelManager.create_default()
    # Create white document with black text/lines
    doc_img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    resp = mm.predict(doc_img)

    assert resp.task_type == "unrecognized"
    assert resp.primary_condition == "non_plant_or_unrecognized"
    assert resp.confidence == 0.0
    assert "No botanical leaf detected" in resp.uncertainty_warning
    assert len(resp.detections) == 0
