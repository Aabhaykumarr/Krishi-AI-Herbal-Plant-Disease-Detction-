"""Inference gateway and types for KrishiAI."""
from krishiai.inference.batch import annotate_image_with_prediction, run_batch_inference
from krishiai.inference.model_manager import ModelManager, infer_category, infer_species
from krishiai.inference.types import BBox, DetectionItem, LatencyInfo, PredictionResponse

__all__ = [
    "ModelManager",
    "PredictionResponse",
    "DetectionItem",
    "BBox",
    "LatencyInfo",
    "infer_category",
    "infer_species",
    "run_batch_inference",
    "annotate_image_with_prediction",
]
