"""Standardized data schemas and types for KrishiAI model inference (Section 6)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class BBox:
    """Normalized bounding box coordinates [0, 1]."""
    x_center: float
    y_center: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class DetectionItem:
    """Individual object detection within an image."""
    class_id: int
    class_name: str
    category: str  # "disease", "deficiency", "condition_and_freshness", "healthy"
    confidence: float
    bbox: BBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
        }


@dataclass
class LatencyInfo:
    """Execution latency breakdown in milliseconds."""
    preprocessing: float
    inference: float
    postprocessing: float
    total: float
    gemini: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "preprocessing": round(self.preprocessing, 2),
            "inference": round(self.inference, 2),
            "postprocessing": round(self.postprocessing, 2),
            "total": round(self.total, 2),
            "gemini": round(self.gemini, 2),
        }


@dataclass
class PredictionResponse:
    """Standardized response schema for KrishiAI inference gateway."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_version: str = "krishiai-yolo26-v1.0"
    task_type: str = "detection"  # "detection" | "classification"
    plant_species: str = "unknown"  # "basil" | "aloe_vera" | "mint" | "unknown"
    primary_condition: str = "unknown"
    confidence: float = 0.0
    uncertainty_warning: Optional[str] = None
    detections: List[DetectionItem] = field(default_factory=list)
    latency_ms: Optional[LatencyInfo] = None
    verification: Optional[Dict[str, Any]] = None
    original_detections: Optional[List[Dict[str, Any]]] = None
    diagnosis_source: Optional[str] = None
    confidence_source: Optional[str] = None
    condition_category: Optional[str] = None
    yolo_result: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "task_type": self.task_type,
            "plant_species": self.plant_species,
            "primary_condition": self.primary_condition,
            "confidence": round(self.confidence, 4),
            "uncertainty_warning": self.uncertainty_warning,
            "detections": [d.to_dict() for d in self.detections],
            "latency_ms": self.latency_ms.to_dict() if self.latency_ms else None,
        }
        if self.verification is not None:
            result["verification"] = self.verification
        if self.original_detections is not None:
            result["original_detections"] = self.original_detections
        if self.diagnosis_source is not None:
            result["diagnosis_source"] = self.diagnosis_source
        if self.confidence_source is not None:
            result["confidence_source"] = self.confidence_source
        if self.condition_category is not None:
            result["condition_category"] = self.condition_category
        if self.yolo_result is not None:
            result["yolo_result"] = self.yolo_result
        if self.recommendation is not None:
            result["recommendation"] = self.recommendation
        return result
