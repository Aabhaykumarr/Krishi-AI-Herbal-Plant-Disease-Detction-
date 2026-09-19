"""ModelManager: Unified Inference Gateway for KrishiAI (Section 51 & Decision D-007)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

from krishiai.config.settings import Settings
from krishiai.inference.types import (
    BBox,
    DetectionItem,
    LatencyInfo,
    PredictionResponse,
)
from krishiai.training.trainer import get_optimal_device
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


def infer_category(class_name: str) -> str:
    """Categorizes a canonical class name into disease, deficiency, condition_and_freshness, or healthy."""
    c = class_name.lower()
    if "healthy" in c:
        return "healthy"
    if "deficiency" in c:
        return "deficiency"
    if any(k in c for k in ("dried", "fresh", "spoiled", "sunburn")):
        return "condition_and_freshness"
    if any(k in c for k in ("spot", "mildew", "rust", "anthracnose", "rot", "blight")):
        return "disease"
    return "disease"


def infer_species(class_name: str) -> str:
    """Infers plant species from class name."""
    c = class_name.lower()
    if "basil" in c:
        return "basil"
    if "mint" in c:
        return "mint"
    if "aloe" in c:
        return "aloe_vera"
    return "unknown"


def is_botanical_image(pil_img: Image.Image) -> tuple[bool, str]:
    """Analyzes image color distribution and texture to detect non-plant documents,

    roadmaps, flowcharts, book pages, packaging boxes, or out-of-distribution subjects.
    """
    thumb = pil_img.resize((128, 128)).convert("RGB")
    arr = np.array(thumb, dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 1. Excess Green Index (ExG)
    exg = 2.0 * g - r - b

    # 2. HSV color space calculation
    cmax = np.max(arr, axis=2)
    cmin = np.min(arr, axis=2)
    delta = cmax - cmin

    sat = np.zeros_like(cmax)
    non_zero = cmax > 0.03
    sat[non_zero] = delta[non_zero] / cmax[non_zero]
    mean_sat = float(np.mean(sat))

    hue = np.zeros_like(cmax)
    r_mask = (cmax == r) & (delta > 0)
    g_mask = (cmax == g) & (delta > 0)
    b_mask = (cmax == b) & (delta > 0)

    hue[r_mask] = 60.0 * (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6)
    hue[g_mask] = 60.0 * (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2)
    hue[b_mask] = 60.0 * (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4)

    # 3. Authentic Green Chlorophyll & Botanical Foliage Tones
    is_chlorophyll = (hue >= 45) & (hue <= 175) & (sat >= 0.06) & (g >= 0.75 * r) & (g >= 0.85 * b)
    is_succulent_green = (hue >= 38) & (hue <= 185) & (sat >= 0.08) & (g >= 0.70 * r) & (exg > 0.03)
    foliage_mask = is_chlorophyll | is_succulent_green
    foliage_ratio = float(np.sum(foliage_mask) / (128 * 128))

    # 4. Document / Book Page / Paper / Packaging Cardboard Detection
    is_neutral_paper = (np.abs(r - g) < 0.045) & (np.abs(g - b) < 0.045) & (cmax > 0.45)
    is_warm_book_paper = (r >= g) & (g >= b) & (sat < 0.28) & (cmax > 0.45)
    is_cardboard_box = (r > g) & (g > b) & (hue >= 18) & (hue <= 42) & (sat >= 0.20) & (sat <= 0.60) & (cmax > 0.35)
    paper_ratio = float(np.sum(is_neutral_paper | is_warm_book_paper) / (128 * 128))
    cardboard_ratio = float(np.sum(is_cardboard_box) / (128 * 128))

    # Gating Rules:
    if paper_ratio > 0.40 and foliage_ratio < 0.10:
        return False, f"Document / Book page detected ({paper_ratio*100:.1f}% paper, {foliage_ratio*100:.1f}% botanical foliage)."

    if cardboard_ratio > 0.30 and foliage_ratio < 0.06:
        return False, f"Packaging / Cardboard surface detected ({cardboard_ratio*100:.1f}% cardboard, {foliage_ratio*100:.1f}% botanical foliage)."

    if foliage_ratio < 0.03:
        return False, f"No botanical plant foliage or chlorophyll signatures detected ({foliage_ratio*100:.1f}% foliage)."

    if mean_sat < 0.05 and foliage_ratio < 0.04:
        return False, f"Low-saturation non-botanical image (Saturation={mean_sat:.2f}, Foliage={foliage_ratio*100:.1f}%)."

    return True, "Valid botanical visual signatures."


class ModelManager:
    """Canonical model management and inference gateway for KrishiAI.

    Provides a unified interface across detection (Basil/Mint) and
    classification (Aloe Vera) models, with latency tracking and OOD handling.
    """

    def __init__(
        self,
        detection_model_path: Optional[Union[str, Path]] = None,
        classification_model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.6,
    ):
        self.settings = Settings.load()
        self.device = get_optimal_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.det_model: Optional[YOLO] = None
        self.cls_model: Optional[YOLO] = None

        if detection_model_path and Path(detection_model_path).exists():
            logger.info(f"Loading detection model from {detection_model_path}")
            self.det_model = YOLO(str(detection_model_path))

        if classification_model_path and Path(classification_model_path).exists():
            logger.info(f"Loading classification model from {classification_model_path}")
            self.cls_model = YOLO(str(classification_model_path))

    @classmethod
    def create_default(
        cls,
        detection_model_path: Optional[Union[str, Path]] = None,
        classification_model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.6,
    ) -> ModelManager:
        """Factory method to create a ModelManager with auto-discovered checkpoints."""
        if detection_model_path is None:
            det_candidates = [
                Path("models/checkpoints/krishiai_yolo26_det_baseline/best.pt"),
                Path("runs/detect/experiments/detection/krishiai_yolo26_det_baseline/weights/best.pt"),
                Path("models/pretrained/yolo26n.pt"),
            ]
            for c in det_candidates:
                if c.exists():
                    detection_model_path = c
                    break

        if classification_model_path is None:
            cls_candidates = [
                Path("models/checkpoints/krishiai_leaf_classifier_balanced/best.pt"),
                Path("runs/classify/models/runs/classify/krishiai_leaf_classifier_balanced/weights/best.pt"),
                Path("models/checkpoints/baseline_cls_run/best.pt"),
                Path("runs/classify/experiments/classification/baseline_cls_run/weights/best.pt"),
                Path("models/pretrained/yolo26n-cls.pt"),
            ]
            for c in cls_candidates:
                if c.exists():
                    classification_model_path = c
                    break

        return cls(
            detection_model_path=detection_model_path,
            classification_model_path=classification_model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        plant_species: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> PredictionResponse:
        """Executes unified inference on an input image.

        Hierarchically orchestrates object detection (for localized spot/lesion
        bounding boxes) and 15-class whole-leaf classification (for holistic
        agronomic condition diagnosis).
        """
        t0 = time.perf_counter()
        active_conf = conf if conf is not None else self.conf_threshold
        active_iou = iou if iou is not None else self.iou_threshold

        # Step 1: Preprocessing
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        t1 = time.perf_counter()
        preproc_ms = (t1 - t0) * 1000.0

        # Step 2: Botanical Visual Gating (Out-Of-Distribution / Non-Plant detection)
        is_botanical, ood_reason = is_botanical_image(pil_img)
        if not is_botanical:
            t_end = time.perf_counter()
            total_ms = (t_end - t0) * 1000.0
            return PredictionResponse(
                task_type="unrecognized",
                plant_species="unknown",
                primary_condition="non_plant_or_unrecognized",
                confidence=0.0,
                uncertainty_warning=f"No botanical leaf detected: {ood_reason} Please upload a clear photo of Basil, Mint, or Aloe Vera foliage.",
                detections=[],
                latency_ms=LatencyInfo(
                    preprocessing=preproc_ms,
                    inference=0.0,
                    postprocessing=total_ms - preproc_ms,
                    total=total_ms,
                ),
            )

        # Step 3: Hierarchical Dual-Pipeline Execution
        # If species is explicitly aloe_vera and cls_model exists -> prioritize classification
        if plant_species == "aloe_vera" and self.cls_model is not None:
            return self._predict_classification(pil_img, preproc_ms, t1, active_conf, requested_species=plant_species)

        # For Basil, Mint, or general: first try object detection for localized symptoms
        if self.det_model is not None:
            det_resp = self._predict_detection(pil_img, preproc_ms, t1, active_conf, active_iou, requested_species=plant_species)
            # If high-confidence bounding boxes were detected, return detection result
            if det_resp.detections and len(det_resp.detections) > 0 and det_resp.primary_condition != "no_detections":
                return det_resp

        # If no localized bounding boxes were found or det_model is unavailable,
        # fallback to fine-grained 15-class leaf classifier
        if self.cls_model is not None:
            return self._predict_classification(pil_img, preproc_ms, t1, active_conf, requested_species=plant_species)
        elif self.det_model is not None:
            return det_resp
        else:
            raise RuntimeError("No model is loaded in ModelManager.")

    def _predict_detection(
        self,
        img: Image.Image,
        preproc_ms: float,
        t_start_infer: float,
        conf: float,
        iou: float,
        requested_species: Optional[str] = None,
    ) -> PredictionResponse:
        results = self.det_model.predict(
            source=img,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )[0]

        t2 = time.perf_counter()
        infer_ms = (t2 - t_start_infer) * 1000.0

        detections: List[DetectionItem] = []
        names = results.names
        boxes = results.boxes

        primary_condition = "healthy"
        max_conf = 0.0
        inferred_species = requested_species or "unknown"

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                c_conf = float(box.conf[0])
                c_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                sp = infer_species(c_name)

                # If species filter is requested, filter out mismatched species detections
                if requested_species and requested_species != "all" and sp != "unknown":
                    if sp != requested_species:
                        continue

                category = infer_category(c_name)

                # Normalized xywh
                xywhn = box.xywhn[0].tolist()
                bbox = BBox(
                    x_center=round(xywhn[0], 4),
                    y_center=round(xywhn[1], 4),
                    width=round(xywhn[2], 4),
                    height=round(xywhn[3], 4),
                )

                detections.append(
                    DetectionItem(
                        class_id=cls_id,
                        class_name=c_name,
                        category=category,
                        confidence=round(c_conf, 4),
                        bbox=bbox,
                    )
                )

                if c_conf > max_conf:
                    max_conf = c_conf
                    primary_condition = c_name
                    if sp != "unknown":
                        inferred_species = sp

        t3 = time.perf_counter()
        postproc_ms = (t3 - t2) * 1000.0
        total_ms = preproc_ms + infer_ms + postproc_ms

        warning = None
        if len(detections) == 0 or max_conf < 0.35:
            warning = "Low confidence prediction - possible out-of-distribution or unclear image."

        return PredictionResponse(
            task_type="detection",
            plant_species=inferred_species,
            primary_condition=primary_condition if detections else "no_detections",
            confidence=round(max_conf, 4),
            uncertainty_warning=warning,
            detections=detections,
            latency_ms=LatencyInfo(
                preprocessing=preproc_ms,
                inference=infer_ms,
                postprocessing=postproc_ms,
                total=total_ms,
            ),
        )

    def _predict_classification(
        self,
        img: Image.Image,
        preproc_ms: float,
        t_start_infer: float,
        conf: float,
        requested_species: Optional[str] = None,
    ) -> PredictionResponse:
        results = self.cls_model.predict(
            source=img,
            device=self.device,
            verbose=False,
        )[0]

        t2 = time.perf_counter()
        infer_ms = (t2 - t_start_infer) * 1000.0

        names = results.names
        probs = results.probs
        raw_probs = probs.data.cpu().numpy()
        unconditioned_top1_conf = float(probs.top1conf)

        # Compute normalized Shannon entropy
        p_safe = raw_probs[raw_probs > 1e-7]
        entropy = float(-np.sum(p_safe * np.log2(p_safe)) / np.log2(len(raw_probs)))

        # Species-conditioned probability selection if species hint is provided
        if requested_species and requested_species != "all":
            filtered_indices = [
                idx for idx, name in (names.items() if isinstance(names, dict) else enumerate(names))
                if infer_species(name) == requested_species
            ]
            if filtered_indices:
                sub_probs = raw_probs[filtered_indices]
                sub_sum = float(np.sum(sub_probs))

                # If model is confident that this image does NOT belong to requested species
                if sub_sum < 0.15:
                    t3 = time.perf_counter()
                    total_ms = preproc_ms + infer_ms + (t3 - t2) * 1000.0
                    return PredictionResponse(
                        task_type="unrecognized",
                        plant_species="unknown",
                        primary_condition="non_plant_or_unrecognized",
                        confidence=0.0,
                        uncertainty_warning=f"Image does not exhibit visual features of {requested_species.replace('_', ' ').title()}. Please upload a clear photo of {requested_species.replace('_', ' ').title()} foliage.",
                        detections=[],
                        latency_ms=LatencyInfo(
                            preprocessing=preproc_ms,
                            inference=infer_ms,
                            postprocessing=(t3 - t2) * 1000.0,
                            total=total_ms,
                        ),
                    )

                if sub_sum > 0:
                    norm_probs = sub_probs / sub_sum
                    best_sub_idx = int(np.argmax(norm_probs))
                    top1_idx = filtered_indices[best_sub_idx]
                    top1_conf = float(norm_probs[best_sub_idx])
                    top1_name = names.get(top1_idx, str(top1_idx)) if isinstance(names, dict) else names[top1_idx]
                else:
                    best_filtered_idx = max(filtered_indices, key=lambda i: raw_probs[i])
                    top1_idx = best_filtered_idx
                    top1_conf = float(raw_probs[best_filtered_idx])
                    top1_name = names.get(top1_idx, str(top1_idx)) if isinstance(names, dict) else names[top1_idx]
            else:
                top1_idx = int(probs.top1)
                top1_conf = float(probs.top1conf)
                top1_name = names.get(top1_idx, str(top1_idx)) if isinstance(names, dict) else names[top1_idx]
        else:
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            top1_name = names.get(top1_idx, str(top1_idx)) if isinstance(names, dict) else names[top1_idx]

        t3 = time.perf_counter()
        postproc_ms = (t3 - t2) * 1000.0
        total_ms = preproc_ms + infer_ms + postproc_ms

        # Low confidence / High entropy out-of-distribution gate for fine-tuned models
        if len(names) <= 50 and (unconditioned_top1_conf < 0.40 or entropy > 0.75):
            return PredictionResponse(
                task_type="unrecognized",
                plant_species="unknown",
                primary_condition="non_plant_or_unrecognized",
                confidence=round(unconditioned_top1_conf, 4),
                uncertainty_warning="Low confidence / ambiguous image. The image does not exhibit distinct botanical disease patterns of Basil, Mint, or Aloe Vera. Please upload a clear photo of herbal plant foliage.",
                detections=[],
                latency_ms=LatencyInfo(
                    preprocessing=preproc_ms,
                    inference=infer_ms,
                    postprocessing=postproc_ms,
                    total=total_ms,
                ),
            )

        warning = None
        if top1_conf < 0.45:
            warning = "Low confidence prediction - possible out-of-distribution or unclear image."

        inferred_sp = infer_species(top1_name)
        if inferred_sp == "unknown" and requested_species:
            inferred_sp = requested_species

        # Generate a holistic whole-leaf bounding box for canvas visualization
        detections = [
            DetectionItem(
                class_id=top1_idx,
                class_name=top1_name,
                category=infer_category(top1_name),
                confidence=round(top1_conf, 4),
                bbox=BBox(
                    x_center=0.5,
                    y_center=0.5,
                    width=0.88,
                    height=0.88,
                ),
            )
        ]

        return PredictionResponse(
            task_type="classification",
            plant_species=inferred_sp,
            primary_condition=top1_name,
            confidence=round(top1_conf, 4),
            uncertainty_warning=warning,
            detections=detections,
            latency_ms=LatencyInfo(
                preprocessing=preproc_ms,
                inference=infer_ms,
                postprocessing=postproc_ms,
                total=total_ms,
            ),
        )
