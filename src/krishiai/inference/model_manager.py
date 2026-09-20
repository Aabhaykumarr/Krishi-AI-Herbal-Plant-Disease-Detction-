"""ModelManager: Unified Inference Gateway for KrishiAI (Section 51 & Decision D-007)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageOps
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
    """Analyzes image color distribution, texture, spatial naturalness, and geometric
    structure to detect non-plant objects (electronic kits, circuit boards, tools, containers,
    devices, cups, laptops, cars, screens), documents, synthetic graphics, solid colors, noise,
    or out-of-distribution subjects.
    """
    img_rgb = pil_img.convert("RGB")

    # 1. Texture Variance Check (Rejects flat solid colors, blank screens, synthetic blocks)
    # Computed on a standardized 256x256 thumbnail for zero memory overhead and fast execution
    thumb_tex = img_rgb.resize((256, 256))
    arr_tex = np.array(thumb_tex, dtype=np.float32)
    std_channels = np.std(arr_tex, axis=(0, 1))
    mean_std = float(np.mean(std_channels))
    if mean_std < 6.0:
        return False, f"Flat or solid-color image detected (Texture variance={mean_std:.1f} < 6.0)."

    # 2. Spatial Autocorrelation Check (Rejects synthetic noise / scrambled pixels)
    gray = np.array(img_rgb.convert("L").resize((128, 128)), dtype=np.float32)
    var_gray = np.var(gray)
    if var_gray > 1.0:
        corr_x = np.corrcoef(gray[:, :-1].flatten(), gray[:, 1:].flatten())[0, 1]
        corr_y = np.corrcoef(gray[:-1, :].flatten(), gray[1:, :].flatten())[0, 1]
        mean_corr = float((corr_x + corr_y) / 2.0)
        if mean_corr < 0.40:
            return False, f"Synthetic noise or non-natural texture detected (Spatial correlation={mean_corr:.2f} < 0.40)."

    # 3. Geometric Rectilinearity & Man-Made Circuit / Device Check
    # Natural biological foliage has isotropic, curved, non-orthogonal edge distributions.
    # Man-made objects (electronic trainer kits, PCBs, breadboards, chips, keyboards, screens, desks)
    # have strong dominant orthogonal horizontal/vertical edges (0, 90, 180, 270 degrees).
    gray_256 = np.array(img_rgb.convert("L").resize((256, 256)), dtype=np.float32)
    gx = (gray_256[:-2, 2:] + 2 * gray_256[1:-1, 2:] + gray_256[2:, 2:]) - (gray_256[:-2, :-2] + 2 * gray_256[1:-1, :-2] + gray_256[2:, :-2])
    gy = (gray_256[2:, :-2] + 2 * gray_256[2:, 1:-1] + gray_256[2:, 2:]) - (gray_256[:-2, :-2] + 2 * gray_256[:-2, 1:-1] + gray_256[:-2, 2:])
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    mag_thresh = float(np.percentile(mag, 85))
    sig_edges = mag > max(15.0, mag_thresh)

    if np.sum(sig_edges) > 100:
        angles = ang[sig_edges]
        ortho_0 = (angles <= 6) | (angles >= 174)
        ortho_90 = (angles >= 84) & (angles <= 96)
        ortho_ratio = float(np.mean(ortho_0 | ortho_90))
        if ortho_ratio > 0.46:
            return False, f"Synthetic / Manufactured object detected ({ortho_ratio*100:.1f}% rectilinear horizontal/vertical edges)."

    # 4. Botanical / Vegetative Spectral Indexing
    thumb = img_rgb.resize((128, 128))
    arr = np.array(thumb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    cmax = np.max(arr, axis=2)
    cmin = np.min(arr, axis=2)
    delta = cmax - cmin

    sat = np.zeros_like(cmax)
    non_zero = cmax > 0.02
    sat[non_zero] = delta[non_zero] / cmax[non_zero]

    hue = np.zeros_like(cmax)
    r_mask = (cmax == r) & (delta > 0)
    g_mask = (cmax == g) & (delta > 0)
    b_mask = (cmax == b) & (delta > 0)

    hue[r_mask] = 60.0 * (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6)
    hue[g_mask] = 60.0 * (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2)
    hue[b_mask] = 60.0 * (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4)

    # Biological vegetation signatures:
    # A. Active Green / Succulent Chlorophyll Foliage (Basil, Mint, Aloe Vera, living foliage):
    is_green_foliage = (
        (hue >= 48.0) & (hue <= 180.0) &
        (sat >= 0.07) &
        (g >= 1.03 * b) &
        (g >= 0.82 * r) &
        ((2.0 * g - r - b) > 0.008)
    )

    # B. Pathological Foliar Necrosis / Chlorosis / Mildew / Lesion on Leaf:
    is_pathology_foliage = (
        (hue >= 14.0) & (hue <= 48.0) &
        (sat >= 0.14) &
        (r >= 1.15 * b) &
        (g >= 1.04 * b) &
        (r >= 0.88 * g)
    )

    # C. Dried Herb Foliage (Harvested dried mint / tea leaves on neutral background):
    is_dried_herb = (
        (r > b) & (g > b) &
        (r >= 0.15) & (r <= 0.80) &
        (g >= 0.15) & (g <= 0.75) &
        (b <= 0.50) &
        (np.abs(r - g) < 0.22) &
        (hue >= 20.0) & (hue <= 75.0)
    )

    green_ratio = float(np.mean(is_green_foliage))
    pathology_ratio = float(np.mean(is_pathology_foliage))
    dried_ratio = float(np.mean(is_dried_herb))
    foliage_ratio = green_ratio + pathology_ratio

    # 5. Out-of-Distribution Non-Botanical Rejection Filters
    is_neutral_bg = (np.abs(r - g) < 0.05) & (np.abs(g - b) < 0.05) & (cmax > 0.60)
    neutral_bg_ratio = float(np.mean(is_neutral_bg))

    is_skin = (r > g) & (g > b) & (r > 0.35) & (r < 0.95) & (hue >= 8.0) & (hue <= 32.0) & (sat >= 0.15) & (sat <= 0.55)
    skin_ratio = float(np.mean(is_skin))

    is_sky_blue = (hue >= 185.0) & (hue <= 255.0) & (sat > 0.18)
    blue_ratio = float(np.mean(is_sky_blue))

    is_purple = (hue >= 260.0) & (hue <= 340.0) & (sat > 0.18)
    purple_ratio = float(np.mean(is_purple))

    is_red = ((hue >= 345.0) | (hue <= 12.0)) & (sat > 0.30) & (r > 1.30 * g)
    red_ratio = float(np.mean(is_red))

    if neutral_bg_ratio > 0.70 and green_ratio < 0.04 and dried_ratio < 0.25:
        return False, f"Document / Plain surface detected ({neutral_bg_ratio*100:.1f}% neutral background)."
    if skin_ratio > 0.40 and green_ratio < 0.04:
        return False, f"Human subject / portrait detected ({skin_ratio*100:.1f}% skin tones)."
    if blue_ratio > 0.50 and green_ratio < 0.04:
        return False, f"Sky / Blue surface detected ({blue_ratio*100:.1f}% blue)."
    if purple_ratio > 0.25 and green_ratio < 0.04:
        return False, f"Non-foliar purple/magenta surface detected ({purple_ratio*100:.1f}% purple)."
    if red_ratio > 0.35 and green_ratio < 0.04:
        return False, f"Non-foliar red/orange surface detected ({red_ratio*100:.1f}% red)."

    # 6. Core Botanical Vegetation Presence Requirement
    has_active_green = green_ratio >= 0.040
    has_pathology_with_green = (green_ratio >= 0.015) and (foliage_ratio >= 0.060)
    has_dried_herb = (dried_ratio >= 0.30) and (neutral_bg_ratio >= 0.15 or green_ratio >= 0.005) and (skin_ratio < 0.25)

    if not (has_active_green or has_pathology_with_green or has_dried_herb):
        return False, f"No authentic botanical foliage or vegetative chlorophyll detected ({green_ratio*100:.1f}% green foliage)."

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

        Hierarchically orchestrates botanical visual gating, object detection (for localized
        spot/lesion bounding boxes), and 15-class whole-leaf classification (for holistic
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
            raw_img = Image.open(img_path)
            pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = ImageOps.exif_transpose(image).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        # Safeguard: resize overly large camera captures to max 1280px to prevent memory spikes
        if max(pil_img.size) > 1280:
            pil_img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)

        t1 = time.perf_counter()
        preproc_ms = (t1 - t0) * 1000.0

        # Step 2: Botanical Visual Gating (Non-Plant / Document / Synthetic Noise detection)
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

        # Run classification analysis to assess herbal species-level evidence
        cls_species_mass = 0.0
        dominant_species = "unknown"
        if self.cls_model is not None:
            with torch.inference_mode():
                cls_check = self.cls_model.predict(source=pil_img, device=self.device, verbose=False)[0]
            names = cls_check.names
            probs = cls_check.probs.data.cpu().numpy()
            aloe_indices = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if "aloe" in n]
            basil_indices = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if "basil" in n]
            mint_indices = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if "mint" in n]
            p_aloe = float(np.sum(probs[aloe_indices]))
            p_basil = float(np.sum(probs[basil_indices]))
            p_mint = float(np.sum(probs[mint_indices]))
            cls_species_mass = max(p_aloe, p_basil, p_mint)
            dominant_species = "aloe_vera" if p_aloe == cls_species_mass else ("basil" if p_basil == cls_species_mass else "mint")

            # If the image is Aloe Vera and not Basil/Mint -> route directly to classification
            if dominant_species == "aloe_vera" and p_aloe >= 0.55 and (plant_species is None or plant_species == "aloe_vera"):
                return self._predict_classification(pil_img, preproc_ms, t1, active_conf, requested_species="aloe_vera")

        # For Basil, Mint, or general: try object detection for localized symptoms
        det_resp = None
        if self.det_model is not None:
            det_resp = self._predict_detection(pil_img, preproc_ms, t1, active_conf, active_iou, requested_species=plant_species)
            if det_resp.detections and len(det_resp.detections) > 0 and det_resp.primary_condition != "no_detections" and det_resp.confidence >= 0.30:
                return det_resp

        # Fallback to fine-grained 15-class leaf classifier
        if self.cls_model is not None:
            return self._predict_classification(pil_img, preproc_ms, t1, active_conf, requested_species=plant_species)
        elif det_resp is not None:
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
        with torch.inference_mode():
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
        with torch.inference_mode():
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

        # Species-level probability mass aggregation
        aloe_indices = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if "aloe" in n]
        basil_indices = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if "basil" in n]
        mint_indices = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if "mint" in n]

        p_aloe = float(np.sum(raw_probs[aloe_indices]))
        p_basil = float(np.sum(raw_probs[basil_indices]))
        p_mint = float(np.sum(raw_probs[mint_indices]))
        max_species_mass = max(p_aloe, p_basil, p_mint)

        # Species-conditioned probability selection if species hint is provided
        if requested_species and requested_species != "all":
            filtered_indices = (
                aloe_indices if requested_species == "aloe_vera"
                else (basil_indices if requested_species == "basil" else mint_indices)
            )
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
                top1_idx = int(probs.top1)
                top1_conf = float(probs.top1conf)
                top1_name = names.get(top1_idx, str(top1_idx)) if isinstance(names, dict) else names[top1_idx]
            inferred_sp = requested_species
        else:
            # Auto-Detect mode: Out-Of-Domain plant rejection
            is_ood = (
                (max_species_mass < 0.60) or
                (entropy > 0.60 and unconditioned_top1_conf < 0.60) or
                (unconditioned_top1_conf < 0.35)
            )
            if is_ood:
                t3 = time.perf_counter()
                total_ms = preproc_ms + infer_ms + (t3 - t2) * 1000.0
                return PredictionResponse(
                    task_type="unrecognized",
                    plant_species="unknown",
                    primary_condition="non_plant_or_unrecognized",
                    confidence=round(unconditioned_top1_conf, 4),
                    uncertainty_warning="Low confidence / Out-of-Domain plant. The foliage does not match the distinct botanical features and disease pathology of the supported herbal crops (Basil, Aloe Vera, Mint). Please upload a clear photo of Basil, Mint, or Aloe Vera.",
                    detections=[],
                    latency_ms=LatencyInfo(
                        preprocessing=preproc_ms,
                        inference=infer_ms,
                        postprocessing=(t3 - t2) * 1000.0,
                        total=total_ms,
                    ),
                )

            inferred_sp = "aloe_vera" if p_aloe == max_species_mass else ("basil" if p_basil == max_species_mass else "mint")
            target_indices = aloe_indices if inferred_sp == "aloe_vera" else (basil_indices if inferred_sp == "basil" else mint_indices)
            sub_probs = raw_probs[target_indices]
            sub_sum = float(np.sum(sub_probs))
            norm_probs = sub_probs / (sub_sum if sub_sum > 0 else 1.0)
            best_sub_idx = int(np.argmax(norm_probs))
            top1_idx = target_indices[best_sub_idx]
            top1_conf = float(norm_probs[best_sub_idx])
            top1_name = names.get(top1_idx, str(top1_idx)) if isinstance(names, dict) else names[top1_idx]

        t3 = time.perf_counter()
        postproc_ms = (t3 - t2) * 1000.0
        total_ms = preproc_ms + infer_ms + postproc_ms

        warning = None
        if top1_conf < 0.35:
            warning = "Low confidence prediction - possible subtle symptoms or non-standard lighting."

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
