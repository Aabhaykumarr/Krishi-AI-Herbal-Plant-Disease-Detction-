"""Robustness, perturbation, and out-of-distribution (OOD) stress testing engine (Section 65)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from krishiai.inference.model_manager import ModelManager
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(img: Image.Image, std: float = 25.0) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


def apply_occlusion(img: Image.Image, patch_ratio: float = 0.25) -> Image.Image:
    img_copy = img.copy()
    w, h = img_copy.size
    pw, ph = int(w * patch_ratio), int(h * patch_ratio)
    x = random.randint(0, max(0, w - pw))
    y = random.randint(0, max(0, h - ph))

    patch = Image.new("RGB", (pw, ph), color=(0, 0, 0))
    img_copy.paste(patch, (x, y))
    return img_copy


PERTURBATIONS: Dict[str, List[Tuple[str, Callable[[Image.Image], Image.Image]]]] = {
    "brightness": [
        ("dark_-50%", lambda img: apply_brightness(img, 0.5)),
        ("bright_+50%", lambda img: apply_brightness(img, 1.5)),
    ],
    "contrast": [
        ("low_contrast_-50%", lambda img: apply_contrast(img, 0.5)),
        ("high_contrast_+50%", lambda img: apply_contrast(img, 1.5)),
    ],
    "blur": [
        ("gaussian_blur_r2", lambda img: apply_gaussian_blur(img, 2.0)),
        ("gaussian_blur_r5", lambda img: apply_gaussian_blur(img, 5.0)),
    ],
    "noise": [
        ("gaussian_noise_std20", lambda img: apply_gaussian_noise(img, 20.0)),
        ("gaussian_noise_std40", lambda img: apply_gaussian_noise(img, 40.0)),
    ],
    "rotation": [
        ("rotate_15deg", lambda img: apply_rotation(img, 15.0)),
        ("rotate_45deg", lambda img: apply_rotation(img, 45.0)),
        ("rotate_90deg", lambda img: apply_rotation(img, 90.0)),
    ],
    "occlusion": [
        ("center_occlusion_20%", lambda img: apply_occlusion(img, 0.20)),
        ("center_occlusion_40%", lambda img: apply_occlusion(img, 0.40)),
    ],
}


def run_robustness_evaluation(
    images: List[Union[str, Path, Image.Image]],
    model_manager: Optional[ModelManager] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluates model stability under systematic environmental perturbations."""
    if model_manager is None:
        model_manager = ModelManager()

    pil_images: List[Image.Image] = []
    for item in images:
        if isinstance(item, (str, Path)):
            pil_images.append(Image.open(item).convert("RGB"))
        elif isinstance(item, Image.Image):
            pil_images.append(item.convert("RGB"))

    if not pil_images:
        raise ValueError("No images provided for robustness testing.")

    logger.info(f"Running Robustness Suite on {len(pil_images)} test images across {len(PERTURBATIONS)} perturbation categories...")

    # 1. Clean Baseline Prediction
    baseline_predictions: List[Dict[str, Any]] = []
    for img in pil_images:
        pred = model_manager.predict(img)
        baseline_predictions.append({
            "condition": pred.primary_condition,
            "confidence": pred.confidence,
            "detections_count": len(pred.detections),
        })

    # 2. Perturbation Evaluations
    perturbation_results: Dict[str, Any] = {}

    for cat, variants in PERTURBATIONS.items():
        perturbation_results[cat] = {}
        for var_name, transform_fn in variants:
            conf_retentions: List[float] = []
            agreement_count = 0
            warning_count = 0

            for i, img in enumerate(pil_images):
                base_pred = baseline_predictions[i]
                pert_img = transform_fn(img)
                pert_pred = model_manager.predict(pert_img)

                if base_pred["confidence"] > 0:
                    conf_retentions.append(pert_pred.confidence / base_pred["confidence"])
                else:
                    conf_retentions.append(1.0)

                if pert_pred.primary_condition == base_pred["condition"]:
                    agreement_count += 1

                if pert_pred.uncertainty_warning:
                    warning_count += 1

            n = len(pil_images)
            perturbation_results[cat][var_name] = {
                "label_agreement_rate": round(agreement_count / n, 4),
                "avg_confidence_retention": round(float(np.mean(conf_retentions)), 4),
                "uncertainty_trigger_rate": round(warning_count / n, 4),
            }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_images_tested": len(pil_images),
        "perturbations": perturbation_results,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"robustness_report_{int(time.time())}.json"
        (out_path / fname).write_text(json.dumps(report, indent=2))
        logger.info(f"Robustness report saved -> {out_path / fname}")

    return report
