"""Batch inference engine for processing image directories in KrishiAI (Section 53)."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ImageDraw, ImageFont

from krishiai.inference.model_manager import ModelManager
from krishiai.inference.types import PredictionResponse
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


def annotate_image_with_prediction(img: Image.Image, pred: PredictionResponse) -> Image.Image:
    """Draws bounding boxes and diagnosis overlay onto a PIL Image."""
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    w, h = draw_img.size

    # Draw detections if any
    if pred.detections:
        for det in pred.detections:
            b = det.bbox
            x1 = int((b.x_center - b.width / 2) * w)
            y1 = int((b.y_center - b.height / 2) * h)
            x2 = int((b.x_center + b.width / 2) * w)
            y2 = int((b.y_center + b.height / 2) * h)

            # Color scheme
            color = "#ef4444" if det.category == "disease" else ("#f59e0b" if det.category == "deficiency" else "#10b981")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, w // 200))
            label = f"{det.class_name} {det.confidence:.2f}"
            draw.text((x1 + 4, max(0, y1 - 15)), label, fill=color)
    else:
        # Header diagnosis for classification
        label = f"{pred.plant_species.upper()}: {pred.primary_condition} ({pred.confidence * 100:.1f}%)"
        draw.text((10, 10), label, fill="#10b981")

    return draw_img


def run_batch_inference(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_manager: Optional[ModelManager] = None,
    plant_species: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    save_annotated: bool = True,
) -> Dict[str, Any]:
    """Runs batch inference on all image files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_annotated:
        (output_dir / "annotated").mkdir(exist_ok=True)

    if model_manager is None:
        model_manager = ModelManager()

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts]

    if not image_paths:
        logger.warning(f"No valid images found in {input_dir}")
        return {"total_images": 0, "results": []}

    logger.info(f"Starting batch inference on {len(image_paths)} images from {input_dir}")
    t0 = time.perf_counter()

    results: List[Dict[str, Any]] = []
    condition_counts: Dict[str, int] = {}
    species_counts: Dict[str, int] = {}

    for idx, img_path in enumerate(image_paths, 1):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            pred = model_manager.predict(
                image=pil_img,
                plant_species=plant_species,
                conf=confidence_threshold,
            )

            rec = {
                "file_name": img_path.name,
                "relative_path": str(img_path.relative_to(input_dir)),
                "plant_species": pred.plant_species,
                "primary_condition": pred.primary_condition,
                "confidence": pred.confidence,
                "task_type": pred.task_type,
                "detection_count": len(pred.detections),
                "latency_total_ms": round(pred.latency_ms.total, 2),
                "uncertainty_warning": pred.uncertainty_warning,
            }
            results.append(rec)

            # Update stats
            cond = pred.primary_condition
            condition_counts[cond] = condition_counts.get(cond, 0) + 1
            sp = pred.plant_species
            species_counts[sp] = species_counts.get(sp, 0) + 1

            # Save annotated image
            if save_annotated:
                annotated = annotate_image_with_prediction(pil_img, pred)
                annotated.save(output_dir / "annotated" / f"annotated_{img_path.name}")

            if idx % 50 == 0 or idx == len(image_paths):
                logger.info(f"Processed {idx}/{len(image_paths)} images...")

        except Exception as exc:
            logger.error(f"Failed to process {img_path}: {exc}")
            results.append({
                "file_name": img_path.name,
                "relative_path": str(img_path.relative_to(input_dir)),
                "error": str(exc),
            })

    total_time = time.perf_counter() - t0
    avg_fps = len(image_paths) / total_time if total_time > 0 else 0

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_images": len(image_paths),
        "total_time_seconds": round(total_time, 2),
        "throughput_fps": round(avg_fps, 2),
        "condition_distribution": condition_counts,
        "species_distribution": species_counts,
        "results": results,
    }

    # Save JSON report
    json_path = output_dir / "batch_predictions.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Save CSV report
    csv_path = output_dir / "batch_summary.csv"
    if results:
        fieldnames = [
            "file_name", "relative_path", "plant_species", "primary_condition",
            "confidence", "task_type", "detection_count", "latency_total_ms", "uncertainty_warning"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"Batch inference complete in {total_time:.2f}s ({avg_fps:.1f} FPS). Report -> {json_path}")
    return summary
