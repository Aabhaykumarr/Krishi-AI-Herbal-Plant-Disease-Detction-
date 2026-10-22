"""FastAPI HTTP service for KrishiAI leaf disease detection and classification (Section 51)."""
from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from krishiai.config.settings import Settings
from krishiai.data.dataset_saver import (
    DatasetSaveError,
    ValidationError,
    save_to_dataset,
)
from krishiai.inference.gemini_verifier import GeminiVisionVerifier
from krishiai.inference.model_manager import ModelManager
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)

# Global model manager instance
_model_manager: Optional[ModelManager] = None
_gemini_verifier: Optional[GeminiVisionVerifier] = None


def get_model_manager() -> ModelManager:
    """Singleton getter for ModelManager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager.create_default(device="auto")
    return _model_manager


def get_gemini_verifier() -> GeminiVisionVerifier:
    """Singleton getter for the optional Gemini Vision reviewer."""
    global _gemini_verifier
    if _gemini_verifier is None:
        _gemini_verifier = GeminiVisionVerifier()
    return _gemini_verifier


def _yolo_snapshot(prediction: Any) -> Dict[str, Any]:
    """Copy YOLO's result before the final, Gemini-first diagnosis is applied."""
    return {
        "task_type": prediction.task_type,
        "plant_species": prediction.plant_species,
        "primary_condition": prediction.primary_condition,
        "confidence": prediction.confidence,
        "uncertainty_warning": prediction.uncertainty_warning,
        "detections": [detection.to_dict() for detection in prediction.detections],
    }


def _apply_gemini_assessment(prediction: Any, analysis: Dict[str, Any]) -> None:
    """Expose Gemini's diagnosis as primary while retaining the full YOLO result."""
    assessment = analysis["assessment"]
    yolo_result = _yolo_snapshot(prediction)
    prediction.yolo_result = yolo_result
    prediction.diagnosis_source = "gemini"
    prediction.confidence_source = "gemini_visual_score_uncalibrated"
    prediction.confidence = assessment["visual_confidence"]
    prediction.condition_category = assessment["category"]

    leaf_detected = assessment["leaf_detected"]
    if leaf_detected and assessment["category"] == "non_herbal_plant":
        # Keep Gemini's supported-crop gate authoritative for the user-facing
        # result, while preserving every original YOLO detection in yolo_result.
        prediction.task_type = "unrecognized"
        prediction.plant_species = "non_herbal_plant"
        prediction.primary_condition = "non_herbal_plant_detected"
        prediction.detections = []
        prediction.recommendation = (
            "Non-herbal plant detected. This app identifies only Aloe Vera, Basil, and Mint."
        )
        prediction.uncertainty_warning = (
            "YOLO26's independent result is retained below for comparison; Gemini did not identify a supported plant."
        )
    elif leaf_detected:
        species = assessment["plant_species"]
        condition_slug = assessment["condition_slug"]
        prediction.task_type = "classification"
        prediction.plant_species = species
        prediction.primary_condition = f"{species}_{condition_slug}" if species != "unknown" else condition_slug
        prediction.uncertainty_warning = (
            "Gemini could not determine the condition clearly; treat this result as uncertain."
            if assessment["category"] == "uncertain"
            else None
        )
        if assessment["category"] == "healthy":
            prediction.recommendation = "Gemini's visual assessment found no obvious disease symptoms. Continue routine monitoring."
        elif assessment["category"] == "uncertain":
            prediction.recommendation = "The visual condition is uncertain. Use a clearer image or seek a local plant-health opinion before taking action."
        else:
            prediction.recommendation = "Gemini identified a possible condition. Confirm the diagnosis locally before applying treatment."
    else:
        prediction.task_type = "unrecognized"
        prediction.plant_species = "unknown"
        prediction.primary_condition = "non_plant_or_unrecognized"
        prediction.recommendation = "Gemini did not identify a clear plant leaf in this image. Upload a closer, well-lit leaf photo for diagnosis."
        prediction.uncertainty_warning = "Gemini did not identify a plant leaf; the independent YOLO26 result is retained below for comparison."

    yolo_species = GeminiVisionVerifier._slug(yolo_result["plant_species"] or "unknown")
    yolo_condition = GeminiVisionVerifier._slug(yolo_result["primary_condition"] or "unknown")
    if yolo_species and yolo_condition.startswith(f"{yolo_species}_"):
        yolo_condition = yolo_condition[len(yolo_species) + 1:]
    matched = (
        leaf_detected
        and assessment["category"] != "uncertain"
        and yolo_species == assessment["plant_species"]
        and yolo_condition == assessment["condition_slug"]
    )
    if assessment["category"] == "non_herbal_plant":
        review_status = "non_herbal_plant_detected"
        note = "Gemini identified a plant outside the supported set (Aloe Vera, Basil, and Mint); YOLO26 is retained below for comparison."
    elif assessment["category"] == "uncertain":
        review_status = "uncertain_gemini_preferred"
        note = "Gemini returned an uncertain visual assessment; it is displayed alongside the YOLO26 cross-check."
    elif matched:
        review_status = "matched"
        note = "Gemini and YOLO26 agree on the plant and visible condition."
    else:
        review_status = "disagreed_gemini_preferred"
        note = "Gemini and YOLO26 disagree; Gemini's diagnosis is displayed and the original YOLO26 result is retained for comparison."

    prediction.verification = {
        "status": review_status,
        "model": analysis.get("model"),
        "reviewed_count": 1,
        "suppressed_suggestions": 0,
        "corrections": [],
        "confidence_is_calibrated": False,
        "gemini_assessment": assessment,
        "yolo_result": yolo_result,
        "note": note,
    }


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="KrishiAI API",
        description="Multi-Class Herbal Plant Leaf Disease Detection and Classification using YOLO26",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_file = Path("static/index.html")
        content = index_file.read_text(encoding="utf-8") if index_file.exists() else "<h1>KrishiAI API is Running.</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>"
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/predict", response_class=HTMLResponse, include_in_schema=False)
    async def prediction_page() -> HTMLResponse:
        """Allow opening the dashboard at /predict without changing POST /predict."""
        return await index()

    @app.get("/**", include_in_schema=False)
    async def malformed_dashboard_path() -> RedirectResponse:
        """Recover when the browser is opened at the literal path /**."""
        return RedirectResponse(url="/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @app.get("/health", status_code=status.HTTP_200_OK)
    async def health_check() -> Dict[str, Any]:
        mm = get_model_manager()
        return {
            "status": "healthy",
            "service": "krishiai-api",
            "version": "1.0.0",
            "models": {
                "detection": mm.det_model is not None,
                "classification": mm.cls_model is not None,
                "device": mm.device,
            },
        }

    @app.get("/model/info", status_code=status.HTTP_200_OK)
    async def model_info() -> Dict[str, Any]:
        mm = get_model_manager()
        return {
            "model_version": "krishiai-yolo26-v1.0",
            "architecture": "YOLO26",
            "supported_crops": ["basil", "aloe_vera", "mint"],
            "models_loaded": {
                "detection": mm.det_model is not None,
                "classification": mm.cls_model is not None,
            },
            "taxonomy": {
                "diseases": [
                    "basil_bacterial_spot",
                    "basil_bug_spot",
                    "basil_downy_mildew",
                    "basil_powdery_mildew",
                    "aloe_vera_rust",
                    "aloe_vera_anthracnose",
                    "aloe_vera_leaf_spot",
                ],
                "deficiencies": [
                    "basil_magnesium_deficiency",
                    "basil_nitrogen_deficiency",
                ],
                "conditions_and_freshness": [
                    "aloe_vera_sunburn",
                    "mint_dried",
                    "mint_fresh",
                    "mint_spoiled",
                ],
                "healthy_states": [
                    "basil_healthy",
                    "aloe_vera_healthy",
                    "mint_fresh",
                ],
            },
        }

    @app.post("/predict", status_code=status.HTTP_200_OK)
    async def predict_image(
        file: UploadFile = File(...),
        plant_species: Optional[str] = Form(None),
        confidence_threshold: Optional[float] = Form(None),
        iou_threshold: Optional[float] = Form(None),
    ) -> Dict[str, Any]:
        """Runs disease detection / classification prediction on an uploaded leaf image."""
        content_type = (file.content_type or "").lower()
        if content_type and not (
            content_type.startswith("image/")
            or content_type in ("application/octet-stream", "binary/octet-stream")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type '{file.content_type}'. Please upload an image file (JPEG, PNG, WebP).",
            )

        try:
            max_upload_bytes = Settings.load().max_upload_size_mb * 1024 * 1024
            content = await file.read(max_upload_bytes + 1)
            if len(content) > max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the configured upload limit of {max_upload_bytes // (1024 * 1024)}MB.",
                )
            raw_img = Image.open(io.BytesIO(content))
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            # Downscale image to standardized 640px to eliminate high memory allocations on cloud containers
            if max(img.size) > 640:
                img.thumbnail((640, 640), Image.Resampling.LANCZOS)
            del content, raw_img
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not decode image file: {exc}",
            )

        try:
            # Run the cloud vision pass first. The local model then performs
            # an independent cross-check even when Gemini cannot be reached.
            gemini_started = time.perf_counter()
            try:
                analysis = await asyncio.to_thread(
                    get_gemini_verifier().analyze,
                    image=img,
                    requested_species=plant_species,
                )
            except Exception as exc:
                logger.warning(
                    "Gemini Vision analysis could not run (%s); using YOLO26 fallback.",
                    type(exc).__name__,
                )
                analysis = {
                    "status": "failed",
                    "note": "Gemini Vision was unavailable; the YOLO26 result is shown as fallback.",
                }
            gemini_ms = (time.perf_counter() - gemini_started) * 1000.0

            mm = get_model_manager()
            supported_yolo_species = {"basil", "aloe_vera", "mint"}
            requested_yolo_species = (plant_species or "").strip().lower()
            yolo_species_hint = requested_yolo_species if requested_yolo_species in supported_yolo_species else None
            resp = await asyncio.to_thread(
                mm.predict,
                image=img,
                plant_species=yolo_species_hint,
                conf=confidence_threshold,
                iou=iou_threshold,
            )

            if (
                analysis.get("status") == "analyzed"
                and isinstance(analysis.get("assessment"), dict)
            ):
                _apply_gemini_assessment(resp, analysis)
            else:
                resp.verification = {
                    "status": analysis.get("status", "failed"),
                    "model": analysis.get("model"),
                    "failure_code": analysis.get("failure_code"),
                    "reviewed_count": 0,
                    "suppressed_suggestions": 0,
                    "corrections": [],
                    "confidence_is_calibrated": False,
                    "note": analysis.get(
                        "note", "Gemini Vision was unavailable; the YOLO26 result is shown as fallback."
                    ),
                }

            if resp.latency_ms is not None:
                resp.latency_ms.gemini = gemini_ms
                resp.latency_ms.total += gemini_ms
            return resp.to_dict()
        except Exception as exc:
            logger.error(f"Prediction failed: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal inference failure: {exc}",
            )

    @app.post("/save-to-dataset", status_code=status.HTTP_201_CREATED)
    async def save_image_to_dataset(
        file: UploadFile = File(...),
        plant_species: str = Form(...),
    ) -> Dict[str, Any]:
        """Save an uploaded leaf image to the incoming dataset directory.

        This endpoint provides an explicit opt-in mechanism for users to contribute
        images to the dataset. Images are NOT automatically saved during prediction.

        Args:
            file: Image file (JPEG, PNG, or WebP)
            plant_species: Required species label (basil, aloe_vera, or mint)

        Returns:
            Success response with saved path and metadata

        Raises:
            400: Validation error (invalid species, file type, size, or duplicate)
            500: Server error during save operation
        """
        # Validate content type
        content_type = (file.content_type or "").lower()
        if content_type and not (
            content_type.startswith("image/")
            or content_type in ("application/octet-stream", "binary/octet-stream")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type '{file.content_type}'. Please upload an image file (JPEG, PNG, WebP).",
            )

        settings = Settings.load()
        max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024

        try:
            # Bound memory use before image validation and dataset writing.
            content = await file.read(max_upload_bytes + 1)
            if len(content) > max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the configured upload limit of {max_upload_bytes // (1024 * 1024)}MB.",
                )

            # Save to dataset with validation
            result = save_to_dataset(
                content=content,
                filename=file.filename or "unnamed.jpg",
                species=plant_species,
                settings=settings,
            )

            logger.info(
                f"Image saved to dataset: species={plant_species}, "
                f"path={result['saved_path']}, size={len(content)} bytes"
            )

            return {
                "success": True,
                "message": f"Image successfully saved to {plant_species} dataset",
                "saved_path": result["saved_path"],
                "species": plant_species,
                "timestamp": result["metadata"]["saved_at"],
                "image_info": result["metadata"]["image_info"],
            }

        except HTTPException:
            raise
        except ValidationError as exc:
            logger.warning(f"Dataset save validation failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        except DatasetSaveError as exc:
            logger.error(f"Dataset save failed: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save image to dataset: {exc}",
            )
        except Exception as exc:
            logger.error(f"Unexpected error during dataset save: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error: {exc}",
            )

    return app


app = create_app()
