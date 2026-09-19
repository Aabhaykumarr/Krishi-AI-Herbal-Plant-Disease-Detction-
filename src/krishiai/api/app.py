"""FastAPI HTTP service for KrishiAI leaf disease detection and classification (Section 51)."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from krishiai.config.settings import Settings
from krishiai.inference.model_manager import ModelManager
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)

# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Singleton getter for ModelManager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager.create_default(device="auto")
    return _model_manager


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
    async def index() -> str:
        index_file = Path("static/index.html")
        if index_file.exists():
            return index_file.read_text(encoding="utf-8")
        return "<h1>KrishiAI API is Running.</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>"

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
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type '{file.content_type}'. Please upload an image file (JPEG, PNG, WebP).",
            )

        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not decode image file: {exc}",
            )

        mm = get_model_manager()
        try:
            resp = mm.predict(
                image=img,
                plant_species=plant_species,
                conf=confidence_threshold,
                iou=iou_threshold,
            )
            return resp.to_dict()
        except Exception as exc:
            logger.error(f"Prediction failed: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal inference failure: {exc}",
            )

    return app


app = create_app()
