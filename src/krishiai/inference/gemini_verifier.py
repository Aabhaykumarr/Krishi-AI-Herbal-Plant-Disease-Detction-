"""Gemini-first image diagnosis with a YOLO26 cross-check and safe fallback.

The API uses ``analyze`` to let Gemini classify the whole image before the
independent YOLO26 pass. The legacy ``verify`` helper remains available for
allow-listed class corrections, but provider failures never replace the YOLO
fallback response.
"""
from __future__ import annotations

import io
import json
import math
import re
import time
import unicodedata
from typing import Literal
from typing import Any, Dict, List, Mapping, Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from krishiai.config.settings import Settings
from krishiai.inference.model_manager import infer_category, infer_species
from krishiai.inference.types import DetectionItem, PredictionResponse
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


class _CorrectionSchema(BaseModel):
    detection_index: int = Field(ge=0)
    proposed_class: str
    visual_confidence: float = Field(ge=0, le=1)
    explanation: str = Field(max_length=300)


class _ReviewSchema(BaseModel):
    corrections: List[_CorrectionSchema] = Field(default_factory=list)


class _DiagnosisSchema(BaseModel):
    leaf_detected: bool
    plant_species: str = Field(min_length=1, max_length=60)
    condition: str = Field(min_length=1, max_length=100)
    category: Literal[
        "healthy", "disease", "deficiency", "pest_damage", "uncertain", "non_plant", "non_herbal_plant"
    ]
    visual_confidence: float = Field(ge=0, le=1)
    explanation: str = Field(max_length=300)


class _GeminiResponseError(ValueError):
    """A provider response arrived but did not match the diagnosis contract."""


class GeminiVisionVerifier:
    """Review detections with Gemini while preserving the YOLO response contract."""

    def __init__(self, settings: Optional[Settings] = None, client: Any = None):
        self.settings = settings or Settings.load()
        self._client = client

    def analyze(self, image: Image.Image, requested_species: Optional[str] = None) -> Dict[str, Any]:
        """Return Gemini's independent whole-image diagnosis before YOLO runs.

        Gemini may classify only Basil, Aloe Vera, and Mint. Other plants are
        normalized to a single out-of-scope result, regardless of provider text.
        Provider errors are returned as statuses so the API can use the local model.
        """
        if not self.settings.gemini_vision_enabled:
            return self._result("disabled", "Gemini Vision is disabled; use the YOLO26 result.")
        if not self.settings.gemini_api_key:
            return self._result("missing_key", "Gemini Vision is not configured; use the YOLO26 result.")

        for response_attempt in range(self.settings.gemini_max_retries + 1):
            try:
                payload = self._request_diagnosis(image.convert("RGB"), requested_species)
                assessment = self._validate_diagnosis(payload)
                return {
                    "status": "analyzed",
                    "model": self.settings.gemini_model,
                    "assessment": assessment,
                    "note": "Gemini analyzed the uploaded image independently before the YOLO26 cross-check.",
                }
            except _GeminiResponseError as exc:
                # A structured response can occasionally be empty or malformed.
                # Retry that case once using the existing retry setting, then
                # leave the original YOLO result available as the safe fallback.
                if response_attempt < self.settings.gemini_max_retries:
                    time.sleep(min(0.25 * (2**response_attempt), 0.75))
                    continue
                return self._failure_result(exc)
            except Exception as exc:
                # Never log provider messages: they can contain request metadata.
                return self._failure_result(exc)

    def _request_diagnosis(self, image: Image.Image, requested_species: Optional[str]) -> Any:
        from google import genai
        from google.genai import types

        contents: List[Any] = [
            "Assess this image using ONLY these supported plants: Basil (Ocimum basilicum), Aloe Vera, and Mint. "
            "First decide whether a plant leaf is clearly visible and whether it is one of those three supported "
            "plants. If it is clearly a different plant, do not name that plant and do not diagnose its condition: "
            "return plant_species exactly 'non_herbal_plant', condition exactly 'Non-herbal plant detected', and "
            "category 'non_herbal_plant'. Explain only that the image is outside the supported set of Basil, Aloe "
            "Vera, and Mint. Never force an unsupported plant into one of the three labels, even when the UI has "
            "a selected crop. For a supported plant, describe only the most visually supported condition in plain "
            "language. Use healthy only when no obvious disease or damage is visible. If the image is too unclear "
            "to decide whether it is a supported plant, use uncertain. For an image with no visible plant leaf, use "
            "leaf_detected=false, plant_species='unknown', and category 'non_plant'. Do not claim a specific "
            "pathogen from ambiguous evidence or provide treatment advice. visual_confidence is an uncalibrated "
            "visual-review score, not a probability. Return only the requested structured fields."
        ]
        allowed_hints = {"basil": "basil", "aloe_vera": "aloe vera", "mint": "mint"}
        hint = allowed_hints.get((requested_species or "").strip().lower())
        if hint:
            contents.append(
                f"The UI selection is {hint}; it is not proof. If the image is another plant, use the required "
                f"non_herbal_plant result instead of naming it or changing it to {hint}."
            )
        contents.append(
            types.Part.from_bytes(data=self._jpeg_bytes(image), mime_type="image/jpeg")
        )

        if self._client is None:
            http_options = types.HttpOptions(
                timeout=int(self.settings.gemini_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            self._client = genai.Client(api_key=self.settings.gemini_api_key, http_options=http_options)

        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=512,
            response_mime_type="application/json",
            response_schema=_DiagnosisSchema,
        )
        for attempt in range(self.settings.gemini_max_retries + 1):
            try:
                return self._client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                if attempt >= self.settings.gemini_max_retries or not self._is_transient(exc):
                    raise
                time.sleep(min(0.25 * (2**attempt), 0.75))
        raise RuntimeError("Gemini diagnosis request did not complete")

    @staticmethod
    def _validate_diagnosis(payload: Any) -> Dict[str, Any]:
        # The Google GenAI SDK parses Pydantic response schemas into `parsed`.
        # Prefer that typed value, but keep text JSON as a compatibility fallback.
        parsed = getattr(payload, "parsed", None)
        if isinstance(parsed, BaseModel):
            data = parsed.model_dump()
        elif isinstance(parsed, Mapping):
            data = dict(parsed)
        else:
            text = getattr(payload, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise _GeminiResponseError("Gemini returned no structured diagnosis")
            try:
                data = json.loads(text)
            except (TypeError, json.JSONDecodeError) as exc:
                raise _GeminiResponseError("Gemini returned invalid diagnosis JSON") from exc

        expected = {"leaf_detected", "plant_species", "condition", "category", "visual_confidence", "explanation"}
        if not isinstance(data, dict) or set(data) != expected:
            raise _GeminiResponseError("Gemini diagnosis does not match the response schema")
        if not isinstance(data["leaf_detected"], bool):
            raise _GeminiResponseError("Gemini returned an invalid leaf-detection flag")
        category = data["category"]
        if category not in {
            "healthy", "disease", "deficiency", "pest_damage", "uncertain", "non_plant", "non_herbal_plant"
        }:
            raise _GeminiResponseError("Gemini returned an unknown diagnosis category")
        if data["leaf_detected"] == (category == "non_plant"):
            raise _GeminiResponseError("Gemini leaf flag and category disagree")

        species = data["plant_species"]
        condition = data["condition"]
        explanation = data["explanation"]
        confidence = data["visual_confidence"]
        if not isinstance(species, str) or not species.strip() or len(species.strip()) > 60:
            raise _GeminiResponseError("Gemini returned an invalid plant species")
        if not isinstance(condition, str) or not condition.strip() or len(condition.strip()) > 100:
            raise _GeminiResponseError("Gemini returned an invalid condition")
        if not isinstance(explanation, str) or len(explanation) > 300:
            raise _GeminiResponseError("Gemini returned an invalid explanation")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise _GeminiResponseError("Gemini returned an invalid visual confidence")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise _GeminiResponseError("Gemini returned an out-of-range visual confidence")

        species_slug = GeminiVisionVerifier._slug(species)
        condition_slug = GeminiVisionVerifier._slug(condition)
        if not species_slug or not condition_slug:
            raise _GeminiResponseError("Gemini returned a label that cannot be normalized")
        if not data["leaf_detected"]:
            species_slug = "unknown"
        else:
            supported_species = GeminiVisionVerifier._supported_species(species_slug)
            if supported_species is None:
                # Enforce the crop boundary locally even if Gemini ignores its prompt.
                species_slug = "non_herbal_plant"
                condition = "Non-herbal plant detected"
                condition_slug = "non_herbal_plant_detected"
                category = "non_herbal_plant"
                explanation = "This image is outside the supported plants: Basil, Aloe Vera, and Mint."
            else:
                species_slug = supported_species
                if category == "non_herbal_plant":
                    category = "uncertain"
                    condition = "Uncertain plant identification"
                    condition_slug = "uncertain_plant_identification"
                    explanation = "The visual assessment did not consistently identify a supported plant."
                elif condition_slug.startswith(f"{species_slug}_"):
                    # Gemini may include the plant name in its free-form
                    # condition text. The API adds that prefix itself.
                    condition_slug = condition_slug[len(species_slug) + 1:]

        return {
            "leaf_detected": data["leaf_detected"],
            "plant_species": species_slug,
            "condition": condition.strip(),
            "condition_slug": condition_slug,
            "category": category,
            "visual_confidence": confidence,
            "explanation": explanation.strip(),
        }

    @staticmethod
    def _slug(value: str) -> str:
        ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        return "_".join(re.findall(r"[a-z0-9]+", ascii_value.lower()))[:80]

    @staticmethod
    def _supported_species(species_slug: str) -> Optional[str]:
        aliases = {
            "basil": "basil",
            "sweet_basil": "basil",
            "ocimum_basilicum": "basil",
            "basil_ocimum_basilicum": "basil",
            "basil_leaf": "basil",
            "basil_plant": "basil",
            "aloe": "aloe_vera",
            "aloe_vera": "aloe_vera",
            "aloe_barbadensis": "aloe_vera",
            "aloe_barbadensis_miller": "aloe_vera",
            "aloe_vera_leaf": "aloe_vera",
            "aloe_vera_plant": "aloe_vera",
            "mint": "mint",
            "mentha": "mint",
            "mentha_spicata": "mint",
            "spearmint": "mint",
            "mentha_piperita": "mint",
            "peppermint": "mint",
            "mint_leaf": "mint",
            "mint_plant": "mint",
        }
        if species_slug in aliases:
            return aliases[species_slug]
        # Gemini often includes a common name and its botanical name together,
        # e.g. "Basil (Ocimum basilicum)". Accept these supported-name variants
        # without allowing an unrelated species name through the crop gate.
        tokens = set(species_slug.split("_"))
        if "basil" in tokens or {"ocimum", "basilicum"}.issubset(tokens):
            return "basil"
        if "aloe" in tokens:
            return "aloe_vera"
        if "mint" in tokens or "mentha" in tokens or species_slug in {"spearmint", "peppermint"}:
            return "mint"
        return None

    def verify(
        self,
        image: Image.Image,
        prediction: PredictionResponse,
        class_names: Mapping[int, str],
    ) -> Dict[str, Any]:
        """Review a YOLO response and apply only validated class-name corrections.

        The returned mapping is safe to attach to the API response. Provider
        details and credentials are intentionally excluded from it and logs.
        """
        if not self.settings.gemini_vision_enabled:
            return self._result("disabled", "Gemini Vision review is disabled.")
        if not self.settings.gemini_api_key:
            return self._result(
                "missing_key", "Gemini Vision is not configured; the YOLO26 result is shown unchanged."
            )
        if prediction.task_type not in {"detection", "classification"} or not prediction.detections:
            return self._result("skipped", "No eligible YOLO26 result was available for visual review.")

        try:
            names = {int(class_id): str(name) for class_id, name in class_names.items()}
            review_items = self._make_review_items(image.convert("RGB"), prediction, names)
            if not review_items:
                return self._result("skipped", "No valid, species-matched image regions were available for review.")

            payload = self._request_review(review_items, prediction)
            corrections = self._validate_review(payload, review_items)

            # Apply only after the complete provider response has passed validation.
            original_detections = [item.to_dict() for item in prediction.detections]
            applied: List[Dict[str, Any]] = []
            suppressed_count = 0
            name_to_id = {name: class_id for class_id, name in names.items()}
            for review_item, suggestion in zip(review_items, corrections):
                if suggestion is None:
                    continue
                detection_index = review_item["detection_index"]
                detection = prediction.detections[detection_index]
                proposed_class = suggestion["proposed_class"]
                if proposed_class == detection.class_name:
                    continue
                if suggestion["visual_confidence"] < self.settings.gemini_correction_threshold:
                    suppressed_count += 1
                    continue

                from_class = detection.class_name
                detection.class_id = name_to_id[proposed_class]
                detection.class_name = proposed_class
                detection.category = infer_category(proposed_class)
                # Keep YOLO confidence and bbox unchanged. Gemini's score is not
                # calibrated and must never be presented as model confidence.
                applied.append(
                    {
                        "detection_index": detection_index,
                        "from_class": from_class,
                        "to_class": proposed_class,
                        "visual_confidence": suggestion["visual_confidence"],
                        "explanation": suggestion["explanation"],
                    }
                )

            if applied:
                top_index = max(
                    range(len(prediction.detections)),
                    key=lambda index: prediction.detections[index].confidence,
                )
                prediction.primary_condition = prediction.detections[top_index].class_name
                top_species = infer_species(prediction.primary_condition)
                if top_species != "unknown":
                    prediction.plant_species = top_species
                status = "corrected"
                note = "Gemini suggested a label correction. YOLO26 confidence and box coordinates are unchanged."
            else:
                status = "reviewed"
                note = "Gemini reviewed the result; no label change passed the configured correction checks."

            return {
                "status": status,
                "model": self.settings.gemini_model,
                "reviewed_count": len(review_items),
                "suppressed_suggestions": suppressed_count,
                "corrections": applied,
                "confidence_is_calibrated": False,
                "note": note,
                "original_detections": original_detections,
            }
        except Exception as exc:
            # Do not log provider messages: they can contain request metadata.
            logger.warning("Gemini Vision review failed (%s); preserving YOLO26 result.", type(exc).__name__)
            return self._result(
                "failed", "Gemini Vision review was unavailable; the YOLO26 result is shown unchanged."
            )

    def _make_review_items(
        self,
        image: Image.Image,
        prediction: PredictionResponse,
        names: Mapping[int, str],
    ) -> List[Dict[str, Any]]:
        ordered_indices = sorted(
            range(len(prediction.detections)),
            key=lambda index: prediction.detections[index].confidence,
            reverse=True,
        )[: self.settings.gemini_max_detections]
        items = []
        for detection_index in ordered_indices:
            detection = prediction.detections[detection_index]
            species = infer_species(detection.class_name)
            if species == "unknown":
                species = prediction.plant_species
            if species not in {"basil", "mint", "aloe_vera"}:
                continue
            prefix = f"{species}_"
            candidates = sorted({name for name in names.values() if name.lower().startswith(prefix)})
            if detection.class_name not in candidates or len(candidates) < 2:
                continue

            crop = image.copy() if prediction.task_type == "classification" else self._expanded_crop(image, detection)
            if crop is None:
                continue
            items.append(
                {
                    "detection_index": detection_index,
                    "species": species,
                    "current_class": detection.class_name,
                    "candidate_classes": candidates,
                    "yolo_confidence": detection.confidence,
                    "bbox": detection.bbox.to_dict(),
                    "image_bytes": self._jpeg_bytes(crop),
                }
            )
        return items

    @staticmethod
    def _expanded_crop(image: Image.Image, detection: DetectionItem) -> Optional[Image.Image]:
        bbox = detection.bbox
        values = (bbox.x_center, bbox.y_center, bbox.width, bbox.height)
        if not all(math.isfinite(float(value)) for value in values):
            return None
        width, height = image.size
        box_width = min(1.0, max(0.0, bbox.width)) * width
        box_height = min(1.0, max(0.0, bbox.height)) * height
        center_x = min(1.0, max(0.0, bbox.x_center)) * width
        center_y = min(1.0, max(0.0, bbox.y_center)) * height
        if box_width <= 0 or box_height <= 0:
            return None
        pad_x = max(8.0, box_width * 0.20)
        pad_y = max(8.0, box_height * 0.20)
        left = max(0, int(math.floor(center_x - box_width / 2 - pad_x)))
        top = max(0, int(math.floor(center_y - box_height / 2 - pad_y)))
        right = min(width, int(math.ceil(center_x + box_width / 2 + pad_x)))
        bottom = min(height, int(math.ceil(center_y + box_height / 2 + pad_y)))
        if right <= left or bottom <= top:
            return None
        return image.crop((left, top, right, bottom)).convert("RGB")

    @staticmethod
    def _jpeg_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return buffer.getvalue()

    def _request_review(self, review_items: List[Dict[str, Any]], prediction: PredictionResponse) -> Any:
        from google import genai
        from google.genai import types

        contents: List[Any] = [
            "Review the supplied plant-leaf image regions only. The YOLO26 class is a candidate, not ground truth. "
            "For each listed detection, either return no correction or propose exactly one better label from that "
            "detection's allowed class list. Do not invent a label, change species, infer an unlisted condition, or "
            "return bounding boxes. Be conservative: only return a correction when the visual evidence clearly "
            "supports it. The numeric visual_confidence is an uncalibrated visual-review score, not a probability. "
            "Return an empty corrections array when uncertain."
        ]
        for item in review_items:
            contents.append(
                "\nDetection index {index}; task={task}; species={species}; current YOLO26 class={current}; "
                "YOLO26 confidence={confidence:.4f}; normalized box={bbox}; allowed classes={classes}. "
                "The next image is the expanded detection crop (or full leaf for classification).".format(
                    index=item["detection_index"],
                    task=prediction.task_type,
                    species=item["species"],
                    current=item["current_class"],
                    confidence=item["yolo_confidence"],
                    bbox=json.dumps(item["bbox"], separators=(",", ":")),
                    classes=json.dumps(item["candidate_classes"], ensure_ascii=True),
                )
            )
            contents.append(types.Part.from_bytes(data=item["image_bytes"], mime_type="image/jpeg"))

        if self._client is None:
            http_options = types.HttpOptions(
                timeout=int(self.settings.gemini_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            self._client = genai.Client(api_key=self.settings.gemini_api_key, http_options=http_options)

        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=_ReviewSchema,
        )
        for attempt in range(self.settings.gemini_max_retries + 1):
            try:
                return self._client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                if attempt >= self.settings.gemini_max_retries or not self._is_transient(exc):
                    raise
                time.sleep(min(0.25 * (2**attempt), 0.75))
        raise RuntimeError("Gemini review request did not complete")

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if code is None and response is not None:
            code = getattr(response, "status_code", None)
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None
        if code in {408, 500, 502, 503, 504}:
            return True
        if code is not None:
            return False  # In particular, do not retry quota/rate-limit 429 errors.
        name = type(exc).__name__.lower()
        return any(token in name for token in ("timeout", "connecterror", "connectionreset", "networkerror"))

    @staticmethod
    def _validate_review(payload: Any, review_items: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        text = getattr(payload, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Gemini returned no structured review")
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Gemini returned invalid JSON") from exc
        if not isinstance(data, dict) or set(data) != {"corrections"} or not isinstance(data["corrections"], list):
            raise ValueError("Gemini response does not match the review schema")

        item_by_index = {item["detection_index"]: item for item in review_items}
        validated: Dict[int, Dict[str, Any]] = {}
        for entry in data["corrections"]:
            if not isinstance(entry, dict) or set(entry) != {
                "detection_index", "proposed_class", "visual_confidence", "explanation"
            }:
                raise ValueError("Gemini correction fields are invalid")
            detection_index = entry["detection_index"]
            confidence = entry["visual_confidence"]
            proposed_class = entry["proposed_class"]
            explanation = entry["explanation"]
            if isinstance(detection_index, bool) or not isinstance(detection_index, int):
                raise ValueError("Gemini returned an invalid detection index")
            if detection_index not in item_by_index or detection_index in validated:
                raise ValueError("Gemini returned an unknown or duplicate detection index")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("Gemini returned an invalid visual confidence")
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Gemini returned an out-of-range visual confidence")
            if not isinstance(proposed_class, str) or proposed_class not in item_by_index[detection_index]["candidate_classes"]:
                raise ValueError("Gemini proposed a class outside the model taxonomy")
            if not isinstance(explanation, str) or len(explanation) > 300:
                raise ValueError("Gemini returned an invalid explanation")
            validated[detection_index] = {
                "proposed_class": proposed_class,
                "visual_confidence": confidence,
                "explanation": explanation.strip(),
            }

        return [validated.get(item["detection_index"]) for item in review_items]

    def _result(self, status: str, note: str) -> Dict[str, Any]:
        return {
            "status": status,
            "model": self.settings.gemini_model,
            "reviewed_count": 0,
            "suppressed_suggestions": 0,
            "corrections": [],
            "confidence_is_calibrated": False,
            "note": note,
        }

    def _failure_result(self, exc: Exception) -> Dict[str, Any]:
        """Return a safe, actionable failure code without exposing provider text."""
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None

        if isinstance(exc, _GeminiResponseError):
            failure_code = "invalid_response"
            note = "Gemini returned an unreadable diagnosis after a retry; the YOLO26 result is shown as fallback."
        elif code in {401, 403}:
            failure_code = "authentication"
            note = "Gemini rejected the configured API key; check GEMINI_API_KEY. The YOLO26 result is shown as fallback."
        elif code == 404:
            failure_code = "model_unavailable"
            note = "The configured Gemini model is unavailable; check GEMINI_MODEL. The YOLO26 result is shown as fallback."
        elif code == 429:
            failure_code = "quota_or_rate_limit"
            note = "Gemini quota or rate limit was reached; the YOLO26 result is shown as fallback."
        elif code == 400:
            failure_code = "invalid_request"
            note = "Gemini rejected the request configuration; check the model and structured-output settings. The YOLO26 result is shown as fallback."
        elif code in {408, 500, 502, 503, 504} or self._is_transient(exc):
            failure_code = "temporary_provider_error"
            note = "Gemini is temporarily unavailable; retry the image diagnosis. The YOLO26 result is shown as fallback."
        else:
            failure_code = "request_failed"
            note = "Gemini could not complete the request; check the connection and Gemini configuration. The YOLO26 result is shown as fallback."

        logger.warning("Gemini Vision analysis failed (%s); using YOLO26 fallback.", failure_code)
        result = self._result("failed", note)
        result["failure_code"] = failure_code
        return result
