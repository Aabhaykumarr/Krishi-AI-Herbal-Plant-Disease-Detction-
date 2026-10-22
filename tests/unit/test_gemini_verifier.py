"""Mock-only tests for optional Gemini review and safe YOLO fallback behavior."""
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from krishiai.config.settings import Settings
from krishiai.inference.gemini_verifier import GeminiVisionVerifier, _DiagnosisSchema
from krishiai.inference.types import BBox, DetectionItem, PredictionResponse


class _FakeModels:
    def __init__(self, responses):
        self.responses = list(responses) if isinstance(responses, list) else [responses]
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if hasattr(response, "parsed") or hasattr(response, "text"):
            return response
        return SimpleNamespace(text=json.dumps(response))


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


class _ProviderError(Exception):
    def __init__(self, code):
        super().__init__("provider failure")
        self.code = code


def _settings(**overrides):
    values = {
        "data_root": __import__("pathlib").Path("data"),
        "device": "cpu",
        "seed": 42,
        "log_level": "INFO",
        "incoming_data_root": __import__("pathlib").Path("data/incoming"),
        "max_upload_size_mb": 10,
        "gemini_api_key": "test-key",
        "gemini_vision_enabled": True,
        "gemini_model": "gemini-test-vision",
        "gemini_timeout_seconds": 5,
        "gemini_max_detections": 4,
        "gemini_correction_threshold": 0.82,
        "gemini_max_retries": 1,
    }
    values.update(overrides)
    return Settings(**values)


def _prediction(task_type="detection"):
    name = "aloe_vera_rust" if task_type == "classification" else "basil_bacterial_spot"
    return PredictionResponse(
        task_type=task_type,
        plant_species="aloe_vera" if task_type == "classification" else "basil",
        primary_condition=name,
        confidence=0.91,
        detections=[
            DetectionItem(
                class_id=0,
                class_name=name,
                category="disease",
                confidence=0.91,
                bbox=BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.3),
            )
        ],
    )


def _correction(class_name="basil_healthy", visual_confidence=0.93, detection_index=0):
    return {
        "corrections": [
            {
                "detection_index": detection_index,
                "proposed_class": class_name,
                "visual_confidence": visual_confidence,
                "explanation": "The reviewed leaf appears uniformly green.",
            }
        ]
    }


def _diagnosis(
    plant_species="grape",
    condition="healthy",
    category="healthy",
    visual_confidence=0.94,
    leaf_detected=True,
):
    return {
        "leaf_detected": leaf_detected,
        "plant_species": plant_species,
        "condition": condition,
        "category": category,
        "visual_confidence": visual_confidence,
        "explanation": "The visible leaf appears mostly healthy.",
    }


def test_valid_same_species_correction_preserves_yolo_confidence_and_box():
    prediction = _prediction()
    original_bbox = prediction.detections[0].bbox.to_dict()
    client = _FakeClient(_correction())
    verifier = GeminiVisionVerifier(settings=_settings(), client=client)

    result = verifier.verify(
        Image.new("RGB", (100, 100), (40, 130, 40)),
        prediction,
        {0: "basil_bacterial_spot", 1: "basil_healthy", 2: "mint_fresh"},
    )

    assert result["status"] == "corrected"
    assert result["original_detections"][0]["class_name"] == "basil_bacterial_spot"
    assert result["corrections"][0]["to_class"] == "basil_healthy"
    assert prediction.detections[0].class_name == "basil_healthy"
    assert prediction.detections[0].class_id == 1
    assert prediction.detections[0].confidence == 0.91
    assert prediction.confidence == 0.91
    assert prediction.detections[0].bbox.to_dict() == original_bbox
    assert prediction.primary_condition == "basil_healthy"
    assert result["confidence_is_calibrated"] is False


def test_missing_key_disables_review_without_mutating_result():
    prediction = _prediction()
    result = GeminiVisionVerifier(settings=_settings(gemini_api_key=None)).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "missing_key"
    assert prediction.primary_condition == "basil_bacterial_spot"
    assert "original_detections" not in result


def test_disabled_review_does_not_call_provider():
    prediction = _prediction()
    client = _FakeClient(_correction())
    result = GeminiVisionVerifier(settings=_settings(gemini_vision_enabled=False), client=client).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "disabled"
    assert client.models.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"corrections": [{"detection_index": 0, "proposed_class": "mint_fresh", "visual_confidence": 0.99, "explanation": "wrong species"}]},
        {"corrections": [{"detection_index": 7, "proposed_class": "basil_healthy", "visual_confidence": 0.99, "explanation": "unknown index"}]},
        {"corrections": [{"detection_index": 0, "proposed_class": "basil_healthy", "visual_confidence": 0.99, "explanation": "ok", "extra": "bad"}]},
        {"corrections": [{"detection_index": 0, "proposed_class": "basil_healthy", "visual_confidence": 1.5, "explanation": "bad score"}]},
    ],
)
def test_invalid_or_out_of_taxonomy_response_falls_back(payload):
    prediction = _prediction()
    result = GeminiVisionVerifier(settings=_settings(), client=_FakeClient(payload)).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy", 2: "mint_fresh"}
    )
    assert result["status"] == "failed"
    assert prediction.detections[0].class_name == "basil_bacterial_spot"
    assert prediction.confidence == 0.91


def test_low_visual_score_does_not_correct():
    prediction = _prediction()
    result = GeminiVisionVerifier(settings=_settings(), client=_FakeClient(_correction(visual_confidence=0.4))).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "reviewed"
    assert result["suppressed_suggestions"] == 1
    assert prediction.detections[0].class_name == "basil_bacterial_spot"


def test_transient_error_gets_one_bounded_retry_and_then_succeeds():
    transient = _ProviderError(503)
    client = _FakeClient([transient, _correction()])
    prediction = _prediction()
    result = GeminiVisionVerifier(settings=_settings(), client=client).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "corrected"
    assert len(client.models.calls) == 2


def test_quota_error_is_not_retried_and_falls_back():
    quota = _ProviderError(429)
    client = _FakeClient([quota, _correction()])
    prediction = _prediction()
    result = GeminiVisionVerifier(settings=_settings(), client=client).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "failed"
    assert len(client.models.calls) == 1
    assert prediction.detections[0].class_name == "basil_bacterial_spot"


def test_authentication_error_is_not_retried_and_provider_text_is_not_returned():
    auth = _ProviderError(401)
    client = _FakeClient([auth, _correction()])
    prediction = _prediction()
    result = GeminiVisionVerifier(settings=_settings(), client=client).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "failed"
    assert len(client.models.calls) == 1
    assert "provider failure" not in result["note"]
    assert prediction.detections[0].class_name == "basil_bacterial_spot"


def test_timeout_exception_gets_one_bounded_retry():
    client = _FakeClient([TimeoutError("private provider details"), {"corrections": []}])
    prediction = _prediction()
    result = GeminiVisionVerifier(settings=_settings(), client=client).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "reviewed"
    assert len(client.models.calls) == 2


def test_duplicate_indices_reject_the_entire_response():
    payload = {
        "corrections": [
            {"detection_index": 0, "proposed_class": "basil_healthy", "visual_confidence": 0.99, "explanation": "first"},
            {"detection_index": 0, "proposed_class": "basil_healthy", "visual_confidence": 0.99, "explanation": "second"},
        ]
    }
    prediction = _prediction()
    result = GeminiVisionVerifier(settings=_settings(), client=_FakeClient(payload)).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "failed"
    assert prediction.detections[0].class_name == "basil_bacterial_spot"


def test_unrecognized_prediction_is_not_sent_to_gemini():
    prediction = _prediction()
    prediction.task_type = "unrecognized"
    client = _FakeClient(_correction())
    result = GeminiVisionVerifier(settings=_settings(), client=client).verify(
        Image.new("RGB", (100, 100)), prediction, {0: "basil_bacterial_spot", 1: "basil_healthy"}
    )
    assert result["status"] == "skipped"
    assert client.models.calls == []


def test_classifier_review_uses_full_leaf_image():
    image = Image.new("RGB", (120, 80), (30, 100, 30))
    client = _FakeClient({"corrections": []})
    prediction = _prediction(task_type="classification")
    verifier = GeminiVisionVerifier(settings=_settings(), client=client)
    result = verifier.verify(
        image,
        prediction,
        {0: "aloe_vera_rust", 1: "aloe_vera_healthy", 2: "basil_healthy"},
    )
    assert result["status"] == "reviewed"
    from google.genai.types import Part
    part = next(content for content in client.models.calls[0]["contents"] if isinstance(content, Part))
    submitted = Image.open(io.BytesIO(part.inline_data.data))
    assert submitted.size == image.size


def test_crop_is_valid_and_expanded_for_localized_detection():
    image = Image.new("RGB", (200, 100), (30, 100, 30))
    prediction = _prediction()
    prediction.detections[0].bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)
    verifier = GeminiVisionVerifier(settings=_settings(), client=_FakeClient({"corrections": []}))
    items = verifier._make_review_items(
        image,
        prediction,
        {0: "basil_bacterial_spot", 1: "basil_healthy"},
    )
    crop = Image.open(io.BytesIO(items[0]["image_bytes"]))
    assert 40 < crop.width < image.width
    assert 20 < crop.height < image.height


def test_gemini_normalizes_an_unsupported_plant_to_non_herbal():
    client = _FakeClient(_diagnosis())
    verifier = GeminiVisionVerifier(settings=_settings(), client=client)

    result = verifier.analyze(Image.new("RGB", (100, 100), (40, 130, 40)))

    assert result["status"] == "analyzed"
    assert result["assessment"]["plant_species"] == "non_herbal_plant"
    assert result["assessment"]["condition"] == "Non-herbal plant detected"
    assert result["assessment"]["condition_slug"] == "non_herbal_plant_detected"
    assert result["assessment"]["category"] == "non_herbal_plant"
    assert "grape" not in result["assessment"]["explanation"].lower()
    assert len(client.models.calls) == 1
    assert client.models.calls[0]["config"].response_schema.__name__ == "_DiagnosisSchema"


def test_gemini_analysis_reports_failure_for_quota_and_keeps_fallback_available():
    client = _FakeClient(_ProviderError(429))
    result = GeminiVisionVerifier(settings=_settings(), client=client).analyze(
        Image.new("RGB", (100, 100), (40, 130, 40))
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "quota_or_rate_limit"
    assert len(client.models.calls) == 1
    assert "YOLO26" in result["note"]


def test_gemini_analysis_rejects_malformed_or_invalid_output():
    client = _FakeClient([
        _diagnosis(visual_confidence=1.5),
        _diagnosis(visual_confidence=1.5),
    ])
    result = GeminiVisionVerifier(settings=_settings(), client=client).analyze(
        Image.new("RGB", (100, 100), (40, 130, 40))
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "invalid_response"
    assert len(client.models.calls) == 2


def test_gemini_analysis_uses_sdk_parsed_structured_response():
    parsed = _DiagnosisSchema(**_diagnosis())
    client = _FakeClient(SimpleNamespace(text=None, parsed=parsed))

    result = GeminiVisionVerifier(settings=_settings(), client=client).analyze(
        Image.new("RGB", (100, 100), (40, 130, 40))
    )

    assert result["status"] == "analyzed"
    assert result["assessment"]["plant_species"] == "non_herbal_plant"


@pytest.mark.parametrize(
    ("species", "expected"),
    [
        ("Basil", "basil"),
        ("Basil (Ocimum basilicum)", "basil"),
        ("Ocimum basilicum", "basil"),
        ("Aloe", "aloe_vera"),
        ("Peppermint", "mint"),
    ],
)
def test_gemini_keeps_supported_species_and_normalizes_common_names(species, expected):
    result = GeminiVisionVerifier(settings=_settings(), client=_FakeClient(_diagnosis(plant_species=species))).analyze(
        Image.new("RGB", (100, 100), (40, 130, 40))
    )

    assert result["status"] == "analyzed"
    assert result["assessment"]["plant_species"] == expected
    assert result["assessment"]["category"] == "healthy"


def test_gemini_strips_species_prefix_from_condition_slug():
    result = GeminiVisionVerifier(
        settings=_settings(),
        client=_FakeClient(_diagnosis(plant_species="basil", condition="Basil downy mildew")),
    ).analyze(Image.new("RGB", (100, 100), (40, 130, 40)))

    assert result["assessment"]["condition_slug"] == "downy_mildew"


def test_species_hint_is_interpolated_in_gemini_prompt():
    client = _FakeClient(_diagnosis(plant_species="basil"))
    GeminiVisionVerifier(settings=_settings(), client=client).analyze(
        Image.new("RGB", (100, 100), (40, 130, 40)), requested_species="basil"
    )

    prompt = " ".join(
        part for part in client.models.calls[0]["contents"] if isinstance(part, str)
    )
    assert "UI selection is basil" in prompt
    assert "changing it to basil" in prompt
    assert "{hint}" not in prompt


def test_gemini_analysis_retries_invalid_structured_response_once():
    client = _FakeClient([{"wrong": "schema"}, _diagnosis()])

    result = GeminiVisionVerifier(settings=_settings(), client=client).analyze(
        Image.new("RGB", (100, 100), (40, 130, 40))
    )

    assert result["status"] == "analyzed"
    assert len(client.models.calls) == 2


def test_gemini_analysis_accepts_non_leaf_as_a_structured_result():
    result = GeminiVisionVerifier(
        settings=_settings(),
        client=_FakeClient(_diagnosis(plant_species="unknown", condition="no leaf visible", category="non_plant", leaf_detected=False)),
    ).analyze(Image.new("RGB", (100, 100), (40, 130, 40)))

    assert result["status"] == "analyzed"
    assert result["assessment"]["leaf_detected"] is False
    assert result["assessment"]["plant_species"] == "unknown"


def test_api_keeps_existing_fields_and_adds_auditable_gemini_result(monkeypatch):
    from fastapi.testclient import TestClient
    from importlib import import_module

    api_module = import_module("krishiai.api.app")

    class _Manager:
        def predict(self, **kwargs):
            return _prediction()

        def get_class_names(self, task_type):
            return {0: "basil_bacterial_spot", 1: "basil_healthy"}

    monkeypatch.setattr(api_module, "get_model_manager", lambda: _Manager())
    monkeypatch.setattr(
        api_module,
        "_gemini_verifier",
        GeminiVisionVerifier(settings=_settings(), client=_FakeClient(_diagnosis())),
    )
    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 100), (40, 130, 40)).save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    response = TestClient(api_module.create_app()).post(
        "/predict",
        files={"file": ("leaf.jpg", image_buffer, "image/jpeg")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["primary_condition"] == "non_herbal_plant_detected"
    assert result["plant_species"] == "non_herbal_plant"
    assert result["diagnosis_source"] == "gemini"
    assert result["task_type"] == "unrecognized"
    assert result["detections"] == []
    assert result["yolo_result"]["primary_condition"] == "basil_bacterial_spot"
    assert result["yolo_result"]["detections"][0]["class_name"] == "basil_bacterial_spot"
    assert result["verification"]["status"] == "non_herbal_plant_detected"
    assert result["confidence"] == 0.94
    assert result["confidence_source"] == "gemini_visual_score_uncalibrated"
    assert set(("request_id", "timestamp", "task_type", "detections", "latency_ms", "yolo_result")).issubset(result)


def test_api_returns_yolo_result_when_verifier_raises(monkeypatch):
    from fastapi.testclient import TestClient
    from importlib import import_module

    api_module = import_module("krishiai.api.app")
    execution_order = []

    class _Manager:
        def predict(self, **kwargs):
            execution_order.append("yolo")
            return _prediction()

        def get_class_names(self, task_type):
            return {0: "basil_bacterial_spot", 1: "basil_healthy"}

    class _BrokenVerifier:
        def analyze(self, **kwargs):
            execution_order.append("gemini")
            raise RuntimeError("provider secret must not escape")

    monkeypatch.setattr(api_module, "get_model_manager", lambda: _Manager())
    monkeypatch.setattr(api_module, "_gemini_verifier", _BrokenVerifier())
    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 100), (40, 130, 40)).save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    response = TestClient(api_module.create_app()).post(
        "/predict",
        files={"file": ("leaf.jpg", image_buffer, "image/jpeg")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["primary_condition"] == "basil_bacterial_spot"
    assert result["verification"]["status"] == "failed"
    assert "diagnosis_source" not in result
    assert "provider secret" not in response.text
    assert execution_order == ["gemini", "yolo"]


def test_api_preserves_safe_gemini_failure_details(monkeypatch):
    from fastapi.testclient import TestClient
    from importlib import import_module

    api_module = import_module("krishiai.api.app")

    class _Manager:
        def predict(self, **kwargs):
            return _prediction()

    class _FailedVerifier:
        def analyze(self, **kwargs):
            return {
                "status": "failed",
                "model": "gemini-test-vision",
                "failure_code": "invalid_response",
                "note": "Gemini returned an unreadable diagnosis after a retry.",
            }

    monkeypatch.setattr(api_module, "get_model_manager", lambda: _Manager())
    monkeypatch.setattr(api_module, "_gemini_verifier", _FailedVerifier())
    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 100), (40, 130, 40)).save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    response = TestClient(api_module.create_app()).post(
        "/predict",
        files={"file": ("leaf.jpg", image_buffer, "image/jpeg")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["verification"]["failure_code"] == "invalid_response"
    assert result["verification"]["note"] == "Gemini returned an unreadable diagnosis after a retry."
    assert result["primary_condition"] == "basil_bacterial_spot"
