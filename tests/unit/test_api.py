"""Unit and integration tests for FastAPI REST service (Section 51 & Contract 6.1)."""
from pathlib import Path
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from krishiai.api.app import create_app


@pytest.fixture
def client(monkeypatch):
    from importlib import import_module

    api_module = import_module("krishiai.api.app")

    class _DisabledGeminiVerifier:
        def analyze(self, **kwargs):
            return {
                "status": "disabled",
                "note": "Gemini Vision is disabled in API tests; use YOLO26 fallback.",
            }

    # Tests must remain local and must never consume a developer's API quota.
    monkeypatch.setattr(api_module, "_gemini_verifier", _DisabledGeminiVerifier())
    app = api_module.create_app()
    return TestClient(app)


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "krishiai-api"
    assert "models" in data
    assert "device" in data["models"]


def test_get_predict_opens_dashboard_while_post_predict_remains_api(client):
    response = client.get("/predict")

    assert response.status_code == 200
    assert "KrishiAI" in response.text
    assert 'id="btn-camera"' in response.text
    assert 'id="btn-gallery"' in response.text


def test_literal_double_asterisk_path_redirects_to_dashboard(client):
    response = client.get("/**", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/"


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "krishiai-yolo26-v1.0"
    assert data["architecture"] == "YOLO26"
    assert "basil" in data["supported_crops"]
    assert "aloe_vera" in data["supported_crops"]
    assert "mint" in data["supported_crops"]
    assert "taxonomy" in data
    assert "diseases" in data["taxonomy"]


def test_predict_endpoint_with_image(client, tmp_path):
    sample_file = Path("static/samples/basil_sample.jpg")
    if sample_file.exists():
        with open(sample_file, "rb") as f:
            buf = io.BytesIO(f.read())
    else:
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

    files = {"file": ("test_leaf.jpg", buf, "image/jpeg")}
    data = {"plant_species": "basil", "confidence_threshold": "0.25"}

    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert "request_id" in result
    assert "timestamp" in result
    assert result["plant_species"] == "basil"
    assert "latency_ms" in result
    assert "total" in result["latency_ms"]


def test_predict_endpoint_invalid_file_type(client):
    buf = io.BytesIO(b"not an image file")
    files = {"file": ("test.txt", buf, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_predict_endpoint_large_mobile_image(client):
    # Simulate high-resolution camera photo from smartphone (e.g. 3000x2000)
    img = Image.new("RGB", (3000, 2000), color=(40, 160, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("mobile_camera_photo.jpg", buf, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "primary_condition" in data
    assert "latency_ms" in data


def test_predict_rejects_upload_over_configured_limit(client, monkeypatch):
    from krishiai.config import settings as settings_module

    mock_settings = settings_module.Settings.load()
    mock_settings.max_upload_size_mb = 1
    monkeypatch.setattr(settings_module.Settings, "load", lambda: mock_settings)

    files = {"file": ("large.jpg", io.BytesIO(b"x" * (1024 * 1024 + 1)), "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]


def test_save_to_dataset_valid_image(client, tmp_path, monkeypatch):
    # Mock settings to use tmp_path
    from krishiai.config import settings as settings_module
    mock_settings = settings_module.Settings.load()
    mock_settings.incoming_data_root = tmp_path / "incoming"
    mock_settings.max_upload_size_mb = 10

    def mock_load():
        return mock_settings

    monkeypatch.setattr(settings_module.Settings, "load", mock_load)

    # Create test image
    img = Image.new("RGB", (100, 100), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("basil_leaf.jpg", buf, "image/jpeg")}
    data = {"plant_species": "basil"}

    response = client.post("/save-to-dataset", files=files, data=data)
    assert response.status_code == 201
    result = response.json()
    assert result["success"] is True
    assert result["species"] == "basil"
    assert "saved_path" in result
    assert "basil" in result["saved_path"]
    assert "timestamp" in result
    assert "image_info" in result


def test_save_to_dataset_missing_species(client):
    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf, "image/jpeg")}
    # Missing plant_species parameter

    response = client.post("/save-to-dataset", files=files)
    assert response.status_code == 422  # FastAPI validation error


def test_save_to_dataset_invalid_species(client, tmp_path, monkeypatch):
    from krishiai.config import settings as settings_module
    mock_settings = settings_module.Settings.load()
    mock_settings.incoming_data_root = tmp_path / "incoming"

    def mock_load():
        return mock_settings

    monkeypatch.setattr(settings_module.Settings, "load", mock_load)

    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf, "image/jpeg")}
    data = {"plant_species": "tomato"}

    response = client.post("/save-to-dataset", files=files, data=data)
    assert response.status_code == 400
    assert "Unsupported species" in response.json()["detail"]


def test_save_to_dataset_invalid_file_type(client, tmp_path, monkeypatch):
    from krishiai.config import settings as settings_module
    mock_settings = settings_module.Settings.load()
    mock_settings.incoming_data_root = tmp_path / "incoming"

    def mock_load():
        return mock_settings

    monkeypatch.setattr(settings_module.Settings, "load", mock_load)

    buf = io.BytesIO(b"not an image")
    files = {"file": ("malicious.exe", buf, "application/octet-stream")}
    data = {"plant_species": "basil"}

    response = client.post("/save-to-dataset", files=files, data=data)
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]


def test_save_to_dataset_without_file(client):
    data = {"plant_species": "basil"}
    response = client.post("/save-to-dataset", data=data)
    assert response.status_code == 422  # Missing required file


def test_save_to_dataset_aloe_vera(client, tmp_path, monkeypatch):
    from krishiai.config import settings as settings_module
    mock_settings = settings_module.Settings.load()
    mock_settings.incoming_data_root = tmp_path / "incoming"

    def mock_load():
        return mock_settings

    monkeypatch.setattr(settings_module.Settings, "load", mock_load)

    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("aloe.jpg", buf, "image/jpeg")}
    data = {"plant_species": "aloe_vera"}

    response = client.post("/save-to-dataset", files=files, data=data)
    assert response.status_code == 201
    result = response.json()
    assert result["species"] == "aloe_vera"
    assert "aloe_vera" in result["saved_path"]


def test_save_to_dataset_mint(client, tmp_path, monkeypatch):
    from krishiai.config import settings as settings_module
    mock_settings = settings_module.Settings.load()
    mock_settings.incoming_data_root = tmp_path / "incoming"

    def mock_load():
        return mock_settings

    monkeypatch.setattr(settings_module.Settings, "load", mock_load)

    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("mint_leaf.jpg", buf, "image/jpeg")}
    data = {"plant_species": "mint"}

    response = client.post("/save-to-dataset", files=files, data=data)
    assert response.status_code == 201
    result = response.json()
    assert result["species"] == "mint"
    assert "mint" in result["saved_path"]

