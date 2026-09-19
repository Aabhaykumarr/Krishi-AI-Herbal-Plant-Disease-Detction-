"""Unit and integration tests for FastAPI REST service (Section 51 & Contract 6.1)."""
from pathlib import Path
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from krishiai.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "krishiai-api"
    assert "models" in data
    assert "device" in data["models"]


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
    # Create a synthetic test image
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
