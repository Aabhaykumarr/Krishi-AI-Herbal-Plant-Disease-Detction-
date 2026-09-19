"""Automated live server test suite for KrishiAI API and Web Dashboard."""
import io
import json
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from PIL import Image


BASE_URL = "http://localhost:8000"


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health_endpoint():
    print_section("Test 1: Health Probe (GET /health)")
    url = f"{BASE_URL}/health"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        data = json.loads(resp.read().decode())
        print("Status Code: 200 OK")
        print("Response Body:", json.dumps(data, indent=2))
        assert data["status"] == "healthy"
        assert data["models"]["detection"] is True
        assert data["models"]["classification"] is True
        print("[PASS] Health endpoint verified.")


def test_model_info_endpoint():
    print_section("Test 2: Model Information & Taxonomy (GET /model/info)")
    url = f"{BASE_URL}/model/info"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print("Status Code: 200 OK")
        print(f"Model Version: {data['model_version']}")
        print(f"Architecture: {data['architecture']}")
        print(f"Supported Crops: {data['supported_crops']}")
        print(f"Taxonomy Categories: {list(data['taxonomy'].keys())}")
        assert "diseases" in data["taxonomy"]
        assert "deficiencies" in data["taxonomy"]
        assert "conditions_and_freshness" in data["taxonomy"]
        assert "healthy_states" in data["taxonomy"]
        print("[PASS] Model info & taxonomy verified.")


def test_dashboard_ui_endpoint():
    print_section("Test 3: Web Dashboard HTML (GET /)")
    url = f"{BASE_URL}/"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        print("Status Code: 200 OK")
        print(f"HTML Size: {len(content)} bytes")
        assert "KrishiAI" in content
        assert "result-canvas" in content
        assert "dropzone" in content
        assert "Quick Test Samples" in content
        assert "Agronomic Management Recommendations" in content or "Agronomic" in content
        print("[PASS] Web dashboard HTML serving verified.")


def test_openapi_docs():
    print_section("Test 4: Swagger Docs & OpenAPI Schema (GET /docs, GET /openapi.json)")
    with urllib.request.urlopen(f"{BASE_URL}/docs") as resp:
        assert resp.status == 200
        print("GET /docs: 200 OK")
    with urllib.request.urlopen(f"{BASE_URL}/openapi.json") as resp:
        assert resp.status == 200
        schema = json.loads(resp.read().decode())
        print("GET /openapi.json: 200 OK")
        print(f"Paths Defined: {list(schema['paths'].keys())}")
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/model/info" in schema["paths"]
        print("[PASS] OpenAPI specifications verified.")


def _send_multipart_predict(
    image_bytes: bytes,
    filename: str = "test.jpg",
    content_type: str = "image/jpeg",
    plant_species: str = None,
    conf: float = None,
    iou: float = None,
):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()

    # File field
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(image_bytes)
    body.extend(b"\r\n")

    # Optional fields
    if plant_species:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="plant_species"\r\n\r\n{plant_species}\r\n'.encode())
    if conf is not None:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="confidence_threshold"\r\n\r\n{conf}\r\n'.encode())
    if iou is not None:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="iou_threshold"\r\n\r\n{iou}\r\n'.encode())

    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return req


def test_predict_basil_detection():
    print_section("Test 5: Live Prediction - Basil Detection Sample (POST /predict)")
    # Find a test image from detection test split
    test_imgs = list(Path("data/final/detection/test/images").glob("basil_*.jpg"))
    assert len(test_imgs) > 0
    img_path = test_imgs[0]
    print(f"Testing with image: {img_path.name}")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    req = _send_multipart_predict(img_bytes, filename=img_path.name, plant_species="basil")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print("Status Code: 200 OK")
        print("Response:", json.dumps(data, indent=2))
        assert data["plant_species"] == "basil"
        assert data["task_type"] in ("detection", "classification")
        assert "latency_ms" in data
        assert data["latency_ms"]["total"] > 0
        print(f"[PASS] Basil prediction verified (Task: {data['task_type']}, Latency: {data['latency_ms']['total']}ms).")


def test_predict_aloe_vera_classification():
    print_section("Test 6: Live Prediction - Aloe Vera Classification Sample (POST /predict)")
    test_imgs = list(Path("data/final/classification/test/aloe_vera_healthy").glob("*.jpg"))
    assert len(test_imgs) > 0
    img_path = test_imgs[0]
    print(f"Testing with image: {img_path.name}")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    req = _send_multipart_predict(img_bytes, filename=img_path.name, plant_species="aloe_vera")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print("Status Code: 200 OK")
        print("Response:", json.dumps(data, indent=2))
        assert data["plant_species"] == "aloe_vera"
        assert data["task_type"] == "classification"
        assert data["confidence"] > 0.0
        print(f"[PASS] Aloe Vera classification verified (Condition: {data['primary_condition']}, Conf: {data['confidence']*100:.1f}%).")


def test_predict_mint_detection():
    print_section("Test 7: Live Prediction - Mint Detection Sample (POST /predict)")
    test_imgs = list(Path("data/final/detection/test/images").glob("mint_*.jpg"))
    assert len(test_imgs) > 0
    img_path = test_imgs[0]
    print(f"Testing with image: {img_path.name}")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    req = _send_multipart_predict(img_bytes, filename=img_path.name, plant_species="mint")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print("Status Code: 200 OK")
        print("Response:", json.dumps(data, indent=2))
        assert data["plant_species"] == "mint"
        assert data["task_type"] in ("detection", "classification")
        assert "latency_ms" in data
        print(f"[PASS] Mint prediction verified (Task: {data['task_type']}, Latency: {data['latency_ms']['total']}ms).")


def test_predict_custom_thresholds():
    print_section("Test 8: Prediction with Custom Confidence and IoU Thresholds (POST /predict)")
    img = Image.new("RGB", (300, 300), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    req = _send_multipart_predict(img_bytes, conf=0.75, iou=0.45)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        print("Status Code: 200 OK")
        print(f"Task Type: {data['task_type']}")
        print(f"Primary Condition: {data['primary_condition']}")
        print(f"Latency: {data['latency_ms']}")
        print("[PASS] Custom thresholds handled successfully.")


def test_invalid_file_rejection():
    print_section("Test 9: Error Handling - Non-Image File Upload (POST /predict)")
    bad_bytes = b"This is a plain text file, not a leaf image."
    req = _send_multipart_predict(bad_bytes, filename="notes.txt", content_type="text/plain")

    try:
        with urllib.request.urlopen(req) as resp:
            raise AssertionError(f"Expected 400 Bad Request, but got {resp.status}")
    except urllib.error.HTTPError as err:
        assert err.code == 400, f"Expected 400, got {err.code}"
        body = json.loads(err.read().decode())
        print(f"HTTP Status: {err.code} Bad Request")
        print(f"Error Detail: {body}")
        assert "Invalid file type" in body["detail"]
        print("[PASS] Invalid file rejection properly handled with 400 status.")


def test_cors_headers():
    print_section("Test 10: CORS Headers Verification")
    url = f"{BASE_URL}/health"
    req = urllib.request.Request(url, headers={"Origin": "http://example.com"})
    with urllib.request.urlopen(req) as resp:
        headers = dict(resp.headers)
        origin_header = headers.get("access-control-allow-origin")
        print(f"Access-Control-Allow-Origin: {origin_header}")
        assert origin_header in ["*", "http://example.com"]
        print("[PASS] CORS middleware verified.")


def main():
    print("\n" + "#" * 60)
    print("   KRISHIAI END-TO-END LIVE SERVICE VERIFICATION SUITE")
    print("#" * 60)

    test_health_endpoint()
    test_model_info_endpoint()
    test_dashboard_ui_endpoint()
    test_openapi_docs()
    test_predict_basil_detection()
    test_predict_aloe_vera_classification()
    test_predict_mint_detection()
    test_predict_custom_thresholds()
    test_invalid_file_rejection()
    test_cors_headers()

    print("\n" + "#" * 60)
    print("   ALL 10 LIVE SERVICE & UI TESTS PASSED SUCCESSFULLY!")
    print("#" * 60 + "\n")
    test_cors_headers()

    print("\n" + "#" * 60)
    print("   ALL 9 LIVE SERVICE & UI TESTS PASSED SUCCESSFULLY!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
