"""
Test script verifying Out-Of-Distribution (OOD) Non-Plant rejection
(Book pages, IoT kit boxes, diagrams, documents) and live botanical leaf prediction.
"""
import io
import json
import urllib.request
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8000"

def create_yellowed_book_page() -> bytes:
    """Creates a synthetic yellowed printed book page with text lines."""
    img = Image.new("RGB", (400, 600), (242, 235, 218))
    draw = ImageDraw.Draw(img)
    for y in range(40, 560, 20):
        draw.line([(40, y), (360, y)], fill=(20, 20, 20), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_white_textbook_page() -> bytes:
    """Creates a white textbook page with headers and paragraphs."""
    img = Image.new("RGB", (450, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "CHAPTER 4: EMBEDDED SYSTEMS & IoT", fill=(0, 0, 0))
    for y in range(80, 560, 22):
        draw.line([(40, y), (410, y)], fill=(80, 80, 80), width=1)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_iot_box_image() -> bytes:
    """Creates an IoT kit cardboard packaging box with blue banner, text, and barcodes."""
    img = Image.new("RGB", (400, 400), (180, 140, 95))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 360, 180], fill=(20, 30, 80))
    draw.text((50, 60), "ARDUINO / ESP32 IoT STARTER KIT", fill=(255, 255, 255))
    draw.rectangle([40, 200, 360, 360], fill=(220, 220, 220))
    draw.text((50, 220), "Sensors: DHT11, Ultrasonic, Relay", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_synthetic_roadmap_image() -> bytes:
    """Creates a synthetic roadmap diagram."""
    img = Image.new("RGB", (600, 400), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 120], outline=(60, 60, 70), width=2, fill=(240, 243, 246))
    draw.text((65, 75), "Step 1: Planning", fill=(20, 20, 30))
    draw.rectangle([250, 50, 400, 120], outline=(60, 60, 70), width=2, fill=(240, 243, 246))
    draw.text((265, 75), "Step 2: Execution", fill=(20, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def send_predict(img_bytes: bytes, filename: str, plant_species: str = None) -> dict:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.extend(f"Content-Type: image/jpeg\r\n\r\n".encode())
    body.extend(img_bytes)
    body.extend(b"\r\n")
    if plant_species:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="plant_species"\r\n\r\n{plant_species}\r\n'.encode())
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())

def main():
    print("=" * 75)
    print("  KRISHIAI OUT-OF-DISTRIBUTION & BOTANICAL INFERENCE ASSURANCE SUITE")
    print("=" * 75)

    # 1. Yellowed Book Page Rejection
    print("\n1. Testing Yellowed Book Page Rejection...")
    res1 = send_predict(create_yellowed_book_page(), "yellowed_book.jpg")
    print(f"   Task: {res1['task_type']} | Condition: {res1['primary_condition']}")
    print(f"   Warning: {res1['uncertainty_warning']}")
    assert res1["primary_condition"] == "non_plant_or_unrecognized"
    assert res1["task_type"] == "unrecognized"
    print("   [PASS] Yellowed book page successfully rejected!")

    # 2. White Textbook Page Rejection
    print("\n2. Testing White Textbook Page Rejection...")
    res2 = send_predict(create_white_textbook_page(), "textbook.jpg")
    print(f"   Task: {res2['task_type']} | Condition: {res2['primary_condition']}")
    assert res2["primary_condition"] == "non_plant_or_unrecognized"
    print("   [PASS] White textbook page successfully rejected!")

    # 3. IoT Kit Box Rejection
    print("\n3. Testing IoT Starter Kit Box Rejection...")
    res3 = send_predict(create_iot_box_image(), "iot_box.jpg")
    print(f"   Task: {res3['task_type']} | Condition: {res3['primary_condition']}")
    print(f"   Warning: {res3['uncertainty_warning']}")
    assert res3["primary_condition"] == "non_plant_or_unrecognized"
    print("   [PASS] IoT kit packaging box successfully rejected!")

    # 4. Roadmap Diagram Rejection
    print("\n4. Testing Roadmap Diagram Rejection...")
    res4 = send_predict(create_synthetic_roadmap_image(), "roadmap.jpg")
    print(f"   Task: {res4['task_type']} | Condition: {res4['primary_condition']}")
    assert res4["primary_condition"] == "non_plant_or_unrecognized"
    print("   [PASS] Roadmap diagram successfully rejected!")

    # 5. Authentic Basil Foliage Test
    print("\n5. Testing Authentic Basil Test Sample...")
    with open("static/samples/basil_sample.jpg", "rb") as f:
        basil_bytes = f.read()
    res5 = send_predict(basil_bytes, "basil.jpg", plant_species="basil")
    print(f"   Task: {res5['task_type']} | Condition: {res5['primary_condition']} | Conf: {res5['confidence']*100:.1f}%")
    assert res5["plant_species"] == "basil"
    assert res5["primary_condition"] != "non_plant_or_unrecognized"
    print("   [PASS] Authentic Basil foliage recognized & diagnosed!")

    # 6. Authentic Aloe Vera Foliage Test
    print("\n6. Testing Authentic Aloe Vera Test Sample...")
    with open("static/samples/aloe_vera_sample.jpg", "rb") as f:
        aloe_bytes = f.read()
    res6 = send_predict(aloe_bytes, "aloe.jpg", plant_species="aloe_vera")
    print(f"   Task: {res6['task_type']} | Condition: {res6['primary_condition']} | Conf: {res6['confidence']*100:.1f}%")
    assert res6["plant_species"] == "aloe_vera"
    assert res6["primary_condition"] != "non_plant_or_unrecognized"
    print("   [PASS] Authentic Aloe Vera recognized & diagnosed!")

    # 7. Authentic Mint Foliage Test
    print("\n7. Testing Authentic Mint Test Sample...")
    with open("static/samples/mint_sample.jpg", "rb") as f:
        mint_bytes = f.read()
    res7 = send_predict(mint_bytes, "mint.jpg", plant_species="mint")
    print(f"   Task: {res7['task_type']} | Condition: {res7['primary_condition']} | Conf: {res7['confidence']*100:.1f}%")
    assert res7["plant_species"] == "mint"
    assert res7["primary_condition"] != "non_plant_or_unrecognized"
    print("   [PASS] Authentic Mint recognized & diagnosed!")

    print("\n" + "=" * 75)
    print("  ALL 7/7 NON-PLANT REJECTION & BOTANICAL DIAGNOSIS TESTS PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    main()
