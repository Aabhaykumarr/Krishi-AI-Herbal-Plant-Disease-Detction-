"""
Comprehensive Real-World Unseen Internet Image Testing for KrishiAI.
Downloads unseen botanical leaf images from Wikimedia Commons and evaluates the live server at http://localhost:8000.
"""
import io
import json
import time
from pathlib import Path
from PIL import Image
import requests

SAMPLES_DIR = Path("tests/unseen_internet_samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

UNSEEN_BOTANICAL_SOURCES = [
    {
        "id": "aloe_vera_field_plant_1",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/76/%22_15_-_ITALY_-_Aloe_Vera_piantate_a_terra_-_Aloe_Vera_plant_in_Italy.jpg",
        "species": "aloe_vera",
        "title": "Aloe Vera Plant in Ground (Italy)",
        "source_license": "Wikimedia Commons (CC BY-SA 4.0)",
    },
    {
        "id": "aloe_vera_finca_leaves_2",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b8/%2B_La_Finca_De_Aloe_Vera._01.jpg",
        "species": "aloe_vera",
        "title": "La Finca De Aloe Vera Succulent Leaves",
        "source_license": "Wikimedia Commons (CC BY-SA 4.0)",
    },
    {
        "id": "basil_natural_plant_1",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/71/2009-365-303_No_More_Basil_%284058715479%29.jpg",
        "species": "basil",
        "title": "Real Basil Plant in Garden Pot",
        "source_license": "Wikimedia Commons (CC BY 2.0)",
    },
    {
        "id": "basil_close_foliage_2",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/1e/205_gnuckx_Basil-Taormina-Sicilia-Italy-castielli_CC0_HQ_%284276828527%29.jpg",
        "species": "basil",
        "title": "Basil Foliage Taormina Sicilia",
        "source_license": "Wikimedia Commons (CC0)",
    },
    {
        "id": "mint_spearmint_field_1",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/87/20140727Mentha_spicata4.jpg",
        "species": "mint",
        "title": "Fresh Spearmint (Mentha spicata) Leaves",
        "source_license": "Wikimedia Commons (CC BY-SA 3.0)",
    },
    {
        "id": "mint_spearmint_foliage_2",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/74/20140727Mentha_spicata6.jpg",
        "species": "mint",
        "title": "Spearmint Foliage Close-Up",
        "source_license": "Wikimedia Commons (CC BY-SA 3.0)",
    },
]

def main():
    print("=" * 80)
    print("      KRISHIAI REAL-WORLD UNSEEN INTERNET INFERENCE & VALIDATION")
    print("=" * 80)

    # 1. Verify Server is Alive
    try:
        health_resp = requests.get("http://localhost:8000/health", timeout=5)
        if health_resp.status_code != 200:
            print(f"[ERROR] Server health probe failed with status {health_resp.status_code}")
            return
        health = health_resp.json()
        print(f"[*] Server Status: {health.get('status')} | Service: {health.get('service')} (v{health.get('version')})")
        print(f"[*] Models Loaded: Detection={health['models']['detection']} | Classification={health['models']['classification']} | Device={health['models']['device']}")
    except Exception as e:
        print(f"[CRITICAL] Cannot connect to http://localhost:8000: {e}")
        return

    # 2. Process each unseen internet image
    headers = {"User-Agent": "KrishiAIResearchBot/1.0 (botanical-research@krishiai.org)"}
    reports = []

    for item in UNSEEN_BOTANICAL_SOURCES:
        print(f"\n" + "-" * 75)
        print(f"[*] Testing Unseen Sample: {item['title']}")
        print(f"    Species Hint: {item['species']}")
        print(f"    Source: {item['url']}")
        print(f"    License: {item['source_license']}")

        local_img_path = SAMPLES_DIR / f"{item['id']}.jpg"

        # Download if not already cached
        if not local_img_path.exists():
            print(f"    Downloading from internet...")
            try:
                r = requests.get(item["url"], headers=headers, timeout=20)
                if r.status_code == 200:
                    with open(local_img_path, "wb") as f:
                        f.write(r.content)
                    print(f"    [OK] Downloaded and cached ({len(r.content)/1024:.1f} KB)")
                else:
                    print(f"    [FAIL] HTTP {r.status_code} while downloading.")
                    continue
            except Exception as e:
                print(f"    [FAIL] Download error: {e}")
                continue
        else:
            print(f"    [OK] Found cached local file ({local_img_path.stat().st_size/1024:.1f} KB)")

        # Verify PIL can open and resize/convert if very large
        try:
            with Image.open(local_img_path) as img:
                orig_w, orig_h = img.size
                print(f"    Original Resolution: {orig_w}x{orig_h}")
                # If image is excessively large (>2000px), create a normalized preview for web upload
                if max(orig_w, orig_h) > 1600:
                    img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    upload_buf = io.BytesIO()
                    img.save(upload_buf, format="JPEG", quality=90)
                    upload_bytes = upload_buf.getvalue()
                else:
                    upload_bytes = local_img_path.read_bytes()
        except Exception as e:
            print(f"    [FAIL] Image validation error: {e}")
            continue

        # Execute Live Server Inference
        try:
            files = {"file": (f"{item['id']}.jpg", upload_bytes, "image/jpeg")}
            data = {"plant_species": item["species"], "confidence_threshold": 0.15}

            t_req_start = time.perf_counter()
            resp = requests.post("http://localhost:8000/predict", files=files, data=data, timeout=30)
            t_req_end = time.perf_counter()
            network_total_ms = (t_req_end - t_req_start) * 1000.0

            if resp.status_code == 200:
                result = resp.json()
                print(f"\n    [+] DIAGNOSTIC PREDICTION RESULTS:")
                print(f"        Task Type:          {result.get('task_type')}")
                print(f"        Plant Species:      {result.get('plant_species')}")
                print(f"        Primary Diagnosis:  {result.get('primary_condition')}")
                print(f"        Confidence:         {result.get('confidence')*100:.2f}%")
                print(f"        Uncertainty Alert:  {result.get('uncertainty_warning')}")
                print(f"        Detections Found:   {len(result.get('detections', []))}")
                for d_idx, det in enumerate(result.get('detections', [])[:3]):
                    print(f"          - Box #{d_idx+1}: {det['class_name']} ({det['confidence']*100:.1f}%) | Coordinates: {det['bbox']}")

                print(f"        Latency Profile:")
                print(f"          - Preprocessing:  {result['latency_ms']['preprocessing']:.2f} ms")
                print(f"          - Model Core:     {result['latency_ms']['inference']:.2f} ms")
                print(f"          - Postprocessing: {result['latency_ms']['postprocessing']:.2f} ms")
                print(f"          - Total Service:  {result['latency_ms']['total']:.2f} ms (E2E HTTP: {network_total_ms:.1f} ms)")

                reports.append({
                    "id": item["id"],
                    "title": item["title"],
                    "species": item["species"],
                    "predicted_species": result.get("plant_species"),
                    "diagnosis": result.get("primary_condition"),
                    "confidence": result.get("confidence"),
                    "uncertainty": result.get("uncertainty_warning"),
                    "detections": len(result.get("detections", [])),
                    "latency_ms": result["latency_ms"]["total"],
                })
            else:
                print(f"    [FAIL] Server returned HTTP {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"    [FAIL] Request failed: {e}")

    # Summary Report
    print("\n" + "=" * 85)
    print("                   KRISHIAI REAL-WORLD INFERENCE AUDIT SUMMARY")
    print("=" * 85)
    print(f"{'Sample / Image Title':<32} | {'Crop':<10} | {'Primary Diagnosis':<22} | {'Conf':<7} | {'Latency':<9}")
    print("-" * 85)
    for rep in reports:
        conf_str = f"{rep['confidence']*100:.1f}%"
        lat_str = f"{rep['latency_ms']:.1f}ms"
        print(f"{rep['title'][:32]:<32} | {rep['predicted_species']:<10} | {rep['diagnosis'][:22]:<22} | {conf_str:<7} | {lat_str:<9}")
    print("=" * 85)

if __name__ == "__main__":
    main()
