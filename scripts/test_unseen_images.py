"""
Download real unseen botanical images from open internet repositories (Wikimedia Commons, open botanical repositories)
and run full diagnostic inference against the live KrishiAI server at http://localhost:8000/predict.
"""
import json
import urllib.request
from pathlib import Path
import requests

SAMPLES_DIR = Path("tests/unseen_internet_samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# List of high-quality, authentic open-licensed botanical leaf images from Wikimedia Commons / botanical repositories
UNSEEN_IMAGE_URLS = {
    "aloe_vera_field_italy": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/76/%22_15_-_ITALY_-_Aloe_Vera_piantate_a_terra_-_Aloe_Vera_plant_in_Italy.jpg",
        "expected_species": "aloe_vera",
        "description": "Aloe Vera Plant in Ground Italy (Wikimedia Commons)",
    },
    "aloe_vera_finca": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b8/%2B_La_Finca_De_Aloe_Vera._01.jpg",
        "expected_species": "aloe_vera",
        "description": "La Finca De Aloe Vera Succulent Leaves (Wikimedia Commons)",
    },
    "basil_natural_pot": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/71/2009-365-303_No_More_Basil_%284058715479%29.jpg",
        "expected_species": "basil",
        "description": "Real Basil Plant in Garden Pot (Wikimedia Commons)",
    },
    "basil_foliage_sicilia": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/1e/205_gnuckx_Basil-Taormina-Sicilia-Italy-castielli_CC0_HQ_%284276828527%29.jpg",
        "expected_species": "basil",
        "description": "Basil Foliage Taormina Sicilia (Wikimedia Commons)",
    },
    "mint_spearmint_field": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/87/20140727Mentha_spicata4.jpg",
        "expected_species": "mint",
        "description": "Fresh Spearmint (Mentha spicata) Leaves (Wikimedia Commons)",
    },
    "mint_spearmint_foliage": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/74/20140727Mentha_spicata6.jpg",
        "expected_species": "mint",
        "description": "Spearmint Foliage Close-Up (Wikimedia Commons)",
    },
}

def run_unseen_tests():
    headers = {"User-Agent": "KrishiAIResearchBot/1.0 (contact: research@krishiai.org)"}

    print("=" * 80)
    print("KRISHIAI UNSEEN REAL-WORLD INTERNET IMAGE INFERENCE & EVALUATION")
    print("Server Target: http://localhost:8000/predict")
    print("=" * 80)

    results = []

    for name, meta in UNSEEN_IMAGE_URLS.items():
        print(f"\n---> [Downloading unseen internet sample]: {name}")
        print(f"     Source: {meta['url']}")
        print(f"     Description: {meta['description']}")

        local_path = SAMPLES_DIR / f"{name}.jpg"

        # Download image
        try:
            req = urllib.request.Request(meta["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response, open(local_path, "wb") as out_file:
                out_file.write(response.read())
            print(f"     Saved local image ({local_path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"     [FAILED TO DOWNLOAD]: {e}")
            continue

        # Send to KrishiAI live server
        print(f"     Sending to POST http://localhost:8000/predict...")
        try:
            with open(local_path, "rb") as f:
                files = {"file": (local_path.name, f, "image/jpeg")}
                data = {"plant_species": meta["expected_species"], "confidence_threshold": 0.20}
                resp = requests.post("http://localhost:8000/predict", files=files, data=data, timeout=30)

            if resp.status_code == 200:
                pred = resp.json()
                print(f"     [HTTP 200 OK]")
                print(f"     Task:              {pred.get('task_type')}")
                print(f"     Species:           {pred.get('plant_species')}")
                print(f"     Primary Condition: {pred.get('primary_condition')}")
                print(f"     Confidence:        {pred.get('confidence') * 100:.2f}%")
                print(f"     Uncertainty Warn:  {pred.get('uncertainty_warning')}")
                print(f"     Detections Count:  {len(pred.get('detections', []))}")
                for idx, det in enumerate(pred.get('detections', [])[:5]):
                    print(f"       Box #{idx+1}: {det['class_name']} ({det['confidence']*100:.1f}%) | BBox: {det['bbox']}")
                print(f"     Latency Breakdown:")
                print(f"       - Preprocess:    {pred['latency_ms']['preprocessing']:.2f} ms")
                print(f"       - Inference:     {pred['latency_ms']['inference']:.2f} ms")
                print(f"       - Postprocess:   {pred['latency_ms']['postprocessing']:.2f} ms")
                print(f"       - Total E2E:     {pred['latency_ms']['total']:.2f} ms")

                results.append({
                    "name": name,
                    "status": "PASS",
                    "species": pred.get("plant_species"),
                    "condition": pred.get("primary_condition"),
                    "confidence": pred.get("confidence"),
                    "latency_total_ms": pred["latency_ms"]["total"],
                    "detections_count": len(pred.get("detections", [])),
                })
            else:
                print(f"     [SERVER ERROR {resp.status_code}]: {resp.text}")
                results.append({"name": name, "status": f"HTTP {resp.status_code}"})
        except Exception as e:
            print(f"     [PREDICTION FAILED]: {e}")
            results.append({"name": name, "status": "FAIL", "error": str(e)})

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF UNSEEN INTERNET IMAGE EVALUATION")
    print("=" * 80)
    print(f"{'Sample Name':<28} | {'Species':<12} | {'Condition':<25} | {'Conf':<8} | {'Latency':<10}")
    print("-" * 90)
    for r in results:
        if r.get("status") == "PASS":
            conf_str = f"{r['confidence']*100:.1f}%"
            lat_str = f"{r['latency_total_ms']:.1f}ms"
            print(f"{r['name']:<28} | {r['species']:<12} | {r['condition']:<25} | {conf_str:<8} | {lat_str:<10}")
        else:
            print(f"{r['name']:<28} | {r.get('status')}")
    print("=" * 80)

if __name__ == "__main__":
    run_unseen_tests()
