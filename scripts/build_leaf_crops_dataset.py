"""
Build unified 15-class cropped leaf dataset for two-stage hierarchical diagnosis.
Extracts leaf bounding boxes from detection dataset and merges with Aloe Vera classification dataset.
"""
import os
import shutil
from pathlib import Path
import yaml
from PIL import Image

def build_leaf_crops():
    print("=" * 60)
    print("Building Canonical 15-Class Leaf Crop Dataset (data/final/leaf_crops/)")
    print("=" * 60)

    # 1. Load detection class names
    det_yaml_path = Path("data/final/detection/data.yaml")
    with open(det_yaml_path) as f:
        det_yaml = yaml.safe_load(f)
    det_classes = det_yaml["names"]

    output_base = Path("data/final/leaf_crops")
    splits = ["train", "val", "test"]

    for split in splits:
        for cls_name in det_classes.values():
            (output_base / split / cls_name).mkdir(parents=True, exist_ok=True)

    # 2. Extract bounding box crops for Basil and Mint
    crop_counts = {cls_name: 0 for cls_name in det_classes.values()}

    for split in splits:
        img_dir = Path(f"data/final/detection/{split}/images")
        lbl_dir = Path(f"data/final/detection/{split}/labels")

        for img_path in img_dir.glob("*.jpg"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size

                    with open(lbl_path) as f:
                        lines = [line.strip() for line in f if line.strip()]

                    for idx, line in enumerate(lines):
                        parts = line.split()
                        cls_id = int(parts[0])
                        cls_name = det_classes[cls_id]

                        if len(parts) == 5:
                            # Standard YOLO box (xc, yc, w, h)
                            xc, yc, w, h = map(float, parts[1:5])
                            xmin = max(0, int((xc - w / 2) * img_w))
                            ymin = max(0, int((yc - h / 2) * img_h))
                            xmax = min(img_w, int((xc + w / 2) * img_w))
                            ymax = min(img_h, int((yc + h / 2) * img_h))
                        elif len(parts) > 5:
                            # Polygon segmentation
                            coords = [float(p) for p in parts[1:]]
                            xs = coords[0::2]
                            ys = coords[1::2]
                            xmin = max(0, int(min(xs) * img_w))
                            ymin = max(0, int(min(ys) * img_h))
                            xmax = min(img_w, int(max(xs) * img_w))
                            ymax = min(img_h, int(max(ys) * img_h))

                        # Ensure valid crop dimension
                        if xmax - xmin < 10 or ymax - ymin < 10:
                            continue

                        # Add 5% margin for botanical context
                        pad_x = int((xmax - xmin) * 0.05)
                        pad_y = int((ymax - ymin) * 0.05)
                        crop_box = (
                            max(0, xmin - pad_x),
                            max(0, ymin - pad_y),
                            min(img_w, xmax + pad_x),
                            min(img_h, ymax + pad_y),
                        )

                        crop_img = img.crop(crop_box)
                        crop_filename = f"{img_path.stem}_crop_{idx}.jpg"
                        crop_path = output_base / split / cls_name / crop_filename
                        crop_img.save(crop_path, quality=95)
                        crop_counts[cls_name] += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print("\nExtracted Detection Leaf Crops:")
    for cls_name, cnt in crop_counts.items():
        print(f"  - {cls_name}: {cnt} crops")

    # 3. Merge Aloe Vera classification dataset
    print("\nMerging Aloe Vera Classification Dataset into leaf_crops/...")
    aloe_base = Path("data/final/classification")
    aloe_counts = {}

    for split in splits:
        split_dir = aloe_base / split
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            target_dir = output_base / split / cls_name
            target_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for img_file in cls_dir.glob("*.jpg"):
                dest = target_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                count += 1
            aloe_counts[cls_name] = aloe_counts.get(cls_name, 0) + count

    print("\nAloe Vera Merged Images:")
    for cls_name, cnt in aloe_counts.items():
        print(f"  - {cls_name}: {cnt} images")

    # Summary
    total_crops = sum(crop_counts.values()) + sum(aloe_counts.values())
    print("\n" + "=" * 60)
    print(f"Total Unified 15-Class Leaf Crop Dataset Created: {total_crops} images!")
    print(f"Dataset location: {output_base.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    build_leaf_crops()
