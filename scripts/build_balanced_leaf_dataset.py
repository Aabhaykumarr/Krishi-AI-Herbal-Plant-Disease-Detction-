"""
Create balanced 15-class leaf crop training dataset.
Ensures equal representation across all 15 classes without class imbalance.
"""
import shutil
from pathlib import Path
import random

random.seed(42)

src_base = Path("data/final/leaf_crops")
dest_base = Path("data/final/leaf_crops_balanced")

max_train_per_class = 280
max_val_per_class = 60

for split in ["train", "val", "test"]:
    split_src = src_base / split
    split_dest = dest_base / split

    for cls_dir in split_src.iterdir():
        if not cls_dir.is_dir():
            continue

        target_dir = split_dest / cls_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        imgs = list(cls_dir.glob("*.jpg"))
        random.shuffle(imgs)

        if split == "train":
            selected = imgs[:max_train_per_class]
        elif split == "val":
            selected = imgs[:max_val_per_class]
        else:
            selected = imgs  # Keep full test set for evaluation

        for img in selected:
            dest_img = target_dir / img.name
            if not dest_img.exists():
                shutil.copy2(img, dest_img)

print("Balanced Dataset Built:")
for split in ["train", "val", "test"]:
    cnt = len(list((dest_base / split).rglob("*.jpg")))
    print(f"  {split}: {cnt} images")
