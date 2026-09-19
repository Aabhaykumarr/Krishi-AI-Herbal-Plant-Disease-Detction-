"""Image and annotation validators (Sections 18-19).

Invalid items are reported, never silently dropped or auto-deleted — Section
18 requires quarantining, not deleting, and a script that both detects and
destructively acts on findings hides a side effect a caller wouldn't expect
(Section 81). Deletion/quarantine is a separate, explicit, logged decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageIssue:
    path: str
    reason: str


def validate_image(path: Path, min_dim: int = 16) -> Optional[ImageIssue]:
    """Returns an ImageIssue if the image is invalid, else None."""
    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        return ImageIssue(str(path), f"unsupported extension '{path.suffix}'")
    if not _PIL_AVAILABLE:
        return ImageIssue(str(path), "Pillow not installed; cannot validate")
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:  # re-open: verify() invalidates the handle
            w, h = img.size
            if w < min_dim or h < min_dim:
                return ImageIssue(str(path), f"dimensions too small ({w}x{h})")
    except Exception as exc:  # noqa: BLE001 - report and continue, never crash a batch scan
        return ImageIssue(str(path), f"failed to open/decode: {exc}")
    return None


@dataclass
class YoloLabelIssue:
    path: str
    line_number: int
    reason: str


def validate_yolo_label_file(
    path: Path, num_classes: Optional[int] = None
) -> list[YoloLabelIssue]:
    """Validates a single YOLO-format .txt label file.

    Supports both:
      - Standard Bounding Box: class_id x_center y_center width height (5 values)
      - Polygon Segmentation: class_id x1 y1 x2 y2 ... xn yn (odd number >= 7 values)

    Checks: class_id in range (if num_classes given), coordinates normalized to [0,1],
    positive dimensions for boxes, and coordinates within image bounds (Section 19).
    Collects every issue instead of stopping at the first one.
    """
    issues: list[YoloLabelIssue] = []
    try:
        lines = path.read_text().splitlines()
    except Exception as exc:  # noqa: BLE001
        return [YoloLabelIssue(str(path), 0, f"could not read file: {exc}")]

    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            issues.append(
                YoloLabelIssue(str(path), i, f"expected at least 5 values, got {len(parts)}")
            )
            continue
        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(YoloLabelIssue(str(path), i, "non-numeric value"))
            continue

        if class_id < 0 or (num_classes is not None and class_id >= num_classes):
            issues.append(YoloLabelIssue(str(path), i, f"class_id {class_id} out of range"))

        # Case 1: Standard Bounding Box (5 or 6 values, where 6th might be confidence)
        if len(parts) in (5, 6):
            x, y, w, h = coords[:4]
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                issues.append(YoloLabelIssue(str(path), i, f"center ({x}, {y}) not normalized"))
            if w <= 0 or h <= 0:
                issues.append(YoloLabelIssue(str(path), i, f"non-positive box size ({w}, {h})"))
            if (
                (x - w / 2) < -1e-6
                or (x + w / 2) > 1 + 1e-6
                or (y - h / 2) < -1e-6
                or (y + h / 2) > 1 + 1e-6
            ):
                issues.append(YoloLabelIssue(str(path), i, "box extends outside image bounds"))
        # Case 2: Polygon Segmentation (class_id + pairs of (x, y))
        elif len(coords) % 2 == 0 and len(coords) >= 6:
            for pt_idx in range(0, len(coords), 2):
                px, py = coords[pt_idx], coords[pt_idx + 1]
                if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
                    issues.append(
                        YoloLabelIssue(str(path), i, f"polygon point ({px}, {py}) not normalized")
                    )
                    break
        else:
            issues.append(
                YoloLabelIssue(
                    str(path), i, f"invalid annotation format with {len(parts)} values"
                )
            )

    return issues


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """Converts polygon coordinates [x1, y1, x2, y2, ...] to YOLO bbox (x_center, y_center, width, height).

    Coordinates are clamped to [0.0, 1.0].
    """
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError(f"Expected at least 6 coordinates in pairs, got {len(coords)}")
    xs = coords[0::2]
    ys = coords[1::2]
    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))
    width = max_x - min_x
    height = max_y - min_y
    x_center = min_x + width / 2.0
    y_center = min_y + height / 2.0
    return (round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6))

