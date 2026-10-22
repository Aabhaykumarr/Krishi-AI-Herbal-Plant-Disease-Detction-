"""Safe dataset image saver with validation, sanitization, and duplicate detection."""
from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image

from krishiai.config.settings import Settings
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical species names matching the model taxonomy
SUPPORTED_SPECIES = {"basil", "aloe_vera", "mint"}

# Allowed image MIME types and extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class DatasetSaveError(Exception):
    """Base exception for dataset saving errors."""
    pass


class ValidationError(DatasetSaveError):
    """Raised when validation fails."""
    pass


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename with only safe characters
    """
    # Get basename to prevent any path components
    name = Path(filename).name

    # Remove or replace unsafe characters, keep only alphanumeric, dash, underscore, dot
    name = re.sub(r'[^\w\-.]', '_', name)

    # Remove leading dots (hidden files) and path traversal attempts
    name = name.lstrip('.')

    # Ensure we have something left
    if not name or name == '_':
        name = 'unnamed_image'

    # Limit length
    if len(name) > 200:
        name = name[:200]

    return name


def compute_content_hash(content: bytes) -> str:
    """Compute SHA256 hash of file content for duplicate detection.

    Args:
        content: File content bytes

    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(content).hexdigest()


def validate_species(species: str) -> str:
    """Validate and normalize species name.

    Args:
        species: Species name to validate

    Returns:
        Normalized species name

    Raises:
        ValidationError: If species is not supported
    """
    if not species:
        raise ValidationError("Species parameter is required")

    normalized = species.lower().strip()

    if normalized not in SUPPORTED_SPECIES:
        raise ValidationError(
            f"Unsupported species '{species}'. Must be one of: {', '.join(sorted(SUPPORTED_SPECIES))}"
        )

    return normalized


def validate_image_file(content: bytes, filename: str, max_size_mb: int) -> Tuple[str, Dict]:
    """Validate image file type, size, and format.

    Args:
        content: File content bytes
        filename: Original filename
        max_size_mb: Maximum allowed size in megabytes

    Returns:
        Tuple of (extension, image_info_dict)

    Raises:
        ValidationError: If validation fails
    """
    # Check file size
    size_bytes = len(content)
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        raise ValidationError(
            f"File size {size_mb:.2f}MB exceeds maximum allowed size of {max_size_mb}MB"
        )

    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Invalid file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Verify it's actually a valid image by trying to open it
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()

        # Re-open for actual info (verify() invalidates the image)
        img = Image.open(io.BytesIO(content))
        width, height = img.size
        format_name = img.format or "UNKNOWN"
        mode = img.mode

        image_info = {
            "width": width,
            "height": height,
            "format": format_name,
            "mode": mode,
            "size_bytes": size_bytes,
        }

        return ext, image_info

    except Exception as e:
        raise ValidationError(f"Invalid or corrupted image file: {e}")


def check_duplicate(content: bytes, species_dir: Path) -> bool:
    """Check if image already exists in the species directory using content hash.

    Args:
        content: Image file content
        species_dir: Directory for the species

    Returns:
        True if duplicate found, False otherwise
    """
    content_hash = compute_content_hash(content)

    # Search for existing files with this hash in metadata
    if not species_dir.exists():
        return False

    for meta_file in species_dir.rglob("*.meta.json"):
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            if metadata.get("content_hash") == content_hash:
                logger.info(f"Duplicate detected: hash {content_hash[:8]} already exists")
                return True
        except Exception:
            continue

    return False


def save_to_dataset(
    content: bytes,
    filename: str,
    species: str,
    settings: Settings | None = None,
) -> Dict[str, any]:
    """Save an image to the incoming dataset directory with validation and metadata.

    Args:
        content: Image file content bytes
        filename: Original filename
        species: Plant species (basil, aloe_vera, or mint)
        settings: Settings instance (loaded if None)

    Returns:
        Dictionary with save result information:
            - success: bool
            - saved_path: str (relative path)
            - metadata: dict

    Raises:
        ValidationError: If validation fails
        DatasetSaveError: If save operation fails
    """
    if settings is None:
        settings = Settings.load()

    # Validate species
    species = validate_species(species)

    # Validate image file
    ext, image_info = validate_image_file(content, filename, settings.max_upload_size_mb)

    # Check for duplicates
    species_dir = settings.incoming_data_root / species
    if check_duplicate(content, species_dir):
        raise ValidationError("This image has already been uploaded to the dataset")

    # Create directory structure: incoming_data/{species}/{YYYY-MM-DD}/
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    save_dir = species_dir / date_str

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise DatasetSaveError(f"Failed to create directory {save_dir}: {e}")

    # Generate unique filename: {species}_{timestamp_ms}_{hash[:8]}.jpg
    timestamp_ms = int(now.timestamp() * 1000)
    content_hash = compute_content_hash(content)
    safe_filename = f"{species}_{timestamp_ms}_{content_hash[:8]}{ext}"

    save_path = save_dir / safe_filename
    meta_path = save_dir / f"{safe_filename}.meta.json"

    # Ensure no overwrite (should be impossible with timestamp + hash, but safety check)
    if save_path.exists():
        raise DatasetSaveError(f"File already exists: {save_path}")

    # Prepare metadata
    metadata = {
        "original_filename": sanitize_filename(filename),
        "species": species,
        "saved_at": now.isoformat(),
        "content_hash": content_hash,
        "image_info": image_info,
        "saved_path": str(save_path.relative_to(settings.incoming_data_root)),
    }

    # Atomic write: write to temp file then rename
    temp_img_path = save_path.with_suffix(f"{ext}.tmp")
    temp_meta_path = meta_path.with_suffix(".json.tmp")

    try:
        # Write image
        temp_img_path.write_bytes(content)

        # Write metadata
        temp_meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Atomic rename
        temp_img_path.rename(save_path)
        temp_meta_path.rename(meta_path)

        logger.info(f"Saved image to dataset: {save_path.relative_to(settings.incoming_data_root)}")

        return {
            "success": True,
            "saved_path": str(save_path.relative_to(settings.incoming_data_root)),
            "metadata": metadata,
        }

    except Exception as e:
        # Cleanup temp files on error
        for temp_path in [temp_img_path, temp_meta_path, save_path, meta_path]:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

        raise DatasetSaveError(f"Failed to save image: {e}")
