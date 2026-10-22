"""Tests for dataset_saver module."""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from krishiai.config.settings import Settings
from krishiai.data.dataset_saver import (
    SUPPORTED_SPECIES,
    DatasetSaveError,
    ValidationError,
    check_duplicate,
    compute_content_hash,
    sanitize_filename,
    save_to_dataset,
    validate_image_file,
    validate_species,
)


@pytest.fixture
def temp_incoming_dir(tmp_path):
    """Create a temporary incoming data directory."""
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    return incoming


@pytest.fixture
def mock_settings(temp_incoming_dir):
    """Create mock settings with temporary directory."""
    settings = Settings.load()
    settings.incoming_data_root = temp_incoming_dir
    settings.max_upload_size_mb = 10
    return settings


@pytest.fixture
def sample_image_content():
    """Create a small valid JPEG image."""
    img = Image.new("RGB", (100, 100), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def large_image_content():
    """Create a large image (>10MB)."""
    # Create 11MB image
    img = Image.new("RGB", (3000, 3000), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=100)
    content = buffer.getvalue()
    # Ensure it's actually > 10MB
    while len(content) < 11 * 1024 * 1024:
        content += b"padding" * 1000
    return content


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_sanitize_basic_filename(self):
        assert sanitize_filename("test.jpg") == "test.jpg"

    def test_sanitize_path_traversal_unix(self):
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        # Path.name extracts only the basename
        assert result == "passwd"

    def test_sanitize_path_traversal_windows(self):
        result = sanitize_filename("..\\..\\windows\\system32\\config")
        assert ".." not in result
        assert "\\" not in result
        # Path.name extracts only the basename
        assert result == "config"

    def test_sanitize_removes_special_chars(self):
        result = sanitize_filename("test<>:|?*.jpg")
        # 7 special chars replaced with underscores, plus .jpg
        assert result == "test______.jpg"

    def test_sanitize_removes_leading_dots(self):
        result = sanitize_filename(".hidden.jpg")
        assert not result.startswith(".")
        assert result == "hidden.jpg"

    def test_sanitize_empty_becomes_default(self):
        result = sanitize_filename("")
        assert result == "unnamed_image"

    def test_sanitize_long_filename_truncated(self):
        long_name = "a" * 300 + ".jpg"
        result = sanitize_filename(long_name)
        assert len(result) <= 200


class TestComputeContentHash:
    """Tests for content hash computation."""

    def test_compute_hash_consistent(self, sample_image_content):
        hash1 = compute_content_hash(sample_image_content)
        hash2 = compute_content_hash(sample_image_content)
        assert hash1 == hash2

    def test_compute_hash_different_content(self, sample_image_content):
        hash1 = compute_content_hash(sample_image_content)
        hash2 = compute_content_hash(sample_image_content + b"extra")
        assert hash1 != hash2

    def test_compute_hash_returns_hex(self, sample_image_content):
        result = compute_content_hash(sample_image_content)
        assert len(result) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in result)


class TestValidateSpecies:
    """Tests for species validation."""

    def test_validate_valid_species(self):
        for species in SUPPORTED_SPECIES:
            assert validate_species(species) == species

    def test_validate_species_case_insensitive(self):
        assert validate_species("BASIL") == "basil"
        assert validate_species("Aloe_Vera") == "aloe_vera"
        assert validate_species("MiNt") == "mint"

    def test_validate_species_strips_whitespace(self):
        assert validate_species("  basil  ") == "basil"

    def test_validate_invalid_species_raises(self):
        with pytest.raises(ValidationError, match="Unsupported species"):
            validate_species("tomato")

    def test_validate_empty_species_raises(self):
        with pytest.raises(ValidationError, match="Species parameter is required"):
            validate_species("")

    def test_validate_none_species_raises(self):
        with pytest.raises(ValidationError, match="Species parameter is required"):
            validate_species(None)


class TestValidateImageFile:
    """Tests for image file validation."""

    def test_validate_valid_jpeg(self, sample_image_content):
        ext, info = validate_image_file(sample_image_content, "test.jpg", 10)
        assert ext == ".jpg"
        assert info["width"] == 100
        assert info["height"] == 100
        assert info["format"] == "JPEG"

    def test_validate_valid_png(self):
        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        content = buffer.getvalue()

        ext, info = validate_image_file(content, "test.png", 10)
        assert ext == ".png"
        assert info["format"] == "PNG"

    def test_validate_oversized_file_raises(self, large_image_content):
        with pytest.raises(ValidationError, match="exceeds maximum allowed size"):
            validate_image_file(large_image_content, "large.jpg", 10)

    def test_validate_invalid_extension_raises(self, sample_image_content):
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_image_file(sample_image_content, "test.exe", 10)

    def test_validate_invalid_extension_sh_raises(self, sample_image_content):
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_image_file(sample_image_content, "malicious.sh", 10)

    def test_validate_invalid_extension_txt_raises(self, sample_image_content):
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_image_file(sample_image_content, "data.txt", 10)

    def test_validate_corrupted_image_raises(self):
        invalid_content = b"This is not an image file"
        with pytest.raises(ValidationError, match="Invalid or corrupted image"):
            validate_image_file(invalid_content, "fake.jpg", 10)


class TestCheckDuplicate:
    """Tests for duplicate detection."""

    def test_check_duplicate_no_existing_files(self, temp_incoming_dir, sample_image_content):
        species_dir = temp_incoming_dir / "basil"
        assert check_duplicate(sample_image_content, species_dir) is False

    def test_check_duplicate_detects_existing(
        self, temp_incoming_dir, sample_image_content
    ):
        species_dir = temp_incoming_dir / "basil" / "2026-10-21"
        species_dir.mkdir(parents=True)

        # Create existing file with metadata
        content_hash = compute_content_hash(sample_image_content)
        meta_file = species_dir / "existing.jpg.meta.json"
        meta_file.write_text(
            json.dumps({"content_hash": content_hash}), encoding="utf-8"
        )

        assert check_duplicate(sample_image_content, temp_incoming_dir / "basil") is True

    def test_check_duplicate_different_content(
        self, temp_incoming_dir, sample_image_content
    ):
        species_dir = temp_incoming_dir / "basil" / "2026-10-21"
        species_dir.mkdir(parents=True)

        # Create existing file with different hash
        meta_file = species_dir / "existing.jpg.meta.json"
        meta_file.write_text(
            json.dumps({"content_hash": "different_hash"}), encoding="utf-8"
        )

        assert check_duplicate(sample_image_content, temp_incoming_dir / "basil") is False


class TestSaveToDataset:
    """Tests for the main save_to_dataset function."""

    def test_save_valid_image_basil(self, mock_settings, sample_image_content):
        result = save_to_dataset(
            content=sample_image_content,
            filename="basil_leaf.jpg",
            species="basil",
            settings=mock_settings,
        )

        assert result["success"] is True
        assert "basil" in result["saved_path"]
        assert result["metadata"]["species"] == "basil"
        assert result["metadata"]["image_info"]["width"] == 100

        # Verify files exist
        full_path = mock_settings.incoming_data_root / result["saved_path"]
        assert full_path.exists()
        assert full_path.with_suffix(full_path.suffix + ".meta.json").exists()

    def test_save_valid_image_aloe_vera(self, mock_settings, sample_image_content):
        result = save_to_dataset(
            content=sample_image_content,
            filename="aloe.jpg",
            species="aloe_vera",
            settings=mock_settings,
        )

        assert result["success"] is True
        assert "aloe_vera" in result["saved_path"]

    def test_save_valid_image_mint(self, mock_settings, sample_image_content):
        result = save_to_dataset(
            content=sample_image_content,
            filename="mint.jpg",
            species="mint",
            settings=mock_settings,
        )

        assert result["success"] is True
        assert "mint" in result["saved_path"]

    def test_save_creates_directory_structure(self, mock_settings, sample_image_content):
        result = save_to_dataset(
            content=sample_image_content,
            filename="test.jpg",
            species="basil",
            settings=mock_settings,
        )

        # Check directory structure: incoming/basil/YYYY-MM-DD/
        full_path = mock_settings.incoming_data_root / result["saved_path"]
        assert full_path.parent.parent.name == "basil"
        # Check date format YYYY-MM-DD
        assert len(full_path.parent.name) == 10
        assert full_path.parent.name.count("-") == 2

    def test_save_generates_unique_filename(self, mock_settings, sample_image_content):
        result = save_to_dataset(
            content=sample_image_content,
            filename="test.jpg",
            species="basil",
            settings=mock_settings,
        )

        saved_filename = Path(result["saved_path"]).name
        # Format: {species}_{timestamp}_{hash}.jpg
        assert saved_filename.startswith("basil_")
        assert saved_filename.endswith(".jpg")
        parts = saved_filename.replace(".jpg", "").split("_")
        assert len(parts) >= 3

    def test_save_includes_metadata(self, mock_settings, sample_image_content):
        result = save_to_dataset(
            content=sample_image_content,
            filename="test.jpg",
            species="basil",
            settings=mock_settings,
        )

        metadata = result["metadata"]
        assert metadata["species"] == "basil"
        assert "saved_at" in metadata
        assert "content_hash" in metadata
        assert "image_info" in metadata
        assert metadata["original_filename"] == "test.jpg"

    def test_save_invalid_species_raises(self, mock_settings, sample_image_content):
        with pytest.raises(ValidationError, match="Unsupported species"):
            save_to_dataset(
                content=sample_image_content,
                filename="test.jpg",
                species="invalid_species",
                settings=mock_settings,
            )

    def test_save_invalid_file_type_raises(self, mock_settings):
        with pytest.raises(ValidationError, match="Invalid file extension"):
            save_to_dataset(
                content=b"fake content",
                filename="malicious.exe",
                species="basil",
                settings=mock_settings,
            )

    def test_save_oversized_file_raises(self, mock_settings, large_image_content):
        with pytest.raises(ValidationError, match="exceeds maximum allowed size"):
            save_to_dataset(
                content=large_image_content,
                filename="large.jpg",
                species="basil",
                settings=mock_settings,
            )

    def test_save_duplicate_raises(self, mock_settings, sample_image_content):
        # Save once
        save_to_dataset(
            content=sample_image_content,
            filename="test.jpg",
            species="basil",
            settings=mock_settings,
        )

        # Try to save same content again
        with pytest.raises(ValidationError, match="already been uploaded"):
            save_to_dataset(
                content=sample_image_content,
                filename="test2.jpg",  # Different filename, same content
                species="basil",
                settings=mock_settings,
            )

    def test_save_path_traversal_attempt_sanitized(
        self, mock_settings, sample_image_content
    ):
        result = save_to_dataset(
            content=sample_image_content,
            filename="../../../etc/passwd.jpg",
            species="basil",
            settings=mock_settings,
        )

        # Should save safely without path traversal
        assert result["success"] is True
        full_path = mock_settings.incoming_data_root / result["saved_path"]
        assert full_path.exists()
        # Ensure it's within the incoming directory
        assert str(full_path).startswith(str(mock_settings.incoming_data_root))

    def test_save_metadata_file_created(self, mock_settings, sample_image_content):
        result = save_to_dataset(
            content=sample_image_content,
            filename="test.jpg",
            species="basil",
            settings=mock_settings,
        )

        # Check metadata file exists and is valid JSON
        full_path = mock_settings.incoming_data_root / result["saved_path"]
        meta_path = full_path.with_suffix(full_path.suffix + ".meta.json")
        assert meta_path.exists()

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["species"] == "basil"
        assert "content_hash" in metadata
