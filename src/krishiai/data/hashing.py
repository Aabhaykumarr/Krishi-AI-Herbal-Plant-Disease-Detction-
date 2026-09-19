"""Hashing utilities for duplicate detection (Section 23).

Exact-duplicate hashing (sha256) is stdlib-only and always available.
Near-duplicate (perceptual) hashing is optional: if `imagehash`/Pillow are
not installed, `perceptual_hash()` returns None rather than raising or
silently pretending it found nothing. Callers MUST treat None as "not
computed," never as "no duplicates" — conflating the two would misreport
real dataset risk (Section 101: no fake results).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

try:
    import imagehash
    from PIL import Image

    _PERCEPTUAL_AVAILABLE = True
except ImportError:
    _PERCEPTUAL_AVAILABLE = False


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Exact-duplicate hash. Always available (stdlib only)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path) -> Optional[str]:
    """Near-duplicate hash (average hash). Returns None if imagehash/Pillow
    are not installed — see module docstring for why callers must not treat
    that as "no duplicates found."
    """
    if not _PERCEPTUAL_AVAILABLE:
        return None
    with Image.open(path) as img:
        return str(imagehash.average_hash(img))


def perceptual_available() -> bool:
    return _PERCEPTUAL_AVAILABLE
