from krishiai.data.hashing import perceptual_available, perceptual_hash, sha256_file


def test_sha256_is_stable(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello world")
    assert sha256_file(f) == sha256_file(f)


def test_sha256_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"hello")
    f2.write_bytes(b"world")
    assert sha256_file(f1) != sha256_file(f2)


def test_perceptual_hash_reports_availability_honestly(tmp_classification_dataset):
    img = next(tmp_classification_dataset.rglob("*.png"))
    result = perceptual_hash(img)
    if perceptual_available():
        assert result is not None
    else:
        assert result is None
