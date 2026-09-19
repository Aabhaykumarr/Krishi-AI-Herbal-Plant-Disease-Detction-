from krishiai.data.manifest import DatasetManifest, DatasetRecord


def test_upsert_and_get():
    manifest = DatasetManifest()
    manifest.upsert(DatasetRecord(dataset_id="mint_kaggle_v1", source="kaggle"))
    assert manifest.get("mint_kaggle_v1") is not None
    assert manifest.get("does_not_exist") is None


def test_upsert_replaces_existing_record_with_same_id():
    manifest = DatasetManifest()
    manifest.upsert(DatasetRecord(dataset_id="basil_v1", source="roboflow", image_count=10))
    manifest.upsert(DatasetRecord(dataset_id="basil_v1", source="roboflow", image_count=25))
    assert len(manifest.records) == 1
    assert manifest.get("basil_v1").image_count == 25


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = DatasetManifest()
    manifest.upsert(DatasetRecord(dataset_id="basil_v1", source="roboflow", image_count=120))
    manifest.save(path)

    loaded = DatasetManifest.load(path)
    record = loaded.get("basil_v1")
    assert record is not None
    assert record.image_count == 120


def test_load_missing_file_returns_empty(tmp_path):
    manifest = DatasetManifest.load(tmp_path / "does_not_exist.json")
    assert manifest.records == []
