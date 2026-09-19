"""Dataset manifest schema and I/O utilities (Section 7 of the project brief).

The manifest is the single source of truth describing every dataset that has
been ingested into KrishiAI: where it came from, its license, its task type,
and its validation status. See docs/dataset.md for the full field reference.

Deliberately stdlib-only (dataclasses + json) so this module has zero
dependency risk, regardless of what's installed on a given machine.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

TaskType = Literal["classification", "detection", "segmentation", "unknown"]
ValidationStatus = Literal["pending", "in_review", "passed", "failed", "quarantined"]


@dataclass
class DatasetRecord:
    """One row of dataset provenance/status. Field set matches Section 7."""

    dataset_id: str
    source: str
    source_url: Optional[str] = None
    license: Optional[str] = None
    version: Optional[str] = None
    local_path: str = ""
    task_type: TaskType = "unknown"
    class_names: list[str] = field(default_factory=list)
    annotation_format: str = "unknown"
    image_count: int = 0
    object_count: int = 0
    acquisition_date: Optional[str] = None
    preprocessing_status: str = "not_started"
    validation_status: ValidationStatus = "pending"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetManifest:
    """Loads/saves a collection of DatasetRecord entries as JSON.

    There is exactly one canonical manifest system (this one) — do not create
    a second one for a later phase (Section 97).
    """

    def __init__(self, records: Optional[list[DatasetRecord]] = None):
        self.records: list[DatasetRecord] = records or []

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls([DatasetRecord(**r) for r in raw.get("datasets", [])])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"datasets": [r.to_dict() for r in self.records]}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def upsert(self, record: DatasetRecord) -> None:
        for i, existing in enumerate(self.records):
            if existing.dataset_id == record.dataset_id:
                self.records[i] = record
                return
        self.records.append(record)

    def get(self, dataset_id: str) -> Optional[DatasetRecord]:
        return next((r for r in self.records if r.dataset_id == dataset_id), None)
