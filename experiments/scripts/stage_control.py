"""Shared resumable/progress helpers for experiment stage CLIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4


SEMANTIC_ENTROPY_SCHEMA_VERSION = "semantic_entropy_nli_likelihood_v1"
SEMANTIC_ENERGY_SCHEMA_VERSION = "semantic_energy_cluster_v1"
CORPUS_AXIS_SCHEMA_VERSION = "corpus_axis_counts_v1"
FEATURE_TABLE_SCHEMA_VERSION = "corpus_axis_feature_table_v1"
GENERATION_FREE_SAMPLE_SCHEMA_VERSION = "generation_free_sample_rows_v1"
GENERATION_CANDIDATE_SCORE_SCHEMA_VERSION = "generation_teacher_forced_candidate_scores_v1"
DATASET_PREPARATION_SCHEMA_VERSION = "paired_dataset_materialization_v1"


@dataclass(frozen=True)
class StageProgress:
    phase: str
    completed: int
    total: int
    percent: float
    message: str
    updated_at: str
    output_path: str


def progress_snapshot(*, phase: str, completed: int, total: int, message: str, output_path: Path) -> StageProgress:
    percent = round(completed / total * 100, 4) if total else 100.0
    return StageProgress(
        phase=phase,
        completed=completed,
        total=total,
        percent=percent,
        message=message,
        updated_at=datetime.now(timezone.utc).isoformat(),
        output_path=str(output_path),
    )


def write_progress(path: Path | None, snapshot: StageProgress) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temp_path.write_text(json.dumps(asdict(snapshot), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def artifact_materialized(path: Path) -> bool:
    return path.exists() or path.with_suffix(path.suffix + ".storage.json").exists() or path.with_suffix(path.suffix + ".jsonl").exists()


def remove_materialized_outputs(path: Path) -> None:
    for materialized_path in (path, path.with_suffix(path.suffix + ".jsonl"), path.with_suffix(path.suffix + ".storage.json")):
        if materialized_path.exists():
            materialized_path.unlink()


def add_schema_version(rows: Iterable[Mapping[str, Any]], schema_version: str) -> list[dict[str, Any]]:
    versioned_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["schema_version"] = schema_version
        versioned_rows.append(payload)
    return versioned_rows


def schema_version_from_storage(storage_report: Mapping[str, Any] | None) -> str | None:
    if storage_report is None:
        return None
    schema_version = storage_report.get("schema_version")
    return schema_version if isinstance(schema_version, str) and schema_version.strip() else None


def validate_rows_schema_version(
    rows: Iterable[Mapping[str, Any]],
    expected_schema_version: str,
    *,
    storage_report: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    storage_schema = schema_version_from_storage(storage_report)
    if storage_schema is not None and storage_schema != expected_schema_version:
        return False, f"storage schema_version={storage_schema!r}; expected {expected_schema_version!r}"
    missing_indexes: list[int] = []
    wrong_versions: list[str] = []
    for index, row in enumerate(rows):
        schema_version = row.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            missing_indexes.append(index)
        elif schema_version != expected_schema_version:
            wrong_versions.append(f"row {index}: {schema_version!r}")
        if len(missing_indexes) + len(wrong_versions) >= 5:
            break
    if missing_indexes:
        return False, f"missing schema_version on row(s) {missing_indexes}; expected {expected_schema_version!r}"
    if wrong_versions:
        return False, f"schema_version mismatch: {', '.join(wrong_versions)}; expected {expected_schema_version!r}"
    return True, f"schema_version {expected_schema_version!r} validated"
