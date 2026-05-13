"""Run-level manifests for reproducible experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentManifest:
    """Stable manifest linking datasets, features, formulas, and outputs."""

    run_id: str
    method_name: str
    dataset_names: tuple[str, ...]
    split_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    formula_manifest_ref: str
    dataset_manifest_ref: str
    artifact_refs: tuple[str, ...] = ()
    corpus_snapshot_ref: str | None = None
    created_by: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class StageProgress:
    """Resumable / progress snapshot for one stage of the pipeline."""

    phase: str
    completed: int
    total: int
    percent: float
    message: str
    updated_at: str
    output_path: str
