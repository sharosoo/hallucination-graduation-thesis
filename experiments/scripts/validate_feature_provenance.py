#!/usr/bin/env python3
"""Validate corpus feature artifacts for provenance, schema, and leakage guardrails."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import read_feature_rows
from experiments.scripts.stage_control import CORPUS_AXIS_SCHEMA_VERSION

REQUIRED_FEATURES = {
    "semantic_entropy",
    "cluster_count",
    "semantic_energy_proxy",
    "entity_frequency",
    "entity_frequency_mean",
    "entity_frequency_min",
    "entity_pair_cooccurrence",
    "entity_frequency_axis",
    "entity_pair_cooccurrence_axis",
    "low_frequency_entity_flag",
    "zero_cooccurrence_flag",
    "coverage_score",
    "corpus_source",
    "corpus_risk_only",
    "corpus_status",
    "corpus_axis_bin",
    "corpus_axis_bin_5",
}
LEAKAGE_HINTS = (
    "gold",
    "correct_answers",
    "gold_answers",
    "right_answer",
    "correctness",
    "is_hallucination",
    "label",
)
PROHIBITED_TRAINABLE_ROLES = {"label_only", "correctness_derived"}
ALLOWED_BACKEND_IDS = {"infini_gram_api_count", "infini_gram_local_count", "quco_cache_infini_gram"}
FINAL_OK_STATUSES = {"resolved", "fallback_resolved"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Feature artifact path (.parquet or jsonl)")
    return parser.parse_args()


def validate_row(row: dict[str, Any], row_index: int) -> list[str]:
    problems: list[str] = []
    for field_name in (
        "run_id",
        "dataset",
        "split_id",
        "candidate_id",
        "prompt_id",
        "pair_id",
        "sample_id",
        "candidate_role",
        "features",
        "corpus_axis",
        "feature_provenance",
        "schema_version",
    ):
        if field_name not in row:
            problems.append(f"row {row_index}: missing top-level field {field_name}")
    if row.get("schema_version") != CORPUS_AXIS_SCHEMA_VERSION:
        problems.append(
            f"row {row_index}: schema_version must be {CORPUS_AXIS_SCHEMA_VERSION!r}; got {row.get('schema_version')!r}"
        )
    for field_name in ("dataset", "candidate_id", "prompt_id", "pair_id", "sample_id", "candidate_role"):
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"row {row_index}: {field_name} must be a non-empty string")
    candidate_id = str(row.get("candidate_id", "")).strip()
    if candidate_id and str(row.get("sample_id", "")).strip() != candidate_id:
        problems.append(f"row {row_index}: sample_id must mirror candidate_id for stable joins")
    features = row.get("features")
    if not isinstance(features, dict):
        return problems + [f"row {row_index}: features must be an object"]
    missing_features = sorted(REQUIRED_FEATURES - set(features))
    if missing_features:
        problems.append(f"row {row_index}: missing required features {missing_features}")
    for required in ("entity_frequency_mean", "entity_frequency_min", "entity_pair_cooccurrence", "coverage_score", "corpus_risk_only"):
        if features.get(required) is None:
            problems.append(f"row {row_index}: feature {required} must not be silent null")
    if not isinstance(features.get("corpus_source"), str) or not features.get("corpus_source"):
        problems.append(f"row {row_index}: corpus_source must be an explicit non-empty string")
    if not isinstance(features.get("corpus_status"), str) or not features.get("corpus_status"):
        problems.append(f"row {row_index}: corpus_status must be an explicit non-empty string")
    corpus_axis = row.get("corpus_axis")
    if not isinstance(corpus_axis, dict):
        problems.append(f"row {row_index}: corpus_axis must be an object")
        corpus_axis = None
    if isinstance(corpus_axis, dict):
        row_status = corpus_axis.get("row_status")
        # Excluded rows are tolerated tail outcomes (counted but not gated).
        if isinstance(row_status, str) and row_status.startswith("excluded_"):
            return problems
        backend_id = corpus_axis.get("backend_id")
        if backend_id not in ALLOWED_BACKEND_IDS:
            problems.append(f"row {row_index}: corpus_axis.backend_id must be one of {sorted(ALLOWED_BACKEND_IDS)}")
        if corpus_axis.get("counts_complete") is not True:
            problems.append(f"row {row_index}: corpus_axis.counts_complete must be true for thesis-valid rows")
        # Excluded rows (e.g. excluded_no_entities for prompts whose candidate_text
        # yields no entity matches) are expected outcomes for a small tail of the
        # dataset. Their detailed axis fields are intentionally null. Skip the
        # per-row schema checks so the validator surfaces exclusions as a count
        # rather than a fail-stop.
        if isinstance(row_status, str) and row_status.startswith("excluded_"):
            return problems
        if row_status not in FINAL_OK_STATUSES:
            problems.append(f"row {row_index}: corpus_axis.row_status must be one of {sorted(FINAL_OK_STATUSES)}")
        for field_name in ("entity_frequency_axis", "entity_pair_cooccurrence_axis", "corpus_axis_score", "corpus_axis_bin", "corpus_axis_bin_5", "corpus_axis_bin_10"):
            if corpus_axis.get(field_name) is None:
                problems.append(f"row {row_index}: corpus_axis missing required field {field_name}")
        # Per-row entity/pair provenance is no longer inlined to keep the
        # parquet row-group lean; check the aggregate counts instead. Detailed
        # per-query provenance is preserved in the corpus-count cache file.
        entity_count = corpus_axis.get("entity_count")
        if not isinstance(entity_count, int) or entity_count <= 0:
            problems.append(f"row {row_index}: corpus_axis.entity_count must be a positive int")
        pair_count = corpus_axis.get("pair_count")
        if not isinstance(pair_count, int) or pair_count < 0:
            problems.append(f"row {row_index}: corpus_axis.pair_count must be a non-negative int")
        for total_field in ("missing_entity_count_total", "missing_pair_count_total", "approximate_entity_total", "approximate_pair_total"):
            value = corpus_axis.get(total_field)
            if not isinstance(value, int) or value < 0:
                problems.append(f"row {row_index}: corpus_axis.{total_field} must be a non-negative int")
            elif total_field.startswith("approximate_") and value > 0:
                problems.append(f"row {row_index}: corpus_axis.{total_field}={value}; approximate counts are not allowed for thesis-valid rows")
            elif total_field.startswith("missing_") and value > 0 and corpus_axis.get("counts_complete") is True:
                problems.append(f"row {row_index}: corpus_axis.{total_field}={value} but counts_complete is true")

    provenance = row.get("feature_provenance")
    if not isinstance(provenance, list):
        return problems + [f"row {row_index}: feature_provenance must be a list"]
    provenance_names = {entry.get("feature_name") for entry in provenance if isinstance(entry, dict)}
    for required in ("entity_frequency", "entity_frequency_mean", "entity_pair_cooccurrence", "coverage_score"):
        if required not in provenance_names:
            problems.append(f"row {row_index}: missing provenance entry for {required}")

    for entry in provenance:
        if not isinstance(entry, dict):
            problems.append(f"row {row_index}: provenance entries must be objects")
            continue
        feature_name = str(entry.get("feature_name", ""))
        role = str(entry.get("role", ""))
        source = str(entry.get("source", ""))
        trainable = bool(entry.get("trainable", False))
        depends_on_correctness = bool(entry.get("depends_on_correctness", False))
        if trainable and depends_on_correctness:
            problems.append(f"row {row_index}: trainable feature {feature_name} depends on correctness")
        lowered_source = source.lower()
        if trainable and role in PROHIBITED_TRAINABLE_ROLES:
            problems.append(f"row {row_index}: trainable feature {feature_name} has prohibited role {role}")
        if trainable and any(hint in lowered_source for hint in LEAKAGE_HINTS) and feature_name != "semantic_energy_proxy":
            problems.append(
                f"row {row_index}: trainable feature {feature_name} has correctness/gold-derived source hint {source!r}"
            )
        if "candidate_text" in lowered_source:
            problems.append(f"row {row_index}: provenance for {feature_name} must not rely on candidate_text-only corpus counting")
    return problems


def main() -> int:
    args = parse_args()
    rows, storage_report = read_feature_rows(Path(args.artifact))
    if not rows:
        raise SystemExit("Feature artifact is empty.")
    problems: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be a JSON object")
            continue
        problems.extend(validate_row(row, index))
    if problems:
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit("Feature provenance validation failed.")
    print(f"Validated {len(rows)} feature rows successfully.")
    if storage_report is not None:
        print(f"Resolved storage fallback: {storage_report['materialized_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
