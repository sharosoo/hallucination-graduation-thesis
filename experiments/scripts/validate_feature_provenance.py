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

REQUIRED_FEATURES = {
    "semantic_entropy",
    "cluster_count",
    "semantic_energy_proxy",
    "entity_frequency",
    "entity_frequency_mean",
    "entity_frequency_min",
    "entity_pair_cooccurrence",
    "low_frequency_entity_flag",
    "zero_cooccurrence_flag",
    "coverage_score",
    "corpus_source",
    "corpus_risk_only",
    "corpus_status",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Feature artifact path (.parquet or jsonl)")
    return parser.parse_args()


def validate_row(row: dict[str, Any], row_index: int) -> list[str]:
    problems: list[str] = []
    for field_name in ("run_id", "dataset", "split_id", "sample_id", "label", "features", "feature_provenance"):
        if field_name not in row:
            problems.append(f"row {row_index}: missing top-level field {field_name}")
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

    provenance = row.get("feature_provenance")
    if not isinstance(provenance, list):
        return problems + [f"row {row_index}: feature_provenance must be a list"]
    provenance_names = {entry.get("feature_name") for entry in provenance if isinstance(entry, dict)}
    for required in ("label", "entity_frequency_mean", "entity_pair_cooccurrence", "coverage_score"):
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
