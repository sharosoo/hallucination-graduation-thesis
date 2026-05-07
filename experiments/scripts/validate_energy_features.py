#!/usr/bin/env python3
"""Validate sampled-response Semantic Energy artifacts plus candidate diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import read_feature_rows
from experiments.scripts.stage_control import SEMANTIC_ENERGY_SCHEMA_VERSION

REQUIRED_TOP_LEVEL_FIELDS = (
    "run_id",
    "dataset",
    "split_id",
    "candidate_id",
    "prompt_id",
    "pair_id",
    "candidate_role",
    "candidate_text",
    "label_source",
    "candidate_token_count",
    "features",
    "energy_provenance",
    "feature_provenance",
    "source_free_sample_path",
    "source_semantic_entropy_path",
    "schema_version",
)
REQUIRED_FEATURE_FIELDS = (
    "semantic_energy_cluster_uncertainty",
    "semantic_energy_sample_energy",
    "semantic_energy_cluster_ids",
    "semantic_energy_cluster_energies",
    "semantic_energy_sample_energies",
    "semantic_energy_sample_cluster_assignments",
    "semantic_energy_token_provenance",
    "semantic_energy_boltzmann",
    "semantic_energy_boltzmann_diagnostic",
    "mean_negative_log_probability",
    "logit_variance",
    "confidence_margin",
    "energy_kind",
    "paper_faithful_energy_available",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Energy feature artifact path (.parquet or .jsonl)")
    return parser.parse_args()


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_row(row: dict[str, Any], row_index: int) -> list[str]:
    problems: list[str] = []
    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in row:
            problems.append(f"row {row_index}: missing top-level field {field_name}")
    if row.get("schema_version") != SEMANTIC_ENERGY_SCHEMA_VERSION:
        problems.append(
            f"row {row_index}: schema_version must be {SEMANTIC_ENERGY_SCHEMA_VERSION!r}; got {row.get('schema_version')!r}"
        )

    for field_name in ("dataset", "candidate_id", "prompt_id", "pair_id", "candidate_role", "label_source", "candidate_text"):
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"row {row_index}: {field_name} must be a non-empty string")

    features = row.get("features")
    if not isinstance(features, dict):
        return problems + [f"row {row_index}: features must be an object"]
    energy_provenance = row.get("energy_provenance")
    if not isinstance(energy_provenance, dict):
        return problems + [f"row {row_index}: energy_provenance must be an object"]
    feature_provenance = row.get("feature_provenance")
    if not isinstance(feature_provenance, list):
        return problems + [f"row {row_index}: feature_provenance must be a list"]

    for field_name in REQUIRED_FEATURE_FIELDS:
        if field_name not in features:
            problems.append(f"row {row_index}: missing feature {field_name}")

    if features.get("energy_kind") != "sampled_response_cluster":
        problems.append(f"row {row_index}: energy_kind must be 'sampled_response_cluster'")
    if features.get("paper_faithful_energy_available") is not True:
        problems.append(f"row {row_index}: paper_faithful_energy_available must be true")
    for sampled_name in ("semantic_energy_cluster_uncertainty", "semantic_energy_sample_energy"):
        if features.get(sampled_name) is None or not _is_number(features.get(sampled_name)):
            problems.append(f"row {row_index}: {sampled_name} must be numeric")
    if features.get("semantic_energy_boltzmann") != features.get("semantic_energy_cluster_uncertainty"):
        problems.append(f"row {row_index}: semantic_energy_boltzmann must be a compatibility alias of semantic_energy_cluster_uncertainty")
    if features.get("semantic_energy_boltzmann") is None or not _is_number(features.get("semantic_energy_boltzmann")):
        problems.append(f"row {row_index}: semantic_energy_boltzmann compatibility field must be numeric")
    if features.get("semantic_energy_boltzmann_diagnostic") is None or not _is_number(features.get("semantic_energy_boltzmann_diagnostic")):
        problems.append(f"row {row_index}: semantic_energy_boltzmann_diagnostic must be numeric")
    if features.get("mean_negative_log_probability") is None or not _is_number(features.get("mean_negative_log_probability")):
        problems.append(f"row {row_index}: mean_negative_log_probability must be numeric")
    for nullable_name in ("logit_variance", "confidence_margin"):
        value = features.get(nullable_name)
        if value is not None and not _is_number(value):
            problems.append(f"row {row_index}: {nullable_name} must be numeric or null")

    for list_name in (
        "semantic_energy_cluster_ids",
        "semantic_energy_cluster_energies",
        "semantic_energy_sample_energies",
        "semantic_energy_sample_cluster_assignments",
        "semantic_energy_token_provenance",
    ):
        if not isinstance(features.get(list_name), list) or not features.get(list_name):
            problems.append(f"row {row_index}: {list_name} must be a non-empty list")
    token_provenance = features.get("semantic_energy_token_provenance")
    if isinstance(token_provenance, list):
        observed_indexes: list[int] = []
        for sample_entry_index, sample_entry in enumerate(token_provenance):
            if not isinstance(sample_entry, dict):
                problems.append(f"row {row_index}: semantic_energy_token_provenance[{sample_entry_index}] must be an object")
                continue
            sample_index = sample_entry.get("sample_index")
            if isinstance(sample_index, int) and not isinstance(sample_index, bool):
                observed_indexes.append(sample_index)
            for vector_name in ("selected_token_logits", "logsumexp", "selected_token_logprobs", "selected_token_probabilities", "generated_token_ids", "selected_token_ids", "generated_tokens"):
                if not isinstance(sample_entry.get(vector_name), list) or not sample_entry.get(vector_name):
                    problems.append(f"row {row_index}: token provenance sample {sample_entry_index} missing non-empty {vector_name}")
            lengths = [len(sample_entry.get(vector_name, [])) for vector_name in ("selected_token_logits", "logsumexp", "selected_token_logprobs", "selected_token_probabilities", "generated_token_ids", "selected_token_ids", "generated_tokens") if isinstance(sample_entry.get(vector_name), list)]
            if lengths and len(set(lengths)) != 1:
                problems.append(f"row {row_index}: token provenance sample {sample_entry_index} token-vector lengths must align")
        if sorted(observed_indexes) != list(range(10)):
            problems.append(f"row {row_index}: token provenance must cover exact sample indexes 0..9")
    if features.get("diagnostic_only") is not False:
        problems.append(f"row {row_index}: sampled-response Energy rows must not be diagnostic_only")
    if features.get("not_for_thesis_claims") is not False:
        problems.append(f"row {row_index}: sampled-response Energy rows must be thesis-claim eligible")
    if features.get("not_for_validation") is True:
        problems.append(f"row {row_index}: sampled-response Energy rows must be validation-eligible")

    if energy_provenance.get("source") != "free_sample_rows_and_semantic_entropy_clusters":
        problems.append(f"row {row_index}: source must be free_sample_rows_and_semantic_entropy_clusters")
    if energy_provenance.get("paper_faithful_energy_available") is not True:
        problems.append(f"row {row_index}: energy_provenance.paper_faithful_energy_available must be true")
    if energy_provenance.get("energy_granularity") != "prompt_level_broadcast_to_candidate_rows":
        problems.append(f"row {row_index}: energy_granularity must be prompt_level_broadcast_to_candidate_rows")
    if energy_provenance.get("sample_energy_formula") != "sample_energy = mean(-selected_token_logits)":
        problems.append(f"row {row_index}: sample_energy_formula must declare mean negative selected-token logit")
    if energy_provenance.get("candidate_scoring_mode") != "teacher_forced":
        problems.append(f"row {row_index}: candidate_scoring_mode must be teacher_forced")
    if energy_provenance.get("prompt_prefix_scoring_excluded") is not True:
        problems.append(f"row {row_index}: prompt_prefix_scoring_excluded must be true")
    if "free-sample generated answer token positions" not in str(energy_provenance.get("token_position_source")):
        problems.append(f"row {row_index}: token_position_source must mention free-sample generated answer token positions")
    for provenance_name in ("source_free_sample_path", "source_semantic_entropy_path", "semantic_clusterer", "nli_model_ref", "model_name", "tokenizer_name"):
        if not isinstance(energy_provenance.get(provenance_name), str) or not str(energy_provenance.get(provenance_name)).strip():
            problems.append(f"row {row_index}: energy_provenance.{provenance_name} must be a non-empty string")

    if row.get("candidate_id") != row.get("sample_id"):
        problems.append(f"row {row_index}: sample_id must mirror candidate_id for stable joins")

    candidate_token_count = row.get("candidate_token_count")
    token_count_from_provenance = energy_provenance.get("candidate_token_count")
    if not isinstance(candidate_token_count, int) or candidate_token_count <= 0:
        problems.append(f"row {row_index}: candidate_token_count must be a positive integer")
    if candidate_token_count != token_count_from_provenance:
        problems.append(f"row {row_index}: candidate_token_count must match energy provenance")

    provenance_names = {entry.get("feature_name") for entry in feature_provenance if isinstance(entry, dict)}
    for required_name in (
        "energy_kind",
        "semantic_energy_cluster_uncertainty",
        "semantic_energy_sample_energy",
        "semantic_energy_boltzmann",
        "semantic_energy_boltzmann_diagnostic",
        "mean_negative_log_probability",
        "logit_variance",
        "confidence_margin",
    ):
        if required_name not in provenance_names:
            problems.append(f"row {row_index}: missing provenance entry for {required_name}")

    return problems


def main() -> int:
    args = parse_args()
    rows, storage_report = read_feature_rows(Path(args.artifact))
    if not rows:
        raise SystemExit("Energy feature artifact is empty.")
    problems: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be a JSON object")
            continue
        problems.extend(validate_row(row, index))
    if problems:
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit("Energy feature validation failed.")
    print(f"Validated {len(rows)} energy feature rows successfully.")
    if storage_report is not None:
        print(f"Resolved storage fallback: {storage_report['materialized_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
