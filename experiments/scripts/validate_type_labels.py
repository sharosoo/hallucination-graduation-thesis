#!/usr/bin/env python3
"""Validate thesis-valid feature-table schema, archived TypeLabel metadata, and leakage guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import read_feature_rows
from experiments.application.labeling import validate_type_labels, write_validation_report


REQUIRED_TOP_LEVEL_FIELDS = (
    "dataset",
    "split_id",
    "prompt_id",
    "pair_id",
    "candidate_id",
    "sample_id",
    "candidate_text",
    "candidate_label",
    "is_correct",
    "is_hallucination",
    "label",
    "archived_type_label",
    "label_source",
    "features",
    "analysis_features",
    "energy_availability",
    "feature_provenance",
    "feature_alignment",
    "thesis_target",
    "correctness_label_source",
)
REQUIRED_FEATURE_FIELDS = (
    "semantic_entropy_nli_likelihood",
    "semantic_entropy_cluster_count",
    "semantic_entropy_discrete_cluster_entropy",
    "semantic_energy_cluster_uncertainty",
    "semantic_energy_sample_energy",
    "semantic_entropy",
    "cluster_count",
    "semantic_energy_boltzmann",
    "semantic_energy_boltzmann_diagnostic",
    "mean_negative_log_probability",
    "logit_variance",
    "confidence_margin",
    "entity_frequency",
    "entity_frequency_axis",
    "entity_frequency_mean",
    "entity_frequency_min",
    "entity_pair_cooccurrence",
    "entity_pair_cooccurrence_axis",
    "low_frequency_entity_flag",
    "zero_cooccurrence_flag",
    "coverage_score",
    "corpus_risk_only",
    "corpus_axis_bin",
    "corpus_axis_bin_5",
)


def _validate_schema(artifact: Path) -> list[str]:
    problems: list[str] = []
    rows, _storage = read_feature_rows(artifact)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be an object")
            continue
        for field_name in REQUIRED_TOP_LEVEL_FIELDS:
            if field_name not in row:
                problems.append(f"row {index}: missing top-level field {field_name}")
        for field_name in (
            "dataset",
            "split_id",
            "prompt_id",
            "pair_id",
            "candidate_id",
            "sample_id",
            "candidate_text",
            "candidate_label",
            "label",
            "archived_type_label",
            "label_source",
        ):
            value = row.get(field_name)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"row {index}: {field_name} must be a non-empty string")
        if str(row.get("sample_id", "")).strip() != str(row.get("candidate_id", "")).strip():
            problems.append(f"row {index}: sample_id must mirror candidate_id for stable joins")
        if not isinstance(row.get("is_correct"), bool):
            problems.append(f"row {index}: is_correct must be boolean")
        if not isinstance(row.get("is_hallucination"), bool):
            problems.append(f"row {index}: is_hallucination must be boolean")
        features = row.get("features")
        if not isinstance(features, dict):
            problems.append(f"row {index}: features must be an object")
            continue
        for field_name in REQUIRED_FEATURE_FIELDS:
            if field_name not in features:
                problems.append(f"row {index}: features missing required field {field_name}")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Feature table artifact path (.parquet or .jsonl)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema_problems = _validate_schema(Path(args.artifact))
    report = validate_type_labels(
        feature_artifact_path=Path(args.artifact),
        dataset_config_path=ROOT / "experiments" / "configs" / "datasets.yaml",
    )
    if schema_problems:
        problems = report.get("problems")
        if isinstance(problems, list):
            report["problems"] = [*schema_problems, *problems]
        else:
            report["problems"] = schema_problems
        report["status"] = "error"
    report_path = write_validation_report(Path(args.artifact), report)
    print(json.dumps({"report_path": str(report_path), **report}, indent=2, ensure_ascii=False))
    if report["problems"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
