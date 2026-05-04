#!/usr/bin/env python3
"""Validate semantic energy artifacts, rejecting proxy-only outputs by default."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import read_feature_rows

REQUIRED_TOP_LEVEL_FIELDS = (
    "run_id",
    "dataset",
    "split_id",
    "sample_id",
    "label",
    "features",
    "energy_provenance",
    "feature_provenance",
)
REQUIRED_FEATURE_FIELDS = (
    "semantic_entropy",
    "cluster_count",
    "semantic_energy_boltzmann",
    "semantic_energy_proxy",
    "energy_kind",
    "logit_variance",
    "confidence_margin",
)
ALLOWED_ENERGY_KINDS = {"true_boltzmann", "proxy_selected_logit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Energy feature artifact path (.parquet or .jsonl)")
    parser.add_argument(
        "--allow-diagnostic-proxy",
        action="store_true",
        help="Validate diagnostic-only proxy artifacts structurally, while keeping them explicitly not_for_thesis_claims/not_for_validation.",
    )
    return parser.parse_args()


def validate_row(row: dict[str, Any], row_index: int, *, allow_diagnostic_proxy: bool) -> list[str]:
    problems: list[str] = []
    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in row:
            problems.append(f"row {row_index}: missing top-level field {field_name}")

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

    energy_kind = features.get("energy_kind")
    if energy_kind not in ALLOWED_ENERGY_KINDS:
        problems.append(f"row {row_index}: energy_kind must be one of {sorted(ALLOWED_ENERGY_KINDS)}")
        return problems

    provenance_names = {entry.get("feature_name") for entry in feature_provenance if isinstance(entry, dict)}
    for required_name in (
        "semantic_entropy",
        "cluster_count",
        "energy_kind",
        "semantic_energy_boltzmann",
        "semantic_energy_proxy",
        "logit_variance",
        "confidence_margin",
    ):
        if required_name not in provenance_names:
            problems.append(f"row {row_index}: missing provenance entry for {required_name}")

    has_full_logits = bool(energy_provenance.get("source_artifact_has_full_logits", False))
    rerun_note = str(energy_provenance.get("rerun_instructions", ""))
    provenance_note = str(energy_provenance.get("note", ""))
    diagnostic_only = bool(row.get("diagnostic_only") or features.get("diagnostic_only") or energy_provenance.get("diagnostic_only"))
    not_for_thesis_claims = bool(
        row.get("not_for_thesis_claims") or features.get("not_for_thesis_claims") or energy_provenance.get("not_for_thesis_claims")
    )
    not_for_validation = bool(
        row.get("not_for_validation") or features.get("not_for_validation") or energy_provenance.get("not_for_validation")
    )

    if energy_kind == "true_boltzmann":
        if features.get("semantic_energy_boltzmann") is None:
            problems.append(f"row {row_index}: true_boltzmann rows must populate semantic_energy_boltzmann")
        if features.get("semantic_energy_proxy") is not None:
            problems.append(f"row {row_index}: true_boltzmann rows must not also populate semantic_energy_proxy")
        if not has_full_logits:
            problems.append(f"row {row_index}: true_boltzmann rows require source_artifact_has_full_logits=true")
        if energy_provenance.get("requires_rerun_for_true_boltzmann"):
            problems.append(f"row {row_index}: true_boltzmann rows must not carry rerun-required provenance")
    else:
        if features.get("semantic_energy_proxy") is None:
            problems.append(f"row {row_index}: proxy_selected_logit rows must populate semantic_energy_proxy")
        if features.get("semantic_energy_boltzmann") is not None:
            problems.append(f"row {row_index}: proxy_selected_logit rows must leave semantic_energy_boltzmann null")
        if has_full_logits:
            problems.append(f"row {row_index}: proxy_selected_logit rows are mislabeled because full logits are available")
        if not energy_provenance.get("requires_rerun_for_true_boltzmann"):
            problems.append(f"row {row_index}: proxy_selected_logit rows must explicitly mark rerun-required provenance")
        if "true boltzmann" in provenance_note.lower() and "does not claim" not in provenance_note.lower():
            problems.append(f"row {row_index}: proxy provenance note must not claim true Boltzmann support")
        if not rerun_note:
            problems.append(f"row {row_index}: proxy_selected_logit rows must include rerun instructions")
        if not diagnostic_only:
            problems.append(f"row {row_index}: proxy_selected_logit rows must be marked diagnostic_only")
        if not not_for_thesis_claims:
            problems.append(f"row {row_index}: proxy_selected_logit rows must be marked not_for_thesis_claims")
        if not not_for_validation:
            problems.append(f"row {row_index}: proxy_selected_logit rows must be marked not_for_validation")
        if not allow_diagnostic_proxy:
            problems.append(
                f"row {row_index}: proxy_selected_logit diagnostic artifacts are not experiment-ready; rerun with full logits or validate with --allow-diagnostic-proxy only for diagnostics"
            )

    for nullable_name in ("logit_variance", "confidence_margin"):
        if features.get(nullable_name) is not None and not has_full_logits:
            problems.append(f"row {row_index}: {nullable_name} must stay null when full logits are unavailable")

    return problems


def load_rerun_required_report(artifact: Path) -> dict[str, Any] | None:
    storage_report_path = artifact.with_suffix(artifact.suffix + ".storage.json")
    if not storage_report_path.exists():
        return None
    payload = json.loads(storage_report_path.read_text(encoding="utf-8"))
    if payload.get("storage_kind") != "rerun_required_report":
        return None
    materialized_path = Path(str(payload["materialized_path"]))
    report_payload = json.loads(materialized_path.read_text(encoding="utf-8"))
    return {"storage": payload, "report": report_payload}


def main() -> int:
    args = parse_args()
    rerun_report = load_rerun_required_report(Path(args.artifact))
    if rerun_report is not None:
        report = rerun_report["report"]
        raise SystemExit(
            "Energy feature validation failed: full_logits_required / rerun_required. "
            + str(report.get("message", ""))
            + " "
            + str(report.get("rerun_instructions", ""))
        )
    rows, storage_report = read_feature_rows(Path(args.artifact))
    if not rows:
        raise SystemExit("Energy feature artifact is empty.")
    problems: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be a JSON object")
            continue
        problems.extend(validate_row(row, index, allow_diagnostic_proxy=args.allow_diagnostic_proxy))
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
