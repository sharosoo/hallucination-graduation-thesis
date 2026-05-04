"""Application-layer labeling and feature-table export helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.adapters.corpus_features import (
    infer_label,
    load_json,
    read_analysis_bin_config,
    read_feature_rows,
    select_analysis_bin,
    write_feature_artifact,
    write_json,
)
from experiments.domain import AnalysisBin, FeatureRole, TypeLabel

EXPECTED_TYPE_LABELS = tuple(label.value for label in TypeLabel)
ENERGY_UNAVAILABLE_STATUS = "full_logits_required_rerun_required"


def assign_operational_label(existing_label: str, semantic_entropy: float) -> TypeLabel:
    """Assign the fixed operational label from correctness state and Semantic Entropy.

    Current corpus rows already carry the correctness-derived operational label from the
    upstream row source. `NORMAL` therefore means the sample was judged correct, while all
    non-normal labels mean the sample was judged incorrect and must be re-thresholded with
    the fixed four-way policy.
    """

    normalized = str(existing_label).strip()
    if normalized == TypeLabel.NORMAL.value:
        return TypeLabel.NORMAL
    return infer_label(True, semantic_entropy)


def serialize_analysis_bin(bin_spec: AnalysisBin | None) -> dict[str, Any] | None:
    if bin_spec is None:
        return None
    return {
        "scheme_name": bin_spec.scheme_name,
        "bin_id": bin_spec.bin_id,
        "lower_bound": bin_spec.lower_bound,
        "upper_bound": bin_spec.upper_bound,
        "includes_upper_bound": bin_spec.includes_upper_bound,
        "note": bin_spec.note,
    }


def build_boundary_self_check() -> dict[str, Any]:
    cases = []
    for semantic_entropy, expected in (
        (0.1, TypeLabel.LOW_DIVERSITY),
        (0.5, TypeLabel.AMBIGUOUS_INCORRECT),
        (0.5001, TypeLabel.HIGH_DIVERSITY),
    ):
        actual = infer_label(True, semantic_entropy)
        cases.append(
            {
                "semantic_entropy": semantic_entropy,
                "expected_label": expected.value,
                "actual_label": actual.value,
                "passes": actual is expected,
            }
        )
    return {
        "cases": cases,
        "all_pass": all(case["passes"] for case in cases),
    }


def _resolve_artifact_path(requested_path: Path) -> Path:
    storage_report_path = requested_path.with_suffix(requested_path.suffix + ".storage.json")
    if storage_report_path.exists():
        payload = load_json(storage_report_path)
        materialized = payload.get("materialized_path")
        if materialized:
            return Path(str(materialized))
    return requested_path


def _load_energy_storage_payload(results_dir: Path) -> dict[str, Any]:
    requested_path = results_dir / "energy_features.parquet"
    storage_report_path = requested_path.with_suffix(requested_path.suffix + ".storage.json")
    if not storage_report_path.exists():
        raise FileNotFoundError(f"Missing energy storage sidecar: {storage_report_path}")
    return load_json(storage_report_path)


def load_energy_availability(results_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    storage_payload = _load_energy_storage_payload(results_dir)
    report = storage_payload.get("report")
    if not isinstance(report, dict):
        raise ValueError("Energy storage sidecar is missing a report object.")

    availability_by_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    selected_artifacts = report.get("selected_artifacts")
    if isinstance(selected_artifacts, list):
        for artifact in selected_artifacts:
            if not isinstance(artifact, dict):
                continue
            dataset_id = str(artifact.get("dataset_id", ""))
            split_id = str(artifact.get("split_id", ""))
            if not dataset_id or not split_id:
                continue
            availability_by_dataset[(dataset_id, split_id)] = {
                "artifact_id": artifact.get("artifact_id"),
                "dataset_name": artifact.get("dataset_name"),
                "dataset_id": dataset_id,
                "split_id": split_id,
                "source_artifact_path": artifact.get("path"),
                "true_boltzmann_available": bool(artifact.get("has_full_logits", False)),
                "has_logits": bool(artifact.get("has_logits", False)),
                "has_full_logits": bool(artifact.get("has_full_logits", False)),
                "energy_status": ENERGY_UNAVAILABLE_STATUS if not artifact.get("has_full_logits", False) else "true_boltzmann_ready",
                "rerun_required": not bool(artifact.get("has_full_logits", False)),
                "full_logits_required": not bool(artifact.get("has_full_logits", False)),
                "diagnostic_proxy_accepted": False,
                "not_for_thesis_claims": not bool(artifact.get("has_full_logits", False)),
                "note": report.get("message", ""),
                "rerun_instructions": report.get("rerun_instructions"),
            }
    return availability_by_dataset, report


def _label_presence_explanations(label_counts: Counter[str]) -> dict[str, str]:
    explanations: dict[str, str] = {}
    if label_counts.get(TypeLabel.AMBIGUOUS_INCORRECT.value, 0) == 0:
        explanations[TypeLabel.AMBIGUOUS_INCORRECT.value] = (
            "No incorrect sample in the current corpus feature rows fell inside 0.1 < SE <= 0.5, "
            "so the gray-zone label is absent in this export instead of being force-filled."
        )
    for label_value in EXPECTED_TYPE_LABELS:
        if label_counts.get(label_value, 0) == 0 and label_value not in explanations:
            explanations[label_value] = "Label absent in current source rows after applying the fixed operational policy."
    return explanations


def _dataset_label_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get("dataset", "unknown"))][str(row.get("label", ""))] += 1
    return {
        dataset: {label: counter.get(label, 0) for label in EXPECTED_TYPE_LABELS}
        for dataset, counter in sorted(counts.items())
    }


def _build_energy_metadata(
    row: dict[str, Any],
    energy_entry: dict[str, Any],
    fallback_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": energy_entry.get("energy_status", ENERGY_UNAVAILABLE_STATUS),
        "true_boltzmann_available": bool(energy_entry.get("true_boltzmann_available", False)),
        "full_logits_required": bool(energy_entry.get("full_logits_required", True)),
        "rerun_required": bool(energy_entry.get("rerun_required", True)),
        "diagnostic_proxy_accepted": False,
        "not_for_thesis_claims": bool(energy_entry.get("not_for_thesis_claims", True)),
        "source_artifact_path": energy_entry.get("source_artifact_path") or row.get("source_artifact_path"),
        "source_artifact_id": energy_entry.get("artifact_id") or row.get("artifact_id"),
        "rerun_instructions": energy_entry.get("rerun_instructions") or fallback_report.get("rerun_instructions"),
        "note": energy_entry.get("note") or fallback_report.get("message", ""),
    }


def _build_feature_provenance(
    existing: list[dict[str, Any]],
    *,
    energy_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        feature_name = str(entry.get("feature_name", ""))
        if not feature_name:
            continue
        payload = deepcopy(entry)
        if feature_name == "semantic_energy_proxy":
            payload["role"] = FeatureRole.ANALYSIS_ONLY.value
            payload["trainable"] = False
            payload["note"] = (
                "Retained as diagnostic metadata from the corpus artifact only. "
                "Do not use it as thesis-valid or fusion-valid Semantic Energy evidence without full logits."
            )
        seen.add(feature_name)
        updated.append(payload)

    if "semantic_energy_boltzmann" not in seen:
        updated.append(
            {
                "feature_name": "semantic_energy_boltzmann",
                "role": FeatureRole.ANALYSIS_ONLY.value,
                "source": "unavailable_without_full_logits",
                "source_artifact_path": energy_metadata.get("source_artifact_path"),
                "depends_on_correctness": False,
                "trainable": False,
                "note": energy_metadata.get("rerun_instructions"),
            }
        )
    if "energy_status" not in seen:
        updated.append(
            {
                "feature_name": "energy_status",
                "role": FeatureRole.ANALYSIS_ONLY.value,
                "source": "energy storage sidecar status",
                "source_artifact_path": energy_metadata.get("source_artifact_path"),
                "depends_on_correctness": False,
                "trainable": False,
                "note": "Availability marker only. Prevents proxy energy from being treated as valid thesis evidence.",
            }
        )
    return updated


def build_feature_table(results_dir: Path, out_path: Path, dataset_config_path: Path) -> dict[str, Any]:
    corpus_path = results_dir / "corpus_features.parquet"
    corpus_rows, corpus_storage = read_feature_rows(corpus_path)
    if not corpus_rows:
        raise ValueError(f"Corpus feature source is empty: {_resolve_artifact_path(corpus_path)}")

    analysis_bins, raw_bin_specs = read_analysis_bin_config(dataset_config_path)
    energy_by_dataset, energy_report = load_energy_availability(results_dir)

    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    missing_energy_matches: Counter[str] = Counter()
    for source_row in corpus_rows:
        if not isinstance(source_row, dict):
            continue
        row = deepcopy(source_row)
        features = row.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"Row missing features object for sample_id={row.get('sample_id')}")
        semantic_entropy = float(features.get("semantic_entropy", 0.0) or 0.0)
        label = assign_operational_label(str(row.get("label", "")), semantic_entropy)
        analysis_bin = select_analysis_bin(semantic_entropy, analysis_bins, raw_bin_specs)

        dataset_key = (str(row.get("dataset_id", "")), str(row.get("split_id", "")))
        energy_entry = energy_by_dataset.get(dataset_key)
        if energy_entry is None:
            missing_energy_matches[f"{dataset_key[0]}::{dataset_key[1]}"] += 1
            energy_entry = {
                "artifact_id": row.get("artifact_id"),
                "source_artifact_path": row.get("source_artifact_path"),
                "energy_status": ENERGY_UNAVAILABLE_STATUS,
                "true_boltzmann_available": False,
                "full_logits_required": True,
                "rerun_required": True,
                "not_for_thesis_claims": True,
                "note": energy_report.get("message", "No dataset-level energy availability match found."),
                "rerun_instructions": energy_report.get("rerun_instructions"),
            }

        energy_metadata = _build_energy_metadata(row, energy_entry, energy_report)
        features["se_bin"] = serialize_analysis_bin(analysis_bin)
        features["semantic_energy_boltzmann"] = None
        features["energy_status"] = energy_metadata["status"]
        features["energy_available"] = energy_metadata["true_boltzmann_available"]
        features["full_logits_required"] = energy_metadata["full_logits_required"]
        features["rerun_required"] = energy_metadata["rerun_required"]
        features["energy_note"] = energy_metadata["note"]

        row["label"] = label.value
        row["energy_availability"] = energy_metadata
        existing_provenance = row.get("feature_provenance")
        provenance_list: list[dict[str, Any]] = existing_provenance if isinstance(existing_provenance, list) else []
        row["feature_provenance"] = _build_feature_provenance(provenance_list, energy_metadata=energy_metadata)
        rows.append(row)
        label_counts[label.value] += 1

    boundary_self_check = build_boundary_self_check()
    report = {
        "run_id": f"feature-table-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_out_path": str(out_path),
        "row_count": len(rows),
        "source_artifacts": {
            "corpus": {
                "requested_path": str(corpus_path),
                "resolved_path": str(_resolve_artifact_path(corpus_path)),
                "storage": corpus_storage,
            },
            "energy": {
                "requested_path": str(results_dir / 'energy_features.parquet'),
                "storage": _load_energy_storage_payload(results_dir),
            },
        },
        "label_counts": {label: label_counts.get(label, 0) for label in EXPECTED_TYPE_LABELS},
        "label_counts_by_dataset": _dataset_label_counts(rows),
        "label_presence_explanations": _label_presence_explanations(label_counts),
        "boundary_self_check": boundary_self_check,
        "energy_status_counts": dict(Counter(row["energy_availability"]["status"] for row in rows)),
        "missing_energy_matches": dict(missing_energy_matches),
        "analysis_bin_scheme": analysis_bins[0].scheme_name if analysis_bins else None,
        "analysis_bin_count": len(analysis_bins),
        "notes": [
            "Rows are sourced from corpus_features as the validated row-level artifact and relabeled with the fixed four-way operational policy.",
            "Energy availability comes from the energy_features rerun-required sidecar. Proxy scalar energy is retained only as diagnostic metadata and not treated as thesis-valid evidence.",
        ],
    }

    storage = write_feature_artifact(out_path, rows, report)
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    write_json(report_path, report)
    return {
        "rows": rows,
        "report": report,
        "storage": storage,
        "report_path": str(report_path),
    }


def validate_type_labels(feature_artifact_path: Path, dataset_config_path: Path) -> dict[str, Any]:
    rows, storage_report = read_feature_rows(feature_artifact_path)
    if not rows:
        raise ValueError("Feature table artifact is empty.")

    analysis_bins, raw_bin_specs = read_analysis_bin_config(dataset_config_path)
    valid_bin_ids = {bin_spec.bin_id for bin_spec in analysis_bins}
    problems: list[str] = []
    label_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be an object")
            continue
        label_value = str(row.get("label", ""))
        if label_value not in EXPECTED_TYPE_LABELS:
            problems.append(f"row {index}: label must be one of {list(EXPECTED_TYPE_LABELS)}, got {label_value!r}")
            continue
        features = row.get("features")
        if not isinstance(features, dict):
            problems.append(f"row {index}: features must be an object")
            continue
        semantic_entropy = float(features.get("semantic_entropy", 0.0) or 0.0)
        expected_label = assign_operational_label(label_value, semantic_entropy)
        if label_value != expected_label.value:
            problems.append(
                f"row {index}: label/SE mismatch, expected {expected_label.value} for SE={semantic_entropy}, got {label_value}"
            )
        se_bin = features.get("se_bin")
        if not isinstance(se_bin, dict):
            problems.append(f"row {index}: features.se_bin must be an object")
        else:
            bin_id = str(se_bin.get("bin_id", ""))
            recalculated_bin = select_analysis_bin(semantic_entropy, analysis_bins, raw_bin_specs)
            if bin_id not in valid_bin_ids:
                problems.append(f"row {index}: unknown SE bin id {bin_id!r}")
            elif recalculated_bin is None or bin_id != recalculated_bin.bin_id:
                problems.append(
                    f"row {index}: SE bin mismatch, expected {None if recalculated_bin is None else recalculated_bin.bin_id}, got {bin_id}"
                )
        energy_availability = row.get("energy_availability")
        if not isinstance(energy_availability, dict):
            problems.append(f"row {index}: missing energy_availability object")
        else:
            if energy_availability.get("true_boltzmann_available") is not False:
                problems.append(f"row {index}: energy availability must keep true_boltzmann_available=false for current artifacts")
            if energy_availability.get("full_logits_required") is not True:
                problems.append(f"row {index}: energy availability must keep full_logits_required=true")
        label_counts[label_value] += 1

    boundary_self_check = build_boundary_self_check()
    explanations = _label_presence_explanations(label_counts)
    missing_without_explanation = [
        label for label in EXPECTED_TYPE_LABELS if label_counts.get(label, 0) == 0 and not explanations.get(label)
    ]
    if missing_without_explanation:
        problems.append(
            "Missing labels without explanation: " + ", ".join(sorted(missing_without_explanation))
        )
    if not boundary_self_check["all_pass"]:
        problems.append("Boundary self-check failed for one or more threshold cases.")

    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(feature_artifact_path),
        "resolved_storage": storage_report,
        "row_count": len(rows),
        "label_counts": {label: label_counts.get(label, 0) for label in EXPECTED_TYPE_LABELS},
        "label_counts_by_dataset": _dataset_label_counts(rows),
        "label_presence_explanations": explanations,
        "boundary_self_check": boundary_self_check,
        "problems": problems,
        "status": "ok" if not problems else "error",
    }
    return report


def write_validation_report(feature_artifact_path: Path, report: dict[str, Any]) -> Path:
    report_path = feature_artifact_path.parent / "type_label_validation_report.json"
    write_json(report_path, report)
    return report_path
