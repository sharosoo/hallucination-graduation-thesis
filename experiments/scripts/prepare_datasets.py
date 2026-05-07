#!/usr/bin/env python3
"""Prepare paired dataset prompt groups and candidate rows for scoring."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import HuggingFaceDatasetLoader, candidate_row_to_json, prompt_group_to_json
from experiments.adapters.hf_datasets import DatasetMaterializationError
from experiments.scripts.stage_control import DATASET_PREPARATION_SCHEMA_VERSION, progress_snapshot, write_json_atomic, write_progress, write_text_atomic
from experiments.scripts.validate_datasets import load_json


ACTIVE_DATASET_CONTRACT = "single_paired_discriminative_experiment_dataset"
CANDIDATE_ROWS_PER_PROMPT = 2
ALLOWED_DATASET_SOURCES = {
    "TruthfulQA": {"hf_id": "truthful_qa", "config": "generation", "split": "validation"},
    "HaluEval-QA": {"hf_id": "pminervini/HaluEval", "config": "qa", "split": "data"},
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the dataset registry config")
    parser.add_argument("--out", required=True, help="Output directory for dataset preparation metadata")
    parser.add_argument("--progress", help="Optional progress JSON path updated atomically.")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    write_json_atomic(path, payload)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows)
    write_text_atomic(path, payload)


def emit_progress(
    progress_path: Path | None,
    *,
    phase: str,
    completed: int,
    total: int,
    message: str,
    output_path: Path,
) -> None:
    write_progress(
        progress_path,
        progress_snapshot(phase=phase, completed=completed, total=total, message=message, output_path=output_path),
    )


def selected_datasets(config: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = config.get("datasets", [])
    active_names = config.get("experiment_dataset_policy", {}).get("active_datasets")
    if isinstance(active_names, list) and active_names:
        active_name_set = {str(name) for name in active_names}
        return [dataset for dataset in datasets if str(dataset.get("name")) in active_name_set]
    return [dataset for dataset in datasets if dataset.get("role") == "core"]


def target_count_for(dataset: dict[str, Any]) -> int:
    return int(dataset["target_sample_count"])


def validate_allowed_source(dataset: dict[str, Any]) -> list[str]:
    name = str(dataset.get("name", "<unknown>"))
    expected = ALLOWED_DATASET_SOURCES.get(name)
    if expected is None:
        return [f"{name}: unsupported active paired dataset"]

    problems: list[str] = []
    for field, expected_value in expected.items():
        actual_value = dataset.get(field)
        if actual_value != expected_value:
            problems.append(f"{name}: {field} must be {expected_value!r}, got {actual_value!r}")
    if "source_id" in dataset and dataset.get("source_id") != expected["hf_id"]:
        problems.append(f"{name}: source_id must match allowed hf_id {expected['hf_id']!r}")
    return problems


def validate_paired_config(config: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    policy = config.get("experiment_dataset_policy")
    if not isinstance(policy, dict):
        return ["experiment_dataset_policy must be an object for paired dataset preparation"]
    if policy.get("dataset_contract") != ACTIVE_DATASET_CONTRACT:
        problems.append(f"experiment_dataset_policy.dataset_contract must be {ACTIVE_DATASET_CONTRACT!r}")
    if policy.get("candidate_rows_per_prompt") != CANDIDATE_ROWS_PER_PROMPT:
        problems.append(f"experiment_dataset_policy.candidate_rows_per_prompt must be {CANDIDATE_ROWS_PER_PROMPT}")

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        problems.append("datasets must be a non-empty list")
        return problems

    selected = selected_datasets(config)
    active_names = policy.get("active_datasets")
    if not isinstance(active_names, list) or not active_names:
        problems.append("experiment_dataset_policy.active_datasets must be a non-empty list")
    elif {str(name) for name in active_names} != set(ALLOWED_DATASET_SOURCES):
        problems.append(
            "experiment_dataset_policy.active_datasets must exactly match "
            f"{sorted(ALLOWED_DATASET_SOURCES)} for paired thesis preparation"
        )
    if isinstance(active_names, list) and active_names:
        selected_names = {str(dataset.get("name")) for dataset in selected}
        missing_names = sorted(str(name) for name in active_names if str(name) not in selected_names)
        if missing_names:
            problems.append(f"active datasets missing registry entries: {missing_names}")
    if not selected:
        problems.append("no datasets selected for paired preparation")

    for dataset in selected:
        name = dataset.get("name", "<unknown>")
        for field in ("split", "split_id", "target_sample_count", "seed", "label_source", "label_policy", "candidate_pair_policy"):
            if field not in dataset:
                problems.append(f"{name}: missing required field {field!r}")
        candidate_policy = dataset.get("candidate_pair_policy")
        if not isinstance(candidate_policy, dict):
            problems.append(f"{name}: candidate_pair_policy must be an object")
        elif candidate_policy.get("candidate_rows_per_prompt") != CANDIDATE_ROWS_PER_PROMPT:
            problems.append(f"{name}: candidate_pair_policy.candidate_rows_per_prompt must be {CANDIDATE_ROWS_PER_PROMPT}")
        if "hf_id" not in dataset and "source_id" not in dataset:
            problems.append(f"{name}: missing hf_id/source_id")
        problems.extend(validate_allowed_source(dataset))
        try:
            if target_count_for(dataset) <= 0:
                problems.append(f"{name}: target_sample_count must be positive")
        except (KeyError, TypeError, ValueError):
            problems.append(f"{name}: target_sample_count must be an integer")
    return problems


def summarize_label_policy(config: dict[str, Any]) -> dict[str, Any]:
    label_policy = config.get("label_policy", {})
    analysis_bins = label_policy.get("analysis_se_bins", {}) if isinstance(label_policy, dict) else {}
    return {
        "operational_labels": label_policy.get("operational_labels", []),
        "fixed_operational_thresholds": label_policy.get("fixed_operational_thresholds", {}),
        "row_level_output": label_policy.get("row_level_output", {}),
        "analysis_se_bins": {
            "scheme_name": analysis_bins.get("scheme_name"),
            "analysis_only": analysis_bins.get("analysis_only"),
            "special_attention_thresholds": analysis_bins.get("special_attention_thresholds", []),
            "bin_count": len(analysis_bins.get("bins", [])) if isinstance(analysis_bins.get("bins"), list) else 0,
        },
    }


def summarize_candidate_pair_policy(config: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    policy = config.get("experiment_dataset_policy", {})
    return {
        "dataset_contract": policy.get("dataset_contract"),
        "active_datasets": policy.get("active_datasets", [dataset.get("name") for dataset in selected]),
        "candidate_rows_per_prompt": policy.get("candidate_rows_per_prompt"),
        "candidate_policy": policy.get("candidate_policy"),
        "datasets": [
            {
                "name": dataset.get("name"),
                "split_id": dataset.get("split_id"),
                "candidate_pair_policy": dataset.get("candidate_pair_policy", {}),
                "label_source": dataset.get("label_source", {}),
            }
            for dataset in selected
        ],
    }


def skipped_rows_by_reason(dataset_reports: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in dataset_reports:
        for skipped_row in report.get("skipped_rows", []):
            reason = str(skipped_row.get("reason", "unknown"))
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def hf_report_info(dataset_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        report["dataset"]: {
            "hf_id": report.get("hf_id"),
            "hf_config": report.get("hf_config"),
            "split": report.get("split"),
            "fingerprint": report.get("fingerprint"),
            "cache_files": report.get("cache_files", []),
            "available_count": report.get("available_count"),
            "selected_source_row_count": report.get("selected_source_row_count"),
            "materialized_source_row_count": report.get("materialized_source_row_count"),
        }
        for report in dataset_reports
    }


def build_report(config_path: Path, config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    selected = selected_datasets(config)
    return {
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "dataset": "experiment_dataset",
        "downloads_performed": False,
        "selected_datasets": [dataset["name"] for dataset in selected],
        "label_policy_summary": summarize_label_policy(config),
        "candidate_pair_policy_summary": summarize_candidate_pair_policy(config, selected),
        "datasets": [
            {
                "name": dataset["name"],
                "role": dataset["role"],
                "hf_id": dataset.get("hf_id"),
                "config": dataset.get("config"),
                "split": dataset["split"],
                "split_id": dataset["split_id"],
                "target_sample_count": dataset["target_sample_count"],
                "materialization_target_count": target_count_for(dataset),
                "seed": dataset["seed"],
                "label_source_type": dataset["label_source"]["type"],
                "label_policy": dataset["label_policy"],
                "candidate_pair_policy": dataset["candidate_pair_policy"],
                "notes": dataset["notes"],
            }
            for dataset in selected
        ],
    }


def materialize(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    selected = selected_datasets(config)
    loader = HuggingFaceDatasetLoader()
    prompt_groups: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    for dataset in selected:
        groups, candidates, report = loader.materialize_paired_rows(dataset, target_count=target_count_for(dataset))
        dataset_reports.append(report)
        prompt_groups.extend(prompt_group_to_json(group) for group in groups)
        candidate_rows.extend(candidate_row_to_json(candidate) for candidate in candidates)

    prompt_groups.sort(key=lambda row: (row.get("dataset", ""), row.get("split_id", ""), row.get("prompt_id", "")))
    candidate_rows.sort(
        key=lambda row: (
            row.get("dataset", ""),
            row.get("split_id", ""),
            row.get("prompt_id", ""),
            0 if row.get("candidate_role") == "right" else 1,
            row.get("candidate_id", ""),
        )
    )

    prompt_groups_path = out_dir / "prompt_groups.jsonl"
    candidate_rows_path = out_dir / "candidate_rows.jsonl"
    manifest_path = out_dir / "dataset_manifest.json"
    write_jsonl(prompt_groups_path, prompt_groups)
    write_jsonl(candidate_rows_path, candidate_rows)
    manifest = {
        "schema_version": DATASET_PREPARATION_SCHEMA_VERSION,
        "manifest_version": 1,
        "manifest_kind": "paired_dataset_materialization",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "experiment_dataset",
        "prompt_groups_path": str(prompt_groups_path),
        "candidate_rows_path": str(candidate_rows_path),
        "prompt_group_count": len(prompt_groups),
        "candidate_row_count": len(candidate_rows),
        "selected_datasets": [dataset["name"] for dataset in selected],
        "skipped_rows_by_reason": skipped_rows_by_reason(dataset_reports),
        "hf_report_info": hf_report_info(dataset_reports),
        "dataset_reports": dataset_reports,
    }
    write_json(manifest_path, manifest)
    return {
        "prompt_groups_path": str(prompt_groups_path),
        "candidate_rows_path": str(candidate_rows_path),
        "dataset_manifest_path": str(manifest_path),
        "prompt_group_count": len(prompt_groups),
        "candidate_row_count": len(candidate_rows),
        "skipped_rows_by_reason": manifest["skipped_rows_by_reason"],
        "hf_report_info": manifest["hf_report_info"],
        "dataset_reports": dataset_reports,
    }


def print_report(report: dict[str, Any]) -> None:
    print("Dataset preparation report")
    print(f"- Dataset: {report['dataset']}")
    print(f"- Downloads performed: {report['downloads_performed']}")
    print(f"- Selected datasets: {', '.join(report['selected_datasets'])}")
    materialized = report.get("materialized", {})
    print(f"- Prompt groups: {materialized.get('prompt_group_count', 0)}")
    print(f"- Candidate rows: {materialized.get('candidate_row_count', 0)}")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.out)
    progress_path = Path(args.progress) if args.progress else None
    out_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(
        progress_path,
        phase="start",
        completed=0,
        total=3,
        message="starting dataset preparation",
        output_path=out_dir / "dataset_manifest.json",
    )

    config = load_json(config_path)
    problems = validate_paired_config(config)
    if problems:
        for problem in problems:
            print(f"- {problem}")
        emit_progress(
            progress_path,
            phase="failed_validation",
            completed=0,
            total=3,
            message="dataset registry validation failed before preparation",
            output_path=out_dir / "dataset_manifest.json",
        )
        raise SystemExit("Dataset registry validation failed before preparation.")
    emit_progress(
        progress_path,
        phase="config_validated",
        completed=1,
        total=3,
        message="dataset registry validation passed",
        output_path=out_dir / "dataset_manifest.json",
    )

    report = build_report(config_path, config, out_dir)
    try:
        materialized = materialize(config, out_dir)
    except DatasetMaterializationError as exc:
        emit_progress(
            progress_path,
            phase="failed",
            completed=1,
            total=3,
            message=f"Dataset materialization failed: {exc}",
            output_path=out_dir / "dataset_manifest.json",
        )
        raise SystemExit(f"Dataset materialization failed: {exc}") from exc
    emit_progress(
        progress_path,
        phase="materialized",
        completed=2,
        total=3,
        message="dataset prompt groups and candidate rows materialized",
        output_path=out_dir / "dataset_manifest.json",
    )
    report["downloads_performed"] = True
    report["schema_version"] = DATASET_PREPARATION_SCHEMA_VERSION
    report["materialized"] = materialized
    write_json(out_dir / "dataset_preparation_report.json", report)
    emit_progress(
        progress_path,
        phase="complete",
        completed=3,
        total=3,
        message="dataset preparation complete",
        output_path=out_dir / "dataset_manifest.json",
    )
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
