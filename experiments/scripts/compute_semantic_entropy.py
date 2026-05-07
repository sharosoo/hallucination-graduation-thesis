#!/usr/bin/env python3
"""Compute prompt-level Semantic Entropy rows from free-sample artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters.semantic_entropy_features import (
    DEFAULT_NLI_MODEL_NAME,
    SemanticEntropyDependencyError,
    SemanticEntropyInputError,
    write_semantic_entropy_artifact,
)
from experiments.adapters import read_feature_rows
from experiments.scripts.stage_control import (
    SEMANTIC_ENTROPY_SCHEMA_VERSION,
    artifact_materialized,
    progress_snapshot,
    remove_materialized_outputs,
    validate_rows_schema_version,
    write_progress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free-samples", required=True, help="Task 7 free_sample_rows artifact")
    parser.add_argument("--out", required=True, help="Output path (.parquet or .jsonl)")
    parser.add_argument(
        "--nli-model",
        default=DEFAULT_NLI_MODEL_NAME,
        help=f"NLI model reference for strict bidirectional entailment clustering (default: {DEFAULT_NLI_MODEL_NAME})",
    )
    parser.add_argument("--progress", help="Optional progress JSON path updated atomically.")
    parser.add_argument("--resume", action="store_true", help="Skip when existing output validates.")
    parser.add_argument("--force", action="store_true", help="Replace existing output instead of refusing overwrite.")
    return parser.parse_args()


def _validate_existing(path: Path, free_samples_path: Path) -> tuple[bool, str]:
    try:
        rows, storage = read_feature_rows(path)
    except Exception as exc:
        return False, f"existing artifact could not be read: {exc}"
    if not rows:
        return False, "existing artifact is empty"
    schema_ok, schema_message = validate_rows_schema_version(rows, SEMANTIC_ENTROPY_SCHEMA_VERSION, storage_report=storage)
    if not schema_ok:
        return False, schema_message
    seen_prompt_ids: set[str] = set()
    problems: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be a JSON object")
            continue
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            problems.append(f"row {index}: missing prompt_id")
        elif prompt_id in seen_prompt_ids:
            problems.append(f"row {index}: duplicate prompt_id {prompt_id!r}")
        else:
            seen_prompt_ids.add(prompt_id)
        for field_name in (
            "semantic_entropy_nli_likelihood",
            "semantic_entropy_discrete_cluster_entropy",
            "semantic_entropy",
        ):
            value = row.get(field_name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                problems.append(f"row {index}: {field_name} must be numeric")
        for field_name in ("semantic_entropy_cluster_count", "cluster_count"):
            value = row.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"row {index}: {field_name} must be an integer")
        if row.get("semantic_entropy_nli_likelihood") != row.get("semantic_entropy"):
            problems.append(f"row {index}: semantic_entropy alias must equal semantic_entropy_nli_likelihood")
        if row.get("semantic_entropy_cluster_count") != row.get("cluster_count"):
            problems.append(f"row {index}: cluster_count alias must equal semantic_entropy_cluster_count")
        if not isinstance(row.get("nli_model_ref"), str) or not str(row.get("nli_model_ref", "")).strip():
            problems.append(f"row {index}: missing nli_model_ref")
        if not isinstance(row.get("sample_log_likelihoods"), list) or len(row.get("sample_log_likelihoods", [])) != 10:
            problems.append(f"row {index}: sample_log_likelihoods must contain 10 records")
        if not isinstance(row.get("cluster_log_likelihoods"), list) or not row.get("cluster_log_likelihoods"):
            problems.append(f"row {index}: cluster_log_likelihoods must be a non-empty list")
        if not isinstance(row.get("pairwise_entailment_decisions"), list):
            problems.append(f"row {index}: pairwise_entailment_decisions must be a list")
        features = row.get("features")
        if not isinstance(features, dict):
            problems.append(f"row {index}: features must be an object")
        else:
            for field_name in (
                "semantic_entropy_nli_likelihood",
                "semantic_entropy_discrete_cluster_entropy",
                "semantic_entropy",
            ):
                if not isinstance(features.get(field_name), int | float) or isinstance(features.get(field_name), bool):
                    problems.append(f"row {index}: features.{field_name} must be numeric")
            for field_name in ("semantic_entropy_cluster_count", "cluster_count"):
                if not isinstance(features.get(field_name), int) or isinstance(features.get(field_name), bool):
                    problems.append(f"row {index}: features.{field_name} must be an integer")
            if features.get("semantic_entropy_nli_likelihood") != features.get("semantic_entropy"):
                problems.append(f"row {index}: features.semantic_entropy alias mismatch")
            if features.get("semantic_entropy_cluster_count") != features.get("cluster_count"):
                problems.append(f"row {index}: features.cluster_count alias mismatch")
        source_path = row.get("source_free_sample_path")
        if not isinstance(source_path, str) or not source_path.strip():
            problems.append(f"row {index}: missing source_free_sample_path")
        else:
            try:
                if Path(source_path).resolve() != free_samples_path.resolve():
                    problems.append(f"row {index}: source_free_sample_path does not match --free-samples")
            except OSError:
                if Path(source_path) != free_samples_path:
                    problems.append(f"row {index}: source_free_sample_path does not match --free-samples")
        if len(problems) >= 20:
            break
    if problems:
        return False, "; ".join(problems[:5])
    return True, f"existing Semantic Entropy artifact validates with {len(rows)} prompt row(s)"


def _emit(progress_path: Path | None, *, phase: str, completed: int, total: int, message: str, out_path: Path) -> None:
    write_progress(progress_path, progress_snapshot(phase=phase, completed=completed, total=total, message=message, output_path=out_path))


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    free_samples_path = Path(args.free_samples)
    progress_path = Path(args.progress) if args.progress else None
    if args.resume and args.force:
        print("--resume and --force are mutually exclusive.", file=sys.stderr)
        return 2
    if artifact_materialized(out_path):
        is_valid, validation_message = _validate_existing(out_path, free_samples_path)
        if args.resume and is_valid:
            _emit(progress_path, phase="complete", completed=1, total=1, message=f"resume skip: {validation_message}", out_path=out_path)
            print(json.dumps({"status": "skipped_existing_valid_output", "message": validation_message}, indent=2, ensure_ascii=False))
            return 0
        if not args.force:
            print(f"Refusing to overwrite existing Semantic Entropy output without --force. Validation status: {validation_message}", file=sys.stderr)
            return 2
        remove_materialized_outputs(out_path)

    _emit(progress_path, phase="start", completed=0, total=0, message="starting Semantic Entropy computation", out_path=out_path)
    try:
        payload = write_semantic_entropy_artifact(free_samples_path, out_path, nli_model_name=str(args.nli_model))
    except (SemanticEntropyInputError, SemanticEntropyDependencyError) as exc:
        _emit(progress_path, phase="failed", completed=0, total=0, message=str(exc), out_path=out_path)
        print(str(exc), file=sys.stderr)
        return 2
    report = payload.get("report")
    row_count = int(report.get("row_count", 0)) if isinstance(report, dict) else 0
    is_valid, validation_message = _validate_existing(out_path, free_samples_path)
    if not is_valid:
        _emit(progress_path, phase="failed_validation", completed=row_count, total=row_count, message=validation_message, out_path=out_path)
        print(f"Semantic Entropy output validation failed after write: {validation_message}", file=sys.stderr)
        return 2
    _emit(progress_path, phase="complete", completed=row_count, total=row_count, message=validation_message, out_path=out_path)
    print(json.dumps({"report": payload["report"], "storage": payload["storage"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
