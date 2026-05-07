#!/usr/bin/env python3
"""Compute sampled-response Semantic Energy plus candidate diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import read_feature_rows, write_feature_artifact
from experiments.adapters.energy_features import EnergyFeatureUnavailableError, build_energy_rows_from_generation_artifacts
from experiments.scripts.stage_control import (
    SEMANTIC_ENERGY_SCHEMA_VERSION,
    add_schema_version,
    artifact_materialized,
    progress_snapshot,
    remove_materialized_outputs,
    validate_rows_schema_version,
    write_progress,
)
from experiments.scripts.validate_energy_features import validate_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output path (.parquet or .jsonl)")
    parser.add_argument(
        "--candidate-scores",
        required=True,
        help="Teacher-forced candidate-score artifact containing candidate_score_rows and token_score_rows for diagnostics",
    )
    parser.add_argument(
        "--free-samples",
        required=True,
        help="N=10 free-sample generation artifact containing selected_token_logits for sampled-response Energy",
    )
    parser.add_argument(
        "--semantic-entropy",
        required=True,
        help="Task 4 Semantic Entropy artifact containing semantic_clusters for shared cluster aggregation",
    )
    parser.add_argument(
        "--progress",
        help="Optional progress JSON path. Updated atomically with phase/completed/total/percent/message.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip computation when the existing output validates. Invalid existing outputs still fail unless --force is used.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing output artifact after validation/removal of stale materialized files.",
    )
    return parser.parse_args()


def _snapshot(
    *,
    phase: str,
    completed: int,
    total: int,
    percent: float,
    message: str,
    out_path: Path,
    candidate_scores_path: Path,
    free_samples_path: Path,
    semantic_entropy_path: Path,
):
    del candidate_scores_path
    del free_samples_path
    del semantic_entropy_path
    del percent
    return progress_snapshot(
        phase=phase,
        completed=completed,
        total=total,
        message=message,
        output_path=out_path,
    )


def _expected_candidate_ids(candidate_scores_path: Path) -> set[str]:
    payload = json.loads(candidate_scores_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate-score artifact must decode to an object")
    candidate_rows = payload.get("candidate_score_rows")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        raise ValueError("candidate-score artifact must contain non-empty candidate_score_rows")
    candidate_ids: set[str] = set()
    for index, row in enumerate(candidate_rows):
        if not isinstance(row, dict):
            raise ValueError(f"candidate_score_rows[{index}] must be an object")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"candidate_score_rows[{index}] must include non-empty candidate_id")
        if candidate_id in candidate_ids:
            raise ValueError(f"candidate-score artifact duplicates candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
    return candidate_ids


def _same_source_path(left: object, right: Path) -> bool:
    if not isinstance(left, str) or not left.strip():
        return False
    left_path = Path(left)
    right_text = str(right)
    if str(left_path) == right_text:
        return True
    try:
        return left_path.resolve() == right.resolve()
    except OSError:
        return False


def _validate_existing_artifact(path: Path, candidate_scores_path: Path, free_samples_path: Path, semantic_entropy_path: Path) -> tuple[bool, str]:
    try:
        expected_ids = _expected_candidate_ids(candidate_scores_path)
    except Exception as exc:
        return False, f"candidate-score input could not be validated for resume: {exc}"
    try:
        rows, storage_report = read_feature_rows(path)
    except Exception as exc:
        return False, f"existing artifact could not be read: {exc}"
    if not rows:
        return False, "existing artifact is empty"
    schema_ok, schema_message = validate_rows_schema_version(
        rows,
        SEMANTIC_ENERGY_SCHEMA_VERSION,
        storage_report=storage_report,
    )
    if not schema_ok:
        return False, schema_message
    if len(rows) != len(expected_ids):
        return False, f"existing artifact row count {len(rows)} does not match candidate-score count {len(expected_ids)}"
    problems: list[str] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be a JSON object")
            continue
        problems.extend(validate_row(row, index))
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str):
            problems.append(f"row {index}: candidate_id must be a string")
        elif candidate_id in seen_ids:
            problems.append(f"row {index}: duplicate candidate_id {candidate_id!r}")
        else:
            seen_ids.add(candidate_id)
        if not _same_source_path(row.get("source_artifact_path"), candidate_scores_path):
            problems.append(f"row {index}: source_artifact_path does not match requested candidate-score input")
        if not _same_source_path(row.get("source_free_sample_path"), free_samples_path):
            problems.append(f"row {index}: source_free_sample_path does not match requested free-sample input")
        if not _same_source_path(row.get("source_semantic_entropy_path"), semantic_entropy_path):
            problems.append(f"row {index}: source_semantic_entropy_path does not match requested semantic-entropy input")
        energy_provenance = row.get("energy_provenance")
        features = row.get("features")
        if isinstance(energy_provenance, dict):
            if energy_provenance.get("paper_faithful_energy_available") is not True:
                problems.append(f"row {index}: paper_faithful_energy_available must be true")
            if energy_provenance.get("energy_granularity") != "prompt_level_broadcast_to_candidate_rows":
                problems.append(f"row {index}: energy_granularity must mark prompt-level broadcast")
            if energy_provenance.get("source_artifact_has_full_logits") is not True:
                problems.append(f"row {index}: source_artifact_has_full_logits must be true")
            if energy_provenance.get("tokens_with_full_logits") != row.get("candidate_token_count"):
                problems.append(f"row {index}: tokens_with_full_logits must match candidate_token_count")
        if isinstance(features, dict):
            if features.get("semantic_energy_cluster_uncertainty") is None:
                problems.append(f"row {index}: semantic_energy_cluster_uncertainty must be present")
            if features.get("semantic_energy_sample_energy") is None:
                problems.append(f"row {index}: semantic_energy_sample_energy must be present")
            if features.get("logit_variance") is None:
                problems.append(f"row {index}: logit_variance must be present for thesis-valid full-logits Energy output")
            if features.get("confidence_margin") is None:
                problems.append(f"row {index}: confidence_margin must be present for thesis-valid full-logits Energy output")
        if len(problems) >= 20:
            break
    missing_ids = expected_ids - seen_ids
    extra_ids = seen_ids - expected_ids
    if missing_ids:
        preview = sorted(missing_ids)[:5]
        problems.append(f"existing artifact is missing {len(missing_ids)} candidate_id(s): {preview}")
    if extra_ids:
        preview = sorted(extra_ids)[:5]
        problems.append(f"existing artifact contains {len(extra_ids)} unexpected candidate_id(s): {preview}")
    if problems:
        return False, "; ".join(problems[:5])
    return True, f"existing artifact validates with {len(rows)} row(s)"


def _coerce_progress(
    update: Mapping[str, object],
    *,
    out_path: Path,
    candidate_scores_path: Path,
    free_samples_path: Path,
    semantic_entropy_path: Path,
):
    phase = update.get("phase")
    completed = update.get("completed")
    total = update.get("total")
    percent = update.get("percent")
    message = update.get("message")
    return _snapshot(
        phase=phase if isinstance(phase, str) else "unknown",
        completed=completed if isinstance(completed, int) else 0,
        total=total if isinstance(total, int) else 0,
        percent=float(percent) if isinstance(percent, int | float) and not isinstance(percent, bool) else 0.0,
        message=message if isinstance(message, str) else "",
        out_path=out_path,
        candidate_scores_path=candidate_scores_path,
        free_samples_path=free_samples_path,
        semantic_entropy_path=semantic_entropy_path,
    )


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    candidate_scores_path = Path(args.candidate_scores)
    free_samples_path = Path(args.free_samples)
    semantic_entropy_path = Path(args.semantic_entropy)
    progress_path = Path(args.progress) if args.progress else None
    if args.resume and args.force:
        print("--resume and --force are mutually exclusive.", file=sys.stderr)
        return 2
    if artifact_materialized(out_path):
        is_valid, validation_message = _validate_existing_artifact(out_path, candidate_scores_path, free_samples_path, semantic_entropy_path)
        if args.resume and is_valid:
            write_progress(
                progress_path,
                _snapshot(
                    phase="complete",
                    completed=1,
                    total=1,
                    percent=100.0,
                    message=f"resume skip: {validation_message}",
                    out_path=out_path,
                    candidate_scores_path=candidate_scores_path,
                    free_samples_path=free_samples_path,
                    semantic_entropy_path=semantic_entropy_path,
                ),
            )
            print(json.dumps({"status": "skipped_existing_valid_output", "message": validation_message}, indent=2, ensure_ascii=False))
            return 0
        if not args.force:
            print(
                f"Refusing to overwrite existing Energy output without --force. Validation status: {validation_message}",
                file=sys.stderr,
            )
            return 2
        remove_materialized_outputs(out_path)

    write_progress(
        progress_path,
        _snapshot(
            phase="start",
            completed=0,
            total=0,
            percent=0.0,
            message="starting sampled-response Semantic Energy computation",
            out_path=out_path,
            candidate_scores_path=candidate_scores_path,
            free_samples_path=free_samples_path,
            semantic_entropy_path=semantic_entropy_path,
        ),
    )

    def progress_callback(update: dict[str, object]) -> None:
        write_progress(
            progress_path,
            _coerce_progress(
                update,
                out_path=out_path,
                candidate_scores_path=candidate_scores_path,
                free_samples_path=free_samples_path,
                semantic_entropy_path=semantic_entropy_path,
            ),
        )

    try:
        rows, report = build_energy_rows_from_generation_artifacts(
            candidate_scores_path=candidate_scores_path,
            free_samples_path=free_samples_path,
            semantic_entropy_path=semantic_entropy_path,
            progress_callback=progress_callback,
        )
    except EnergyFeatureUnavailableError as exc:
        write_progress(
            progress_path,
            _snapshot(
                phase="failed",
                completed=0,
                total=0,
                percent=0.0,
                message=str(exc),
                out_path=out_path,
                candidate_scores_path=candidate_scores_path,
                free_samples_path=free_samples_path,
                semantic_entropy_path=semantic_entropy_path,
            ),
        )
        print(json.dumps({"report": {"message": str(exc)}}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        write_progress(
            progress_path,
            _snapshot(
                phase="failed",
                completed=0,
                total=0,
                percent=0.0,
                message=str(exc),
                out_path=out_path,
                candidate_scores_path=candidate_scores_path,
                free_samples_path=free_samples_path,
                semantic_entropy_path=semantic_entropy_path,
            ),
        )
        print(str(exc), file=sys.stderr)
        return 2
    write_progress(
        progress_path,
        _snapshot(
            phase="write_output",
            completed=len(rows),
            total=len(rows),
            percent=100.0,
            message="writing Energy feature artifact with atomic replace",
            out_path=out_path,
            candidate_scores_path=candidate_scores_path,
            free_samples_path=free_samples_path,
            semantic_entropy_path=semantic_entropy_path,
        ),
    )
    report["schema_version"] = SEMANTIC_ENERGY_SCHEMA_VERSION
    versioned_rows = add_schema_version(rows, SEMANTIC_ENERGY_SCHEMA_VERSION)
    storage = write_feature_artifact(out_path, versioned_rows, report, schema_version=SEMANTIC_ENERGY_SCHEMA_VERSION)
    is_valid, validation_message = _validate_existing_artifact(out_path, candidate_scores_path, free_samples_path, semantic_entropy_path)
    if not is_valid:
        write_progress(
            progress_path,
            _snapshot(
                phase="failed_validation",
                completed=len(rows),
                total=len(rows),
                percent=100.0,
                message=validation_message,
                out_path=out_path,
                candidate_scores_path=candidate_scores_path,
                free_samples_path=free_samples_path,
                semantic_entropy_path=semantic_entropy_path,
            ),
        )
        print(f"Energy output validation failed after write: {validation_message}", file=sys.stderr)
        return 2
    write_progress(
        progress_path,
        _snapshot(
            phase="complete",
            completed=len(rows),
            total=len(rows),
            percent=100.0,
            message=validation_message,
            out_path=out_path,
            candidate_scores_path=candidate_scores_path,
            free_samples_path=free_samples_path,
            semantic_entropy_path=semantic_entropy_path,
        ),
    )
    print(json.dumps({"report": report, "storage": storage}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
