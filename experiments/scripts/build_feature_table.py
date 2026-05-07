#!/usr/bin/env python3
"""Build the thesis-valid candidate-level feature table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import read_feature_rows
from experiments.application.labeling import build_feature_table, validate_type_labels
from experiments.scripts.stage_control import (
    FEATURE_TABLE_SCHEMA_VERSION,
    artifact_materialized,
    progress_snapshot,
    remove_materialized_outputs,
    validate_rows_schema_version,
    write_progress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Directory containing corpus_features and energy_features results")
    parser.add_argument("--out", required=True, help="Output path (.parquet or .jsonl)")
    parser.add_argument("--progress", help="Optional progress JSON path updated atomically.")
    parser.add_argument("--resume", action="store_true", help="Skip when existing output validates.")
    parser.add_argument("--force", action="store_true", help="Replace existing output instead of refusing overwrite.")
    return parser.parse_args()


def _validate_existing(path: Path, dataset_config_path: Path) -> tuple[bool, str]:
    try:
        rows, storage = read_feature_rows(path)
    except Exception as exc:
        return False, f"existing artifact could not be read: {exc}"
    if not rows:
        return False, "existing artifact is empty"
    schema_ok, schema_message = validate_rows_schema_version(rows, FEATURE_TABLE_SCHEMA_VERSION, storage_report=storage)
    if not schema_ok:
        return False, schema_message
    try:
        report = validate_type_labels(path, dataset_config_path)
    except Exception as exc:
        return False, f"existing feature table failed type-label validation: {exc}"
    row_count = report.get("row_count", len(rows)) if isinstance(report, dict) else len(rows)
    return True, f"existing feature table validates with {row_count} row(s)"


def _emit(progress_path: Path | None, *, phase: str, completed: int, total: int, message: str, out_path: Path) -> None:
    write_progress(progress_path, progress_snapshot(phase=phase, completed=completed, total=total, message=message, output_path=out_path))


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    progress_path = Path(args.progress) if args.progress else None
    dataset_config_path = ROOT / "experiments" / "configs" / "datasets.yaml"
    if args.resume and args.force:
        print("--resume and --force are mutually exclusive.", file=sys.stderr)
        return 2
    if artifact_materialized(out_path):
        is_valid, validation_message = _validate_existing(out_path, dataset_config_path)
        if args.resume and is_valid:
            _emit(progress_path, phase="complete", completed=1, total=1, message=f"resume skip: {validation_message}", out_path=out_path)
            print(json.dumps({"status": "skipped_existing_valid_output", "message": validation_message}, indent=2, ensure_ascii=False))
            return 0
        if not args.force:
            print(f"Refusing to overwrite existing feature table without --force. Validation status: {validation_message}", file=sys.stderr)
            return 2
        remove_materialized_outputs(out_path)

    _emit(progress_path, phase="start", completed=0, total=0, message="starting feature table build", out_path=out_path)
    payload = build_feature_table(
        results_dir=Path(args.inputs),
        out_path=out_path,
        dataset_config_path=dataset_config_path,
    )
    row_count = int(payload["report"].get("row_count", 0)) if isinstance(payload.get("report"), dict) else 0
    is_valid, validation_message = _validate_existing(out_path, dataset_config_path)
    if not is_valid:
        _emit(progress_path, phase="failed_validation", completed=row_count, total=row_count, message=validation_message, out_path=out_path)
        print(f"Feature table validation failed after write: {validation_message}", file=sys.stderr)
        return 2
    _emit(progress_path, phase="complete", completed=row_count, total=row_count, message=validation_message, out_path=out_path)
    print(
        json.dumps(
            {
                "storage": payload["storage"],
                "report_path": payload["report_path"],
                "row_identity": payload["report"]["row_identity"],
                "prompt_broadcast_key": payload["report"]["prompt_broadcast_key"],
                "target_label_field": payload["report"]["target_label_field"],
                "target_counts_by_dataset": payload["report"]["target_counts_by_dataset"],
                "prompt_balance": {
                    "prompt_count": payload["report"]["prompt_balance"]["prompt_count"],
                    "valid_prompt_count": payload["report"]["prompt_balance"]["valid_prompt_count"],
                    "invalid_prompt_count": payload["report"]["prompt_balance"]["invalid_prompt_count"],
                },
                "energy_status_counts": payload["report"]["energy_status_counts"],
                "feature_alignment_summary": payload["report"]["feature_alignment_summary"],
                "boundary_self_check": payload["report"]["boundary_self_check"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
