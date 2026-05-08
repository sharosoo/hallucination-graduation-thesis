#!/usr/bin/env python3
"""Compute direct corpus feature rows for the thesis pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import write_feature_artifact
from experiments.adapters.corpus_features import CorpusFeatureAdapter, read_feature_rows
from experiments.scripts.stage_control import (
    CORPUS_AXIS_SCHEMA_VERSION,
    add_schema_version,
    artifact_materialized,
    progress_snapshot,
    remove_materialized_outputs,
    validate_rows_schema_version,
    write_progress,
)
from experiments.scripts.validate_feature_provenance import validate_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output path (.parquet or .jsonl)")
    parser.add_argument("--candidates", required=True, help="candidate_rows artifact whose adjacent sidecar provides direct corpus count provenance")
    parser.add_argument("--progress", help="Optional progress JSON path updated atomically.")
    parser.add_argument("--resume", action="store_true", help="Skip when existing output validates.")
    parser.add_argument("--force", action="store_true", help="Replace existing output instead of refusing overwrite.")
    parser.add_argument(
        "--entity-extractor",
        choices=("regex", "quco"),
        default="regex",
        help=(
            "Entity extractor backend. 'regex' (default) = legacy "
            "phrase_candidates heuristic. 'quco' = ZhishanQ/QuCo-extractor-0.5B "
            "(Qwen2.5-0.5B-Instruct fine-tuned, knowledge triplet output)."
        ),
    )
    parser.add_argument(
        "--entity-extractor-model-ref",
        default="ZhishanQ/QuCo-extractor-0.5B",
        help="HF model id or local path used when --entity-extractor=quco.",
    )
    parser.add_argument(
        "--entity-extractor-device",
        default=None,
        help="Override device for QuCo extractor (e.g. 'cuda:0', 'cpu').",
    )
    return parser.parse_args()


def _build_entity_extractor(args: argparse.Namespace) -> Any:
    if args.entity_extractor == "regex":
        from experiments.adapters.entity_extractor_regex import RegexEntityExtractor
        return RegexEntityExtractor()
    if args.entity_extractor == "quco":
        from experiments.adapters.entity_extractor_quco import QucoEntityExtractor
        return QucoEntityExtractor(
            model_ref=args.entity_extractor_model_ref,
            device=args.entity_extractor_device,
        )
    raise ValueError(f"Unknown entity extractor: {args.entity_extractor}")


def _validate_existing(path: Path, expected_source_path: Path) -> tuple[bool, str]:
    if path.suffix == ".parquet" and path.exists():
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(
                path,
                columns=[
                    "schema_version",
                    "candidate_id",
                    "sample_id",
                    "dataset",
                    "prompt_id",
                    "pair_id",
                    "candidate_role",
                    "source_artifact_path",
                ],
            )
            payload = table.to_pydict()
            row_count = table.num_rows
            if row_count <= 0:
                return False, "existing artifact is empty"
            schema_ok, schema_message = validate_rows_schema_version(table.to_pylist(), CORPUS_AXIS_SCHEMA_VERSION)
            if not schema_ok:
                return False, schema_message
            candidate_ids = payload.get("candidate_id", [])
            sample_ids = payload.get("sample_id", [])
            source_paths = payload.get("source_artifact_path", [])
            if len(set(candidate_ids)) != row_count:
                return False, "existing artifact has duplicate candidate_id values"
            expected = expected_source_path.resolve()
            for index, candidate_id in enumerate(candidate_ids):
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    return False, f"row {index}: candidate_id must be a non-empty string"
                if sample_ids[index] != candidate_id:
                    return False, f"row {index}: sample_id must mirror candidate_id"
            for field_name in ("dataset", "prompt_id", "pair_id", "candidate_role"):
                for index, value in enumerate(payload.get(field_name, [])):
                    if not isinstance(value, str) or not value.strip():
                        return False, f"row {index}: {field_name} must be a non-empty string"
            for index, value in enumerate(source_paths):
                if not isinstance(value, str) or not value.strip():
                    return False, f"row {index}: missing source_artifact_path"
                if Path(value).resolve() != expected:
                    return False, f"row {index}: source_artifact_path does not match --candidates"
            return True, f"existing corpus artifact validates with {row_count} row(s)"
        except Exception as exc:
            return False, f"existing parquet artifact could not be validated by projected columns: {exc}"
    try:
        rows, storage = read_feature_rows(path)
    except Exception as exc:
        return False, f"existing artifact could not be read: {exc}"
    if not rows:
        return False, "existing artifact is empty"
    schema_ok, schema_message = validate_rows_schema_version(rows, CORPUS_AXIS_SCHEMA_VERSION, storage_report=storage)
    if not schema_ok:
        return False, schema_message
    problems: list[str] = []
    seen_candidate_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be a JSON object")
            continue
        problems.extend(validate_row(row, index))
        candidate_id = row.get("candidate_id")
        if isinstance(candidate_id, str):
            if candidate_id in seen_candidate_ids:
                problems.append(f"row {index}: duplicate candidate_id {candidate_id!r}")
            seen_candidate_ids.add(candidate_id)
        source_path = row.get("source_artifact_path")
        if isinstance(source_path, str) and source_path.strip() and Path(source_path) != expected_source_path:
            try:
                if Path(source_path).resolve() != expected_source_path.resolve():
                    problems.append(f"row {index}: source_artifact_path does not match --candidates")
            except OSError:
                problems.append(f"row {index}: source_artifact_path does not match --candidates")
        if len(problems) >= 20:
            break
    if problems:
        return False, "; ".join(problems[:5])
    return True, f"existing corpus artifact validates with {len(rows)} row(s)"


def _emit(progress_path: Path | None, *, phase: str, completed: int, total: int, message: str, out_path: Path) -> None:
    write_progress(progress_path, progress_snapshot(phase=phase, completed=completed, total=total, message=message, output_path=out_path))


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    candidates_path = Path(args.candidates)
    progress_path = Path(args.progress) if args.progress else None
    if args.resume and args.force:
        print("--resume and --force are mutually exclusive.", file=sys.stderr)
        return 2
    if artifact_materialized(out_path):
        is_valid, validation_message = _validate_existing(out_path, candidates_path)
        if args.resume and is_valid:
            _emit(progress_path, phase="complete", completed=1, total=1, message=f"resume skip: {validation_message}", out_path=out_path)
            print(json.dumps({"status": "skipped_existing_valid_output", "message": validation_message}, indent=2, ensure_ascii=False))
            return 0
        if not args.force:
            print(f"Refusing to overwrite existing corpus output without --force. Validation status: {validation_message}", file=sys.stderr)
            return 2
        remove_materialized_outputs(out_path)

    _emit(progress_path, phase="start", completed=0, total=0, message="starting corpus feature computation", out_path=out_path)
    dataset_config_path = ROOT / "experiments" / "configs" / "datasets.yaml"
    extractor = _build_entity_extractor(args)
    direct_adapter = CorpusFeatureAdapter(
        candidates_path=candidates_path,
        dataset_config_path=dataset_config_path,
        entity_extractor=extractor,
    )
    _emit(progress_path, phase="build_rows", completed=0, total=len(direct_adapter.rows), message="building corpus feature rows from required count backend", out_path=out_path)
    rows, report = direct_adapter.build_feature_rows()
    _emit(progress_path, phase="write_output", completed=len(rows), total=len(rows), message="writing corpus feature artifact", out_path=out_path)
    report["schema_version"] = CORPUS_AXIS_SCHEMA_VERSION
    versioned_rows = add_schema_version(rows, CORPUS_AXIS_SCHEMA_VERSION)
    storage = write_feature_artifact(out_path, versioned_rows, report, schema_version=CORPUS_AXIS_SCHEMA_VERSION)
    # Write per-row entity/pair provenance to a sidecar JSONL so the lean
    # parquet stays fast for downstream stages while keeping a paper-trail
    # of which entities/pairs were matched per candidate. Downstream code
    # never touches this file; it's read on-demand for spot checks.
    provenance_records = getattr(direct_adapter, "_provenance_records", None)
    if isinstance(provenance_records, list) and provenance_records:
        provenance_path = out_path.with_suffix(out_path.suffix + ".provenance.jsonl")
        provenance_tmp = provenance_path.with_name(f".{provenance_path.name}.tmp-{os.getpid()}")
        provenance_tmp.parent.mkdir(parents=True, exist_ok=True)
        with provenance_tmp.open("w", encoding="utf-8") as handle:
            for record in provenance_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        provenance_tmp.replace(provenance_path)
        report["provenance_sidecar"] = {
            "path": str(provenance_path),
            "row_count": len(provenance_records),
            "schema_note": "Per-candidate entity/pair matches with raw counts. Heavy nested fields not stored in main parquet to keep row-group reads lean.",
        }
    is_valid, validation_message = _validate_existing(out_path, candidates_path)
    if not is_valid:
        _emit(progress_path, phase="failed_validation", completed=len(rows), total=len(rows), message=validation_message, out_path=out_path)
        print(f"Corpus output validation failed after write: {validation_message}", file=sys.stderr)
        return 2
    _emit(progress_path, phase="complete", completed=len(rows), total=len(rows), message=validation_message, out_path=out_path)
    print(json.dumps({"report": report, "storage": storage}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
