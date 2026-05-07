#!/usr/bin/env python3
"""Build annotation-driven correctness labels from candidate rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters.correctness_dataset import build_annotation_correctness_dataset  # type: ignore[reportMissingImports]
from experiments.scripts.stage_control import progress_snapshot, write_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidate_rows.jsonl from prepare_datasets.py; this is the only label-building input.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output correctness dataset directory; writes data/correctness_judgments.jsonl and dataset_manifest.json.",
    )
    parser.add_argument(
        "--evidence",
        help="Optional path for label-manifest evidence JSON with forbidden-field and no-judge markers.",
    )
    parser.add_argument(
        "--halueval-evidence",
        help="Optional path for HaluEval-QA direct annotation evidence JSON.",
    )
    parser.add_argument("--progress", help="Optional progress JSON path updated atomically.")
    return parser.parse_args()


def _emit(
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


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    progress_path = Path(args.progress) if args.progress else None
    output_path = out_dir / "dataset_manifest.json"
    _emit(
        progress_path,
        phase="start",
        completed=0,
        total=1,
        message="starting correctness label dataset build",
        output_path=output_path,
    )
    try:
        result = build_annotation_correctness_dataset(
            candidate_rows_path=Path(args.candidates),
            out_dir=out_dir,
            evidence_path=Path(args.evidence) if args.evidence else None,
            halueval_evidence_path=Path(args.halueval_evidence) if args.halueval_evidence else None,
        )
    except Exception as exc:
        _emit(
            progress_path,
            phase="failed",
            completed=0,
            total=1,
            message=str(exc),
            output_path=output_path,
        )
        raise
    _emit(
        progress_path,
        phase="complete",
        completed=1,
        total=1,
        message="correctness label dataset build complete",
        output_path=output_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
