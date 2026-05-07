#!/usr/bin/env python3
"""Run leakage-safe fusion baseline evaluation for the feature table."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.stage_control import progress_snapshot, write_progress


def _load_run_fusion():
    module = importlib.import_module("experiments.application.fusion")
    return module.run_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Feature table artifact path (.parquet or .jsonl)")
    parser.add_argument("--config", required=True, help="Fusion config path (.yaml stored as JSON-compatible text)")
    parser.add_argument("--out", required=True, help="Output directory for fusion artifacts")
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
    output_path = out_dir / "summary.json"
    _emit(
        progress_path,
        phase="start",
        completed=0,
        total=1,
        message="starting fusion evaluation",
        output_path=output_path,
    )
    try:
        run_fusion = _load_run_fusion()
        payload = run_fusion(Path(args.features), Path(args.config), out_dir)
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
        message="fusion evaluation complete",
        output_path=output_path,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
