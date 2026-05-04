#!/usr/bin/env python3
"""Run type-specific signal analysis for the merged feature table."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_run_type_analysis():
    module = importlib.import_module("experiments.application.type_analysis")
    return module.run_type_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Feature table artifact path (.parquet or .jsonl)")
    parser.add_argument("--out", required=True, help="Output directory for type-analysis artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_type_analysis = _load_run_type_analysis()
    summary = run_type_analysis(Path(args.features), Path(args.out))
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "row_count": summary["row_count"],
                "datasets": summary["datasets"],
                "artifacts": summary["artifacts"],
                "self_check": summary["self_check"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
