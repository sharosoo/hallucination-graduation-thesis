#!/usr/bin/env python3
"""Validate feature-table type labels and boundary behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.application.labeling import validate_type_labels, write_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Feature table artifact path (.parquet or .jsonl)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_type_labels(
        feature_artifact_path=Path(args.artifact),
        dataset_config_path=ROOT / "experiments" / "configs" / "datasets.yaml",
    )
    report_path = write_validation_report(Path(args.artifact), report)
    print(json.dumps({"report_path": str(report_path), **report}, indent=2, ensure_ascii=False))
    if report["problems"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
