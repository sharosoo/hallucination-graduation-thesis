#!/usr/bin/env python3
"""Build the merged feature table with fixed type labels and explicit energy availability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.application.labeling import build_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Directory containing corpus_features and energy_features results")
    parser.add_argument("--out", required=True, help="Output path (.parquet or .jsonl)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_feature_table(
        results_dir=Path(args.inputs),
        out_path=Path(args.out),
        dataset_config_path=ROOT / "experiments" / "configs" / "datasets.yaml",
    )
    print(
        json.dumps(
            {
                "storage": payload["storage"],
                "report_path": payload["report_path"],
                "label_counts": payload["report"]["label_counts"],
                "energy_status_counts": payload["report"]["energy_status_counts"],
                "boundary_self_check": payload["report"]["boundary_self_check"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
