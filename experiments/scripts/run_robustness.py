#!/usr/bin/env python3
"""Run robustness, threshold-sensitivity, and selective-risk reporting."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_run_robustness():
    module = importlib.import_module("experiments.application.robustness")
    return module.run_robustness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Feature table artifact path (.parquet or .jsonl)")
    parser.add_argument("--fusion", required=True, help="Fusion result directory containing summary.json and predictions.jsonl")
    parser.add_argument("--out", required=True, help="Output directory for robustness artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_robustness = _load_run_robustness()
    payload = run_robustness(Path(args.features), Path(args.fusion), Path(args.out))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
