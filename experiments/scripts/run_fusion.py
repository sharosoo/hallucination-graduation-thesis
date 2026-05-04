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


def _load_run_fusion():
    module = importlib.import_module("experiments.application.fusion")
    return module.run_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Feature table artifact path (.parquet or .jsonl)")
    parser.add_argument("--config", required=True, help="Fusion config path (.yaml stored as JSON-compatible text)")
    parser.add_argument("--out", required=True, help="Output directory for fusion artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_fusion = _load_run_fusion()
    payload = run_fusion(Path(args.features), Path(args.config), Path(args.out))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
