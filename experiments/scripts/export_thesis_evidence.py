#!/usr/bin/env python3
"""Export thesis-ready evidence from current experiment artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_export_thesis_evidence():
    module = importlib.import_module("experiments.application.thesis_evidence")
    return module.export_thesis_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Root experiments/results directory")
    parser.add_argument("--out", required=True, help="LaTeX table output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_thesis_evidence = _load_export_thesis_evidence()
    payload = export_thesis_evidence(Path(args.results), Path(args.out))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
