#!/usr/bin/env python3
"""Validate robustness summary wording against CI-crossing overclaims."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_validator():
    module = importlib.import_module("experiments.application.robustness")
    return module.validate_report_claims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Robustness summary.json path to validate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate = _load_validator()
    payload = validate(Path(args.summary))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
