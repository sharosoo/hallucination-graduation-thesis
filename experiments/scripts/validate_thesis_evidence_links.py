#!/usr/bin/env python3
"""Validate thesis TeX sources against evidence notes and exported summary."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_validate_thesis_evidence_links():
    module = importlib.import_module("experiments.application.thesis_evidence")
    return module.validate_thesis_evidence_links


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thesis", help="Path to thesis main.tex")
    parser.add_argument("evidence_notes", help="Path to evidence notes directory")
    parser.add_argument("summary", help="Path to thesis_evidence_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_thesis_evidence_links = _load_validate_thesis_evidence_links()
    problems = validate_thesis_evidence_links(Path(args.thesis), Path(args.evidence_notes), Path(args.summary))
    if problems:
        print("Thesis evidence link validation failed.", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Thesis evidence link validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
