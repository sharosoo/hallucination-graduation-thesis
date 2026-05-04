#!/usr/bin/env python3
"""Validate the binding experiment pipeline document."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REQUIRED_PHRASES = (
    "Smoke results are never thesis evidence",
    "Full logits must be repo-owned",
    "Corpus statistics must be repo-owned",
    "Infini-gram-compatible count backend",
    "Elasticsearch/BM25 is used for retrieval",
    "custom stdlib L2 logistic regression",
    "full-core",
    "7817 rows",
    "validate_paper_feature_alignment.py",
    "run_pipeline.py",
    "cache_upstream_corpus_stats",
    "proxy_cached_artifact_index",
)

REQUIRED_STAGES = tuple(f"### S{index}." for index in range(10))
FORBIDDEN_PATTERNS = (
    re.compile(r"200\s+rows?\s+(?:is|are)\s+(?:final|thesis-valid)", re.IGNORECASE),
    re.compile(r"Elasticsearch\s+(?:computes|is used for)\s+entity\s+frequency", re.IGNORECASE),
)


def validate(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    for phrase in REQUIRED_PHRASES:
        if phrase not in text:
            problems.append(f"missing required pipeline phrase: {phrase}")
    for marker in REQUIRED_STAGES:
        if marker not in text:
            problems.append(f"missing pipeline stage marker: {marker}")
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        if match:
            problems.append(f"forbidden pipeline wording: {match.group(0)}")
    if "Thesis-valid?" not in text or "No |" not in text or "Yes, after gates" not in text:
        problems.append("evaluation mode table must distinguish smoke/dev from thesis-valid full runs")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", help="Path to experiments/PIPELINE.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problems = validate(Path(args.pipeline))
    if problems:
        print("Pipeline contract validation failed.", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Pipeline contract validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
