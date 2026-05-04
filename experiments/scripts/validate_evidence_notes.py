#!/usr/bin/env python3
"""Validate evidence notes for required guardrails and citation caveats."""

from __future__ import annotations

import re
import sys
from pathlib import Path


NUMERIC_CLAIM_RE = re.compile(r"\b(?:\d+(?:\.\d+)?%|AUROC|AUPRC|\d+\.\d+)\b")
REFERENCE_RE = re.compile(r"\b(page|table|section|equation)\b", re.IGNORECASE)
UNVERIFIED_MARKER = "UNVERIFIED_DO_NOT_CITE"


def require_file(path: Path, problems: list[str]) -> str:
    if not path.exists():
        problems.append(f"missing evidence note: {path.name}")
        return ""
    return path.read_text(encoding="utf-8")


def validate_numeric_claims(path: Path, text: str, problems: list[str]) -> None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not NUMERIC_CLAIM_RE.search(line):
            continue
        if REFERENCE_RE.search(line) or UNVERIFIED_MARKER in line:
            continue
        problems.append(
            f"uncaveated numeric claim in {path.name}:{line_number} -> add page/table/section/equation reference or {UNVERIFIED_MARKER}"
        )


def validate(directory: Path) -> list[str]:
    problems: list[str] = []
    quco_path = directory / "quco_vs_probe.md"
    energy_path = directory / "semantic_energy_single_cluster.md"

    quco_text = require_file(quco_path, problems)
    energy_text = require_file(energy_path, problems)

    if quco_text:
        required_phrases = [
            "corpus-grounded proxies",
            "model-internal supervised signal",
            "objective/model-agnostic caveat",
        ]
        for phrase in required_phrases:
            if phrase not in quco_text:
                problems.append(f"missing phrase in quco_vs_probe.md: {phrase}")
        validate_numeric_claims(quco_path, quco_text, problems)

    if energy_text:
        if UNVERIFIED_MARKER not in energy_text and not REFERENCE_RE.search(energy_text):
            problems.append(
                "semantic_energy_single_cluster.md must include PDF references or UNVERIFIED_DO_NOT_CITE for single-cluster/Zero-SE claims"
            )
        validate_numeric_claims(energy_path, energy_text, problems)

    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: python3 experiments/scripts/validate_evidence_notes.py <evidence_notes_dir>",
            file=sys.stderr,
        )
        return 2

    problems = validate(Path(argv[1]))
    if problems:
        print("Evidence note validation failed.", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print("Evidence note validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
