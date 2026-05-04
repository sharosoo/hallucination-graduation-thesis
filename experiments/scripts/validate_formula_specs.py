#!/usr/bin/env python3
"""Validate literature formula notes against required formula specs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


FORMULA_MARKER = re.compile(r"^## Formula: (?P<id>[a-z0-9_]+)\s*$", re.MULTILINE)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON-compatible YAML file {path}: {exc}") from exc


def split_formula_sections(text: str) -> dict[str, str]:
    matches = list(FORMULA_MARKER.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group("id")] = text[start:end]
    return sections


def has_reference(section: str) -> bool:
    reference_patterns = [
        r"Page Reference:\s*.+",
        r"Section Reference:\s*.+",
        r"Equation Reference:\s*.+",
    ]
    return all(re.search(pattern, section) for pattern in reference_patterns)


def validate(config_path: Path, notes_path: Path) -> list[str]:
    config = load_json(config_path)
    notes_text = notes_path.read_text(encoding="utf-8")
    sections = split_formula_sections(notes_text)
    problems: list[str] = []

    for required in config.get("required_formulas", []):
        formula_id = required["id"]
        section = sections.get(formula_id)
        if not section:
            problems.append(f"missing formula section: {formula_id}")
            continue
        if f"Source ID: {required['source_id']}" not in section:
            problems.append(
                f"missing source id for {formula_id}: expected Source ID: {required['source_id']}"
            )
        if not has_reference(section):
            if required.get("allowed_caveat") and "UNVERIFIED_DO_NOT_CITE" in section:
                continue
            problems.append(
                f"missing source reference fields for {formula_id}: require page/section/equation or UNVERIFIED_DO_NOT_CITE"
            )
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: python3 experiments/scripts/validate_formula_specs.py <formulas.yaml> <formula_notes.md>",
            file=sys.stderr,
        )
        return 2

    problems = validate(Path(argv[1]), Path(argv[2]))
    if problems:
        print("Formula spec validation failed.", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print("Formula spec validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
