#!/usr/bin/env python3
"""Validate that paper-derived feature claims are traceable to formula notes and pipeline gates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


FORMULA_MARKER = re.compile(r"^## Formula: (?P<id>[a-z0-9_]+)\s*$", re.MULTILINE)
REFERENCE_FIELDS = ("Page Reference:", "Section Reference:", "Equation Reference:")

PIPELINE_FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "semantic_entropy": ("semantic_entropy", "cluster_count", "Farquhar"),
    "semantic_energy_boltzmann_or_proxy": ("semantic_energy_boltzmann", "full_logits", "logsumexp", "Ma Semantic Energy"),
    "quco_entity_frequency": ("entity_frequency_mean", "entity_frequency_min", "Infini-gram-compatible"),
    "quco_entity_pair_cooccurrence": ("entity_pair_cooccurrence", "head AND tail", "QuCo-RAG"),
    "selective_risk_metrics": ("learned fusion with corpus", "logistic coefficients", "non-probe"),
}

IMPLEMENTATION_FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "semantic_entropy": ("experiments/adapters/corpus_features.py", "semantic_entropy"),
    "semantic_energy_boltzmann_or_proxy": ("experiments/adapters/energy_features.py", "semantic_energy_boltzmann"),
    "quco_entity_frequency": ("experiments/adapters/corpus_features.py", "entity_frequency_mean"),
    "quco_entity_pair_cooccurrence": ("experiments/adapters/corpus_features.py", "entity_pair_cooccurrence"),
    "selective_risk_metrics": ("experiments/application/fusion.py", "LogisticRegressionModel"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_formula_sections(text: str) -> dict[str, str]:
    matches = list(FORMULA_MARKER.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group("id")] = text[start:end]
    return sections


def validate(args: argparse.Namespace) -> list[str]:
    formulas_path = Path(args.formulas)
    notes_path = Path(args.notes)
    pipeline_path = Path(args.pipeline)
    repo_root = formulas_path.resolve().parents[2]

    formulas = load_json(formulas_path)
    sections = split_formula_sections(notes_path.read_text(encoding="utf-8"))
    pipeline_text = pipeline_path.read_text(encoding="utf-8")
    problems: list[str] = []

    for entry in formulas.get("required_formulas", []):
        formula_id = str(entry.get("id"))
        section = sections.get(formula_id)
        if section is None:
            problems.append(f"missing formula notes section for {formula_id}")
            continue
        if f"Source ID: {entry.get('source_id')}" not in section:
            problems.append(f"formula {formula_id} missing source id {entry.get('source_id')}")
        for field in REFERENCE_FIELDS:
            if field not in section and "UNVERIFIED_DO_NOT_CITE" not in section:
                problems.append(f"formula {formula_id} missing {field}")
        for phrase in PIPELINE_FEATURE_REQUIREMENTS.get(formula_id, ()): 
            if phrase not in pipeline_text:
                problems.append(f"pipeline missing feature alignment phrase for {formula_id}: {phrase}")
        implementation_requirement = IMPLEMENTATION_FEATURE_REQUIREMENTS.get(formula_id)
        if implementation_requirement is not None:
            relative_path, token = implementation_requirement
            implementation_path = repo_root / relative_path
            if not implementation_path.exists():
                problems.append(f"missing implementation file for {formula_id}: {relative_path}")
            elif token not in implementation_path.read_text(encoding="utf-8"):
                problems.append(f"implementation file {relative_path} missing token for {formula_id}: {token}")

    if "proxy selected-logit scalars are not thesis-valid" not in pipeline_text:
        problems.append("pipeline must explicitly reject proxy selected-logit Energy as thesis-valid")
    if "Elasticsearch/BM25 may be used for retrieval evidence" not in pipeline_text:
        problems.append("pipeline must separate Elasticsearch retrieval from QuCo count semantics")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulas", required=True, help="Path to experiments/configs/formulas.yaml")
    parser.add_argument("--notes", required=True, help="Path to experiments/literature/formula_notes.md")
    parser.add_argument("--pipeline", required=True, help="Path to experiments/PIPELINE.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problems = validate(args)
    if problems:
        print("Paper-feature alignment validation failed.", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Paper-feature alignment validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
