#!/usr/bin/env python3
"""Validate that paper-derived feature claims are traceable to formula notes and pipeline gates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


FORMULA_MARKER = re.compile(r"^## Formula: (?P<id>[a-z0-9_]+)\s*$", re.MULTILINE)
REFERENCE_FIELDS = ("Page Reference:", "Section Reference:", "Equation Reference:")

PIPELINE_FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "semantic_entropy": (
        "semantic_entropy_nli_likelihood",
        "N=10 free samples",
        "NLI semantic clustering",
        "likelihood-based cluster probability",
        "Farquhar Semantic Entropy",
    ),
    "semantic_energy_cluster_uncertainty": (
        "semantic_energy_cluster_uncertainty",
        "multiple generated answers",
        "selected-token logit-derived energy",
        "cluster-level aggregation",
        "Ma Semantic Energy",
    ),
    "candidate_logit_diagnostics": (
        "semantic_energy_boltzmann_diagnostic",
        "mean_negative_log_probability",
        "logit_variance",
        "confidence_margin",
        "candidate-level diagnostic",
    ),
    "quco_entity_frequency": (
        "entity_frequency_axis",
        "Infini-gram-compatible",
        "continuous corpus-support axis",
        "QuCo-RAG",
    ),
    "quco_entity_pair_cooccurrence": (
        "entity_pair_cooccurrence_axis",
        "head AND tail",
        "relation-level corpus-support axis",
        "QuCo-RAG",
    ),
    "condition_aware_fusion": (
        "condition-aware fusion",
        "corpus-axis bin",
        "global fusion",
        "axis interaction",
    ),
}

IMPLEMENTATION_FEATURE_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "semantic_entropy": ("experiments/scripts/compute_semantic_entropy.py", "semantic_entropy"),
    "semantic_energy_cluster_uncertainty": ("experiments/scripts/compute_energy_features.py", "energy"),
    "candidate_logit_diagnostics": ("experiments/adapters/energy_features.py", "confidence_margin"),
    "quco_entity_frequency": ("experiments/adapters/corpus_features.py", "entity_frequency"),
    "quco_entity_pair_cooccurrence": ("experiments/adapters/corpus_features.py", "cooccurrence"),
    "condition_aware_fusion": ("experiments/application/fusion.py", "LogisticRegressionModel"),
}


def load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"formula config must contain a JSON object: {path}")
    return loaded


def split_formula_sections(text: str) -> dict[str, str]:
    matches = list(FORMULA_MARKER.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group("id")] = text[start:end]
    return sections


def required_formula_entries(formulas: dict[str, object]) -> list[dict[str, object]]:
    entries = formulas.get("required_formulas")
    if not isinstance(entries, list):
        raise ValueError("formula config missing list field required_formulas")
    checked: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each required_formulas entry must be an object")
        checked.append(entry)
    return checked


def validate(args: argparse.Namespace) -> list[str]:
    formulas_path = Path(args.formulas)
    notes_path = Path(args.notes)
    pipeline_path = Path(args.pipeline)
    repo_root = formulas_path.resolve().parents[2]

    formulas = load_json(formulas_path)
    sections = split_formula_sections(notes_path.read_text(encoding="utf-8"))
    pipeline_text = pipeline_path.read_text(encoding="utf-8")
    problems: list[str] = []

    for entry in required_formula_entries(formulas):
        formula_id = str(entry.get("id"))
        source_id = str(entry.get("source_id"))
        section = sections.get(formula_id)
        if section is None:
            problems.append(f"missing formula notes section for {formula_id}")
            continue
        if f"Source ID: {source_id}" not in section:
            problems.append(f"formula {formula_id} missing source id {source_id}")
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

    required_pipeline_caveats = (
        "multi-generation semantic clustering and cluster-level energy aggregation",
        "not direct correctness labels",
        "Elasticsearch/BM25 may be used for retrieval evidence",
        "not a RAG system",
    )
    for caveat in required_pipeline_caveats:
        if caveat not in pipeline_text:
            problems.append(f"pipeline missing required caveat: {caveat}")
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
