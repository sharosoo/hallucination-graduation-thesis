#!/usr/bin/env python3
"""Validate the binding experiment pipeline document."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REQUIRED_PHRASES = (
    "Full logits must be repo-owned",
    "Corpus statistics must be repo-owned",
    "Infini-gram-compatible count backend",
    "Elasticsearch/BM25 is used for retrieval",
    "custom stdlib L2 logistic regression",
    "paired discriminative dataset",
    "continuous corpus-support axis",
    "hallucination metric reliability",
    "candidate_rows.jsonl",
    "teacher-forced",
    "answer-only free samples N=10",
    "NLI likelihood Semantic Entropy",
    "paper-faithful Semantic Energy",
    "condition-aware fusion",
    "corpus-bin reliability",
    "validate_paper_feature_alignment.py",
    "run_pipeline.py",
)

REQUIRED_STAGES = tuple(f"### S{index}." for index in range(10))
FORBIDDEN_PHRASES = (
    "pdflatex",
    "export_thesis_evidence.py",
    "prompt_rows.jsonl",
    "judge_correctness.py",
    "--judge-config",
    "true Semantic Energy",
    "prompt free samples N=5",
    "prompt별 5개 group",
    "Corpus-Grounded Selective Fusion Detector",
)
FORBIDDEN_PATTERNS = (
    re.compile(r"200\s+rows?\s+(?:is|are)\s+(?:final|thesis-valid)", re.IGNORECASE),
    re.compile(r"Elasticsearch\s+(?:computes|is used for)\s+entity\s+frequency", re.IGNORECASE),
    re.compile(r"run_pipeline\.py[^\n`]*--mode\b", re.IGNORECASE),
    re.compile(r"\bfull-core\b", re.IGNORECASE),
    re.compile(r"\bfull-extended\b", re.IGNORECASE),
    re.compile(r"\bsmoke\b", re.IGNORECASE),
    re.compile(r"\bdataset_mode\b", re.IGNORECASE),
    re.compile(r"\bprompt_rows\b", re.IGNORECASE),
    re.compile(r"SE\s*<=\s*0\.1.*LOW_DIVERSITY", re.IGNORECASE),
    re.compile(r"SE\s*>\s*0\.5.*HIGH_DIVERSITY", re.IGNORECASE),
)

REQUIRED_COMMANDS = (
    "validate_pipeline_contract.py experiments/PIPELINE.md",
    "prepare_datasets.py --config experiments/configs/datasets.yaml",
    "run_generation.py --config experiments/configs/generation.yaml --prompt-groups experiments/results/datasets/prompt_groups.jsonl --candidates experiments/results/datasets/candidate_rows.jsonl --out-free-samples experiments/results/generation/free_sample_rows.json --out-candidate-scores experiments/results/generation/candidate_scores.json",
    "validate_generation_logits.py experiments/results/generation/free_sample_rows.json",
    "validate_generation_logits.py experiments/results/generation/candidate_scores.json",
    "compute_semantic_entropy.py --free-samples experiments/results/generation/free_sample_rows.json",
    "compute_corpus_features.py --candidates experiments/results/datasets/candidate_rows.jsonl",
    "compute_energy_features.py --candidate-scores experiments/results/generation/candidate_scores.json",
    "validate_energy_features.py experiments/results/energy_features.parquet",
    "validate_type_labels.py experiments/results/features.parquet",
)


def validate_generation_config(pipeline_path: Path) -> list[str]:
    problems: list[str] = []
    generation_path = pipeline_path.parent / "configs" / "generation.yaml"
    if not generation_path.exists():
        return [f"missing generation config: {generation_path}"]
    try:
        config = json.loads(generation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"generation config must remain JSON-compatible YAML: {exc}"]
    generation = config.get("generation")
    if not isinstance(generation, dict):
        return ["generation config missing object field: generation"]
    if generation.get("free_sample_count") != 10:
        problems.append("generation.free_sample_count must be 10 for thesis-valid N=10 SE")
    if generation.get("do_sample") is not True:
        problems.append("generation.do_sample must be true for repeated answer sampling")
    if "prompt_rows" in config:
        problems.append("generation config must not include prompt_rows fixture data in the thesis path")
    text = generation_path.read_text(encoding="utf-8")
    forbidden = ("smoke", "exactly five free samples", "deterministic_fixture")
    for needle in forbidden:
        if needle in text:
            problems.append(f"generation config contains stale fixture/N=5 wording: {needle}")
    return problems


def validate(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    for phrase in REQUIRED_PHRASES:
        if phrase not in text:
            problems.append(f"missing required pipeline phrase: {phrase}")
    for marker in REQUIRED_STAGES:
        if marker not in text:
            problems.append(f"missing pipeline stage marker: {marker}")
    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            problems.append(f"forbidden downstream thesis wording in experiment pipeline: {phrase}")
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        if match:
            problems.append(f"forbidden pipeline wording: {match.group(0)}")
    for command in REQUIRED_COMMANDS:
        if command not in text:
            problems.append(f"missing required pipeline command/path: {command}")
    problems.extend(validate_generation_config(path))
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
