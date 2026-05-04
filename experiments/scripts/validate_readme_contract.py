#!/usr/bin/env python3
"""Validate the README-first experiment contract."""

from __future__ import annotations

import sys
from pathlib import Path


REQUIRED_SECTIONS = [
    "## 1. Method name and thesis claims",
    "## 2. Label contract",
    "## 3. Dataset contract",
    "## 4. Feature contract",
    "## 5. Baseline contract",
    "## 6. PC Probe guardrail",
    "## 7. Metrics contract",
    "## 8. Output schema contract",
    "## 9. Hexagonal architecture convention",
    "## 10. Reference-status caveats",
]

REQUIRED_ITEMS = [
    ("method name", "Corpus-Grounded Selective Fusion Detector"),
    (
        "high-diversity Semantic Entropy thesis claim",
        "High-diversity hallucinations are targeted by Semantic Entropy.",
    ),
    (
        "low-diversity Semantic Energy thesis claim",
        "Low-diversity hallucinations are targeted by Semantic Energy.",
    ),
    (
        "QuCo-RAG selective fusion thesis claim",
        "QuCo-RAG-style corpus statistics support selective fusion",
    ),
    ("NORMAL label", "NORMAL"),
    ("HIGH_DIVERSITY label", "HIGH_DIVERSITY"),
    ("LOW_DIVERSITY label", "LOW_DIVERSITY"),
    ("AMBIGUOUS_INCORRECT label", "AMBIGUOUS_INCORRECT"),
    ("NORMAL label rule", "correct -> NORMAL"),
    ("HIGH_DIVERSITY label rule", "incorrect and SE > 0.5 -> HIGH_DIVERSITY"),
    ("LOW_DIVERSITY label rule", "incorrect and SE <= 0.1 -> LOW_DIVERSITY"),
    (
        "AMBIGUOUS_INCORRECT label rule",
        "incorrect and 0.1 < SE <= 0.5 -> AMBIGUOUS_INCORRECT",
    ),
    ("ambiguous incorrect guardrail", "Ambiguous incorrect samples are never treated as normal."),
    (
        "fixed operational labels guardrail",
        "The four operational labels above keep their fixed thresholds.",
    ),
    (
        "row-level operational type label marking",
        "Every row-level output must include an explicit operational type label marker for every sample",
    ),
    ("TruthfulQA dataset", "TruthfulQA"),
    ("TriviaQA dataset", "TriviaQA"),
    ("HaluEval-QA dataset", "HaluEval-QA"),
    ("Natural Questions future dataset", "Natural Questions"),
    ("HotpotQA future dataset", "HotpotQA"),
    ("FEVER future dataset", "FEVER"),
    ("BioASQ future dataset", "BioASQ"),
    (
        "dataset expansion policy",
        "Dataset expansion is allowed when clearly configured and reported",
    ),
    (
        "mandatory core datasets policy",
        "TruthfulQA, TriviaQA, and HaluEval-QA remain mandatory core datasets",
    ),
    (
        "stretch datasets promotion policy",
        "remain future or stretch datasets unless a later checked-in config explicitly promotes them",
    ),
    ("semantic_entropy feature", "semantic_entropy"),
    ("cluster_count feature", "cluster_count"),
    ("semantic energy feature option", "semantic_energy_boltzmann"),
    ("semantic energy proxy feature option", "semantic_energy_proxy"),
    ("logit_variance feature", "logit_variance"),
    ("confidence_margin feature", "confidence_margin"),
    ("entity_frequency feature", "entity_frequency"),
    ("entity_pair_cooccurrence feature", "entity_pair_cooccurrence"),
    ("low_frequency_entity_flag feature", "low_frequency_entity_flag"),
    ("SE-only baseline", "SE-only"),
    ("Energy-only baseline", "Energy-only"),
    ("corpus-risk-only baseline", "corpus-risk-only"),
    ("fixed linear 0.1/0.9 baseline", "fixed linear 0.1/0.9"),
    ("fixed linear 0.5/0.5 baseline", "fixed linear 0.5/0.5"),
    ("fixed linear 0.9/0.1 baseline", "fixed linear 0.9/0.1"),
    ("hard cascade baseline", "hard cascade"),
    ("learned fusion without corpus baseline", "learned fusion without corpus"),
    ("learned fusion with corpus baseline", "learned fusion with corpus"),
    (
        "PC Probe reference-only guardrail",
        "PC Probe is reference-only and not implemented. Hidden-state/probe features are excluded.",
    ),
    (
        "corpus direct learned-fusion guardrail",
        "corpus features as direct learned-fusion features, not a scalar coverage-only proposed method",
    ),
    ("AUROC metric", "AUROC"),
    ("AUPRC metric", "AUPRC"),
    ("accuracy metric", "accuracy"),
    ("precision metric", "precision"),
    ("recall metric", "recall"),
    ("F1 metric", "F1"),
    (
        "fine-grained SE bin analysis",
        "Fine-grained SE bin analysis for crossover and threshold sensitivity",
    ),
    (
        "SE bin coverage",
        "deciles or config-defined bins spanning `[0, +inf)`",
    ),
    (
        "SE bins are analysis only",
        "Fine-grained SE bins are for analysis, crossover checks, and threshold sensitivity only.",
    ),
    ("run_id output element", "run_id"),
    ("method_name output element", "method_name"),
    ("dataset output element", "dataset"),
    ("split_id output element", "split_id"),
    ("sample_id output element", "sample_id"),
    ("label output element", "label"),
    ("features output element", "features"),
    ("prediction_score output element", "prediction_score"),
    ("prediction_label output element", "prediction_label"),
    ("metrics output element", "metrics"),
    ("feature_importance output element", "feature_importance"),
    ("formula_manifest_ref output element", "formula_manifest_ref"),
    ("dataset_manifest_ref output element", "dataset_manifest_ref"),
    ("domain architecture element", "domain"),
    ("ports architecture element", "ports"),
    ("adapters architecture element", "adapters"),
    ("application architecture element", "application"),
    ("scripts architecture element", "scripts"),
    ("configs architecture element", "configs"),
    ("manifests architecture element", "manifests"),
    ("results architecture element", "results"),
    ("literature architecture element", "literature"),
    ("QuCo-RAG caveat", "QuCo-RAG is used as corpus frequency and entity-pair co-occurrence inspiration."),
    ("Ma Semantic Energy caveat", "Ma Semantic Energy is used as the Semantic Energy source or motivation."),
    ("Phillips/PC Probe caveat", "Phillips/PC Probe is reference-only."),
]


def validate(readme_path: Path) -> list[str]:
    if not readme_path.exists():
        return [f"README path does not exist: {readme_path}"]

    text = readme_path.read_text(encoding="utf-8")
    missing: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in text:
            missing.append(f"section: {section}")

    for name, needle in REQUIRED_ITEMS:
        if needle not in text:
            missing.append(f"{name}: {needle}")

    if "PC Probe" in text and "implemented baseline" in text:
        missing.append("PC Probe guardrail conflict: README suggests an implemented baseline")

    return missing


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python3 experiments/scripts/validate_readme_contract.py <README.md>", file=sys.stderr)
        return 2

    readme_path = Path(argv[1])
    missing = validate(readme_path)
    if missing:
        print(f"README contract validation failed for {readme_path}.", file=sys.stderr)
        print("Missing or invalid contract items:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 1

    print(f"README contract validation passed: {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
