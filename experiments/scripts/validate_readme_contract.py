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
    ("method name", "Corpus-Conditioned Hallucination Metric Reliability Study"),
    (
        "corpus axis reliability thesis claim",
        "Corpus entity frequency and entity-pair co-occurrence define continuous conditioning axes for hallucination metric reliability.",
    ),
    (
        "paper faithful metric reliability thesis claim",
        "Paper-faithful Semantic Entropy and Semantic Energy must be evaluated as metric families whose reliability can change across corpus-axis bins",
    ),
    (
        "condition-aware fusion thesis claim",
        "Condition-aware fusion is evaluated by comparing global fusion against corpus-bin-aware or axis-interaction fusion",
    ),
    ("not RAG system caveat", "The proposed contribution is not a RAG system"),
    ("low frequency not label caveat", "Low entity frequency or zero entity-pair co-occurrence is not itself a hallucination label."),
    ("is_correct label", "is_correct"),
    ("is_hallucination label", "is_hallucination"),
    ("candidate_label field", "candidate_label"),
    (
        "archived operational label caveat",
        "may appear only as archived operational labels from the earlier diagnostic run",
    ),
    (
        "paired active dataset contract",
        "The active thesis dataset is exactly one paired discriminative experiment dataset built from:",
    ),
    ("TruthfulQA active paired dataset", "- TruthfulQA"),
    ("HaluEval-QA active paired dataset", "- HaluEval-QA"),
    ("exactly two candidate rows contract", "Each prompt contributes exactly two candidate rows"),
    ("teacher-forced fixed candidate scoring contract", "The model scores those fixed candidates with teacher forcing."),
    ("no generated-answer relabeling contract", "It does not generate an answer for later correctness labeling."),
    ("excluded datasets heading", "Excluded datasets:"),
    ("TriviaQA excluded dataset", "- TriviaQA"),
    ("Natural Questions excluded dataset", "- Natural Questions"),
    ("HotpotQA excluded dataset", "- HotpotQA"),
    ("FEVER excluded dataset", "- FEVER"),
    ("BioASQ excluded dataset", "- BioASQ"),
    (
        "excluded datasets clean-pair rationale",
        "do not provide clean dataset-level `(right_answer, hallucinated_answer)` candidate pairs",
    ),
    ("semantic entropy likelihood feature", "semantic_entropy_nli_likelihood"),
    ("semantic entropy cluster feature", "semantic_entropy_cluster_count"),
    ("semantic entropy discrete feature", "semantic_entropy_discrete_cluster_entropy"),
    ("semantic energy cluster feature", "semantic_energy_cluster_uncertainty"),
    ("semantic energy sample feature", "semantic_energy_sample_energy"),
    ("mean nll diagnostic", "mean_negative_log_probability"),
    ("logit variance diagnostic", "logit_variance"),
    ("confidence margin diagnostic", "confidence_margin"),
    ("boltzmann diagnostic", "semantic_energy_boltzmann_diagnostic"),
    ("entity frequency feature", "entity_frequency"),
    ("entity frequency axis", "entity_frequency_axis"),
    ("entity pair cooccurrence feature", "entity_pair_cooccurrence"),
    ("entity pair cooccurrence axis", "entity_pair_cooccurrence_axis"),
    ("corpus bin", "corpus_axis_bin"),
    ("N=10 SE requirement", "N=10 answer-only samples"),
    ("NLI SE requirement", "NLI-based semantic clustering"),
    ("likelihood SE requirement", "likelihood-based cluster probabilities"),
    ("Semantic Energy cluster requirement", "cluster-level aggregation"),
    ("corpus condition not label", "not direct correctness labels"),
    ("SE-only baseline", "SE-only"),
    ("Energy-only baseline", "Energy-only"),
    ("logit diagnostic baseline", "logit-diagnostic-only"),
    ("corpus-axis-only baseline", "corpus-axis-only"),
    ("global fusion without corpus", "global learned fusion without corpus axis"),
    ("global fusion with corpus", "global learned fusion with corpus axis"),
    ("corpus-bin feature selection", "corpus-bin feature selection"),
    ("corpus-bin weighted fusion", "corpus-bin weighted fusion"),
    ("axis interaction logistic fusion", "axis-interaction logistic fusion"),
    ("PC Probe reference-only guardrail", "PC Probe is reference-only and not implemented. Hidden-state/probe features are excluded."),
    ("AUROC metric", "AUROC"),
    ("AUPRC metric", "AUPRC"),
    ("paired win rate metric", "paired win rate"),
    ("prompt bootstrap metric", "Prompt-grouped bootstrap confidence intervals"),
    ("corpus bin metrics", "per corpus-axis bin"),
    ("feature alignment output", "feature_alignment"),
    ("corpus_axis output element", "corpus_axis"),
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

FORBIDDEN_ITEMS = [
    "The method name is exactly `Corpus-Grounded Selective Fusion Detector`.",
    "High-diversity hallucinations are targeted by Semantic Entropy.",
    "Low-diversity hallucinations are targeted by Semantic Energy.",
    "The four operational labels above keep their fixed thresholds.",
    "corpus features as direct learned-fusion features, not a scalar coverage-only proposed method",
    "The main method comparison is learned fusion with corpus against learned fusion without corpus",
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

    for needle in FORBIDDEN_ITEMS:
        if needle in text:
            missing.append(f"stale README contract item: {needle}")

    forbidden_conflicts = [
        (
            "TriviaQA active/core/mandatory conflict",
            [
                "TriviaQA active dataset",
                "active TriviaQA",
                "TriviaQA is active",
                "TriviaQA core dataset",
                "core TriviaQA",
                "TriviaQA is core",
                "TriviaQA mandatory",
                "mandatory TriviaQA",
                "TruthfulQA, TriviaQA, and HaluEval-QA remain mandatory core datasets",
            ],
        ),
        (
            "LLM-as-judge or heuristic label construction conflict",
            [
                "LLM-as-judge label construction",
                "LLM as judge label construction",
                "judge-based label construction is allowed",
                "heuristic matching is allowed",
                "heuristic matching label construction",
                "synthetic hallucinated candidates are allowed",
                "generates an answer for later correctness labeling",
                "may generate an answer for later correctness labeling",
                "can generate an answer for later correctness labeling",
            ],
        ),
        (
            "RAG thesis framing conflict",
            [
                "RAG system thesis",
                "retrieval trigger thesis",
                "retrieval trigger is the main contribution",
            ],
        ),
    ]

    for conflict_name, needles in forbidden_conflicts:
        for needle in needles:
            if needle in text:
                missing.append(f"{conflict_name}: {needle}")

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

    print(f"README contract validation passed for {readme_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
