#!/usr/bin/env python3
"""Derive reproducibility manifests from upstream experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.validate_datasets import load_json


EXPECTED_BRANCHES: tuple[dict[str, Any], ...] = (
    {
        "branch": "exp01_truthfulqa",
        "category": "row_results",
        "required": True,
        "dataset_id": "truthfulqa",
        "dataset_name": "TruthfulQA",
        "paths": ["experiment_notes/exp01_truthfulqa/results.json"],
    },
    {
        "branch": "exp02_halueval",
        "category": "row_results",
        "required": False,
        "dataset_id": "halueval_qa",
        "dataset_name": "HaluEval-QA",
        "paths": ["experiment_notes/exp02_halueval/results.json"],
    },
    {
        "branch": "exp04_corpus_adaptive",
        "category": "corpus_augmented",
        "required": True,
        "dataset_id": "mixed",
        "dataset_name": "mixed",
        "paths": [
            "experiment_notes/exp04_corpus_adaptive/analysis_results_latest.json",
            "experiment_notes/exp04_corpus_adaptive/analysis_results_20260120_230157.json",
            "experiment_notes/exp04_corpus_adaptive/truthfulqa_with_corpus.json",
            "experiment_notes/exp04_corpus_adaptive/halueval_with_corpus.json",
        ],
    },
    {
        "branch": "exp05_quintile_analysis",
        "category": "corpus_augmented",
        "required": True,
        "dataset_id": "mixed",
        "dataset_name": "mixed",
        "paths": [
            "experiment_notes/exp05_quintile_analysis/truthfulqa_with_corpus.json",
            "experiment_notes/exp05_quintile_analysis/halueval_with_corpus.json",
        ],
    },
    {
        "branch": "exp07_zero_se_analysis",
        "category": "row_results_and_analysis",
        "required": True,
        "dataset_id": "mixed",
        "dataset_name": "mixed",
        "paths": [
            "experiment_notes/exp07_zero_se_analysis/analysis_results.json",
            "experiment_notes/exp07_zero_se_analysis/results_triviaqa.json",
            "experiment_notes/exp07_zero_se_analysis/results_naturalquestions.json",
            "experiment_notes/exp07_zero_se_analysis/results_halueval_dialogue.json",
            "experiment_notes/exp07_zero_se_analysis/results_triviaqa_llm_judge.json",
            "experiment_notes/exp07_zero_se_analysis/results_naturalquestions_llm_judge.json",
        ],
    },
    {
        "branch": "exp08_robustness",
        "category": "analysis",
        "required": True,
        "dataset_id": "mixed",
        "dataset_name": "mixed",
        "paths": ["experiment_notes/exp08_robustness/robustness_results.json"],
    },
    {
        "branch": "exp09_thesis_final",
        "category": "row_results",
        "required": False,
        "dataset_id": "truthfulqa",
        "dataset_name": "TruthfulQA",
        "paths": [
            "experiment_notes/exp09_thesis_final/results_full.json",
            "experiment_notes/exp09_thesis_final/results_checkpoint.json",
        ],
    },
    {
        "branch": "exp10_thesis_complete",
        "category": "row_results_and_analysis",
        "required": True,
        "dataset_id": "truthfulqa",
        "dataset_name": "TruthfulQA",
        "paths": [
            "experiment_notes/exp10_thesis_complete/results.json",
            "experiment_notes/exp10_thesis_complete/analysis.json",
        ],
    },
)

LABEL_KEYS = {
    "is_hallucination",
    "label",
    "labels",
    "correctness",
    "correct",
    "hallucination_label",
    "hallucination_annotations",
    "right_answer",
    "gold_answers",
    "correct_answers",
}
LABEL_METRIC_KEYS = {
    "n_hallucination",
    "n_hallucinations",
    "n_normal",
    "hallucination_rate",
    "hall_rate_overall",
    "hall_rate_in_zero_se",
    "n_hall_in_zero_se",
    "n_normal_in_zero_se",
}
CORPUS_KEYS = {
    "corpus_stats",
    "corpus",
    "entity_frequency",
    "entity_frequencies",
    "entity_pair_cooccurrence",
    "cooccurrence",
    "coverage",
    "low_frequency",
    "low_frequency_entity_flag",
    "freq_score",
    "cooc_score",
}
LOGIT_HINT_KEYS = {"logits", "full_logits", "token_logits", "selected_logits", "top_logits", "logprobs"}
FULL_LOGIT_HINT_KEYS = {"full_logits", "token_logits"}
MODEL_METADATA_KEYS = {
    "llm_model",
    "nli_model",
    "model",
    "model_name",
    "dataset",
    "dataset_name",
    "dataset_split",
    "split",
    "num_responses",
    "num_samples",
    "max_samples",
    "max_new_tokens",
    "temperature",
    "seed",
    "epsilon",
}
SAMPLE_COLLECTION_KEYS = ("samples", "zero_se_analysis", "complementarity_sensitivity", "datasets", "se_bins", "cluster_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to the upstream hallucination_lfe repo")
    parser.add_argument("--out", required=True, help="Output directory for derived manifests")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str) -> str:
    letters = []
    for character in value.lower():
        if character.isalnum():
            letters.append(character)
        else:
            letters.append("_")
    slug = "".join(letters)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def load_dataset_registry() -> tuple[dict[str, dict[str, Any]], str]:
    config_path = ROOT / "experiments" / "configs" / "datasets.yaml"
    config = load_json(config_path)
    registry = {}
    for dataset in config.get("datasets", []):
        registry[slugify(dataset["name"])] = dataset
        hf_id = dataset.get("hf_id")
        if isinstance(hf_id, str) and hf_id.strip():
            registry[slugify(hf_id)] = dataset
    aliases = {
        "truthfulqa": "TruthfulQA",
        "halueval": "HaluEval-QA",
        "halueval_qa": "HaluEval-QA",
        "halueval qa": "HaluEval-QA",
        "halueval_dialogue": None,
        "triviaqa": "TriviaQA",
        "naturalquestions": "Natural Questions",
        "natural_questions": "Natural Questions",
    }
    for alias, target_name in aliases.items():
        if target_name is None:
            continue
        target = registry.get(slugify(target_name))
        if target is not None:
            registry[slugify(alias)] = target
    return registry, str(config.get("registry_name", "unknown_registry"))


def find_dataset_info(payload: dict[str, Any], registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        payload.get("dataset_name"),
        payload.get("dataset"),
    ]
    config = payload.get("config")
    if isinstance(config, dict):
        candidates.extend([config.get("dataset"), config.get("dataset_name")])

    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        normalized = slugify(candidate)
        dataset = registry.get(normalized)
        if dataset is not None:
            return {
                "dataset_id": slugify(dataset["name"]),
                "dataset_name": dataset["name"],
                "registry_role": dataset.get("role"),
                "registry_split_id": dataset.get("split_id"),
                "registry_hf_id": dataset.get("hf_id"),
                "source_dataset_value": candidate,
            }
        return {
            "dataset_id": normalized,
            "dataset_name": candidate,
            "registry_role": None,
            "registry_split_id": None,
            "registry_hf_id": None,
            "source_dataset_value": candidate,
        }

    return {
        "dataset_id": "unknown",
        "dataset_name": "unknown",
        "registry_role": None,
        "registry_split_id": None,
        "registry_hf_id": None,
        "source_dataset_value": None,
    }


def maybe_collection(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    return None


def detect_primary_collection(payload: dict[str, Any]) -> tuple[str | None, list[Any] | None]:
    for key in SAMPLE_COLLECTION_KEYS:
        collection = maybe_collection(payload.get(key))
        if collection is not None:
            return key, collection
    return None, None


def infer_sample_count(payload: dict[str, Any], collection_name: str | None, collection: list[Any] | None) -> int | None:
    for key in ("n_samples", "n_total", "num_samples", "max_samples"):
        value = payload.get(key)
        if isinstance(value, int) and value >= 0:
            return value

    if collection_name == "samples" and collection is not None:
        return len(collection)

    stats = payload.get("statistics")
    if isinstance(stats, dict):
        total = 0
        found = False
        for key in ("n_hallucination", "n_normal"):
            value = stats.get(key)
            if isinstance(value, int):
                total += value
                found = True
        if found:
            return total

    if isinstance(payload.get("datasets"), dict):
        nested_counts = []
        for value in payload["datasets"].values():
            if isinstance(value, dict):
                nested_count = value.get("n_samples")
                if isinstance(nested_count, int) and nested_count >= 0:
                    nested_counts.append(nested_count)
        if nested_counts and all(count == nested_counts[0] for count in nested_counts):
            return nested_counts[0]

    for key in ("zero_se", "zero_se_analysis"):
        values = payload.get(key)
        if isinstance(values, list):
            nested_counts = []
            for item in values:
                if isinstance(item, dict):
                    nested_count = item.get("n")
                    if not isinstance(nested_count, int):
                        nested_count = item.get("n_total")
                    if isinstance(nested_count, int) and nested_count >= 0:
                        nested_counts.append(nested_count)
            if nested_counts and all(count == nested_counts[0] for count in nested_counts):
                return nested_counts[0]

    if collection is not None and collection_name == "datasets":
        return len(collection)

    return None


def extract_field_names(collection: list[Any] | None) -> list[str]:
    if collection is None:
        return []
    names: set[str] = set()
    for item in collection[:25]:
        if isinstance(item, dict):
            names.update(item.keys())
    return sorted(names)


def scalar_sequence(values: Any) -> bool:
    return isinstance(values, list) and bool(values) and all(isinstance(item, (int, float)) for item in values)


def nested_scalar_matrix(values: Any) -> bool:
    return isinstance(values, list) and bool(values) and all(scalar_sequence(item) for item in values)


def detect_logits_in_value(key: str, value: Any) -> tuple[bool, bool]:
    lowered = key.lower()
    has_logits = False
    has_full_logits = False

    if any(hint in lowered for hint in LOGIT_HINT_KEYS):
        has_logits = True
        if any(hint in lowered for hint in FULL_LOGIT_HINT_KEYS) and (scalar_sequence(value) or nested_scalar_matrix(value)):
            has_full_logits = True
        elif nested_scalar_matrix(value):
            has_full_logits = True

    if lowered == "logits" and nested_scalar_matrix(value):
        has_full_logits = True

    return has_logits, has_full_logits


def detect_capabilities(payload: dict[str, Any], sample_fields: list[str], collection: list[Any] | None) -> dict[str, bool]:
    has_logits = False
    has_full_logits = False
    has_corpus_stats = False
    has_labels = False

    def walk_mapping(mapping: dict[str, Any]) -> None:
        nonlocal has_logits, has_full_logits, has_corpus_stats, has_labels
        for key, value in mapping.items():
            lowered = key.lower()
            logits_here, full_logits_here = detect_logits_in_value(lowered, value)
            has_logits = has_logits or logits_here
            has_full_logits = has_full_logits or full_logits_here
            if any(marker in lowered for marker in CORPUS_KEYS):
                has_corpus_stats = True
            if lowered in LABEL_KEYS:
                has_labels = True
            if lowered in LABEL_METRIC_KEYS or "hallucination" in lowered:
                has_labels = True
            if isinstance(value, dict):
                walk_mapping(value)

    walk_mapping(payload)
    if collection is not None:
        for item in collection[:25]:
            if isinstance(item, dict):
                walk_mapping(item)

    for field in sample_fields:
        lowered = field.lower()
        if any(marker in lowered for marker in CORPUS_KEYS):
            has_corpus_stats = True
        if lowered in LABEL_KEYS:
            has_labels = True

    if isinstance(payload.get("statistics"), dict):
        stats = payload["statistics"]
        if any(key in stats for key in ("n_hallucination", "n_normal")):
            has_labels = True

    return {
        "has_logits": has_logits,
        "has_full_logits": has_full_logits,
        "has_corpus_stats": has_corpus_stats,
        "has_labels": has_labels,
    }


def prompt_model_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    config = payload.get("config")
    if isinstance(config, dict):
        for key, value in config.items():
            if key in MODEL_METADATA_KEYS and value is not None:
                metadata[key] = value

    for key in ("dataset_name", "dataset", "timestamp", "experiment"):
        value = payload.get(key)
        if value is not None:
            metadata[key] = value

    return metadata


def collect_metric_keys(payload: dict[str, Any]) -> list[str]:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return sorted(metrics.keys())
    return []


def collect_statistics_keys(payload: dict[str, Any]) -> list[str]:
    statistics = payload.get("statistics")
    if isinstance(statistics, dict):
        return sorted(statistics.keys())
    return []


def infer_artifact_kind(relative_path: str, payload: dict[str, Any], collection_name: str | None) -> str:
    if collection_name == "samples":
        if "with_corpus" in relative_path:
            return "row_results_with_corpus"
        return "row_results"
    if "with_corpus" in relative_path:
        return "row_results_with_corpus"
    if "analysis" in relative_path or collection_name in {"zero_se_analysis", "complementarity_sensitivity", "se_bins", "cluster_analysis"}:
        return "analysis"
    if collection_name == "datasets":
        return "dataset_summary"
    if isinstance(payload.get("datasets"), dict):
        return "dataset_summary"
    return "unknown"


def build_artifact_record(
    *,
    source_root: Path,
    relative_path: str,
    branch: str,
    branch_category: str,
    payload: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = source_root / relative_path
    collection_name, collection = detect_primary_collection(payload)
    sample_count = infer_sample_count(payload, collection_name, collection)
    sample_fields = extract_field_names(collection)
    capabilities = detect_capabilities(payload, sample_fields, collection)
    dataset_info = find_dataset_info(payload, registry)
    prompt_metadata = prompt_model_metadata(payload)
    artifact_kind = infer_artifact_kind(relative_path, payload, collection_name)
    artifact_id = slugify(f"{branch}_{path.stem}_{dataset_info['dataset_id']}")

    return {
        "artifact_id": artifact_id,
        "branch": branch,
        "branch_category": branch_category,
        "artifact_kind": artifact_kind,
        "relative_path": relative_path,
        "absolute_path": str(path.resolve()),
        "source_root_relative_dir": str(path.parent.relative_to(source_root)),
        "source_root_name": source_root.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "top_level_fields": sorted(payload.keys()),
        "primary_collection": collection_name,
        "sample_count": sample_count,
        "sample_fields": sample_fields,
        "metric_fields": collect_metric_keys(payload),
        "statistics_fields": collect_statistics_keys(payload),
        "dataset": dataset_info,
        "prompt_model_metadata": prompt_metadata,
        "availability": capabilities,
        "metadata": {
            "timestamp": payload.get("timestamp"),
            "experiment": payload.get("experiment"),
            "config": payload.get("config") if isinstance(payload.get("config"), dict) else None,
        },
    }


def branch_status_records(source_root: Path) -> list[dict[str, Any]]:
    statuses = []
    for branch in EXPECTED_BRANCHES:
        present = []
        missing = []
        for relative_path in branch["paths"]:
            if (source_root / relative_path).exists():
                present.append(relative_path)
            else:
                missing.append(relative_path)
        statuses.append(
            {
                "branch": branch["branch"],
                "category": branch["category"],
                "required": branch["required"],
                "dataset_id": branch["dataset_id"],
                "dataset_name": branch["dataset_name"],
                "expected_paths": branch["paths"],
                "present_paths": present,
                "missing_paths": missing,
                "status": "complete" if not missing else ("partial" if present else "missing"),
            }
        )
    return statuses


def main() -> int:
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    out_dir = Path(args.out)

    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit("source path not found")

    out_dir.mkdir(parents=True, exist_ok=True)

    registry, registry_name = load_dataset_registry()
    artifacts: list[dict[str, Any]] = []

    for branch in EXPECTED_BRANCHES:
        for relative_path in branch["paths"]:
            path = source_root / relative_path
            if not path.exists() or not path.is_file():
                continue
            payload = load_json(path)
            artifacts.append(
                build_artifact_record(
                    source_root=source_root,
                    relative_path=relative_path,
                    branch=branch["branch"],
                    branch_category=branch["category"],
                    payload=payload,
                    registry=registry,
                )
            )

    artifacts.sort(key=lambda item: (item["branch"], item["relative_path"]))
    branch_status = branch_status_records(source_root)

    manifest = {
        "manifest_version": 1,
        "manifest_kind": "upstream_artifacts",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_root_name": source_root.name,
        "source_input": args.source,
        "dataset_registry_name": registry_name,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "expected_branches": branch_status,
    }
    summary = {
        "manifest_version": 1,
        "manifest_kind": "upstream_branch_status",
        "generated_at": manifest["generated_at"],
        "source_root": str(source_root),
        "dataset_registry_name": registry_name,
        "artifact_count": len(artifacts),
        "branch_status": branch_status,
    }

    write_json(out_dir / "upstream_artifacts_manifest.json", manifest)
    write_json(out_dir / "upstream_branch_status.json", summary)

    print(
        json.dumps(
            {
                "manifest_path": str(out_dir / "upstream_artifacts_manifest.json"),
                "branch_status_path": str(out_dir / "upstream_branch_status.json"),
                "artifact_count": len(artifacts),
                "branches": [record["branch"] for record in branch_status],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
