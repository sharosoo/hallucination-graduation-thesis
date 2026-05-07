"""Application-layer helpers for thesis-valid feature-table export and validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from experiments.adapters.corpus_features import (
    load_json,
    read_analysis_bin_config,
    read_feature_rows,
    select_analysis_bin,
    write_feature_artifact,
    write_json,
)
from experiments.domain import AnalysisBin, FeatureRole, TypeLabel
from experiments.scripts.stage_control import FEATURE_TABLE_SCHEMA_VERSION, add_schema_version

EXPECTED_TYPE_LABELS = tuple(label.value for label in TypeLabel)
CORPUS_FEATURES_PATH = Path("corpus_features.parquet")
ENERGY_FEATURES_PATH = Path("energy_features.parquet")
SEMANTIC_ENTROPY_FEATURES_PATH = Path("semantic_entropy_features.parquet")
CORRECTNESS_DATASET_PATH = Path("correctness/data/correctness_judgments.jsonl")

FORBIDDEN_FEATURE_KEYS = {
    "candidate_role",
    "candidate_label",
    "label",
    "archived_type_label",
    "is_correct",
    "is_hallucination",
    "label_source",
    "gold",
    "gold_answer",
    "gold_answers",
    "reference_answer",
    "right_answer",
    "correct_answers",
    "hallucinated_answer",
    "raw_annotation",
    "raw_annotations",
    "raw_judge_response",
    "judge_name",
    "judge_mode",
    "rationale",
}
FORBIDDEN_PROVENANCE_HINTS = (
    "gold",
    "correct_answers",
    "gold_answers",
    "right_answer",
    "hallucinated_answer",
    "correctness",
    "annotation",
    "is_hallucination",
    "is_correct",
    "candidate_label",
    "label",
)
PAPER_FAITHFUL_FEATURES = (
    "semantic_entropy_nli_likelihood",
    "semantic_entropy_cluster_count",
    "semantic_entropy_discrete_cluster_entropy",
    "semantic_energy_cluster_uncertainty",
    "semantic_energy_sample_energy",
)
ADAPTED_FEATURES = (
    "semantic_entropy",
    "cluster_count",
    "semantic_energy_boltzmann",
    "entity_frequency_mean",
    "entity_frequency_min",
    "coverage_score",
    "corpus_risk_only",
)
DIAGNOSTIC_FEATURES = (
    "semantic_energy_boltzmann_diagnostic",
    "mean_negative_log_probability",
    "logit_variance",
    "confidence_margin",
)
EXTERNAL_CORPUS_FEATURES = (
    "entity_frequency",
    "entity_frequency_axis",
    "entity_pair_cooccurrence",
    "entity_pair_cooccurrence_axis",
    "low_frequency_entity_flag",
    "zero_cooccurrence_flag",
    "corpus_axis_bin",
    "corpus_axis_bin_5",
)
ANALYSIS_ONLY_FIELDS = (
    "archived_type_label",
    "is_correct",
    "is_hallucination",
    "candidate_label",
)
TRAINABLE_FEATURE_ORDER = (
    *PAPER_FAITHFUL_FEATURES,
    "semantic_entropy",
    "cluster_count",
    "semantic_energy_boltzmann",
    *DIAGNOSTIC_FEATURES,
    "entity_frequency",
    "entity_frequency_axis",
    "entity_frequency_mean",
    "entity_frequency_min",
    "entity_pair_cooccurrence",
    "entity_pair_cooccurrence_axis",
    "low_frequency_entity_flag",
    "zero_cooccurrence_flag",
    "coverage_score",
    "corpus_risk_only",
    "corpus_axis_bin",
    "corpus_axis_bin_5",
)
PROMPT_BROADCAST_FEATURES = (
    "semantic_entropy_nli_likelihood",
    "semantic_entropy_cluster_count",
    "semantic_entropy_discrete_cluster_entropy",
    "semantic_entropy",
    "cluster_count",
    "semantic_energy_cluster_uncertainty",
    "semantic_energy_sample_energy",
    "semantic_energy_boltzmann",
)
FEATURE_ALIGNMENT_CATEGORY_BY_NAME = {
    **{name: "paper-faithful" for name in PAPER_FAITHFUL_FEATURES},
    **{name: "adapted" for name in ADAPTED_FEATURES},
    **{name: "diagnostic" for name in DIAGNOSTIC_FEATURES},
    **{name: "external_corpus" for name in EXTERNAL_CORPUS_FEATURES},
    **{name: "analysis-only" for name in ANALYSIS_ONLY_FIELDS},
}


@dataclass(frozen=True)
class CandidateLabelRecord:
    prompt_id: str
    candidate_id: str
    pair_id: str
    candidate_role: str
    candidate_text: str
    is_correct: bool
    label_source: str
    dataset: str
    split_id: str
    sample_id: str

    @property
    def is_hallucination(self) -> bool:
        return not self.is_correct

    @classmethod
    def from_row(cls, row: object) -> "CandidateLabelRecord | None":
        if not isinstance(row, dict):
            return None
        candidate_id = str(row.get("candidate_id", "")).strip()
        prompt_id = str(row.get("prompt_id", "")).strip()
        if not candidate_id or not prompt_id:
            return None
        return cls(
            prompt_id=prompt_id,
            candidate_id=candidate_id,
            pair_id=str(row.get("pair_id", "")).strip(),
            candidate_role=str(row.get("candidate_role", "")).strip(),
            candidate_text=str(row.get("candidate_text", "")).strip(),
            is_correct=bool(row.get("is_correct", False)),
            label_source=str(row.get("label_source", "")).strip(),
            dataset=str(row.get("dataset", "")).strip(),
            split_id=str(row.get("split_id", "")).strip(),
            sample_id=str(row.get("sample_id", "")).strip(),
        )


def loads_json_object(line: str) -> object:
    return json.loads(line)


def row_features(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features")
    return features if isinstance(features, dict) else {}


def _required_finite_float(value: Any, *, field_name: str, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Missing numeric {field_name} for {context}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {field_name} for {context}: {value!r}") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError(f"Non-finite numeric {field_name} for {context}: {value!r}")
    return numeric


def _required_nonnegative_int(value: Any, *, field_name: str, context: str) -> int:
    numeric = _required_finite_float(value, field_name=field_name, context=context)
    if numeric < 0 or int(numeric) != numeric:
        raise ValueError(f"Invalid integer {field_name} for {context}: {value!r}")
    return int(numeric)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def assign_operational_label(is_correct: bool, semantic_entropy: float) -> TypeLabel:
    if is_correct:
        return TypeLabel.NORMAL
    if semantic_entropy <= 0.1:
        return TypeLabel.LOW_DIVERSITY
    if semantic_entropy <= 0.5:
        return TypeLabel.AMBIGUOUS_INCORRECT
    return TypeLabel.HIGH_DIVERSITY


def serialize_analysis_bin(bin_spec: AnalysisBin | None) -> dict[str, Any] | None:
    if bin_spec is None:
        return None
    return {
        "scheme_name": bin_spec.scheme_name,
        "bin_id": bin_spec.bin_id,
        "lower_bound": bin_spec.lower_bound,
        "upper_bound": bin_spec.upper_bound,
        "includes_upper_bound": bin_spec.includes_upper_bound,
        "note": bin_spec.note,
    }


def build_boundary_self_check() -> dict[str, Any]:
    cases = []
    for semantic_entropy, expected in (
        (0.1, TypeLabel.LOW_DIVERSITY),
        (0.5, TypeLabel.AMBIGUOUS_INCORRECT),
        (0.5001, TypeLabel.HIGH_DIVERSITY),
    ):
        actual = assign_operational_label(False, semantic_entropy)
        cases.append(
            {
                "semantic_entropy": semantic_entropy,
                "expected_label": expected.value,
                "actual_label": actual.value,
                "passes": actual is expected,
            }
        )
    return {
        "cases": cases,
        "all_pass": all(case["passes"] for case in cases),
    }


def _resolve_artifact_path(requested_path: Path) -> Path:
    storage_report_path = requested_path.with_suffix(requested_path.suffix + ".storage.json")
    if storage_report_path.exists():
        payload = load_json(storage_report_path)
        materialized = payload.get("materialized_path")
        if materialized:
            return Path(str(materialized))
    return requested_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = loads_json_object(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_candidate_labels(results_dir: Path) -> dict[str, CandidateLabelRecord]:
    correctness_path = results_dir / CORRECTNESS_DATASET_PATH
    if not correctness_path.exists():
        raise FileNotFoundError(f"Missing candidate correctness rows: {correctness_path}")
    records: dict[str, CandidateLabelRecord] = {}
    for row in _read_jsonl(correctness_path):
        record = CandidateLabelRecord.from_row(row)
        if record is None:
            continue
        if record.candidate_id in records:
            raise ValueError(f"Duplicate candidate correctness row: {record.candidate_id}")
        records[record.candidate_id] = record
    return records


def load_prompt_semantic_entropy(results_dir: Path) -> dict[str, dict[str, Any]]:
    rows, _storage = read_feature_rows(results_dir / SEMANTIC_ENTROPY_FEATURES_PATH)
    by_prompt: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt_id = str(row.get("prompt_id", "")).strip()
        if not prompt_id:
            continue
        if prompt_id in by_prompt:
            raise ValueError(f"Duplicate prompt_id in semantic entropy rows: {prompt_id}")
        by_prompt[prompt_id] = row
    return by_prompt


def load_energy_rows(results_dir: Path) -> dict[str, dict[str, Any]]:
    rows, _storage = read_feature_rows(results_dir / ENERGY_FEATURES_PATH)
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        if candidate_id in by_candidate:
            raise ValueError(f"Duplicate candidate_id in energy rows: {candidate_id}")
        by_candidate[candidate_id] = row
    return by_candidate


def load_corpus_feature_rows(corpus_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    return read_feature_rows(corpus_path)


def _feature_value(source: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in source:
            return source.get(name)
    features = row_features(source)
    for name in names:
        if name in features:
            return features.get(name)
    return None


def _canonical_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _feature_alignment_entry(feature_name: str) -> dict[str, Any]:
    category = FEATURE_ALIGNMENT_CATEGORY_BY_NAME[feature_name]
    entry = {
        "category": category,
        "trainable": category != "analysis-only",
    }
    if feature_name in PAPER_FAITHFUL_FEATURES:
        entry["note"] = "Required thesis-facing paper-faithful signal."
    elif feature_name in DIAGNOSTIC_FEATURES:
        entry["note"] = "Candidate-level diagnostic signal kept separate from paper-faithful uncertainty features."
    elif feature_name in EXTERNAL_CORPUS_FEATURES:
        entry["note"] = "Corpus-support axis feature derived from external corpus count provenance."
    elif feature_name in ADAPTED_FEATURES:
        entry["note"] = "Compatibility or adapted feature retained for downstream stability."
    else:
        entry["note"] = "Analysis-only metadata; never use as a trainable feature."
    return entry


def build_feature_alignment() -> dict[str, dict[str, Any]]:
    return {
        feature_name: _feature_alignment_entry(feature_name)
        for feature_name in (
            *TRAINABLE_FEATURE_ORDER,
            *ANALYSIS_ONLY_FIELDS,
        )
    }


def _feature_alignment_summary() -> dict[str, int]:
    counts: Counter[str] = Counter()
    for feature_name in (*TRAINABLE_FEATURE_ORDER, *ANALYSIS_ONLY_FIELDS):
        counts[FEATURE_ALIGNMENT_CATEGORY_BY_NAME[feature_name]] += 1
    return dict(sorted(counts.items()))


def _build_energy_metadata(energy_row: dict[str, Any]) -> dict[str, Any]:
    energy_provenance = energy_row.get("energy_provenance")
    provenance = energy_provenance if isinstance(energy_provenance, dict) else {}
    features = row_features(energy_row)
    paper_faithful_available = (
        features.get("semantic_energy_cluster_uncertainty") is not None
        and features.get("semantic_energy_sample_energy") is not None
    )
    diagnostic_available = features.get("semantic_energy_boltzmann_diagnostic") is not None
    return {
        "status": "paper_faithful_prompt_level_broadcast" if paper_faithful_available else "paper_faithful_energy_unavailable",
        "paper_faithful_energy_available": paper_faithful_available,
        "true_boltzmann_available": False,
        "candidate_boltzmann_diagnostic_available": diagnostic_available,
        "full_logits_required": False,
        "rerun_required": not paper_faithful_available,
        "not_for_thesis_claims": not paper_faithful_available,
        "energy_granularity": provenance.get("energy_granularity", "prompt_level_broadcast_to_candidate_rows"),
        "source_artifact_path": energy_row.get("source_artifact_path"),
        "source_artifact_id": energy_row.get("artifact_id", energy_row.get("run_id")),
        "energy_kind": provenance.get("energy_kind", features.get("energy_kind")),
        "note": provenance.get(
            "candidate_diagnostics_note",
            "semantic_energy_cluster_uncertainty and semantic_energy_sample_energy are paper-faithful prompt-level broadcasts; semantic_energy_boltzmann_diagnostic and related logit signals remain candidate diagnostics.",
        ),
    }


def _build_feature_provenance(
    source_row: dict[str, Any],
    semantic_entropy_row: dict[str, Any],
    energy_row: dict[str, Any],
    *,
    results_dir: Path,
) -> list[dict[str, Any]]:
    corpus_features = row_features(source_row)
    corpus_axis = source_row.get("corpus_axis")
    corpus_source = corpus_features.get("corpus_source")
    corpus_status = corpus_features.get("corpus_status")
    corpus_backend = corpus_axis.get("backend_id") if isinstance(corpus_axis, dict) else None
    semantic_path = str(results_dir / SEMANTIC_ENTROPY_FEATURES_PATH)
    energy_path = str(results_dir / ENERGY_FEATURES_PATH)
    source_artifact_path = str(source_row.get("source_artifact_path") or _resolve_artifact_path(results_dir / CORPUS_FEATURES_PATH))

    provenance: list[dict[str, Any]] = []
    for feature_name in PAPER_FAITHFUL_FEATURES:
        source_note = (
            "prompt-level NLI Semantic Entropy broadcast by prompt_id"
            if feature_name.startswith("semantic_entropy_")
            else "paper-faithful sampled-response Semantic Energy broadcast by prompt_id"
        )
        provenance.append(
            {
                "feature_name": feature_name,
                "role": FeatureRole.TRAINABLE.value,
                "source": source_note,
                "source_artifact_path": semantic_path if feature_name.startswith("semantic_entropy_") else energy_path,
                "depends_on_correctness": False,
                "trainable": True,
            }
        )

    provenance.extend(
        [
            {
                "feature_name": "semantic_entropy",
                "role": FeatureRole.TRAINABLE.value,
                "source": "legacy compatibility alias of semantic_entropy_nli_likelihood",
                "source_artifact_path": semantic_path,
                "depends_on_correctness": False,
                "trainable": True,
            },
            {
                "feature_name": "cluster_count",
                "role": FeatureRole.TRAINABLE.value,
                "source": "legacy compatibility alias of semantic_entropy_cluster_count",
                "source_artifact_path": semantic_path,
                "depends_on_correctness": False,
                "trainable": True,
            },
            {
                "feature_name": "semantic_energy_boltzmann",
                "role": FeatureRole.TRAINABLE.value,
                "source": "legacy compatibility alias of semantic_energy_cluster_uncertainty",
                "source_artifact_path": energy_path,
                "depends_on_correctness": False,
                "trainable": True,
            },
        ]
    )

    for feature_name in DIAGNOSTIC_FEATURES:
        provenance.append(
            {
                "feature_name": feature_name,
                "role": FeatureRole.TRAINABLE.value,
                "source": "candidate-level teacher-forced diagnostic joined by candidate_id",
                "source_artifact_path": energy_path,
                "depends_on_correctness": False,
                "trainable": True,
            }
        )

    for feature_name in EXTERNAL_CORPUS_FEATURES + ("entity_frequency_mean", "entity_frequency_min", "coverage_score", "corpus_risk_only"):
        provenance.append(
            {
                "feature_name": feature_name,
                "role": FeatureRole.EXTERNAL_CORPUS.value,
                "source": corpus_source or "external corpus count backend",
                "source_artifact_path": source_artifact_path,
                "depends_on_correctness": False,
                "trainable": True,
                "corpus_status": corpus_status,
                "backend_id": corpus_backend,
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for entry in provenance:
        deduped[str(entry["feature_name"])] = entry
    return [deduped[name] for name in TRAINABLE_FEATURE_ORDER if name in deduped]


def _build_trainable_features(
    *,
    corpus_features: dict[str, Any],
    semantic_entropy_row: dict[str, Any],
    energy_row: dict[str, Any],
) -> dict[str, Any]:
    energy_features = row_features(energy_row)
    trainable_features: dict[str, Any] = {
        "semantic_entropy_nli_likelihood": _feature_value(semantic_entropy_row, "semantic_entropy_nli_likelihood", "semantic_entropy"),
        "semantic_entropy_cluster_count": _feature_value(semantic_entropy_row, "semantic_entropy_cluster_count", "cluster_count"),
        "semantic_entropy_discrete_cluster_entropy": _feature_value(semantic_entropy_row, "semantic_entropy_discrete_cluster_entropy"),
        "semantic_entropy": _feature_value(semantic_entropy_row, "semantic_entropy", "semantic_entropy_nli_likelihood"),
        "cluster_count": _feature_value(semantic_entropy_row, "cluster_count", "semantic_entropy_cluster_count"),
        "semantic_energy_cluster_uncertainty": energy_features.get("semantic_energy_cluster_uncertainty"),
        "semantic_energy_sample_energy": energy_features.get("semantic_energy_sample_energy"),
        "semantic_energy_boltzmann": energy_features.get("semantic_energy_boltzmann"),
        "semantic_energy_boltzmann_diagnostic": energy_features.get("semantic_energy_boltzmann_diagnostic"),
        "mean_negative_log_probability": energy_features.get("mean_negative_log_probability"),
        "logit_variance": energy_features.get("logit_variance"),
        "confidence_margin": energy_features.get("confidence_margin"),
        "entity_frequency": corpus_features.get("entity_frequency"),
        "entity_frequency_axis": corpus_features.get("entity_frequency_axis", corpus_features.get("entity_frequency")),
        "entity_frequency_mean": corpus_features.get("entity_frequency_mean"),
        "entity_frequency_min": corpus_features.get("entity_frequency_min"),
        "entity_pair_cooccurrence": corpus_features.get("entity_pair_cooccurrence"),
        "entity_pair_cooccurrence_axis": corpus_features.get("entity_pair_cooccurrence_axis", corpus_features.get("entity_pair_cooccurrence")),
        "low_frequency_entity_flag": corpus_features.get("low_frequency_entity_flag"),
        "zero_cooccurrence_flag": corpus_features.get("zero_cooccurrence_flag"),
        "coverage_score": corpus_features.get("coverage_score"),
        "corpus_risk_only": corpus_features.get("corpus_risk_only"),
        "corpus_axis_bin": corpus_features.get("corpus_axis_bin"),
        "corpus_axis_bin_5": corpus_features.get("corpus_axis_bin_5"),
    }
    ordered = {key: trainable_features.get(key) for key in TRAINABLE_FEATURE_ORDER}
    leakage = sorted(FORBIDDEN_FEATURE_KEYS.intersection(ordered))
    if leakage:
        raise ValueError(f"Trainable feature object contains forbidden keys: {', '.join(leakage)}")
    return ordered


def _build_analysis_metadata(
    *,
    semantic_entropy: float,
    archived_type_label: TypeLabel,
    analysis_bin: AnalysisBin | None,
    energy_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "archived_type_label": archived_type_label.value,
        "se_bin": serialize_analysis_bin(analysis_bin),
        "energy_status": energy_metadata["status"],
        "energy_granularity": energy_metadata.get("energy_granularity"),
        "semantic_entropy_threshold_source": "fixed_operational_thresholds_for_archived_diagnostics_only",
        "semantic_entropy": semantic_entropy,
        "thesis_target": "is_hallucination",
    }


def _archived_type_label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("label", "")) for row in rows)
    return {label: counts.get(label, 0) for label in EXPECTED_TYPE_LABELS}


def _dataset_target_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        counts[dataset]["hallucinated"] += 1 if bool(row.get("is_hallucination", False)) else 0
        counts[dataset]["right"] += 0 if bool(row.get("is_hallucination", False)) else 1
    return {dataset: dict(sorted(counter.items())) for dataset, counter in sorted(counts.items())}


def _dataset_archived_type_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get("dataset", "unknown"))][str(row.get("label", ""))] += 1
    return {
        dataset: {label: counter.get(label, 0) for label in EXPECTED_TYPE_LABELS}
        for dataset, counter in sorted(counts.items())
    }


def _label_presence_explanations(label_counts: Counter[str]) -> dict[str, str]:
    explanations: dict[str, str] = {}
    if label_counts.get(TypeLabel.AMBIGUOUS_INCORRECT.value, 0) == 0:
        explanations[TypeLabel.AMBIGUOUS_INCORRECT.value] = (
            "No incorrect candidate in the current feature table fell inside 0.1 < SE <= 0.5, "
            "so the archived gray-zone TypeLabel is absent instead of being force-filled."
        )
    for label_value in EXPECTED_TYPE_LABELS:
        if label_counts.get(label_value, 0) == 0 and label_value not in explanations:
            explanations[label_value] = "Archived operational label absent in current source rows after fixed threshold assignment."
    return explanations


def _prompt_balance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("prompt_id", ""))].append(row)

    valid_prompt_count = 0
    violations: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for prompt_id, prompt_rows in sorted(grouped.items()):
        correct_count = sum(1 for row in prompt_rows if bool(row.get("is_correct", False)))
        hallucination_count = sum(1 for row in prompt_rows if bool(row.get("is_hallucination", False)))
        candidate_labels = sorted(str(row.get("candidate_label", "")) for row in prompt_rows)
        prompt_feature_values = {
            feature_name: sorted({_canonical_value((row.get("features") or {}).get(feature_name)) for row in prompt_rows})
            for feature_name in PROMPT_BROADCAST_FEATURES
        }
        examples.append(
            {
                "prompt_id": prompt_id,
                "candidate_ids": [row.get("candidate_id") for row in prompt_rows],
                "candidate_labels": candidate_labels,
                "archived_type_labels": [row.get("label") for row in prompt_rows],
                "prompt_feature_values": prompt_feature_values,
            }
        )
        prompt_fields_valid = all(len(values) == 1 for values in prompt_feature_values.values())
        if (
            len(prompt_rows) == 2
            and correct_count == 1
            and hallucination_count == 1
            and candidate_labels == ["hallucinated", "right"]
            and prompt_fields_valid
        ):
            valid_prompt_count += 1
            continue
        violations.append(
            {
                "prompt_id": prompt_id,
                "row_count": len(prompt_rows),
                "correct_count": correct_count,
                "hallucination_count": hallucination_count,
                "candidate_labels": candidate_labels,
                "prompt_feature_values": prompt_feature_values,
                "candidate_ids": [row.get("candidate_id") for row in prompt_rows],
            }
        )

    return {
        "prompt_count": len(grouped),
        "valid_prompt_count": valid_prompt_count,
        "invalid_prompt_count": len(violations),
        "violations": violations,
        "examples": examples[:25],
    }


def _validate_trainable_features(features: dict[str, Any], *, row_index: int, problems: list[str]) -> None:
    forbidden_present = sorted(FORBIDDEN_FEATURE_KEYS.intersection(features))
    if forbidden_present:
        problems.append(f"row {row_index}: features contain forbidden leakage keys {forbidden_present}")


def _validate_feature_alignment(feature_alignment: dict[str, Any], *, row_index: int, problems: list[str]) -> None:
    for feature_name in (*TRAINABLE_FEATURE_ORDER, *ANALYSIS_ONLY_FIELDS):
        entry = feature_alignment.get(feature_name)
        if not isinstance(entry, dict):
            problems.append(f"row {row_index}: feature_alignment missing object for {feature_name}")
            continue
        expected_category = FEATURE_ALIGNMENT_CATEGORY_BY_NAME[feature_name]
        if entry.get("category") != expected_category:
            problems.append(
                f"row {row_index}: feature_alignment[{feature_name!r}].category must be {expected_category!r}"
            )
        if bool(entry.get("trainable", False)) != (expected_category != "analysis-only"):
            problems.append(
                f"row {row_index}: feature_alignment[{feature_name!r}].trainable inconsistent with category {expected_category!r}"
            )


def _validate_feature_provenance(provenance: list[Any], *, row_index: int, problems: list[str]) -> None:
    provenance_names = {entry.get("feature_name") for entry in provenance if isinstance(entry, dict)}
    for feature_name in TRAINABLE_FEATURE_ORDER:
        if feature_name not in provenance_names:
            problems.append(f"row {row_index}: missing feature_provenance entry for {feature_name}")
    for entry in provenance:
        if not isinstance(entry, dict):
            problems.append(f"row {row_index}: feature_provenance entries must be objects")
            continue
        feature_name = str(entry.get("feature_name", "")).strip()
        role = str(entry.get("role", "")).strip()
        source = str(entry.get("source", "")).strip()
        trainable = bool(entry.get("trainable", False))
        depends_on_correctness = bool(entry.get("depends_on_correctness", False))
        lowered = " ".join(
            str(value).lower()
            for value in (feature_name, source, entry.get("note", ""), entry.get("source_artifact_path", ""))
            if value is not None
        )
        if trainable and depends_on_correctness:
            problems.append(f"row {row_index}: trainable feature {feature_name} depends on correctness")
        if trainable and role in {FeatureRole.LABEL_ONLY.value, FeatureRole.CORRECTNESS_DERIVED.value}:
            problems.append(f"row {row_index}: trainable feature {feature_name} has prohibited role {role}")
        if trainable and any(hint in lowered for hint in FORBIDDEN_PROVENANCE_HINTS):
            problems.append(
                f"row {row_index}: trainable feature provenance for {feature_name} leaks correctness/gold/annotation hints"
            )


def _resolve_candidate_text(
    *,
    label_record: CandidateLabelRecord,
    source_row: dict[str, Any],
    energy_row: dict[str, Any],
) -> str:
    for value in (
        label_record.candidate_text,
        str(source_row.get("candidate_text", "")).strip(),
        str(energy_row.get("candidate_text", "")).strip(),
    ):
        if value:
            return value
    raise ValueError(f"Missing candidate_text for candidate_id={label_record.candidate_id}")


def _required_candidate_label(label_record: CandidateLabelRecord, *, candidate_id: str) -> str:
    candidate_label = (label_record.candidate_role or "").strip().lower()
    if candidate_label not in {"right", "hallucinated"}:
        raise ValueError(f"Unsupported candidate_label for candidate_id={candidate_id}: {label_record.candidate_role!r}")
    if candidate_label == "right" and not label_record.is_correct:
        raise ValueError(f"candidate_id={candidate_id} has candidate_label=right with is_correct=false")
    if candidate_label == "hallucinated" and label_record.is_correct:
        raise ValueError(f"candidate_id={candidate_id} has candidate_label=hallucinated with is_correct=true")
    return candidate_label


def build_feature_table(results_dir: Path, out_path: Path, dataset_config_path: Path) -> dict[str, Any]:
    corpus_path = results_dir / CORPUS_FEATURES_PATH
    corpus_rows, corpus_storage = load_corpus_feature_rows(corpus_path)
    if not corpus_rows:
        raise ValueError(f"Corpus feature source is empty: {_resolve_artifact_path(corpus_path)}")

    analysis_bins, raw_bin_specs = read_analysis_bin_config(dataset_config_path)
    semantic_entropy_by_prompt = load_prompt_semantic_entropy(results_dir)
    energy_by_candidate = load_energy_rows(results_dir)
    labels_by_candidate = load_candidate_labels(results_dir)
    feature_alignment = build_feature_alignment()

    rows: list[dict[str, Any]] = []
    archived_type_label_counts: Counter[str] = Counter()
    seen_candidate_ids: set[str] = set()

    for source_row in corpus_rows:
        if not isinstance(source_row, dict):
            continue

        candidate_id = str(source_row.get("candidate_id", "")).strip()
        prompt_id = str(source_row.get("prompt_id", "")).strip()
        if not candidate_id:
            raise ValueError("Corpus feature row is missing candidate_id")
        if not prompt_id:
            raise ValueError(f"Corpus feature row is missing prompt_id for candidate_id={candidate_id}")
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"Duplicate candidate_id in corpus features: {candidate_id}")
        seen_candidate_ids.add(candidate_id)

        label_record = labels_by_candidate.get(candidate_id)
        if label_record is None:
            raise ValueError(f"Missing label row for candidate_id={candidate_id}")
        if label_record.prompt_id != prompt_id:
            raise ValueError(
                f"Label prompt_id mismatch for candidate_id={candidate_id}: corpus={prompt_id} label={label_record.prompt_id}"
            )

        semantic_entropy_row = semantic_entropy_by_prompt.get(prompt_id)
        if semantic_entropy_row is None:
            raise ValueError(f"Missing prompt-level Semantic Entropy row for prompt_id={prompt_id}")

        energy_row = energy_by_candidate.get(candidate_id)
        if energy_row is None:
            raise ValueError(f"Missing candidate-level energy row for candidate_id={candidate_id}")

        corpus_features = row_features(source_row)
        semantic_entropy = _required_finite_float(
            _feature_value(semantic_entropy_row, "semantic_entropy_nli_likelihood", "semantic_entropy"),
            field_name="semantic_entropy_nli_likelihood",
            context=f"prompt_id={prompt_id}",
        )
        _required_nonnegative_int(
            _feature_value(semantic_entropy_row, "semantic_entropy_cluster_count", "cluster_count"),
            field_name="semantic_entropy_cluster_count",
            context=f"prompt_id={prompt_id}",
        )
        analysis_bin = select_analysis_bin(semantic_entropy, analysis_bins, raw_bin_specs)
        archived_type_label = assign_operational_label(label_record.is_correct, semantic_entropy)
        candidate_label = _required_candidate_label(label_record, candidate_id=candidate_id)
        energy_metadata = _build_energy_metadata(energy_row)
        if not energy_metadata["paper_faithful_energy_available"]:
            raise ValueError(f"Paper-faithful Semantic Energy missing for candidate_id={candidate_id}")
        trainable_features = _build_trainable_features(
            corpus_features=corpus_features,
            semantic_entropy_row=semantic_entropy_row,
            energy_row=energy_row,
        )

        row = deepcopy(source_row)
        row["sample_id"] = candidate_id
        row["candidate_id"] = candidate_id
        row["prompt_id"] = prompt_id
        row["pair_id"] = label_record.pair_id or row.get("pair_id")
        row["dataset"] = label_record.dataset or row.get("dataset")
        row["split_id"] = label_record.split_id or row.get("split_id")
        row["candidate_text"] = _resolve_candidate_text(
            label_record=label_record,
            source_row=source_row,
            energy_row=energy_row,
        )
        row["candidate_label"] = candidate_label
        row["candidate_role"] = candidate_label
        row["is_correct"] = label_record.is_correct
        row["is_hallucination"] = label_record.is_hallucination
        row["label_source"] = label_record.label_source
        row["label"] = archived_type_label.value
        row["archived_type_label"] = archived_type_label.value
        row["label_status"] = "archived_operational_type_label_for_diagnostics_only"
        row["features"] = trainable_features
        row["analysis_features"] = _build_analysis_metadata(
            semantic_entropy=semantic_entropy,
            archived_type_label=archived_type_label,
            analysis_bin=analysis_bin,
            energy_metadata=energy_metadata,
        )
        row["feature_alignment"] = deepcopy(feature_alignment)
        row["thesis_target"] = {
            "field": "is_hallucination",
            "positive_class": True,
            "source": "annotation-backed correctness",
            "trainable": False,
            "note": "Use top-level is_hallucination as the supervised target. Archived TypeLabel values remain analysis-only metadata.",
        }
        row["correctness_label_source"] = {
            "prompt_id": label_record.prompt_id,
            "candidate_id": label_record.candidate_id,
            "candidate_label": candidate_label,
            "is_correct": label_record.is_correct,
            "is_hallucination": label_record.is_hallucination,
            "label_source": label_record.label_source,
            "role": FeatureRole.LABEL_ONLY.value,
            "trainable": False,
        }
        row["energy_availability"] = energy_metadata
        row["feature_provenance"] = _build_feature_provenance(
            source_row,
            semantic_entropy_row,
            energy_row,
            results_dir=results_dir,
        )
        rows.append(row)
        archived_type_label_counts[archived_type_label.value] += 1

    prompt_balance = _prompt_balance_summary(rows)
    if prompt_balance["violations"]:
        raise ValueError(
            "Feature table prompt balance contract violated: "
            + "; ".join(
                f"{entry['prompt_id']} rows={entry['row_count']} correct={entry['correct_count']} hallucination={entry['hallucination_count']}"
                for entry in prompt_balance["violations"][:10]
            )
        )

    boundary_self_check = build_boundary_self_check()
    report = {
        "schema_version": FEATURE_TABLE_SCHEMA_VERSION,
        "run_id": f"feature-table-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_out_path": str(out_path),
        "row_count": len(rows),
        "row_identity": "candidate_id",
        "prompt_broadcast_key": "prompt_id",
        "target_label_field": "is_hallucination",
        "target_label_source": "annotation-backed correctness",
        "source_artifacts": {
            "corpus": {
                "requested_path": str(corpus_path),
                "resolved_path": str(_resolve_artifact_path(corpus_path)),
                "storage": corpus_storage,
            },
            "semantic_entropy": {
                "requested_path": str(results_dir / SEMANTIC_ENTROPY_FEATURES_PATH),
                "resolved_path": str(_resolve_artifact_path(results_dir / SEMANTIC_ENTROPY_FEATURES_PATH)),
                "row_count": len(semantic_entropy_by_prompt),
                "join_key": "prompt_id",
            },
            "energy": {
                "requested_path": str(results_dir / ENERGY_FEATURES_PATH),
                "resolved_path": str(_resolve_artifact_path(results_dir / ENERGY_FEATURES_PATH)),
                "row_count": len(energy_by_candidate),
                "join_key": "candidate_id",
            },
            "correctness": {
                "requested_path": str(results_dir / CORRECTNESS_DATASET_PATH),
                "row_count": len(labels_by_candidate),
                "join_key": "candidate_id",
                "role": FeatureRole.LABEL_ONLY.value,
            },
        },
        "target_counts_by_dataset": _dataset_target_counts(rows),
        "archived_type_label_counts": {label: archived_type_label_counts.get(label, 0) for label in EXPECTED_TYPE_LABELS},
        "archived_type_label_counts_by_dataset": _dataset_archived_type_counts(rows),
        "label_presence_explanations": _label_presence_explanations(archived_type_label_counts),
        "prompt_balance": prompt_balance,
        "boundary_self_check": boundary_self_check,
        "energy_status_counts": dict(Counter(row["energy_availability"]["status"] for row in rows)),
        "analysis_bin_scheme": analysis_bins[0].scheme_name if analysis_bins else None,
        "analysis_bin_count": len(analysis_bins),
        "feature_alignment_summary": _feature_alignment_summary(),
        "notes": [
            "Rows are keyed by candidate_id. Prompt-level Semantic Entropy and paper-faithful Semantic Energy are broadcast to both paired candidates by prompt_id.",
            "The supervised thesis target is top-level is_hallucination / annotation-backed correctness.",
            "Archived TypeLabel values remain diagnostic metadata only and must never become the primary thesis-valid target.",
            "The trainable features object excludes correctness labels, gold/reference answers, and dataset annotation leakage fields.",
        ],
    }

    versioned_rows = add_schema_version(rows, FEATURE_TABLE_SCHEMA_VERSION)
    storage = write_feature_artifact(out_path, versioned_rows, report, schema_version=FEATURE_TABLE_SCHEMA_VERSION)
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    write_json(report_path, report)
    return {
        "rows": versioned_rows,
        "report": report,
        "storage": storage,
        "report_path": str(report_path),
    }


def validate_type_labels(feature_artifact_path: Path, dataset_config_path: Path) -> dict[str, Any]:
    rows, storage_report = read_feature_rows(feature_artifact_path)
    if not rows:
        raise ValueError("Feature table artifact is empty.")

    analysis_bins, raw_bin_specs = read_analysis_bin_config(dataset_config_path)
    valid_bin_ids = {bin_spec.bin_id for bin_spec in analysis_bins}
    problems: list[str] = []
    archived_type_label_counts: Counter[str] = Counter()
    candidate_ids: set[str] = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {index}: row must be an object")
            continue
        if row.get("schema_version") != FEATURE_TABLE_SCHEMA_VERSION:
            problems.append(
                f"row {index}: schema_version must be {FEATURE_TABLE_SCHEMA_VERSION!r}; got {row.get('schema_version')!r}"
            )

        candidate_id = str(row.get("candidate_id", "")).strip()
        prompt_id = str(row.get("prompt_id", "")).strip()
        pair_id = str(row.get("pair_id", "")).strip()
        if not candidate_id:
            problems.append(f"row {index}: missing candidate_id")
            continue
        if candidate_id in candidate_ids:
            problems.append(f"row {index}: duplicate candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
        if str(row.get("sample_id", "")).strip() != candidate_id:
            problems.append(f"row {index}: sample_id must mirror candidate_id for stable joins")
        for field_name in ("dataset", "split_id", "prompt_id", "pair_id", "candidate_id", "candidate_text", "candidate_label", "label_source"):
            value = row.get(field_name)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"row {index}: {field_name} must be a non-empty string")
        if not prompt_id:
            problems.append(f"row {index}: missing prompt_id")
        if not pair_id:
            problems.append(f"row {index}: missing pair_id")

        is_correct = row.get("is_correct")
        is_hallucination = row.get("is_hallucination")
        if not isinstance(is_correct, bool):
            problems.append(f"row {index}: is_correct must be boolean")
        if not isinstance(is_hallucination, bool):
            problems.append(f"row {index}: is_hallucination must be boolean")
        elif isinstance(is_correct, bool) and is_hallucination != (not is_correct):
            problems.append(f"row {index}: is_hallucination must equal not is_correct")

        candidate_label = str(row.get("candidate_label", "")).strip().lower()
        if candidate_label not in {"right", "hallucinated"}:
            problems.append(f"row {index}: candidate_label must be 'right' or 'hallucinated'")
        elif isinstance(is_correct, bool):
            if candidate_label == "right" and not is_correct:
                problems.append(f"row {index}: candidate_label=right requires is_correct=true")
            if candidate_label == "hallucinated" and is_correct:
                problems.append(f"row {index}: candidate_label=hallucinated requires is_correct=false")

        label_value = str(row.get("label", ""))
        archived_type_label = str(row.get("archived_type_label", ""))
        if label_value not in EXPECTED_TYPE_LABELS:
            problems.append(f"row {index}: label must be one of {list(EXPECTED_TYPE_LABELS)}, got {label_value!r}")
            continue
        if archived_type_label != label_value:
            problems.append(f"row {index}: archived_type_label must mirror legacy label field")

        features = row.get("features")
        if not isinstance(features, dict):
            problems.append(f"row {index}: features must be an object")
            continue
        _validate_trainable_features(features, row_index=index, problems=problems)
        for required_name in TRAINABLE_FEATURE_ORDER:
            if required_name not in features:
                problems.append(f"row {index}: features missing required field {required_name}")

        try:
            semantic_entropy = _required_finite_float(
                features.get("semantic_entropy_nli_likelihood"),
                field_name="features.semantic_entropy_nli_likelihood",
                context=f"row {index}",
            )
            cluster_count = _required_nonnegative_int(
                features.get("semantic_entropy_cluster_count"),
                field_name="features.semantic_entropy_cluster_count",
                context=f"row {index}",
            )
            _required_finite_float(
                features.get("semantic_entropy_discrete_cluster_entropy"),
                field_name="features.semantic_entropy_discrete_cluster_entropy",
                context=f"row {index}",
            )
            _required_finite_float(
                features.get("semantic_energy_cluster_uncertainty"),
                field_name="features.semantic_energy_cluster_uncertainty",
                context=f"row {index}",
            )
            _required_finite_float(
                features.get("semantic_energy_sample_energy"),
                field_name="features.semantic_energy_sample_energy",
                context=f"row {index}",
            )
            _required_finite_float(
                features.get("mean_negative_log_probability"),
                field_name="features.mean_negative_log_probability",
                context=f"row {index}",
            )
            _required_finite_float(
                features.get("semantic_energy_boltzmann_diagnostic"),
                field_name="features.semantic_energy_boltzmann_diagnostic",
                context=f"row {index}",
            )
        except ValueError as exc:
            problems.append(str(exc))
            continue

        if features.get("semantic_entropy") != features.get("semantic_entropy_nli_likelihood"):
            problems.append(f"row {index}: semantic_entropy must alias semantic_entropy_nli_likelihood")
        if features.get("cluster_count") != cluster_count:
            problems.append(f"row {index}: cluster_count must alias semantic_entropy_cluster_count")
        if features.get("semantic_energy_boltzmann") != features.get("semantic_energy_cluster_uncertainty"):
            problems.append(f"row {index}: semantic_energy_boltzmann must alias semantic_energy_cluster_uncertainty")

        expected_label = assign_operational_label(bool(is_correct), semantic_entropy)
        if label_value != expected_label.value:
            problems.append(
                f"row {index}: archived TypeLabel mismatch, expected {expected_label.value} for is_correct={bool(is_correct)} and SE={semantic_entropy}, got {label_value}"
            )

        analysis_features = row.get("analysis_features")
        if not isinstance(analysis_features, dict):
            problems.append(f"row {index}: analysis_features must be an object")
        else:
            if analysis_features.get("archived_type_label") != label_value:
                problems.append(f"row {index}: analysis_features.archived_type_label must mirror label")
            se_bin = analysis_features.get("se_bin")
            if not isinstance(se_bin, dict):
                problems.append(f"row {index}: analysis_features.se_bin must be an object")
            else:
                bin_id = str(se_bin.get("bin_id", ""))
                recalculated_bin = select_analysis_bin(semantic_entropy, analysis_bins, raw_bin_specs)
                if bin_id not in valid_bin_ids:
                    problems.append(f"row {index}: unknown SE bin id {bin_id!r}")
                elif recalculated_bin is None or bin_id != recalculated_bin.bin_id:
                    problems.append(
                        f"row {index}: SE bin mismatch, expected {None if recalculated_bin is None else recalculated_bin.bin_id}, got {bin_id}"
                    )

        energy_availability = row.get("energy_availability")
        if not isinstance(energy_availability, dict):
            problems.append(f"row {index}: missing energy_availability object")
        else:
            if energy_availability.get("paper_faithful_energy_available") is not True:
                problems.append(f"row {index}: paper_faithful_energy_available must be true")
            if energy_availability.get("true_boltzmann_available") is not False:
                problems.append(f"row {index}: true_boltzmann_available must remain false")
            if energy_availability.get("candidate_boltzmann_diagnostic_available") is not True:
                problems.append(f"row {index}: candidate_boltzmann_diagnostic_available must be true")
            if energy_availability.get("energy_granularity") != "prompt_level_broadcast_to_candidate_rows":
                problems.append(f"row {index}: energy_granularity must be prompt_level_broadcast_to_candidate_rows")
            if energy_availability.get("not_for_thesis_claims") is not False:
                problems.append(f"row {index}: paper-faithful Energy rows must not be marked not_for_thesis_claims")

        feature_alignment = row.get("feature_alignment")
        if not isinstance(feature_alignment, dict):
            problems.append(f"row {index}: feature_alignment must be an object")
        else:
            _validate_feature_alignment(feature_alignment, row_index=index, problems=problems)

        thesis_target = row.get("thesis_target")
        if not isinstance(thesis_target, dict):
            problems.append(f"row {index}: thesis_target must be an object")
        else:
            if thesis_target.get("field") != "is_hallucination":
                problems.append(f"row {index}: thesis_target.field must equal 'is_hallucination'")
            if thesis_target.get("positive_class") is not True:
                problems.append(f"row {index}: thesis_target.positive_class must be true")

        provenance = row.get("feature_provenance")
        if not isinstance(provenance, list):
            problems.append(f"row {index}: feature_provenance must be a list")
        else:
            _validate_feature_provenance(provenance, row_index=index, problems=problems)

        correctness_source = row.get("correctness_label_source")
        if not isinstance(correctness_source, dict):
            problems.append(f"row {index}: correctness_label_source must be an object")
        else:
            if correctness_source.get("role") != FeatureRole.LABEL_ONLY.value:
                problems.append(f"row {index}: correctness_label_source.role must be label_only")
            if correctness_source.get("trainable") is not False:
                problems.append(f"row {index}: correctness_label_source.trainable must be false")

        archived_type_label_counts[label_value] += 1

    prompt_balance = _prompt_balance_summary(rows)
    for violation in prompt_balance["violations"]:
        problems.append(
            "prompt balance violation for "
            f"{violation['prompt_id']}: rows={violation['row_count']} correct={violation['correct_count']} hallucination={violation['hallucination_count']} "
            f"candidate_labels={violation['candidate_labels']}"
        )

    boundary_self_check = build_boundary_self_check()
    explanations = _label_presence_explanations(archived_type_label_counts)
    missing_without_explanation = [
        label for label in EXPECTED_TYPE_LABELS if archived_type_label_counts.get(label, 0) == 0 and not explanations.get(label)
    ]
    if missing_without_explanation:
        problems.append("Missing archived TypeLabel values without explanation: " + ", ".join(sorted(missing_without_explanation)))
    if not boundary_self_check["all_pass"]:
        problems.append("Boundary self-check failed for one or more threshold cases.")

    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(feature_artifact_path),
        "resolved_storage": storage_report,
        "row_count": len(rows),
        "row_identity": "candidate_id",
        "prompt_broadcast_key": "prompt_id",
        "target_label_field": "is_hallucination",
        "target_label_source": "annotation-backed correctness",
        "target_counts_by_dataset": _dataset_target_counts(rows),
        "archived_type_label_counts": {label: archived_type_label_counts.get(label, 0) for label in EXPECTED_TYPE_LABELS},
        "archived_type_label_counts_by_dataset": _dataset_archived_type_counts(rows),
        "label_presence_explanations": explanations,
        "prompt_balance": prompt_balance,
        "boundary_self_check": boundary_self_check,
        "feature_alignment_summary": _feature_alignment_summary(),
        "problems": problems,
        "status": "ok" if not problems else "error",
        "notes": [
            "Validator enforces that thesis target is top-level is_hallucination and that archived TypeLabel remains diagnostic-only metadata.",
            "Validator rejects correctness, gold-answer, label, or dataset-annotation leakage inside trainable features and trainable feature provenance.",
        ],
    }
    return report


def write_validation_report(feature_artifact_path: Path, report: dict[str, Any]) -> Path:
    report_path = feature_artifact_path.parent / "type_label_validation_report.json"
    write_json(report_path, report)
    return report_path
