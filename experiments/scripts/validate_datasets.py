#!/usr/bin/env python3
"""Validate the paired discriminative dataset registry contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTIVE_DATASETS = ("TruthfulQA", "HaluEval-QA")
EXCLUDED_ACTIVE_DATASETS = ("TriviaQA", "Natural Questions", "HotpotQA", "FEVER", "BioASQ")
SUPERVISED_TARGET_FIELD = "is_hallucination"
PRIMARY_ANALYSIS_AXIS = "corpus_axis_bin"
SENSITIVITY_AXES = ("corpus_axis_bin_5", "corpus_axis_bin_10")
AMBIGUOUS_POLICY_MARKERS = ("same as", "tbd", "todo", "implicit")
FORBIDDEN_ACTIVE_POLICY_TOKENS = (
    "llm_as_judge",
    "alias_match",
    "gold_alias_match",
    "substring",
    "exact match",
    "heuristic",
    "human review",
    "human-review",
    "manual review",
    "judge fallback",
)
FORBIDDEN_TOP_LEVEL_FIELDS = (
    "future_datasets",
    "dataset_expansion_policy",
)
FORBIDDEN_DATASET_FIELDS = (
    "promotion",
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON-compatible YAML file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Dataset config must decode to an object: {path}")
    return payload


def is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _walk_strings(value: Any, path: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk_strings(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_strings(child, f"{path}[{index}]") )
    elif isinstance(value, str):
        found.append((path, value))
    return found


def validate_operational_labels(config: dict[str, Any], problems: list[str]) -> None:
    label_policy = config.get("label_policy")
    if not isinstance(label_policy, dict):
        problems.append("missing top-level label_policy object")
        return

    target = label_policy.get("supervised_target")
    if not isinstance(target, dict):
        problems.append("label_policy.supervised_target must be an object")
    else:
        if target.get("field") != SUPERVISED_TARGET_FIELD:
            problems.append(f"label_policy.supervised_target.field must be {SUPERVISED_TARGET_FIELD!r}")
        if target.get("binary") is not True:
            problems.append("label_policy.supervised_target.binary must be true")

    if label_policy.get("primary_analysis_axis") != PRIMARY_ANALYSIS_AXIS:
        problems.append(f"label_policy.primary_analysis_axis must equal {PRIMARY_ANALYSIS_AXIS!r}")
    sensitivity = label_policy.get("sensitivity_analysis_axes") or []
    missing_sensitivity = [axis for axis in SENSITIVITY_AXES if axis not in sensitivity]
    if missing_sensitivity:
        problems.append(f"label_policy.sensitivity_analysis_axes must include {list(SENSITIVITY_AXES)}; missing {missing_sensitivity}")

    row_level_output = label_policy.get("row_level_output")
    if not isinstance(row_level_output, dict):
        problems.append("label_policy.row_level_output must be an object")
    else:
        required_fields = row_level_output.get("required_fields") or []
        for needed in ("is_hallucination", "is_correct", "candidate_label"):
            if needed not in required_fields:
                problems.append(f"label_policy.row_level_output.required_fields must include {needed!r}")


def validate_analysis_bins(config: dict[str, Any], problems: list[str]) -> None:
    label_policy = config.get("label_policy")
    if not isinstance(label_policy, dict):
        return

    analysis = label_policy.get("analysis_se_bins")
    if not isinstance(analysis, dict):
        problems.append("label_policy.analysis_se_bins must be an object")
        return

    if analysis.get("analysis_only") is not True:
        problems.append("label_policy.analysis_se_bins.analysis_only must be true")
    if analysis.get("spans") != "[0, +inf)":
        problems.append("label_policy.analysis_se_bins.spans must be '[0, +inf)'")
    if analysis.get("special_attention_thresholds") != [0.1, 0.5]:
        problems.append("label_policy.analysis_se_bins.special_attention_thresholds must be [0.1, 0.5]")

    bins = analysis.get("bins")
    if not isinstance(bins, list) or not bins:
        problems.append("label_policy.analysis_se_bins.bins must be a non-empty list")
        return

    boundaries: set[float] = set()
    saw_open_ended_tail = False
    last_upper: float | None = None
    previous_upper: float | None = None
    previous_upper_inclusive: bool | None = None

    for index, bin_spec in enumerate(bins):
        if not isinstance(bin_spec, dict):
            problems.append(f"label_policy.analysis_se_bins.bins[{index}] must be an object")
            continue
        if not is_nonempty_string(bin_spec.get("bin_id")):
            problems.append(f"label_policy.analysis_se_bins.bins[{index}] is missing bin_id")
        lower_bound = bin_spec.get("lower_bound")
        upper_bound = bin_spec.get("upper_bound")
        lower_inclusive = bin_spec.get("lower_inclusive")
        upper_inclusive = bin_spec.get("upper_inclusive")
        includes_upper_bound = bin_spec.get("includes_upper_bound")
        if lower_bound is None or not isinstance(lower_bound, (int, float)):
            problems.append(f"analysis bin {bin_spec.get('bin_id', index)!r} must define numeric lower_bound")
            continue
        if upper_bound is not None and not isinstance(upper_bound, (int, float)):
            problems.append(f"analysis bin {bin_spec.get('bin_id', index)!r} upper_bound must be numeric or null")
            continue
        if not isinstance(lower_inclusive, bool):
            problems.append(f"analysis bin {bin_spec.get('bin_id', index)!r} must define boolean lower_inclusive")
            continue
        if not isinstance(upper_inclusive, bool):
            problems.append(f"analysis bin {bin_spec.get('bin_id', index)!r} must define boolean upper_inclusive")
            continue
        if includes_upper_bound != upper_inclusive:
            problems.append(
                f"analysis bin {bin_spec.get('bin_id', index)!r} includes_upper_bound must match upper_inclusive"
            )
        if upper_bound is not None and upper_bound < lower_bound:
            problems.append(f"analysis bin {bin_spec.get('bin_id', index)!r} has upper_bound < lower_bound")
            continue
        if upper_bound is not None and upper_bound == lower_bound and not (lower_inclusive and upper_inclusive):
            problems.append(f"analysis bin {bin_spec.get('bin_id', index)!r} is zero-width and must include both bounds")
        if index == 0 and (float(lower_bound) != 0.0 or lower_inclusive is not True):
            problems.append("analysis_se_bins must start at 0.0 with an inclusive first bin")
        if previous_upper is not None:
            current_lower = float(lower_bound)
            if current_lower < previous_upper:
                problems.append(
                    f"analysis bins overlap or are out of order at {bin_spec.get('bin_id', index)!r}: lower_bound {current_lower} is below previous upper_bound {previous_upper}"
                )
            elif current_lower > previous_upper:
                problems.append(
                    f"analysis bins leave a gap before {bin_spec.get('bin_id', index)!r}: lower_bound {current_lower} is above previous upper_bound {previous_upper}"
                )
            elif previous_upper_inclusive and lower_inclusive:
                problems.append(
                    f"analysis bins overlap at boundary {current_lower} near {bin_spec.get('bin_id', index)!r}; adjacent bins cannot both include the shared boundary"
                )
            elif not previous_upper_inclusive and not lower_inclusive:
                problems.append(
                    f"analysis bins leave boundary {current_lower} uncovered near {bin_spec.get('bin_id', index)!r}; one adjacent bin must include the shared boundary"
                )
        if upper_bound is None:
            saw_open_ended_tail = True
        else:
            boundaries.add(float(upper_bound))
            last_upper = float(upper_bound)
            previous_upper = float(upper_bound)
            previous_upper_inclusive = upper_inclusive
        boundaries.add(float(lower_bound))
        if upper_bound is None:
            previous_upper = None
            previous_upper_inclusive = None

    if 0.1 not in boundaries:
        problems.append("analysis_se_bins must include a boundary at 0.1")
    if 0.5 not in boundaries:
        problems.append("analysis_se_bins must include a boundary at 0.5")
    if not saw_open_ended_tail:
        problems.append("analysis_se_bins must include an open-ended final bin covering +inf")
    if last_upper is not None and last_upper < 0.9:
        problems.append("analysis_se_bins must cover the high-end range before the open-ended tail")


def validate_top_level_contract(config: dict[str, Any], problems: list[str]) -> None:
    if config.get("version") != 1:
        problems.append(f"version must equal 1; got {config.get('version')!r}")
    if config.get("registry_name") != "paired_discriminative_experiment_datasets":
        problems.append(
            "registry_name must equal 'paired_discriminative_experiment_datasets'"
        )
    for field_name in FORBIDDEN_TOP_LEVEL_FIELDS:
        if field_name in config:
            problems.append(f"forbidden stale top-level field: {field_name}")

    experiment_policy = config.get("experiment_dataset_policy")
    if not isinstance(experiment_policy, dict):
        problems.append("missing top-level experiment_dataset_policy object")
        return
    if experiment_policy.get("dataset_contract") != "single_paired_discriminative_experiment_dataset":
        problems.append(
            "experiment_dataset_policy.dataset_contract must equal 'single_paired_discriminative_experiment_dataset'"
        )
    active_datasets = experiment_policy.get("active_datasets")
    if active_datasets != list(ACTIVE_DATASETS):
        problems.append(
            f"experiment_dataset_policy.active_datasets must equal {list(ACTIVE_DATASETS)}; got {active_datasets!r}"
        )
    if experiment_policy.get("candidate_rows_per_prompt") != 2:
        problems.append("experiment_dataset_policy.candidate_rows_per_prompt must equal 2")
    excluded = experiment_policy.get("excluded_active_datasets")
    if excluded != list(EXCLUDED_ACTIVE_DATASETS):
        problems.append(
            f"experiment_dataset_policy.excluded_active_datasets must equal {list(EXCLUDED_ACTIVE_DATASETS)}; got {excluded!r}"
        )
    if config.get("mandatory_core_datasets") != list(ACTIVE_DATASETS):
        problems.append(f"mandatory_core_datasets must equal {list(ACTIVE_DATASETS)}")


def _validate_policy_text(text: str, path: str, problems: list[str]) -> None:
    lowered = text.lower()
    for marker in AMBIGUOUS_POLICY_MARKERS:
        if marker in lowered:
            problems.append(f"{path} is too ambiguous for audit: contains {marker!r}")
    for token in FORBIDDEN_ACTIVE_POLICY_TOKENS:
        if token in lowered:
            problems.append(f"{path} contains forbidden stale active-path phrase {token!r}")


def validate_dataset_entry(dataset: dict[str, Any], problems: list[str]) -> None:
    name = dataset.get("name", "<unknown>")
    path_prefix = f"datasets[{name!r}]"
    required_fields = (
        "name",
        "role",
        "hf_id",
        "config",
        "split",
        "split_id",
        "target_sample_count",
        "seed",
        "candidate_pair_policy",
        "label_policy",
        "label_source",
        "notes",
    )
    for field in required_fields:
        if field not in dataset:
            problems.append(f"{path_prefix} is missing required field: {field}")
    for field in FORBIDDEN_DATASET_FIELDS:
        if field in dataset:
            problems.append(f"{path_prefix} contains forbidden stale field: {field}")

    if dataset.get("role") != "core":
        problems.append(f"{path_prefix}.role must be 'core'; got {dataset.get('role')!r}")
    if not is_nonempty_string(dataset.get("split_id")):
        problems.append(f"{path_prefix}.split_id must be a non-empty string")
    target_sample_count = dataset.get("target_sample_count")
    if not isinstance(target_sample_count, int) or target_sample_count <= 0:
        problems.append(f"{path_prefix}.target_sample_count must be a positive integer")
    if not isinstance(dataset.get("seed"), int):
        problems.append(f"{path_prefix}.seed must be an integer")

    candidate_pair_policy = dataset.get("candidate_pair_policy")
    if not isinstance(candidate_pair_policy, dict):
        problems.append(f"{path_prefix}.candidate_pair_policy must be an object")
    else:
        if candidate_pair_policy.get("candidate_rows_per_prompt") != 2:
            problems.append(f"{path_prefix}.candidate_pair_policy.candidate_rows_per_prompt must equal 2")
        selection = candidate_pair_policy.get("selection")
        if not isinstance(selection, dict):
            problems.append(f"{path_prefix}.candidate_pair_policy.selection must be an object")
        else:
            for field_name in ("correct_candidate", "incorrect_candidate", "determinism"):
                value = selection.get(field_name)
                if not is_nonempty_string(value):
                    problems.append(f"{path_prefix}.candidate_pair_policy.selection.{field_name} must be a non-empty string")
                elif isinstance(value, str):
                    _validate_policy_text(value, f"{path_prefix}.candidate_pair_policy.selection.{field_name}", problems)

    raw_label_source = dataset.get("label_source")
    label_source = raw_label_source if isinstance(raw_label_source, dict) else None
    if label_source is None:
        problems.append(f"{path_prefix}.label_source must be an object")
    else:
        if label_source.get("type") != "dataset_provided_candidate_pair":
            problems.append(f"{path_prefix}.label_source.type must equal 'dataset_provided_candidate_pair'")
        if not is_nonempty_string(label_source.get("details")):
            problems.append(f"{path_prefix}.label_source.details must be a non-empty string")
        else:
            _validate_policy_text(str(label_source["details"]), f"{path_prefix}.label_source.details", problems)

    label_policy = dataset.get("label_policy")
    if not isinstance(label_policy, dict):
        problems.append(f"{path_prefix}.label_policy must be an object")
        return
    for field_name in (
        "correctness_judgment",
        "supervised_target",
        "primary_analysis_axis",
        "row_level_output",
    ):
        value = label_policy.get(field_name)
        if not is_nonempty_string(value):
            problems.append(f"{path_prefix}.label_policy.{field_name} must be a non-empty string")
            continue
        _validate_policy_text(str(value), f"{path_prefix}.label_policy.{field_name}", problems)

    if name == "TruthfulQA":
        if dataset.get("hf_id") != "truthful_qa":
            problems.append(f"{path_prefix}.hf_id must equal 'truthful_qa'")
        if dataset.get("config") != "generation":
            problems.append(f"{path_prefix}.config must equal 'generation'")
        if dataset.get("split") != "validation":
            problems.append(f"{path_prefix}.split must equal 'validation'")
        if label_source is None or label_source.get("correct_candidate_source") != "correct_answers[]":
            problems.append(f"{path_prefix}.label_source.correct_candidate_source must equal 'correct_answers[]'")
        if label_source is None or label_source.get("incorrect_candidate_source") != "incorrect_answers[]":
            problems.append(f"{path_prefix}.label_source.incorrect_candidate_source must equal 'incorrect_answers[]'")
    elif name == "HaluEval-QA":
        if dataset.get("hf_id") != "pminervini/HaluEval":
            problems.append(f"{path_prefix}.hf_id must equal 'pminervini/HaluEval'")
        if dataset.get("config") != "qa":
            problems.append(f"{path_prefix}.config must equal 'qa'")
        if dataset.get("split") != "data":
            problems.append(f"{path_prefix}.split must equal 'data'")
        if label_source is None or label_source.get("correct_candidate_source") != "right_answer":
            problems.append(f"{path_prefix}.label_source.correct_candidate_source must equal 'right_answer'")
        if label_source is None or label_source.get("incorrect_candidate_source") != "hallucinated_answer":
            problems.append(f"{path_prefix}.label_source.incorrect_candidate_source must equal 'hallucinated_answer'")
    else:
        problems.append(f"{path_prefix}.name must be one of {list(ACTIVE_DATASETS)}; got {name!r}")


def validate_dataset_registry(config: dict[str, Any], problems: list[str]) -> None:
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        problems.append("datasets must be a non-empty list")
        return

    names: list[str] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            problems.append("every dataset entry must be an object")
            continue
        validate_dataset_entry(dataset, problems)
        if is_nonempty_string(dataset.get("name")):
            names.append(str(dataset.get("name")))

    if len(names) != len(set(names)):
        problems.append("dataset names must be unique")
    if names != list(ACTIVE_DATASETS):
        problems.append(f"datasets must appear exactly in paired-contract order {list(ACTIVE_DATASETS)}; got {names!r}")


def validate_forbidden_stale_phrases(config: dict[str, Any], problems: list[str]) -> None:
    for path, text in _walk_strings(config, "config"):
        if "--mode" in text:
            problems.append(f"{path} contains forbidden stale path '--mode'")
        lowered = text.lower()
        if "full-core" in lowered:
            problems.append(f"{path} contains forbidden stale branch phrase 'full-core'")
        if "full-extended" in lowered:
            problems.append(f"{path} contains forbidden stale branch phrase 'full-extended'")

    mandatory = config.get("mandatory_core_datasets")
    if isinstance(mandatory, list) and "TriviaQA" in mandatory:
        index = mandatory.index("TriviaQA")
        problems.append(f"mandatory_core_datasets[{index}] contains forbidden stale active dataset 'TriviaQA'")


def validate_config(config: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    validate_top_level_contract(config, problems)
    validate_operational_labels(config, problems)
    validate_analysis_bins(config, problems)
    validate_dataset_registry(config, problems)
    validate_forbidden_stale_phrases(config, problems)
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python3 experiments/scripts/validate_datasets.py <datasets.yaml>", file=sys.stderr)
        return 2

    config_path = Path(argv[1])
    config = load_json(config_path)
    problems = validate_config(config)
    if problems:
        print("Dataset registry validation failed.", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print("Dataset registry validation passed.")
    print(
        json.dumps(
            {
                "config_path": str(config_path),
                "mandatory_core_datasets": list(ACTIVE_DATASETS),
                "selected_core_datasets": list(ACTIVE_DATASETS),
                "candidate_rows_per_prompt": 2,
                "label_source_type": "dataset_provided_candidate_pair",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
