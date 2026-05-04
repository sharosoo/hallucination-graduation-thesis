#!/usr/bin/env python3
"""Validate the dataset registry and label policy contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.domain import TypeLabel


MANDATORY_CORE_DATASETS = ("TruthfulQA", "TriviaQA", "HaluEval-QA")
FUTURE_DATASETS = ("Natural Questions", "HotpotQA", "FEVER", "BioASQ")
EXPECTED_LABELS = [label.value for label in TypeLabel]
EXPECTED_THRESHOLD_RULES = {
    "NORMAL": {"rule": "correct -> NORMAL"},
    "HIGH_DIVERSITY": {
        "rule": "incorrect and SE > 0.5 -> HIGH_DIVERSITY",
        "min_exclusive": 0.5,
    },
    "LOW_DIVERSITY": {
        "rule": "incorrect and SE <= 0.1 -> LOW_DIVERSITY",
        "max_inclusive": 0.1,
    },
    "AMBIGUOUS_INCORRECT": {
        "rule": "incorrect and 0.1 < SE <= 0.5 -> AMBIGUOUS_INCORRECT",
        "min_exclusive": 0.1,
        "max_inclusive": 0.5,
    },
}
AMBIGUOUS_POLICY_MARKERS = ("same as", "tbd", "todo", "implicit")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON-compatible YAML file {path}: {exc}") from exc


def is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_operational_labels(config: dict, problems: list[str]) -> None:
    label_policy = config.get("label_policy")
    if not isinstance(label_policy, dict):
        problems.append("missing top-level label_policy object")
        return

    labels = label_policy.get("operational_labels")
    if labels != EXPECTED_LABELS:
        problems.append(f"operational_labels must equal {EXPECTED_LABELS}; got {labels!r}")

    row_level_output = label_policy.get("row_level_output")
    if not isinstance(row_level_output, dict):
        problems.append("label_policy.row_level_output must be an object")
    else:
        if row_level_output.get("required_label_field") != "label":
            problems.append("row_level_output.required_label_field must be 'label'")
        if row_level_output.get("require_explicit_operational_label_marker") is not True:
            problems.append("row_level_output must require an explicit operational label marker")
        if row_level_output.get("required_for_every_sample") is not True:
            problems.append("row_level_output must require the label marker for every sample")
        if row_level_output.get("allowed_values") != EXPECTED_LABELS:
            problems.append("row_level_output.allowed_values must match the four operational labels")

    thresholds = label_policy.get("fixed_operational_thresholds")
    if thresholds != EXPECTED_THRESHOLD_RULES:
        problems.append("fixed_operational_thresholds must exactly match the README label rules and thresholds")


def validate_analysis_bins(config: dict, problems: list[str]) -> None:
    label_policy = config.get("label_policy")
    if not isinstance(label_policy, dict):
        return

    analysis = label_policy.get("analysis_se_bins")
    if not isinstance(analysis, dict):
        problems.append("label_policy.analysis_se_bins must be an object")
        return

    if analysis.get("analysis_only") is not True:
        problems.append("analysis_se_bins must be marked analysis_only=true")
    if analysis.get("spans") != "[0, +inf)":
        problems.append("analysis_se_bins.spans must be '[0, +inf)'")
    if analysis.get("special_attention_thresholds") != [0.1, 0.5]:
        problems.append("analysis_se_bins.special_attention_thresholds must be [0.1, 0.5]")

    bins = analysis.get("bins")
    if not isinstance(bins, list) or not bins:
        problems.append("analysis_se_bins.bins must be a non-empty list")
        return

    boundaries: set[float] = set()
    saw_open_ended_tail = False
    last_upper: float | None = None
    previous_upper: float | None = None
    previous_upper_inclusive: bool | None = None

    for index, bin_spec in enumerate(bins):
        if not isinstance(bin_spec, dict):
            problems.append(f"analysis bin at index {index} must be an object")
            continue
        if not is_nonempty_string(bin_spec.get("bin_id")):
            problems.append(f"analysis bin at index {index} is missing bin_id")
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
            problems.append(
                f"analysis bin {bin_spec.get('bin_id', index)!r} is zero-width and must include both bounds"
            )
        if index == 0:
            if float(lower_bound) != 0.0 or lower_inclusive is not True:
                problems.append("analysis_se_bins must start at 0.0 with an inclusive first bin")
        if previous_upper is not None:
            current_lower = float(lower_bound)
            if current_lower < previous_upper:
                problems.append(
                    f"analysis bins overlap or are out of order at {bin_spec.get('bin_id', index)!r}: "
                    f"lower_bound {current_lower} is below previous upper_bound {previous_upper}"
                )
            elif current_lower > previous_upper:
                problems.append(
                    f"analysis bins leave a gap before {bin_spec.get('bin_id', index)!r}: "
                    f"lower_bound {current_lower} is above previous upper_bound {previous_upper}"
                )
            elif previous_upper_inclusive and lower_inclusive:
                problems.append(
                    f"analysis bins overlap at boundary {current_lower} near {bin_spec.get('bin_id', index)!r}; "
                    "adjacent bins cannot both include the shared boundary"
                )
            elif not previous_upper_inclusive and not lower_inclusive:
                problems.append(
                    f"analysis bins leave boundary {current_lower} uncovered near {bin_spec.get('bin_id', index)!r}; "
                    "one adjacent bin must include the shared boundary"
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


def validate_dataset_entry(dataset: dict, problems: list[str]) -> None:
    name = dataset.get("name", "<unknown>")
    required_fields = (
        "name",
        "role",
        "config",
        "split",
        "split_id",
        "target_sample_count",
        "seed",
        "label_policy",
        "label_source",
        "notes",
    )
    for field in required_fields:
        if field not in dataset:
            problems.append(f"dataset {name!r} is missing required field: {field}")
    if "hf_id" not in dataset and "source_id" not in dataset:
        problems.append(f"dataset {name!r} must define hf_id or source_id")

    role = dataset.get("role")
    if role not in {"core", "future"}:
        problems.append(f"dataset {name!r} role must be 'core' or 'future'; got {role!r}")

    target_sample_count = dataset.get("target_sample_count")
    if not isinstance(target_sample_count, int) or target_sample_count <= 0:
        problems.append(f"dataset {name!r} target_sample_count must be a positive integer")

    seed = dataset.get("seed")
    if not isinstance(seed, int):
        problems.append(f"dataset {name!r} seed must be an integer")

    split_id = dataset.get("split_id")
    if not is_nonempty_string(split_id):
        problems.append(f"dataset {name!r} split_id must be a non-empty string")

    label_source = dataset.get("label_source")
    if not isinstance(label_source, dict):
        problems.append(f"dataset {name!r} label_source must be an object")
    else:
        if not is_nonempty_string(label_source.get("type")):
            problems.append(f"dataset {name!r} label_source.type must be a non-empty string")
        if not is_nonempty_string(label_source.get("details")):
            problems.append(f"dataset {name!r} label_source.details must be a non-empty string")

    label_policy = dataset.get("label_policy")
    if not isinstance(label_policy, dict):
        problems.append(f"dataset {name!r} label_policy must be an object")
        return

    for field in (
        "correctness_judgment",
        "operational_label_assignment",
        "se_threshold_reference",
        "row_level_output",
    ):
        value = label_policy.get(field)
        if not is_nonempty_string(value):
            problems.append(f"dataset {name!r} label_policy.{field} must be a non-empty string")
            continue
        lowered = str(value).lower()
        if any(marker in lowered for marker in AMBIGUOUS_POLICY_MARKERS):
            problems.append(f"dataset {name!r} label_policy.{field} is too ambiguous for audit")

    if label_policy.get("se_threshold_reference") != "label_policy.fixed_operational_thresholds":
        problems.append(f"dataset {name!r} must reference top-level fixed operational thresholds")

    combined_policy_text = " ".join(
        str(part)
        for part in (
            label_source.get("type") if isinstance(label_source, dict) else "",
            label_source.get("details") if isinstance(label_source, dict) else "",
            label_policy.get("correctness_judgment", ""),
            label_policy.get("operational_label_assignment", ""),
        )
    ).lower()

    if name == "TriviaQA" and not any(token in combined_policy_text for token in ("alias", "canonical", "gold")):
        problems.append("TriviaQA must define an explicit alias-aware/gold-answer label policy")
    if name == "HaluEval-QA" and not any(token in combined_policy_text for token in ("hallucination", "annotation", "right-answer", "right answer")):
        problems.append("HaluEval-QA must define an explicit annotation-based label policy")

    promotion = dataset.get("promotion")
    if promotion is not None:
        if not isinstance(promotion, dict):
            problems.append(f"dataset {name!r} promotion must be an object when present")
        else:
            for flag_name in ("promoted_to_core", "reported_in_run"):
                if flag_name in promotion and not isinstance(promotion[flag_name], bool):
                    problems.append(f"dataset {name!r} promotion.{flag_name} must be a boolean")


def validate_dataset_registry(config: dict, problems: list[str]) -> None:
    if config.get("mandatory_core_datasets") != list(MANDATORY_CORE_DATASETS):
        problems.append(f"mandatory_core_datasets must equal {list(MANDATORY_CORE_DATASETS)}")
    if config.get("future_datasets") != list(FUTURE_DATASETS):
        problems.append(f"future_datasets must equal {list(FUTURE_DATASETS)}")

    expansion_policy = config.get("dataset_expansion_policy")
    if not isinstance(expansion_policy, dict):
        problems.append("dataset_expansion_policy must be an object")
    else:
        if expansion_policy.get("default_selected_roles") != ["core"]:
            problems.append("dataset_expansion_policy.default_selected_roles must be ['core']")
        if expansion_policy.get("allow_optional_promotions") is not True:
            problems.append("dataset_expansion_policy.allow_optional_promotions must be true")
        if expansion_policy.get("require_explicit_promotion") is not True:
            problems.append("dataset_expansion_policy.require_explicit_promotion must be true")
        if expansion_policy.get("require_reported_expansion") is not True:
            problems.append("dataset_expansion_policy.require_reported_expansion must be true")

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        problems.append("datasets must be a non-empty list")
        return

    names: list[str] = []
    core_names: list[str] = []
    promoted_names: list[str] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            problems.append("every dataset entry must be an object")
            continue
        validate_dataset_entry(dataset, problems)
        name = dataset.get("name")
        if is_nonempty_string(name):
            normalized_name = str(name)
            names.append(normalized_name)
            if dataset.get("role") == "core":
                core_names.append(normalized_name)
        promotion = dataset.get("promotion") or {}
        if isinstance(promotion, dict) and promotion.get("promoted_to_core") is True:
            promoted_name = str(name) if is_nonempty_string(name) else "<unknown>"
            promoted_names.append(promoted_name)
            if dataset.get("role") != "future":
                problems.append(f"promoted dataset {promoted_name!r} must originate from role='future'")
            if promotion.get("reported_in_run") is not True:
                problems.append(f"promoted dataset {promoted_name!r} must set promotion.reported_in_run=true")
            if not is_nonempty_string(promotion.get("reason")):
                problems.append(f"promoted dataset {promoted_name!r} must include promotion.reason for auditability")
            if not is_nonempty_string(promotion.get("approved_by")):
                problems.append(f"promoted dataset {promoted_name!r} must include promotion.approved_by for auditability")

    if len(names) != len(set(names)):
        problems.append("dataset names must be unique")

    if set(core_names) != set(MANDATORY_CORE_DATASETS):
        problems.append(
            f"core role entries must be exactly the mandatory core datasets {list(MANDATORY_CORE_DATASETS)}; got {core_names}"
        )
    if len(core_names) != 3:
        problems.append(f"there must be exactly three role='core' datasets by default; got {len(core_names)}")

    missing_mandatory = sorted(set(str(name) for name in MANDATORY_CORE_DATASETS) - set(names))
    if missing_mandatory:
        problems.append(f"missing mandatory datasets: {missing_mandatory}")

    missing_future = sorted(set(str(name) for name in FUTURE_DATASETS) - set(names))
    if missing_future:
        problems.append(f"missing future/stretch datasets: {missing_future}")

    selected_core_count = len(core_names) + len(promoted_names)
    if selected_core_count < 3:
        problems.append("selected core set must contain at least the three mandatory core datasets")


def validate_config(config: dict) -> list[str]:
    problems: list[str] = []
    validate_operational_labels(config, problems)
    validate_analysis_bins(config, problems)
    validate_dataset_registry(config, problems)
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: python3 experiments/scripts/validate_datasets.py <datasets.yaml>",
            file=sys.stderr,
        )
        return 2

    config_path = Path(argv[1])
    config = load_json(config_path)
    problems = validate_config(config)
    if problems:
        print("Dataset registry validation failed.", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    datasets = config.get("datasets", [])
    selected = [dataset["name"] for dataset in datasets if dataset.get("role") == "core"]
    promoted = [
        dataset["name"]
        for dataset in datasets
        if isinstance(dataset.get("promotion"), dict) and dataset["promotion"].get("promoted_to_core")
    ]
    if promoted:
        selected.extend(promoted)

    print("Dataset registry validation passed.")
    print(json.dumps({
        "config_path": str(config_path),
        "mandatory_core_datasets": list(MANDATORY_CORE_DATASETS),
        "selected_core_datasets": selected,
        "promoted_optional_datasets": promoted,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
