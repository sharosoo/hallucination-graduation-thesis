"""Corpus-bin reliability robustness reporting."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.adapters.corpus_features import (
    FIVE_BIN_RULES,
    THREE_BIN_RULES,
    load_json,
    read_feature_rows,
    write_json,
)
from experiments.application.fusion import (
    DATASET_MANIFEST_REF,
    FORMULA_MANIFEST_REF,
    _coerce_float,
    _compute_auprc,
    _compute_auroc,
)

BOOTSTRAP_SEED = 20260507
BOOTSTRAP_ITERATIONS = 2000
PRIMARY_BIN_FIELD = "corpus_axis_bin"
SENSITIVITY_BIN_FIELD = "corpus_axis_bin_5"
CLAIM_KEYS = {"claim_text", "summary_statement", "headline", "conclusion", "wording", "note"}
FORBIDDEN_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bsignificant improvement\b", "significant improvement"),
    (r"\bsignificantly (?:improves?|improved|better)\b", "significantly improves"),
    (r"\bproven\b", "proven"),
    (r"\bproves\b", "proves"),
    (r"\bconfirmed\b", "confirmed"),
    (r"\bimproves?\b", "improves"),
    (r"\bbetter\b", "better"),
    (r"\bbest\b", "best"),
    (r"\bsuperior\b", "superior"),
    (r"\boutperform(?:s|ed)?\b", "outperforms"),
    (r"\bwin(?:s|ning)?\b", "wins"),
    (r"\bcauses?\b", "causes"),
    (r"\bdue to\b", "due to"),
    (r"\bbecause of\b", "because of"),
)
METHOD_COMPARISON_CANDIDATES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("global learned fusion with corpus axis", "learned fusion with corpus"), ("global learned fusion without corpus axis", "learned fusion without corpus")),
    (("global learned fusion with corpus axis", "learned fusion with corpus"), ("SE-only",)),
    (("global learned fusion without corpus axis", "learned fusion without corpus"), ("SE-only",)),
    (("corpus-bin weighted fusion",), ("global learned fusion without corpus axis", "learned fusion without corpus")),
    (("axis-interaction logistic fusion",), ("global learned fusion without corpus axis", "learned fusion without corpus")),
    (("corpus-bin feature selection",), ("global learned fusion without corpus axis", "learned fusion without corpus")),
    (("corpus-axis-only", "corpus-risk-only"), ("SE-only",)),
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _target_value(row: dict[str, Any]) -> int:
    is_hallucination = _coerce_bool(row.get("is_hallucination"))
    if is_hallucination is not None:
        return 1 if is_hallucination else 0
    is_correct = _coerce_bool(row.get("is_correct"))
    if is_correct is not None:
        return 0 if is_correct else 1
    raise ValueError(f"Missing annotation-backed hallucination target for sample {row.get('sample_id')!r}")


def _prompt_id(row: dict[str, Any]) -> str:
    return str(row.get("prompt_id") or "unknown_prompt")


def _pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or row.get("prompt_id") or row.get("sample_id") or "unknown_pair")


def _sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("candidate_id") or "unknown_sample")


def _dataset(row: dict[str, Any]) -> str:
    return str(row.get("dataset") or "unknown")


def _candidate_role(row: dict[str, Any]) -> str | None:
    raw = row.get("candidate_label")
    if raw is None:
        raw = row.get("candidate_role")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def _bin_value(row: dict[str, Any], field_name: str) -> str | None:
    features = row.get("features")
    if not isinstance(features, dict):
        return None
    raw = features.get(field_name)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _expected_bins(field_name: str) -> tuple[str, ...]:
    rules = THREE_BIN_RULES if field_name == PRIMARY_BIN_FIELD else FIVE_BIN_RULES
    return tuple(str(name) for name, _cutoff in rules)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = rank - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _brier_score(labels: list[int], probabilities: list[float]) -> float | None:
    if not labels or len(labels) != len(probabilities):
        return None
    return sum((probability - label) ** 2 for label, probability in zip(labels, probabilities, strict=True)) / len(labels)


def _is_probability_like(scores: list[float]) -> bool:
    return bool(scores) and all(0.0 <= score <= 1.0 for score in scores)


def _group_by_prompt(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_prompt_id(row), []).append(row)
    return grouped


def _build_prompt_units(
    truth_rows: list[dict[str, Any]],
    candidate_predictions: dict[str, dict[str, Any]],
    reference_predictions: dict[str, dict[str, Any]],
) -> tuple[list[list[tuple[int, float, float]]], int]:
    truth_lookup = {_sample_id(row): row for row in truth_rows}
    prompt_groups = _group_by_prompt(truth_rows)
    prompt_units: list[list[tuple[int, float, float]]] = []
    sample_count = 0
    for prompt_id in sorted(prompt_groups):
        unit: list[tuple[int, float, float]] = []
        for truth_row in sorted(prompt_groups[prompt_id], key=_sample_id):
            sample_id = _sample_id(truth_row)
            candidate_row = candidate_predictions.get(sample_id)
            reference_row = reference_predictions.get(sample_id)
            if candidate_row is None or reference_row is None or sample_id not in truth_lookup:
                unit = []
                break
            candidate_score = _coerce_float(candidate_row.get("prediction_score"))
            reference_score = _coerce_float(reference_row.get("prediction_score"))
            if candidate_score is None or reference_score is None:
                unit = []
                break
            unit.append((_target_value(truth_lookup[sample_id]), candidate_score, reference_score))
        if unit:
            sample_count += len(unit)
            prompt_units.append(unit)
    return prompt_units, sample_count


def _metric_from_scores(labels: list[int], scores: list[float], metric_name: str) -> float | None:
    if metric_name == "auroc":
        return _compute_auroc(labels, scores)
    if metric_name == "auprc":
        return _compute_auprc(labels, scores)
    raise ValueError(f"Unsupported metric {metric_name!r}")


def _bootstrap_metric_delta_by_prompt(
    prompt_units: list[list[tuple[int, float, float]]],
    *,
    metric_name: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    labels = [label for unit in prompt_units for label, _candidate, _reference in unit]
    candidate_scores = [candidate for unit in prompt_units for _label, candidate, _reference in unit]
    reference_scores = [reference for unit in prompt_units for _label, _candidate, reference in unit]
    observed_candidate = _metric_from_scores(labels, candidate_scores, metric_name)
    observed_reference = _metric_from_scores(labels, reference_scores, metric_name)
    if observed_candidate is None or observed_reference is None:
        return {
            "metric": metric_name,
            "observed_delta": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
            "bootstrap_mean_delta": None,
            "bootstrap_seed": seed,
            "bootstrap_iterations": iterations,
            "valid_iterations": 0,
            "ci_crosses_zero": None,
            "statistically_significant": False,
            "claim_text": "Observed delta only. Prompt-grouped resampling cannot estimate this metric because the target collapses in the available units.",
        }

    observed_delta = observed_candidate - observed_reference
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = [prompt_units[rng.randrange(len(prompt_units))] for _unused in range(len(prompt_units))]
        boot_labels = [label for unit in sampled for label, _candidate, _reference in unit]
        if len(set(boot_labels)) < 2:
            continue
        boot_candidate = [candidate for unit in sampled for _label, candidate, _reference in unit]
        boot_reference = [reference for unit in sampled for _label, _candidate, reference in unit]
        candidate_metric = _metric_from_scores(boot_labels, boot_candidate, metric_name)
        reference_metric = _metric_from_scores(boot_labels, boot_reference, metric_name)
        if candidate_metric is None or reference_metric is None:
            continue
        deltas.append(candidate_metric - reference_metric)
    ci_lower = _percentile(deltas, 0.025)
    ci_upper = _percentile(deltas, 0.975)
    crosses_zero = None if ci_lower is None or ci_upper is None else ci_lower <= 0.0 <= ci_upper
    return {
        "metric": metric_name,
        "observed_delta": observed_delta,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "bootstrap_mean_delta": _mean(deltas),
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "valid_iterations": len(deltas),
        "ci_crosses_zero": crosses_zero,
        "statistically_significant": False,
        "claim_text": (
            "Observed delta only. The prompt-grouped interval includes zero or is unavailable."
            if crosses_zero is not False
            else "Observed delta only. The prompt-grouped interval stays on one side of zero in this run."
        ),
    }


def _baseline_lookup(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    baselines = summary.get("baselines")
    if not isinstance(baselines, list):
        return {}
    return {
        str(item.get("method_name")): item
        for item in baselines
        if isinstance(item, dict) and item.get("method_name")
    }


def _predictions_by_method(predictions: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in predictions:
        method_name = str(row.get("method_name") or "unknown_method")
        grouped.setdefault(method_name, {})[_sample_id(row)] = row
    return grouped


def _find_available_method(available: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _build_method_bootstrap_report(
    truth_rows: list[dict[str, Any]],
    predictions_by_method: dict[str, dict[str, dict[str, Any]]],
    baseline_lookup: dict[str, dict[str, Any]],
    *,
    seed: int,
    iterations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available = set(predictions_by_method) | set(baseline_lookup)
    comparisons: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for comparison_index, (candidate_names, reference_names) in enumerate(METHOD_COMPARISON_CANDIDATES):
        candidate_method = _find_available_method(available, candidate_names)
        reference_method = _find_available_method(available, reference_names)
        if candidate_method is None or reference_method is None:
            continue
        prompt_units, sample_count = _build_prompt_units(
            truth_rows,
            predictions_by_method.get(candidate_method, {}),
            predictions_by_method.get(reference_method, {}),
        )
        if not prompt_units:
            comparisons.append(
                {
                    "candidate_method": candidate_method,
                    "reference_method": reference_method,
                    "status": "unavailable",
                    "reason": "no_common_prompt_grouped_predictions",
                    "metrics": [],
                }
            )
            continue
        metrics: list[dict[str, Any]] = []
        for metric_offset, metric_name in enumerate(("auroc", "auprc")):
            result = _bootstrap_metric_delta_by_prompt(
                prompt_units,
                metric_name=metric_name,
                seed=seed + (comparison_index * 10) + metric_offset,
                iterations=iterations,
            )
            result.update(
                {
                    "candidate_method": candidate_method,
                    "reference_method": reference_method,
                    "sample_count": sample_count,
                    "prompt_group_count": len(prompt_units),
                    "resampling_unit": "prompt_id",
                }
            )
            metrics.append(result)
            claim_rows.append(
                {
                    "comparison_id": f"{candidate_method}__minus__{reference_method}__{metric_name}".replace(" ", "_"),
                    "candidate_method": candidate_method,
                    "reference_method": reference_method,
                    "metric": metric_name,
                    "observed_delta": result.get("observed_delta"),
                    "ci_95_lower": result.get("ci_95_lower"),
                    "ci_95_upper": result.get("ci_95_upper"),
                    "statistically_significant": False,
                    "claim_text": result.get("claim_text"),
                }
            )
        comparisons.append(
            {
                "candidate_method": candidate_method,
                "reference_method": reference_method,
                "status": "ok",
                "sample_count": sample_count,
                "prompt_group_count": len(prompt_units),
                "resampling_unit": "prompt_id",
                "metrics": metrics,
            }
        )
    return {
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "comparisons": comparisons,
    }, claim_rows


def _resolve_fusion_artifacts(fusion_path: Path) -> tuple[Path, Path, Path, Path]:
    if fusion_path.is_file():
        if fusion_path.name != "summary.json":
            raise ValueError(f"--fusion file must be summary.json when a file path is provided: {fusion_path}")
        fusion_dir = fusion_path.parent
        summary_path = fusion_path
    else:
        fusion_dir = fusion_path
        summary_path = fusion_dir / "summary.json"
    baseline_metrics_path = fusion_dir / "baseline_metrics.csv"
    predictions_path = fusion_dir / "predictions.jsonl"
    missing = [str(path) for path in (summary_path, baseline_metrics_path, predictions_path) if not path.exists()]
    if missing:
        raise ValueError("Missing fusion artifact(s): " + ", ".join(missing))
    return fusion_dir, summary_path, baseline_metrics_path, predictions_path


def _score_summary(scores: list[float]) -> dict[str, Any]:
    return {
        "mean": _mean(scores),
        "min": None if not scores else min(scores),
        "max": None if not scores else max(scores),
        "std": _stdev(scores),
    }


def _prediction_confusion(labels: list[int], predictions: list[int]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for label, predicted in zip(labels, predictions, strict=True):
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    confusion = _prediction_confusion(labels, predictions)
    tp = confusion["tp"]
    fp = confusion["fp"]
    tn = confusion["tn"]
    fn = confusion["fn"]
    total = tp + fp + tn + fn
    accuracy = None if total == 0 else (tp + tn) / total
    precision = None if (tp + fp) == 0 else tp / (tp + fp)
    recall = None if (tp + fn) == 0 else tp / (tp + fn)
    if precision is None or recall is None or precision == 0.0 or recall == 0.0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": None if total == 0 else (tp + fp) / total,
        "confusion": confusion,
    }


def _pair_bootstrap(units: list[dict[str, Any]], *, seed: int, iterations: int) -> dict[str, Any]:
    if not units:
        return {
            "bootstrap_seed": seed,
            "bootstrap_iterations": iterations,
            "valid_iterations": 0,
            "win_rate_ci_95": [None, None],
            "mean_delta_ci_95": [None, None],
            "auroc_ci_95": [None, None],
            "auprc_ci_95": [None, None],
            "claim_text": "Observed paired reliability summary only. No prompt-grouped units are available for resampling.",
        }
    rng = random.Random(seed)
    win_rates: list[float] = []
    mean_deltas: list[float] = []
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(iterations):
        sampled = [units[rng.randrange(len(units))] for _unused in range(len(units))]
        wins = [float(item["win_value"]) for item in sampled]
        deltas = [float(item["delta"]) for item in sampled]
        win_rates.append(sum(wins) / len(wins))
        mean_deltas.append(sum(deltas) / len(deltas))
        labels: list[int] = []
        scores: list[float] = []
        for unit in sampled:
            labels.extend((0, 1))
            scores.extend((float(unit["correct_score"]), float(unit["hallucinated_score"])))
        auroc = _compute_auroc(labels, scores)
        auprc = _compute_auprc(labels, scores)
        if auroc is not None:
            aurocs.append(auroc)
        if auprc is not None:
            auprcs.append(auprc)
    return {
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "valid_iterations": len(win_rates),
        "win_rate_ci_95": [_percentile(win_rates, 0.025), _percentile(win_rates, 0.975)],
        "mean_delta_ci_95": [_percentile(mean_deltas, 0.025), _percentile(mean_deltas, 0.975)],
        "auroc_ci_95": [_percentile(aurocs, 0.025), _percentile(aurocs, 0.975)],
        "auprc_ci_95": [_percentile(auprcs, 0.025), _percentile(auprcs, 0.975)],
        "claim_text": "Observed paired reliability summary only. Prompt-grouped bootstrap intervals are descriptive and do not license directional or causal wording.",
    }


def _build_method_records(
    truth_rows: list[dict[str, Any]],
    predictions_by_method: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    truth_lookup = {_sample_id(row): row for row in truth_rows}
    method_records: dict[str, list[dict[str, Any]]] = {}
    for method_name, sample_map in predictions_by_method.items():
        records: list[dict[str, Any]] = []
        for sample_id, prediction in sample_map.items():
            truth_row = truth_lookup.get(sample_id)
            if truth_row is None:
                continue
            score = _coerce_float(prediction.get("prediction_score"))
            if score is None:
                continue
            prediction_label = _coerce_bool(prediction.get("prediction_label"))
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": _dataset(truth_row),
                    "prompt_id": _prompt_id(truth_row),
                    "pair_id": _pair_id(truth_row),
                    "candidate_role": _candidate_role(truth_row) or _candidate_role(prediction),
                    "target": _target_value(truth_row),
                    "score": score,
                    "prediction_label": None if prediction_label is None else int(prediction_label),
                    "threshold": _coerce_float(prediction.get("threshold")),
                    PRIMARY_BIN_FIELD: _bin_value(truth_row, PRIMARY_BIN_FIELD),
                    SENSITIVITY_BIN_FIELD: _bin_value(truth_row, SENSITIVITY_BIN_FIELD),
                }
            )
        method_records[method_name] = records
    return method_records


def _pair_records(records: list[dict[str, Any]], field_name: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["pair_id"]), []).append(record)
    pairs: list[dict[str, Any]] = []
    diagnostics = {
        "complete_pair_count": 0,
        "missing_candidate_role_count": 0,
        "missing_bin_count": 0,
        "incomplete_pair_count": 0,
    }
    for pair_id in sorted(grouped):
        members = grouped[pair_id]
        correct_record: dict[str, Any] | None = None
        hallucinated_record: dict[str, Any] | None = None
        for record in members:
            target = int(record["target"])
            role = str(record.get("candidate_role") or "")
            if target == 0 and correct_record is None:
                correct_record = record
            if target == 1 and hallucinated_record is None:
                hallucinated_record = record
            if role in {"right", "correct", "normal"} and correct_record is None:
                correct_record = record
            if role in {"hallucinated", "incorrect", "wrong"} and hallucinated_record is None:
                hallucinated_record = record
        if correct_record is None or hallucinated_record is None:
            diagnostics["incomplete_pair_count"] += 1
            continue
        bin_value = correct_record.get(field_name) or hallucinated_record.get(field_name)
        if bin_value is None:
            diagnostics["missing_bin_count"] += 1
            continue
        delta = float(hallucinated_record["score"]) - float(correct_record["score"])
        win_value = 1.0 if delta > 0 else 0.0 if delta < 0 else 0.5
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt_id": correct_record["prompt_id"],
                "dataset": correct_record["dataset"],
                "bin": str(bin_value),
                "correct_score": float(correct_record["score"]),
                "hallucinated_score": float(hallucinated_record["score"]),
                "delta": delta,
                "win_value": win_value,
            }
        )
        diagnostics["complete_pair_count"] += 1
    return pairs, diagnostics


def _insufficient_bin_entry(
    *,
    bin_name: str,
    dataset_breakdown: dict[str, int],
    row_count: int,
    prompt_count: int,
    pair_count: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "bin": bin_name,
        "status": "insufficient",
        "reason": reason,
        "row_count": row_count,
        "prompt_count": prompt_count,
        "pair_count": pair_count,
        "dataset_breakdown": dataset_breakdown,
        "row_metrics": {
            "auroc": None,
            "auprc": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "predicted_positive_rate": None,
            "confusion": None,
        },
        "paired": {
            "win_rate": None,
            "mean_delta": None,
            "median_delta": None,
            "positive_delta_fraction": None,
            "bootstrap": {
                "bootstrap_seed": None,
                "bootstrap_iterations": 0,
                "valid_iterations": 0,
                "win_rate_ci_95": [None, None],
                "mean_delta_ci_95": [None, None],
                "auroc_ci_95": [None, None],
                "auprc_ci_95": [None, None],
                "claim_text": "Observed paired reliability summary is unavailable because this corpus bin has insufficient paired support.",
            },
        },
        "claim_text": f"Observed {bin_name} corpus bin omitted from directional interpretation because support is insufficient ({reason}).",
    }


def _analyze_bin(
    *,
    method_name: str,
    bin_name: str,
    bin_records: list[dict[str, Any]],
    bin_pairs: list[dict[str, Any]],
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    labels = [int(record["target"]) for record in bin_records]
    scores = [float(record["score"]) for record in bin_records]
    predictions = [int(record["prediction_label"]) for record in bin_records if record.get("prediction_label") is not None]
    row_count = len(bin_records)
    prompt_count = len({str(record["prompt_id"]) for record in bin_records})
    pair_count = len(bin_pairs)
    dataset_breakdown: dict[str, int] = {}
    for record in bin_records:
        dataset_breakdown[str(record["dataset"])] = dataset_breakdown.get(str(record["dataset"]), 0) + 1
    if row_count == 0:
        return _insufficient_bin_entry(
            bin_name=bin_name,
            dataset_breakdown=dataset_breakdown,
            row_count=0,
            prompt_count=0,
            pair_count=0,
            reason="bin_not_present",
        )
    if len(set(labels)) < 2:
        return _insufficient_bin_entry(
            bin_name=bin_name,
            dataset_breakdown=dataset_breakdown,
            row_count=row_count,
            prompt_count=prompt_count,
            pair_count=pair_count,
            reason="single_class_rows",
        )
    if pair_count < 2 or prompt_count < 2:
        return _insufficient_bin_entry(
            bin_name=bin_name,
            dataset_breakdown=dataset_breakdown,
            row_count=row_count,
            prompt_count=prompt_count,
            pair_count=pair_count,
            reason="fewer_than_two_prompt_pairs",
        )
    threshold_values = [float(record["threshold"]) for record in bin_records if record.get("threshold") is not None]
    row_metrics = {
        "auroc": _compute_auroc(labels, scores),
        "auprc": _compute_auprc(labels, scores),
        **(
            _classification_metrics(labels, [int(record["prediction_label"]) for record in bin_records])
            if len(predictions) == row_count
            else {"accuracy": None, "precision": None, "recall": None, "f1": None, "predicted_positive_rate": None, "confusion": None}
        ),
        "score_summary": _score_summary(scores),
        "threshold_summary": {
            "count": len(threshold_values),
            "unique_count": len({round(value, 12) for value in threshold_values}),
            "min": None if not threshold_values else min(threshold_values),
            "max": None if not threshold_values else max(threshold_values),
            "mean": _mean(threshold_values),
        },
    }
    deltas = [float(pair["delta"]) for pair in bin_pairs]
    wins = [float(pair["win_value"]) for pair in bin_pairs]
    bootstrap = _pair_bootstrap(bin_pairs, seed=seed, iterations=iterations)
    return {
        "bin": bin_name,
        "status": "ok",
        "reason": None,
        "row_count": row_count,
        "prompt_count": prompt_count,
        "pair_count": pair_count,
        "dataset_breakdown": dataset_breakdown,
        "row_metrics": row_metrics,
        "paired": {
            "win_rate": _mean(wins),
            "mean_delta": _mean(deltas),
            "median_delta": _percentile(deltas, 0.5),
            "positive_delta_fraction": None if not deltas else sum(1 for value in deltas if value > 0) / len(deltas),
            "bootstrap": bootstrap,
        },
        "claim_text": (
            f"Observed {bin_name} corpus-bin reliability for {method_name} from paired prompt units only. "
            "Intervals are descriptive and do not justify superiority or causal wording."
        ),
    }


def _analyze_scheme(
    *,
    field_name: str,
    method_name: str,
    records: list[dict[str, Any]],
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    expected_bins = _expected_bins(field_name)
    pairs, pair_diagnostics = _pair_records(records, field_name)
    records_by_bin = {bin_name: [] for bin_name in expected_bins}
    pairs_by_bin = {bin_name: [] for bin_name in expected_bins}
    for record in records:
        bin_name = record.get(field_name)
        if bin_name in records_by_bin:
            records_by_bin[str(bin_name)].append(record)
    for pair in pairs:
        bin_name = pair.get("bin")
        if bin_name in pairs_by_bin:
            pairs_by_bin[str(bin_name)].append(pair)
    bins: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    sufficient_bins = 0
    for index, bin_name in enumerate(expected_bins):
        bin_report = _analyze_bin(
            method_name=method_name,
            bin_name=bin_name,
            bin_records=records_by_bin[bin_name],
            bin_pairs=pairs_by_bin[bin_name],
            seed=seed + index,
            iterations=iterations,
        )
        bins.append(bin_report)
        if bin_report.get("status") == "ok":
            sufficient_bins += 1
        claim_rows.append(
            {
                "method_name": method_name,
                "bin_field": field_name,
                "bin": bin_name,
                "metric": "paired_mean_delta",
                "observed_delta": ((bin_report.get("paired") or {}).get("mean_delta")),
                "ci_95_lower": (((bin_report.get("paired") or {}).get("bootstrap") or {}).get("mean_delta_ci_95") or [None, None])[0],
                "ci_95_upper": (((bin_report.get("paired") or {}).get("bootstrap") or {}).get("mean_delta_ci_95") or [None, None])[1],
                "statistically_significant": False,
                "claim_text": bin_report.get("claim_text"),
            }
        )
    valid_bin_rows = [row for row in bins if row.get("status") == "ok"]
    valid_aurocs = [float(row["row_metrics"]["auroc"]) for row in valid_bin_rows if _coerce_float((row.get("row_metrics") or {}).get("auroc")) is not None]
    valid_win_rates = [float(row["paired"]["win_rate"]) for row in valid_bin_rows if _coerce_float((row.get("paired") or {}).get("win_rate")) is not None]
    return {
        "field_name": field_name,
        "expected_bins": list(expected_bins),
        "method_name": method_name,
        "pair_diagnostics": pair_diagnostics,
        "sufficient_bin_count": sufficient_bins,
        "insufficient_bin_count": len(expected_bins) - sufficient_bins,
        "bins": bins,
        "scheme_summary": {
            "worst_bin_auroc": None if not valid_aurocs else min(valid_aurocs),
            "best_bin_auroc": None if not valid_aurocs else max(valid_aurocs),
            "worst_bin_win_rate": None if not valid_win_rates else min(valid_win_rates),
            "best_bin_win_rate": None if not valid_win_rates else max(valid_win_rates),
        },
        "claim_rows": claim_rows,
    }


def _scheme_sensitivity(primary_scheme: dict[str, Any], sensitivity_scheme: dict[str, Any]) -> dict[str, Any]:
    primary_summary = primary_scheme.get("scheme_summary") or {}
    sensitivity_summary = sensitivity_scheme.get("scheme_summary") or {}
    return {
        "primary_field": primary_scheme.get("field_name"),
        "sensitivity_field": sensitivity_scheme.get("field_name"),
        "primary_sufficient_bin_count": primary_scheme.get("sufficient_bin_count"),
        "sensitivity_sufficient_bin_count": sensitivity_scheme.get("sufficient_bin_count"),
        "worst_bin_auroc_delta": _difference(primary_summary.get("worst_bin_auroc"), sensitivity_summary.get("worst_bin_auroc")),
        "best_bin_auroc_delta": _difference(primary_summary.get("best_bin_auroc"), sensitivity_summary.get("best_bin_auroc")),
        "worst_bin_win_rate_delta": _difference(primary_summary.get("worst_bin_win_rate"), sensitivity_summary.get("worst_bin_win_rate")),
        "best_bin_win_rate_delta": _difference(primary_summary.get("best_bin_win_rate"), sensitivity_summary.get("best_bin_win_rate")),
        "claim_text": "Observed 3-bin versus 5-bin sensitivity only. Bin-count changes are descriptive stratification checks, not optimized bin selection evidence.",
    }


def _difference(left: Any, right: Any) -> float | None:
    left_value = _coerce_float(left)
    right_value = _coerce_float(right)
    if left_value is None or right_value is None:
        return None
    return right_value - left_value


def _build_corpus_bin_reliability(
    truth_rows: list[dict[str, Any]],
    predictions_by_method: dict[str, dict[str, dict[str, Any]]],
    *,
    seed: int,
    iterations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    method_records = _build_method_records(truth_rows, predictions_by_method)
    methods: dict[str, Any] = {}
    claim_rows: list[dict[str, Any]] = []
    for method_index, method_name in enumerate(sorted(method_records)):
        records = method_records[method_name]
        primary_scheme = _analyze_scheme(
            field_name=PRIMARY_BIN_FIELD,
            method_name=method_name,
            records=records,
            seed=seed + (method_index * 100),
            iterations=iterations,
        )
        sensitivity_scheme = _analyze_scheme(
            field_name=SENSITIVITY_BIN_FIELD,
            method_name=method_name,
            records=records,
            seed=seed + (method_index * 100) + 50,
            iterations=iterations,
        )
        methods[method_name] = {
            "row_count": len(records),
            "primary": {key: value for key, value in primary_scheme.items() if key != "claim_rows"},
            "sensitivity": {key: value for key, value in sensitivity_scheme.items() if key != "claim_rows"},
            "binning_sensitivity": _scheme_sensitivity(primary_scheme, sensitivity_scheme),
        }
        claim_rows.extend(primary_scheme.get("claim_rows", []))
        claim_rows.extend(sensitivity_scheme.get("claim_rows", []))
    return {
        "primary_field": PRIMARY_BIN_FIELD,
        "sensitivity_field": SENSITIVITY_BIN_FIELD,
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "methods": methods,
    }, claim_rows


def _leave_one_dataset_out_report(summary: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    per_dataset: dict[str, dict[str, Any]] = {}
    for row in metrics_rows:
        method_name = str(row.get("method_name") or "unknown_method")
        scope = str(row.get("scope") or "unknown_scope")
        dataset_name = str(row.get("dataset") or "unknown_dataset")
        payload: dict[str, Any] = {}
        for key, value in row.items():
            if value in {"", None}:
                payload[key] = None
                continue
            numeric = _coerce_float(value)
            payload[key] = numeric if numeric is not None else value
        if scope == "aggregate":
            aggregate[method_name] = payload
        elif scope == "dataset_holdout":
            per_dataset.setdefault(dataset_name, {})[method_name] = payload
    return {
        "evaluation": summary.get("evaluation", {}),
        "aggregate": aggregate,
        "per_dataset": per_dataset,
        "learned_fusion_comparison": summary.get("learned_fusion_comparison", {}),
    }


def _build_calibration_curve(labels: list[int], scores: list[float], *, bin_count: int = 5) -> dict[str, Any]:
    if not scores:
        return {"status": "unavailable", "reason": "empty_scores"}
    if not _is_probability_like(scores):
        return {"status": "unavailable", "reason": "scores_not_probability_like"}
    buckets: list[dict[str, Any]] = []
    ece_total = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        bucket_items = [
            (label, score)
            for label, score in zip(labels, scores, strict=True)
            if (lower <= score < upper) or (index == bin_count - 1 and lower <= score <= upper)
        ]
        if not bucket_items:
            buckets.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_score": None,
                    "observed_positive_rate": None,
                    "absolute_gap": None,
                }
            )
            continue
        count = len(bucket_items)
        mean_score = sum(score for _label, score in bucket_items) / count
        observed_rate = sum(label for label, _score in bucket_items) / count
        gap = abs(mean_score - observed_rate)
        ece_total += gap * count
        buckets.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_score": mean_score,
                "observed_positive_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    return {
        "status": "ok",
        "ece": ece_total / len(scores),
        "bins": buckets,
    }


def _threshold_calibration_report(
    fusion_summary: dict[str, Any],
    predictions_by_method: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    baseline_lookup = _baseline_lookup(fusion_summary)
    methods: dict[str, Any] = {}
    for method_name in sorted(predictions_by_method):
        rows = list(predictions_by_method[method_name].values())
        labels = []
        scores = []
        thresholds = []
        for row in rows:
            label = _coerce_bool(row.get("is_hallucination"))
            if label is None:
                archived = str(row.get("label") or "").strip().lower()
                if archived == "normal":
                    label = False
                elif archived:
                    label = True
            score = _coerce_float(row.get("prediction_score"))
            threshold = _coerce_float(row.get("threshold"))
            if label is None or score is None:
                continue
            labels.append(1 if label else 0)
            scores.append(score)
            if threshold is not None:
                thresholds.append(threshold)
        baseline = baseline_lookup.get(method_name, {})
        aggregate = baseline.get("aggregate") if isinstance(baseline, dict) else {}
        methods[method_name] = {
            "row_count": len(scores),
            "probability_like_scores": _is_probability_like(scores),
            "thresholds": {
                "count": len(thresholds),
                "unique_count": len({round(value, 12) for value in thresholds}),
                "min": None if not thresholds else min(thresholds),
                "max": None if not thresholds else max(thresholds),
                "mean": _mean(thresholds),
            },
            "aggregate_summary_metrics": {
                "threshold": _coerce_float((aggregate or {}).get("threshold")),
                "predicted_positive_rate": _coerce_float((aggregate or {}).get("predicted_positive_rate")),
                "brier_score": _coerce_float((aggregate or {}).get("brier_score")),
                "auroc": _coerce_float((aggregate or {}).get("auroc")),
                "auprc": _coerce_float((aggregate or {}).get("auprc")),
            },
            "calibration_curve": _build_calibration_curve(labels, scores),
            "claim_text": "Observed threshold and calibration diagnostics only. These values describe score behavior and do not imply reliable superiority.",
        }
    return {"methods": methods}


def _ordered_datasets_from_rows(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    configured = ((summary.get("evaluation") or {}).get("dataset_order") or []) if isinstance(summary.get("evaluation"), dict) else []
    actual = sorted({_dataset(row) for row in rows})
    if isinstance(configured, list) and configured:
        configured_names = [str(item) for item in configured]
        return [name for name in configured_names if name in actual] + [name for name in actual if name not in configured_names]
    return actual


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    corpus = summary.get("corpus_bin_reliability", {})
    lodo = summary.get("leave_one_dataset_out", {})
    calibration = summary.get("threshold_calibration", {})
    bootstrap = summary.get("bootstrap", {})
    lines = [
        "# Robustness report",
        "",
        "Observed corpus-bin reliability summary for the current fusion outputs. All wording is intentionally descriptive.",
        "",
        "## Corpus-bin reliability",
        "",
    ]
    for method_name, payload in sorted((corpus.get("methods") or {}).items()):
        lines.append(f"### {method_name}")
        lines.append("")
        lines.append("#### Primary 3-bin analysis")
        lines.append("")
        lines.append("| bin | status | rows | prompts | pairs | auroc | auprc | paired_win_rate | mean_delta |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in ((payload.get("primary") or {}).get("bins") or []):
            row_metrics = row.get("row_metrics") or {}
            paired = row.get("paired") or {}
            lines.append(
                "| {bin_name} | {status} | {rows} | {prompts} | {pairs} | {auroc} | {auprc} | {win_rate} | {mean_delta} |".format(
                    bin_name=row.get("bin"),
                    status=row.get("status"),
                    rows=row.get("row_count"),
                    prompts=row.get("prompt_count"),
                    pairs=row.get("pair_count"),
                    auroc=_format_metric(row_metrics.get("auroc")),
                    auprc=_format_metric(row_metrics.get("auprc")),
                    win_rate=_format_metric(paired.get("win_rate")),
                    mean_delta=_format_metric(paired.get("mean_delta")),
                )
            )
        lines.extend([
            "",
            "#### 5-bin sensitivity",
            "",
            "| bin | status | rows | prompts | pairs | auroc | auprc | paired_win_rate | mean_delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in ((payload.get("sensitivity") or {}).get("bins") or []):
            row_metrics = row.get("row_metrics") or {}
            paired = row.get("paired") or {}
            lines.append(
                "| {bin_name} | {status} | {rows} | {prompts} | {pairs} | {auroc} | {auprc} | {win_rate} | {mean_delta} |".format(
                    bin_name=row.get("bin"),
                    status=row.get("status"),
                    rows=row.get("row_count"),
                    prompts=row.get("prompt_count"),
                    pairs=row.get("pair_count"),
                    auroc=_format_metric(row_metrics.get("auroc")),
                    auprc=_format_metric(row_metrics.get("auprc")),
                    win_rate=_format_metric(paired.get("win_rate")),
                    mean_delta=_format_metric(paired.get("mean_delta")),
                )
            )
        lines.append("")
    lines.extend([
        "## Prompt-grouped method bootstrap deltas",
        "",
        "| candidate | reference | metric | observed_delta | ci_95_lower | ci_95_upper |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for comparison in bootstrap.get("comparisons", []):
        for metric in comparison.get("metrics", []):
            lines.append(
                "| {candidate} | {reference} | {metric_name} | {delta} | {lower} | {upper} |".format(
                    candidate=comparison.get("candidate_method"),
                    reference=comparison.get("reference_method"),
                    metric_name=metric.get("metric"),
                    delta=_format_metric(metric.get("observed_delta")),
                    lower=_format_metric(metric.get("ci_95_lower")),
                    upper=_format_metric(metric.get("ci_95_upper")),
                )
            )
    lines.extend([
        "",
        "## Leave-one-dataset-out",
        "",
        f"Datasets: {', '.join(summary.get('datasets') or [])}",
        "",
        "## Threshold and calibration diagnostics",
        "",
        "| method | probability_like_scores | threshold_mean | threshold_unique_count | brier_score | calibration_status | ece |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ])
    for method_name, payload in sorted((calibration.get("methods") or {}).items()):
        thresholds = payload.get("thresholds") or {}
        aggregate = payload.get("aggregate_summary_metrics") or {}
        curve = payload.get("calibration_curve") or {}
        lines.append(
            "| {method} | {prob_like} | {threshold_mean} | {threshold_unique} | {brier} | {curve_status} | {ece} |".format(
                method=method_name,
                prob_like=payload.get("probability_like_scores"),
                threshold_mean=_format_metric(thresholds.get("mean")),
                threshold_unique=thresholds.get("unique_count"),
                brier=_format_metric(aggregate.get("brier_score")),
                curve_status=curve.get("status"),
                ece=_format_metric(curve.get("ece")),
            )
        )
    lines.extend(["", "## Caveats", ""])
    for caveat in summary.get("caveats", []):
        lines.append(f"- {caveat}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    numeric = _coerce_float(value)
    return "null" if numeric is None else f"{numeric:.6f}"


def _caveats(rows: list[dict[str, Any]], corpus_report: dict[str, Any]) -> list[str]:
    caveats = [
        "Prompt-grouped bootstrap uses prompt-level units and never resamples candidate rows independently.",
        "Corpus support bins are descriptive analysis strata, not new correctness labels.",
        "Bins with too few prompt pairs are reported explicitly as insufficient instead of being dropped.",
        "Threshold and calibration diagnostics summarize score behavior only and do not justify causal or superiority wording.",
    ]
    insufficient = 0
    for method_payload in (corpus_report.get("methods") or {}).values():
        for scheme_name in ("primary", "sensitivity"):
            insufficient += int((method_payload.get(scheme_name) or {}).get("insufficient_bin_count") or 0)
    caveats.append(f"Observed insufficient-bin entries across all method/scheme combinations: {insufficient}.")
    if any(_bin_value(row, SENSITIVITY_BIN_FIELD) is None for row in rows):
        caveats.append("Some rows are missing 5-bin corpus labels, so sensitivity analysis may be partially unavailable.")
    return caveats


def run_robustness(
    features_path: Path,
    fusion_dir: Path,
    out_dir: Path,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    rows, storage_report = read_feature_rows(features_path)
    if not rows:
        raise ValueError(f"Robustness features artifact is empty: {features_path}")
    fusion_root, fusion_summary_path, baseline_metrics_path, predictions_path = _resolve_fusion_artifacts(fusion_dir)
    fusion_summary = load_json(fusion_summary_path)
    metrics_rows = _read_csv(baseline_metrics_path)
    predictions = _read_jsonl(predictions_path)
    predictions_by_method = _predictions_by_method(predictions)
    datasets = _ordered_datasets_from_rows(fusion_summary, rows)
    bootstrap, bootstrap_claims = _build_method_bootstrap_report(
        rows,
        predictions_by_method,
        _baseline_lookup(fusion_summary),
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )
    corpus_bin_reliability, corpus_claims = _build_corpus_bin_reliability(
        rows,
        predictions_by_method,
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )
    leave_one_dataset_out = _leave_one_dataset_out_report(fusion_summary, metrics_rows)
    threshold_calibration = _threshold_calibration_report(fusion_summary, predictions_by_method)
    report_claims = bootstrap_claims + corpus_claims
    summary = {
        "run_id": f"robustness-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": _iso_now(),
        "method_name": str(fusion_summary.get("method_name") or "Corpus-Conditioned Hallucination Metric Reliability Study"),
        "features_path": str(features_path),
        "features_storage": storage_report,
        "fusion_dir": str(fusion_root),
        "fusion_summary_path": str(fusion_summary_path),
        "baseline_metrics_path": str(baseline_metrics_path),
        "predictions_path": str(predictions_path),
        "row_count": len(rows),
        "datasets": datasets,
        "formula_manifest_ref": FORMULA_MANIFEST_REF,
        "dataset_manifest_ref": DATASET_MANIFEST_REF,
        "bootstrap": bootstrap,
        "corpus_bin_reliability": corpus_bin_reliability,
        "leave_one_dataset_out": leave_one_dataset_out,
        "threshold_calibration": threshold_calibration,
        "report_claims": report_claims,
        "caveats": _caveats(rows, corpus_bin_reliability),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    bootstrap_path = out_dir / "bootstrap_ci.json"
    corpus_path = out_dir / "corpus_bin_reliability.json"
    lodo_path = out_dir / "leave_one_dataset_out.json"
    calibration_path = out_dir / "threshold_calibration.json"
    write_json(summary_path, summary)
    write_json(bootstrap_path, bootstrap)
    write_json(corpus_path, corpus_bin_reliability)
    write_json(lodo_path, leave_one_dataset_out)
    write_json(calibration_path, threshold_calibration)
    _write_markdown_report(report_path, summary)
    return {
        "run_id": summary["run_id"],
        "row_count": len(rows),
        "datasets": datasets,
        "artifacts": {
            "summary": str(summary_path),
            "bootstrap_ci_json": str(bootstrap_path),
            "corpus_bin_reliability": str(corpus_path),
            "leave_one_dataset_out": str(lodo_path),
            "threshold_calibration": str(calibration_path),
            "report": str(report_path),
        },
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_iterations": bootstrap_iterations,
    }


def _collect_claim_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        items: list[str] = []
        for entry in value:
            items.extend(_collect_claim_strings(entry))
        return items
    if isinstance(value, dict):
        items: list[str] = []
        for key, entry in value.items():
            if any(token in str(key).lower() for token in ("claim", "headline", "conclusion", "summary", "wording", "note")):
                items.extend(_collect_claim_strings(entry))
        return items
    return []


def validate_report_claims(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    claim_entries = summary.get("report_claims", [])
    if not isinstance(claim_entries, list):
        raise ValueError("Report-claim validation failed:\n- report_claims must be a list")
    problems: list[str] = []
    checked_strings = 0
    for index, entry in enumerate(claim_entries):
        if not isinstance(entry, dict):
            problems.append(f"report_claims[{index}] must be an object")
            continue
        strings = _collect_claim_strings(entry)
        if not strings:
            continue
        checked_strings += len(strings)
        for string_index, claim_text in enumerate(strings):
            normalized = re.sub(r"\s+", " ", claim_text.strip().lower())
            for pattern, label in FORBIDDEN_CLAIM_PATTERNS:
                if re.search(pattern, normalized):
                    problems.append(
                        f"report_claims[{index}] claim_text[{string_index}] uses forbidden phrase '{label}'"
                    )
        ci_lower = _coerce_float(entry.get("ci_95_lower"))
        ci_upper = _coerce_float(entry.get("ci_95_upper"))
        if ci_lower is not None and ci_upper is not None and ci_lower > ci_upper:
            problems.append(f"report_claims[{index}] has inverted interval bounds")
        if entry.get("statistically_significant") is True:
            problems.append(f"report_claims[{index}] must keep statistically_significant=false for descriptive robustness summaries")
    if problems:
        raise ValueError("Report-claim validation failed:\n- " + "\n- ".join(problems))
    return {
        "summary_path": str(summary_path),
        "checked_claims": len(claim_entries),
        "checked_claim_strings": checked_strings,
        "status": "ok",
        "forbidden_phrases": [label for _pattern, label in FORBIDDEN_CLAIM_PATTERNS],
    }
