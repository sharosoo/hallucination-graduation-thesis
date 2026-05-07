"""Leakage-safe condition-aware fusion baseline evaluation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.adapters.corpus_features import load_json, read_feature_rows, write_json

FORMULA_MANIFEST_REF = "experiments/literature/formula_notes.md"
DATASET_MANIFEST_REF = "experiments/manifests/upstream_artifacts_manifest.json"

EMPTY_DATASET_REASON = "empty_dataset"
SINGLE_CLASS_REASON = "single_class_subset"
MISSING_SIGNAL_REASON = "missing_signal"
MISSING_TARGET_REASON = "missing_target"
FULL_LOGITS_REQUIRED_REASON = "full_logits_required"
RERUN_REQUIRED_REASON = "rerun_required"
PARTIAL_DATASET_AVAILABILITY_REASON = "partial_dataset_availability"
BIN_NOT_PRESENT_REASON = "bin_not_present"
INSUFFICIENT_BIN_TRAINING_REASON = "insufficient_bin_training"
INSUFFICIENT_BIN_TEST_REASON = "insufficient_bin_test"

PRIMARY_BIN_FIELD = "corpus_axis_bin"
SENSITIVITY_BIN_FIELD = "corpus_axis_bin_5"
BIN_SCHEME_DEFAULTS = {
    PRIMARY_BIN_FIELD: ("low_support", "medium_support", "high_support"),
    SENSITIVITY_BIN_FIELD: (
        "very_low_support",
        "low_support",
        "mid_support",
        "high_support",
        "very_high_support",
    ),
}

SIGNAL_ALIASES: dict[str, tuple[str, ...]] = {
    "semantic_entropy_score": ("semantic_entropy_nli_likelihood", "semantic_entropy"),
    "semantic_energy_score": ("semantic_energy_cluster_uncertainty", "semantic_energy_boltzmann"),
    "semantic_energy_sample": ("semantic_energy_sample_energy",),
    "semantic_energy_diagnostic": ("semantic_energy_boltzmann_diagnostic",),
    "corpus_risk_score": ("corpus_risk_only",),
    "entity_frequency_axis": ("entity_frequency_axis", "entity_frequency"),
    "entity_pair_cooccurrence_axis": ("entity_pair_cooccurrence_axis", "entity_pair_cooccurrence"),
}


@dataclass(frozen=True)
class StandardScalerModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        transformed: list[list[float]] = []
        for row in rows:
            transformed.append(
                [
                    (value - mean) / scale
                    for value, mean, scale in zip(row, self.means, self.scales, strict=True)
                ]
            )
        return transformed


@dataclass(frozen=True)
class LogisticRegressionModel:
    scaler: StandardScalerModel | None
    weights: tuple[float, ...]
    bias: float
    feature_names: tuple[str, ...]
    l2_lambda: float
    learning_rate: float
    max_iterations: int
    tolerance: float
    iterations_run: int

    def predict_scores(self, rows: list[list[float]]) -> list[float]:
        inputs = self.scaler.transform(rows) if self.scaler is not None else rows
        scores: list[float] = []
        for row in inputs:
            raw = self.bias
            for weight, value in zip(self.weights, row, strict=True):
                raw += weight * value
            scores.append(_sigmoid(raw))
        return scores


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class FoldResult:
    dataset: str
    split_id: str
    row_count: int
    positive_count: int
    threshold: float | None
    threshold_source: str | None
    metrics: dict[str, Any]
    unavailable_reason: str | None
    full_logits_required: bool
    rerun_required: bool
    train_datasets: tuple[str, ...]
    prediction_rows: tuple[dict[str, Any], ...]
    feature_importance: dict[str, Any] | None
    fold_safety: dict[str, Any] | None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _features(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("features")
    return payload if isinstance(payload, dict) else {}


def _energy_availability(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("energy_availability")
    return payload if isinstance(payload, dict) else {}


def _dataset_name(row: dict[str, Any]) -> str:
    return str(row.get("dataset", "unknown"))


def _split_id(row: dict[str, Any]) -> str:
    return str(row.get("split_id", "unknown_split"))


def _sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("candidate_id") or "unknown_sample")


def _candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or _sample_id(row))


def _prompt_id(row: dict[str, Any]) -> str:
    return str(row.get("prompt_id", "unknown_prompt"))


def _pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id", "unknown_pair"))


def _archived_label(row: dict[str, Any]) -> str:
    return str(row.get("label", ""))


def _candidate_label(row: dict[str, Any]) -> str | None:
    raw = row.get("candidate_label")
    if raw is None:
        raw = row.get("candidate_role")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _expected_bins(config: dict[str, Any], field_name: str) -> tuple[str, ...]:
    configured = config.get("evaluation", {}).get("bin_schemes", {}).get(field_name)
    if isinstance(configured, list) and configured:
        return tuple(str(item) for item in configured)
    return BIN_SCHEME_DEFAULTS[field_name]


def _bin_value(row: dict[str, Any], field_name: str) -> str | None:
    feature_value = _features(row).get(field_name)
    if feature_value is None:
        return None
    text = str(feature_value).strip()
    return text or None


def _target_value(row: dict[str, Any]) -> int:
    is_hallucination = _coerce_bool(row.get("is_hallucination"))
    if is_hallucination is not None:
        return 1 if is_hallucination else 0
    is_correct = _coerce_bool(row.get("is_correct"))
    if is_correct is not None:
        return 0 if is_correct else 1
    raise ValueError(
        f"Missing annotation-backed hallucination target for sample {_sample_id(row)!r}; "
        "expected top-level is_hallucination or is_correct"
    )


def _has_target(row: dict[str, Any]) -> bool:
    try:
        _target_value(row)
        return True
    except ValueError:
        return False


def _lookup_feature_value(row: dict[str, Any], feature_name: str) -> float | None:
    keys = SIGNAL_ALIASES.get(feature_name, (feature_name,))
    features = _features(row)
    for key in keys:
        if key in features:
            raw = features.get(key)
        else:
            raw = row.get(key)
        if isinstance(raw, bool):
            return 1.0 if raw else 0.0
        value = _coerce_float(raw)
        if value is not None:
            return value
    return None


def _missing_energy_requires_rerun(rows: list[dict[str, Any]]) -> tuple[bool, bool]:
    full_logits_required = False
    rerun_required = False
    for row in rows:
        availability = _energy_availability(row)
        if _coerce_bool(availability.get("full_logits_required")):
            full_logits_required = True
        if _coerce_bool(availability.get("rerun_required")):
            rerun_required = True
    return full_logits_required, rerun_required


def _feature_available(rows: list[dict[str, Any]], feature_name: str) -> bool:
    return all(_lookup_feature_value(row, feature_name) is not None for row in rows)


def _extract_vector(row: dict[str, Any], feature_names: tuple[str, ...]) -> list[float]:
    vector: list[float] = []
    for feature_name in feature_names:
        value = _lookup_feature_value(row, feature_name)
        if value is None:
            raise ValueError(f"Missing numeric feature {feature_name!r} for sample {_sample_id(row)!r}")
        vector.append(value)
    return vector


def _extract_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> list[list[float]]:
    return [_extract_vector(row, feature_names) for row in rows]


def _rank_average_pairs(scores: list[float]) -> list[tuple[float, float]]:
    ordered = sorted(enumerate(scores), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        score = ordered[index][1]
        while end < len(ordered) and ordered[end][1] == score:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for tied_index in range(index, end):
            original_index = ordered[tied_index][0]
            ranks[original_index] = average_rank
        index = end
    return list(zip(scores, ranks, strict=True))


def _compute_auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = _rank_average_pairs(scores)
    positive_rank_sum = sum(rank for label, (_score, rank) in zip(labels, ranked, strict=True) if label == 1)
    return (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)


def _compute_auprc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels, range(len(labels)), strict=True), key=lambda item: (-item[0], item[2]))
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        score = ordered[index][0]
        while end < len(ordered) and ordered[end][0] == score:
            end += 1
        for tied_index in range(index, end):
            if ordered[tied_index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += precision * (recall - previous_recall)
        previous_recall = recall
        index = end
    return area


def _brier_score(labels: list[int], scores: list[float]) -> float | None:
    if not labels:
        return None
    return sum((score - label) ** 2 for label, score in zip(labels, scores, strict=True)) / len(labels)


def _score_summary(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"mean": None, "min": None, "max": None, "std": None}
    mean_value = sum(scores) / len(scores)
    variance = sum((score - mean_value) ** 2 for score in scores) / len(scores)
    return {
        "mean": mean_value,
        "min": min(scores),
        "max": max(scores),
        "std": math.sqrt(variance),
    }


def _confusion(labels: list[int], predictions: list[int]) -> dict[str, int]:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for label, prediction in zip(labels, predictions, strict=True):
        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 1:
            false_positive += 1
        elif label == 0 and prediction == 0:
            true_negative += 1
        else:
            false_negative += 1
    return {"tp": true_positive, "fp": false_positive, "tn": true_negative, "fn": false_negative}


def _threshold_candidates(scores: list[float]) -> list[float]:
    ordered = sorted(set(scores))
    if not ordered:
        return [0.5]
    return [0.0, *ordered, 1.0]


def _metrics_at_threshold(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    predictions = [1 if score >= threshold else 0 for score in scores]
    confusion = _confusion(labels, predictions)
    tp = confusion["tp"]
    fp = confusion["fp"]
    tn = confusion["tn"]
    fn = confusion["fn"]
    total = len(labels)
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": sum(predictions) / total if total else None,
        "confusion": confusion,
    }


def _select_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics = _metrics_at_threshold(labels, scores, best_threshold)
    for threshold in _threshold_candidates(scores):
        metrics = _metrics_at_threshold(labels, scores, threshold)
        current_key = (
            metrics["f1"],
            metrics["accuracy"] if metrics["accuracy"] is not None else -1.0,
            metrics["precision"],
            metrics["recall"],
            -threshold,
        )
        best_key = (
            best_metrics["f1"],
            best_metrics["accuracy"] if best_metrics["accuracy"] is not None else -1.0,
            best_metrics["precision"],
            best_metrics["recall"],
            -best_threshold,
        )
        if current_key > best_key:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def _unavailable_metric_payload(reason: str) -> dict[str, Any]:
    return {
        "row_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "auroc": None,
        "auprc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "threshold": None,
        "predicted_positive_rate": None,
        "confusion": None,
        "score_summary": {"mean": None, "min": None, "max": None, "std": None},
        "brier_score": None,
        "unavailable_reason": reason,
    }


def _evaluate_rows(rows: list[dict[str, Any]], scores: list[float], threshold: float) -> dict[str, Any]:
    labels = [_target_value(row) for row in rows]
    threshold_metrics = _metrics_at_threshold(labels, scores, threshold)
    return {
        "row_count": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "auroc": _compute_auroc(labels, scores),
        "auprc": _compute_auprc(labels, scores),
        "accuracy": threshold_metrics["accuracy"],
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
        "threshold": threshold,
        "predicted_positive_rate": threshold_metrics["predicted_positive_rate"],
        "confusion": threshold_metrics["confusion"],
        "score_summary": _score_summary(scores),
        "brier_score": _brier_score(labels, scores),
        "unavailable_reason": None,
    }


def _evaluate_bin_group(rows: list[dict[str, Any]], scores: list[float], threshold: float, reason_if_empty: str) -> dict[str, Any]:
    if not rows:
        return _unavailable_metric_payload(reason_if_empty)
    labels = [_target_value(row) for row in rows]
    if len(set(labels)) < 2:
        return {
            **_unavailable_metric_payload(SINGLE_CLASS_REASON),
            "row_count": len(rows),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
        }
    metrics = _evaluate_rows(rows, scores, threshold)
    metrics["unavailable_reason"] = None
    return metrics


def _group_metrics_by_bin(
    rows: list[dict[str, Any]],
    scores: list[float],
    threshold: float,
    *,
    config: dict[str, Any],
    field_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        bin_id = _bin_value(row, field_name)
        if bin_id is None:
            continue
        grouped.setdefault(bin_id, []).append((row, score))
    payload: list[dict[str, Any]] = []
    for bin_id in _expected_bins(config, field_name):
        items = grouped.get(bin_id, [])
        bin_rows = [row for row, _score in items]
        bin_scores = [score for _row, score in items]
        metrics = _evaluate_bin_group(bin_rows, bin_scores, threshold, BIN_NOT_PRESENT_REASON)
        payload.append(
            {
                "bin_field": field_name,
                "bin_id": bin_id,
                "row_count": metrics.get("row_count"),
                "positive_count": metrics.get("positive_count"),
                "negative_count": metrics.get("negative_count"),
                "metrics": metrics,
            }
        )
    return payload


def _evaluate_partition(rows: list[dict[str, Any]], scores: list[float], threshold: float, config: dict[str, Any]) -> dict[str, Any]:
    metrics = _evaluate_rows(rows, scores, threshold)
    metrics[PRIMARY_BIN_FIELD] = _group_metrics_by_bin(rows, scores, threshold, config=config, field_name=PRIMARY_BIN_FIELD)
    metrics[SENSITIVITY_BIN_FIELD] = _group_metrics_by_bin(
        rows,
        scores,
        threshold,
        config=config,
        field_name=SENSITIVITY_BIN_FIELD,
    )
    return metrics


def _fit_standard_scaler(rows: list[list[float]]) -> StandardScalerModel:
    if not rows:
        raise ValueError("Cannot fit StandardScaler on empty rows")
    width = len(rows[0])
    means: list[float] = []
    scales: list[float] = []
    for column_index in range(width):
        values = [row[column_index] for row in rows]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        scale_value = math.sqrt(variance)
        means.append(mean_value)
        scales.append(scale_value if scale_value != 0.0 else 1.0)
    return StandardScalerModel(means=tuple(means), scales=tuple(scales))


def _fit_logistic_regression(
    rows: list[list[float]],
    labels: list[int],
    *,
    feature_names: tuple[str, ...],
    standardize: bool,
    l2_lambda: float,
    learning_rate: float,
    max_iterations: int,
    tolerance: float,
) -> LogisticRegressionModel:
    scaler = _fit_standard_scaler(rows) if standardize else None
    inputs = scaler.transform(rows) if scaler is not None else rows
    weights = [0.0 for _ in feature_names]
    positive_rate = sum(labels) / len(labels) if labels else 0.5
    positive_rate = min(max(positive_rate, 1e-6), 1.0 - 1e-6)
    bias = math.log(positive_rate / (1.0 - positive_rate))
    iterations_run = 0
    for iteration in range(max_iterations):
        grad_w = [0.0 for _ in feature_names]
        grad_b = 0.0
        for row, label in zip(inputs, labels, strict=True):
            raw = bias
            for weight, value in zip(weights, row, strict=True):
                raw += weight * value
            prediction = _sigmoid(raw)
            error = prediction - label
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value
        sample_count = float(len(inputs))
        grad_b /= sample_count
        for index in range(len(grad_w)):
            grad_w[index] = (grad_w[index] / sample_count) + (l2_lambda * weights[index])
        max_gradient = max(abs(grad_b), *(abs(value) for value in grad_w))
        step = learning_rate / math.sqrt(iteration + 1.0)
        bias -= step * grad_b
        for index in range(len(weights)):
            weights[index] -= step * grad_w[index]
        iterations_run = iteration + 1
        if max_gradient <= tolerance:
            break
    return LogisticRegressionModel(
        scaler=scaler,
        weights=tuple(weights),
        bias=bias,
        feature_names=feature_names,
        l2_lambda=l2_lambda,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
        iterations_run=iterations_run,
    )


def _logistic_training_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("learned_fusion", {}) if isinstance(config.get("learned_fusion"), dict) else {}


def _feature_set(config: dict[str, Any], feature_set_name: str) -> tuple[str, ...]:
    raw = _logistic_training_config(config).get("feature_sets", {}).get(feature_set_name, [])
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw)


def _feature_pool(config: dict[str, Any], feature_pool_name: str) -> tuple[str, ...]:
    raw = _logistic_training_config(config).get("feature_pools", {}).get(feature_pool_name, [])
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw)


def _fit_feature_set_model(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    config: dict[str, Any],
) -> tuple[list[float], list[float], float, LogisticRegressionModel]:
    training = _logistic_training_config(config)
    train_matrix = _extract_matrix(train_rows, feature_names)
    test_matrix = _extract_matrix(test_rows, feature_names)
    train_labels = [_target_value(row) for row in train_rows]
    model = _fit_logistic_regression(
        train_matrix,
        train_labels,
        feature_names=feature_names,
        standardize=bool(training.get("standardize", True)),
        l2_lambda=float(training.get("l2_lambda", 0.1)),
        learning_rate=float(training.get("learning_rate", 0.2)),
        max_iterations=int(training.get("max_iterations", 1500)),
        tolerance=float(training.get("tolerance", 1e-6)),
    )
    train_scores = model.predict_scores(train_matrix)
    test_scores = model.predict_scores(test_matrix)
    threshold, _ = _select_threshold(train_labels, train_scores)
    return train_scores, test_scores, threshold, model


def _coefficient_payload(model: LogisticRegressionModel, fold_name: str, *, context: str | None = None) -> dict[str, Any]:
    coefficients = {name: weight for name, weight in zip(model.feature_names, model.weights, strict=True)}
    scaler = None
    if model.scaler is not None:
        scaler = {
            "means": {name: value for name, value in zip(model.feature_names, model.scaler.means, strict=True)},
            "scales": {name: value for name, value in zip(model.feature_names, model.scaler.scales, strict=True)},
        }
    payload = {
        "fold": fold_name,
        "coefficients": coefficients,
        "bias": model.bias,
        "iterations_run": model.iterations_run,
        "training": {
            "l2_lambda": model.l2_lambda,
            "learning_rate": model.learning_rate,
            "max_iterations": model.max_iterations,
            "tolerance": model.tolerance,
            "standardized": model.scaler is not None,
        },
        "scaler": scaler,
    }
    if context is not None:
        payload["context"] = context
    return payload


def _aggregate_flat_importance(per_fold: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not per_fold:
        return None
    grouped: dict[str, list[float]] = {}
    for fold in per_fold:
        flat = fold.get("flat_importance")
        if not isinstance(flat, dict):
            continue
        for feature_name, weight in flat.items():
            numeric = _coerce_float(weight)
            if numeric is None:
                continue
            grouped.setdefault(str(feature_name), []).append(numeric)
    if not grouped:
        return None
    summary: dict[str, Any] = {}
    for feature_name, values in sorted(grouped.items()):
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary[feature_name] = {
            "mean": mean_value,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
            "fold_count": len(values),
        }
    return summary


def _prediction_rows(
    rows: list[dict[str, Any]],
    scores: list[float],
    *,
    baseline_name: str,
    run_id: str,
    threshold: float,
    threshold_source: str,
    train_datasets: tuple[str, ...],
    test_dataset: str,
    fold_id: str,
) -> list[dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        is_hallucination = _target_value(row) == 1
        prediction_rows.append(
            {
                "run_id": run_id,
                "method_name": baseline_name,
                "dataset": _dataset_name(row),
                "split_id": _split_id(row),
                "fold_id": fold_id,
                "sample_id": _sample_id(row),
                "candidate_id": _candidate_id(row),
                "prompt_id": _prompt_id(row),
                "pair_id": _pair_id(row),
                "candidate_label": _candidate_label(row),
                "label": _archived_label(row),
                "is_correct": _coerce_bool(row.get("is_correct")),
                "is_hallucination": is_hallucination,
                PRIMARY_BIN_FIELD: _bin_value(row, PRIMARY_BIN_FIELD),
                SENSITIVITY_BIN_FIELD: _bin_value(row, SENSITIVITY_BIN_FIELD),
                "prediction_score": score,
                "prediction_label": bool(score >= threshold),
                "threshold": threshold,
                "threshold_source": threshold_source,
                "train_datasets": list(train_datasets),
                "test_dataset": test_dataset,
                "formula_manifest_ref": FORMULA_MANIFEST_REF,
                "dataset_manifest_ref": DATASET_MANIFEST_REF,
                "features": _features(row),
            }
        )
    return prediction_rows


def _rows_from_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in predictions:
        rows.append(
            {
                "dataset": item.get("dataset"),
                "split_id": item.get("split_id"),
                "sample_id": item.get("sample_id"),
                "candidate_id": item.get("candidate_id"),
                "prompt_id": item.get("prompt_id"),
                "pair_id": item.get("pair_id"),
                "candidate_label": item.get("candidate_label"),
                "label": item.get("label"),
                "is_correct": item.get("is_correct"),
                "is_hallucination": item.get("is_hallucination"),
                "features": item.get("features", {}),
            }
        )
    return rows


def _baseline_specs(config: dict[str, Any]) -> tuple[BaselineSpec, ...]:
    baselines = config.get("baselines")
    if not isinstance(baselines, list):
        raise ValueError("Fusion config is missing a baselines list")
    specs: list[BaselineSpec] = []
    for entry in baselines:
        if not isinstance(entry, dict):
            continue
        params = {key: value for key, value in entry.items() if key not in {"name", "kind"}}
        specs.append(BaselineSpec(name=str(entry["name"]), kind=str(entry["kind"]), params=params))
    return tuple(specs)


def _ordered_datasets(config: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, ...]:
    configured = config.get("evaluation", {}).get("dataset_order")
    if isinstance(configured, list) and configured:
        allowed = {str(item) for item in configured}
        actual = {_dataset_name(row) for row in rows}
        ordered = [str(item) for item in configured if str(item) in actual]
        remaining = sorted(actual - allowed)
        return tuple(ordered + remaining)
    return tuple(sorted({_dataset_name(row) for row in rows}))


def _rows_for_dataset(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if _dataset_name(row) == dataset_name]


def _require_target(rows: list[dict[str, Any]]) -> tuple[bool, str | None]:
    missing = [row for row in rows if not _has_target(row)]
    if missing:
        return False, MISSING_TARGET_REASON
    return True, None


def _single_signal_scores(rows: list[dict[str, Any]], signal_name: str) -> list[float]:
    scores: list[float] = []
    for row in rows:
        value = _lookup_feature_value(row, signal_name)
        if value is None:
            raise ValueError(f"Missing signal {signal_name!r} for sample {_sample_id(row)!r}")
        scores.append(value)
    return scores


def _evaluate_candidate_signals(train_rows: list[dict[str, Any]], feature_pool: tuple[str, ...]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    labels = [_target_value(row) for row in train_rows]
    for feature_name in feature_pool:
        if not _feature_available(train_rows, feature_name):
            continue
        scores = _single_signal_scores(train_rows, feature_name)
        threshold, threshold_metrics = _select_threshold(labels, scores)
        candidates.append(
            {
                "feature_name": feature_name,
                "auroc": _compute_auroc(labels, scores),
                "auprc": _compute_auprc(labels, scores),
                "f1": threshold_metrics["f1"],
                "accuracy": threshold_metrics["accuracy"],
                "precision": threshold_metrics["precision"],
                "recall": threshold_metrics["recall"],
                "threshold": threshold,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["auroc"] if item["auroc"] is not None else -1.0,
            item["auprc"] if item["auprc"] is not None else -1.0,
            item["f1"],
            item["accuracy"] if item["accuracy"] is not None else -1.0,
            item["feature_name"],
        ),
        reverse=True,
    )
    return candidates


def _feature_selection_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    bin_field: str,
    feature_pool: tuple[str, ...],
    config: dict[str, Any],
    fold_name: str,
) -> tuple[list[float], float, dict[str, Any], dict[str, Any]]:
    test_scores: list[float] = []
    all_train_scores: list[float] = []
    all_train_labels: list[int] = []
    provenance: dict[str, Any] = {
        "fold": fold_name,
        "bin_field": bin_field,
        "selection_policy": "best_train_bin_single_feature_by_auroc_then_auprc_then_f1",
        "train_only": True,
        "bins": [],
    }
    flat_importance: dict[str, float] = {}
    for bin_id in _expected_bins(config, bin_field):
        train_bin_rows = [row for row in train_rows if _bin_value(row, bin_field) == bin_id]
        test_bin_rows = [row for row in test_rows if _bin_value(row, bin_field) == bin_id]
        bin_record: dict[str, Any] = {
            "bin_id": bin_id,
            "train_row_count": len(train_bin_rows),
            "test_row_count": len(test_bin_rows),
            "train_positive_count": sum(_target_value(row) for row in train_bin_rows) if train_bin_rows else 0,
            "test_positive_count": sum(_target_value(row) for row in test_bin_rows) if test_bin_rows else 0,
            "candidate_features": list(feature_pool),
            "train_only": True,
        }
        if not test_bin_rows:
            bin_record["status"] = "na"
            bin_record["reason"] = BIN_NOT_PRESENT_REASON
            provenance["bins"].append(bin_record)
            continue
        if len(train_bin_rows) < 2 or len({_target_value(row) for row in train_bin_rows}) < 2:
            raise ValueError(f"{INSUFFICIENT_BIN_TRAINING_REASON}:{bin_id}")
        ranked = _evaluate_candidate_signals(train_bin_rows, feature_pool)
        if not ranked:
            raise ValueError(f"{MISSING_SIGNAL_REASON}:{bin_id}")
        selected = ranked[0]
        selected_feature = str(selected["feature_name"])
        threshold = float(selected["threshold"])
        train_bin_scores = _single_signal_scores(train_bin_rows, selected_feature)
        test_bin_scores = _single_signal_scores(test_bin_rows, selected_feature)
        all_train_scores.extend(train_bin_scores)
        all_train_labels.extend(_target_value(row) for row in train_bin_rows)
        test_scores.extend(test_bin_scores)
        flat_importance[f"selected::{bin_id}::{selected_feature}"] = 1.0
        bin_record.update(
            {
                "status": "ok",
                "selected_feature": selected_feature,
                "selected_threshold": threshold,
                "ranking": ranked,
            }
        )
        for row, score in zip(test_bin_rows, test_bin_scores, strict=True):
            row.setdefault("_tmp_feature_selection_score", score)
        provenance["bins"].append(bin_record)
    ordered_scores = [float(row.pop("_tmp_feature_selection_score")) for row in test_rows]
    global_threshold, _ = _select_threshold(all_train_labels, all_train_scores)
    return ordered_scores, global_threshold, provenance, {"fold": fold_name, "flat_importance": flat_importance, "bins": provenance["bins"]}


def _corpus_bin_weighted_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    bin_field: str,
    feature_names: tuple[str, ...],
    config: dict[str, Any],
    fold_name: str,
) -> tuple[list[float], float, dict[str, Any], dict[str, Any]]:
    all_train_scores: list[float] = []
    all_train_labels: list[int] = []
    provenance: dict[str, Any] = {
        "fold": fold_name,
        "bin_field": bin_field,
        "weight_policy": "train_bin_logistic_regression_coefficients",
        "train_only": True,
        "bins": [],
    }
    flat_importance: dict[str, float] = {}
    for bin_id in _expected_bins(config, bin_field):
        train_bin_rows = [row for row in train_rows if _bin_value(row, bin_field) == bin_id]
        test_bin_rows = [row for row in test_rows if _bin_value(row, bin_field) == bin_id]
        bin_record: dict[str, Any] = {
            "bin_id": bin_id,
            "train_row_count": len(train_bin_rows),
            "test_row_count": len(test_bin_rows),
            "feature_names": list(feature_names),
            "train_only": True,
        }
        if not test_bin_rows:
            bin_record["status"] = "na"
            bin_record["reason"] = BIN_NOT_PRESENT_REASON
            provenance["bins"].append(bin_record)
            continue
        if len(train_bin_rows) < 2 or len({_target_value(row) for row in train_bin_rows}) < 2:
            raise ValueError(f"{INSUFFICIENT_BIN_TRAINING_REASON}:{bin_id}")
        if not all(_feature_available(train_bin_rows + test_bin_rows, feature_name) for feature_name in feature_names):
            raise ValueError(f"{MISSING_SIGNAL_REASON}:{bin_id}")
        train_bin_scores, test_bin_scores, _threshold, model = _fit_feature_set_model(
            train_bin_rows,
            test_bin_rows,
            feature_names=feature_names,
            config=config,
        )
        all_train_scores.extend(train_bin_scores)
        all_train_labels.extend(_target_value(row) for row in train_bin_rows)
        for feature_name, weight in zip(model.feature_names, model.weights, strict=True):
            flat_importance[f"{bin_id}::{feature_name}"] = weight
        coefficient_payload = _coefficient_payload(model, fold_name, context=bin_id)
        bin_record.update({"status": "ok", "coefficients": coefficient_payload["coefficients"], "bias": coefficient_payload["bias"]})
        for row, score in zip(test_bin_rows, test_bin_scores, strict=True):
            row.setdefault("_tmp_weighted_score", score)
        provenance["bins"].append(bin_record)
    ordered_scores = [float(row.pop("_tmp_weighted_score")) for row in test_rows]
    threshold, _ = _select_threshold(all_train_labels, all_train_scores)
    return ordered_scores, threshold, provenance, {"fold": fold_name, "flat_importance": flat_importance, "bins": provenance["bins"]}


def _interaction_feature_names(base_features: tuple[str, ...], axis_features: tuple[str, ...]) -> tuple[str, ...]:
    names = list(base_features)
    names.extend(axis_features)
    for base_feature in base_features:
        for axis_feature in axis_features:
            names.append(f"interaction::{base_feature}__x__{axis_feature}")
    return tuple(names)


def _interaction_matrix(rows: list[dict[str, Any]], base_features: tuple[str, ...], axis_features: tuple[str, ...]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        base_values = [_lookup_feature_value(row, name) for name in base_features]
        axis_values = [_lookup_feature_value(row, name) for name in axis_features]
        if any(value is None for value in base_values + axis_values):
            missing = [
                name
                for name, value in zip((*base_features, *axis_features), (*base_values, *axis_values), strict=True)
                if value is None
            ]
            raise ValueError(f"Missing interaction features {missing} for sample {_sample_id(row)!r}")
        base_numeric = [float(value) for value in base_values if value is not None]
        axis_numeric = [float(value) for value in axis_values if value is not None]
        vector = [*base_numeric, *axis_numeric]
        for base_value in base_numeric:
            for axis_value in axis_numeric:
                vector.append(base_value * axis_value)
        matrix.append(vector)
    return matrix


def _fit_interaction_model(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    base_features: tuple[str, ...],
    axis_features: tuple[str, ...],
    config: dict[str, Any],
) -> tuple[list[float], list[float], float, LogisticRegressionModel]:
    feature_names = _interaction_feature_names(base_features, axis_features)
    train_matrix = _interaction_matrix(train_rows, base_features, axis_features)
    test_matrix = _interaction_matrix(test_rows, base_features, axis_features)
    train_labels = [_target_value(row) for row in train_rows]
    training = _logistic_training_config(config)
    model = _fit_logistic_regression(
        train_matrix,
        train_labels,
        feature_names=feature_names,
        standardize=bool(training.get("standardize", True)),
        l2_lambda=float(training.get("l2_lambda", 0.1)),
        learning_rate=float(training.get("learning_rate", 0.2)),
        max_iterations=int(training.get("max_iterations", 1500)),
        tolerance=float(training.get("tolerance", 1e-6)),
    )
    train_scores = model.predict_scores(train_matrix)
    test_scores = model.predict_scores(test_matrix)
    threshold, _ = _select_threshold(train_labels, train_scores)
    return train_scores, test_scores, threshold, model


def _baseline_unavailable_fold(
    dataset_name: str,
    split_id: str,
    row_count: int,
    positive_count: int,
    train_datasets: tuple[str, ...],
    *,
    reason: str,
    full_logits_required: bool = False,
    rerun_required: bool = False,
    fold_safety: dict[str, Any] | None = None,
) -> FoldResult:
    return FoldResult(
        dataset=dataset_name,
        split_id=split_id,
        row_count=row_count,
        positive_count=positive_count,
        threshold=None,
        threshold_source=None,
        metrics=_unavailable_metric_payload(reason),
        unavailable_reason=reason,
        full_logits_required=full_logits_required,
        rerun_required=rerun_required,
        train_datasets=train_datasets,
        prediction_rows=(),
        feature_importance=None,
        fold_safety=fold_safety,
    )


def _evaluate_baseline_on_fold(
    baseline: BaselineSpec,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    run_id: str,
    config: dict[str, Any],
) -> FoldResult:
    dataset_name = _dataset_name(test_rows[0]) if test_rows else "unknown"
    split_id = _split_id(test_rows[0]) if test_rows else "unknown_split"
    train_datasets = tuple(sorted({_dataset_name(row) for row in train_rows}))
    positive_count = sum(_target_value(row) for row in test_rows) if test_rows and all(_has_target(row) for row in test_rows) else 0
    row_count = len(test_rows)
    fold_id = f"leave_one_dataset_out::{dataset_name}"
    if not test_rows:
        return _baseline_unavailable_fold(
            dataset_name,
            split_id,
            row_count,
            positive_count,
            train_datasets,
            reason=EMPTY_DATASET_REASON,
        )
    target_ok, target_reason = _require_target(train_rows + test_rows)
    if not target_ok:
        return _baseline_unavailable_fold(
            dataset_name,
            split_id,
            row_count,
            positive_count,
            train_datasets,
            reason=target_reason or MISSING_TARGET_REASON,
        )
    train_labels = [_target_value(row) for row in train_rows]
    if len(set(train_labels)) < 2:
        return _baseline_unavailable_fold(
            dataset_name,
            split_id,
            row_count,
            positive_count,
            train_datasets,
            reason=SINGLE_CLASS_REASON,
        )

    threshold_source = "train_max_f1"
    feature_importance = None
    fold_safety: dict[str, Any] | None = {
        "fold": fold_id,
        "train_datasets": list(train_datasets),
        "test_dataset": dataset_name,
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "target_field": "is_hallucination",
        "target_fallback": "not is_correct",
        "uses_archived_label_as_target": False,
        "train_only": True,
    }

    try:
        if baseline.kind == "single_signal":
            signal = str(baseline.params.get("signal"))
            if not _feature_available(train_rows + test_rows, signal):
                requires_energy = bool(baseline.params.get("requires_energy"))
                full_logits_required, rerun_required = _missing_energy_requires_rerun(train_rows + test_rows)
                return _baseline_unavailable_fold(
                    dataset_name,
                    split_id,
                    row_count,
                    positive_count,
                    train_datasets,
                    reason=FULL_LOGITS_REQUIRED_REASON if requires_energy and full_logits_required else MISSING_SIGNAL_REASON,
                    full_logits_required=requires_energy and full_logits_required,
                    rerun_required=requires_energy and (rerun_required or full_logits_required),
                    fold_safety=fold_safety,
                )
            train_scores = _single_signal_scores(train_rows, signal)
            test_scores = _single_signal_scores(test_rows, signal)
            threshold, _ = _select_threshold(train_labels, train_scores)
            fold_safety["signal"] = signal
        elif baseline.kind == "logistic_feature_set":
            feature_set_name = str(baseline.params.get("feature_set"))
            feature_names = _feature_set(config, feature_set_name)
            if not feature_names or not all(_feature_available(train_rows + test_rows, feature_name) for feature_name in feature_names):
                return _baseline_unavailable_fold(
                    dataset_name,
                    split_id,
                    row_count,
                    positive_count,
                    train_datasets,
                    reason=MISSING_SIGNAL_REASON,
                    fold_safety=fold_safety,
                )
            _train_scores, test_scores, threshold, model = _fit_feature_set_model(
                train_rows,
                test_rows,
                feature_names=feature_names,
                config=config,
            )
            payload = _coefficient_payload(model, fold_id)
            payload["flat_importance"] = dict(payload["coefficients"])
            feature_importance = payload
            fold_safety["feature_set"] = list(feature_names)
        elif baseline.kind == "corpus_bin_feature_selection":
            bin_field = str(baseline.params.get("bin_field", PRIMARY_BIN_FIELD))
            feature_pool = _feature_pool(config, str(baseline.params.get("feature_pool")))
            test_scores, threshold, provenance, flat_importance = _feature_selection_fold(
                train_rows,
                test_rows,
                bin_field=bin_field,
                feature_pool=feature_pool,
                config=config,
                fold_name=fold_id,
            )
            feature_importance = {**flat_importance, "kind": "selected_feature"}
            fold_safety.update(provenance)
        elif baseline.kind == "corpus_bin_weighted_fusion":
            bin_field = str(baseline.params.get("bin_field", PRIMARY_BIN_FIELD))
            feature_names = _feature_set(config, str(baseline.params.get("feature_set")))
            if not feature_names:
                return _baseline_unavailable_fold(
                    dataset_name,
                    split_id,
                    row_count,
                    positive_count,
                    train_datasets,
                    reason=MISSING_SIGNAL_REASON,
                    fold_safety=fold_safety,
                )
            test_scores, threshold, provenance, flat_importance = _corpus_bin_weighted_fold(
                train_rows,
                test_rows,
                bin_field=bin_field,
                feature_names=feature_names,
                config=config,
                fold_name=fold_id,
            )
            feature_importance = {**flat_importance, "kind": "coefficients"}
            fold_safety.update(provenance)
        elif baseline.kind == "axis_interaction_logistic_fusion":
            base_features = _feature_set(config, str(baseline.params.get("feature_set")))
            axis_features = tuple(str(item) for item in baseline.params.get("axis_features", []))
            needed = [*base_features, *axis_features]
            if not base_features or not axis_features or not all(_feature_available(train_rows + test_rows, name) for name in needed):
                return _baseline_unavailable_fold(
                    dataset_name,
                    split_id,
                    row_count,
                    positive_count,
                    train_datasets,
                    reason=MISSING_SIGNAL_REASON,
                    fold_safety=fold_safety,
                )
            _train_scores, test_scores, threshold, model = _fit_interaction_model(
                train_rows,
                test_rows,
                base_features=base_features,
                axis_features=axis_features,
                config=config,
            )
            payload = _coefficient_payload(model, fold_id)
            payload["flat_importance"] = dict(payload["coefficients"])
            feature_importance = payload
            fold_safety["feature_set"] = list(base_features)
            fold_safety["axis_features"] = list(axis_features)
            fold_safety["interaction_features"] = list(model.feature_names)
        else:
            raise ValueError(f"Unsupported baseline kind: {baseline.kind}")
    except ValueError as exc:
        text = str(exc)
        full_logits_required, rerun_required = _missing_energy_requires_rerun(train_rows + test_rows)
        if text.startswith(f"{INSUFFICIENT_BIN_TRAINING_REASON}:"):
            fold_safety["bin_failure"] = text
            return _baseline_unavailable_fold(
                dataset_name,
                split_id,
                row_count,
                positive_count,
                train_datasets,
                reason=INSUFFICIENT_BIN_TRAINING_REASON,
                fold_safety=fold_safety,
            )
        if text.startswith(f"{MISSING_SIGNAL_REASON}:"):
            fold_safety["bin_failure"] = text
            return _baseline_unavailable_fold(
                dataset_name,
                split_id,
                row_count,
                positive_count,
                train_datasets,
                reason=MISSING_SIGNAL_REASON,
                full_logits_required=full_logits_required,
                rerun_required=rerun_required,
                fold_safety=fold_safety,
            )
        raise

    metrics = _evaluate_partition(test_rows, test_scores, threshold, config)
    predictions = _prediction_rows(
        test_rows,
        test_scores,
        baseline_name=baseline.name,
        run_id=run_id,
        threshold=threshold,
        threshold_source=threshold_source,
        train_datasets=train_datasets,
        test_dataset=dataset_name,
        fold_id=fold_id,
    )
    return FoldResult(
        dataset=dataset_name,
        split_id=split_id,
        row_count=row_count,
        positive_count=positive_count,
        threshold=threshold,
        threshold_source=threshold_source,
        metrics=metrics,
        unavailable_reason=None,
        full_logits_required=False,
        rerun_required=False,
        train_datasets=train_datasets,
        prediction_rows=tuple(predictions),
        feature_importance=feature_importance,
        fold_safety=fold_safety,
    )


def _summarize_baseline(baseline: BaselineSpec, folds: list[FoldResult], config: dict[str, Any]) -> dict[str, Any]:
    unavailable = [fold for fold in folds if fold.unavailable_reason is not None]
    feature_importance_per_fold = [fold.feature_importance for fold in folds if fold.feature_importance is not None]
    fold_safety = [fold.fold_safety for fold in folds if fold.fold_safety is not None]
    per_dataset: list[dict[str, Any]] = []
    for fold in folds:
        per_dataset.append(
            {
                "dataset": fold.dataset,
                "split_id": fold.split_id,
                "method_name": baseline.name,
                "row_count": fold.row_count,
                "positive_count": fold.positive_count,
                "metrics": fold.metrics,
                "unavailable_reason": fold.unavailable_reason,
                "full_logits_required": fold.full_logits_required,
                "rerun_required": fold.rerun_required,
                "threshold": fold.threshold,
                "threshold_source": fold.threshold_source,
                "train_datasets": list(fold.train_datasets),
                "fold_safety": fold.fold_safety,
            }
        )
    if unavailable:
        reason = (
            FULL_LOGITS_REQUIRED_REASON
            if any(fold.full_logits_required for fold in unavailable)
            else str(unavailable[0].unavailable_reason or MISSING_SIGNAL_REASON)
        )
        return {
            "method_name": baseline.name,
            "status": "unavailable",
            "unavailable_reason": reason,
            "full_logits_required": reason == FULL_LOGITS_REQUIRED_REASON,
            "rerun_required": reason in {FULL_LOGITS_REQUIRED_REASON, RERUN_REQUIRED_REASON},
            "aggregate": _unavailable_metric_payload(reason if len(unavailable) == len(folds) else PARTIAL_DATASET_AVAILABILITY_REASON),
            "per_dataset": per_dataset,
            "feature_importance": {
                "per_fold": feature_importance_per_fold,
                "aggregate": _aggregate_flat_importance([item for item in feature_importance_per_fold if item is not None]),
            }
            if feature_importance_per_fold
            else None,
            "fold_safety": fold_safety,
        }

    all_predictions = [prediction for fold in folds for prediction in fold.prediction_rows]
    aggregate_rows = _rows_from_predictions(all_predictions)
    aggregate_scores = [float(item["prediction_score"]) for item in all_predictions]
    aggregate_threshold = sum(float(fold.threshold or 0.0) for fold in folds) / len(folds)
    aggregate_metrics = _evaluate_partition(aggregate_rows, aggregate_scores, aggregate_threshold, config)
    return {
        "method_name": baseline.name,
        "status": "ok",
        "unavailable_reason": None,
        "full_logits_required": False,
        "rerun_required": False,
        "aggregate": aggregate_metrics,
        "per_dataset": per_dataset,
        "feature_importance": {
            "per_fold": feature_importance_per_fold,
            "aggregate": _aggregate_flat_importance([item for item in feature_importance_per_fold if item is not None]),
        }
        if feature_importance_per_fold
        else None,
        "fold_safety": fold_safety,
    }


def _flatten_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in summary.get("baselines", []):
        aggregate = baseline.get("aggregate", {})
        rows.append(
            {
                "method_name": baseline.get("method_name"),
                "scope": "aggregate",
                "dataset": "AGGREGATE",
                "bin_field": None,
                "bin_id": None,
                "status": baseline.get("status"),
                "unavailable_reason": baseline.get("unavailable_reason"),
                "full_logits_required": baseline.get("full_logits_required"),
                "rerun_required": baseline.get("rerun_required"),
                "row_count": aggregate.get("row_count"),
                "positive_count": aggregate.get("positive_count"),
                "negative_count": aggregate.get("negative_count"),
                "auroc": aggregate.get("auroc"),
                "auprc": aggregate.get("auprc"),
                "accuracy": aggregate.get("accuracy"),
                "precision": aggregate.get("precision"),
                "recall": aggregate.get("recall"),
                "f1": aggregate.get("f1"),
                "threshold": aggregate.get("threshold"),
                "predicted_positive_rate": aggregate.get("predicted_positive_rate"),
                "brier_score": aggregate.get("brier_score"),
            }
        )
        for dataset_row in baseline.get("per_dataset", []):
            metrics = dataset_row.get("metrics", {})
            rows.append(
                {
                    "method_name": baseline.get("method_name"),
                    "scope": "dataset_holdout",
                    "dataset": dataset_row.get("dataset"),
                    "bin_field": None,
                    "bin_id": None,
                    "status": baseline.get("status") if dataset_row.get("unavailable_reason") is None else "unavailable",
                    "unavailable_reason": dataset_row.get("unavailable_reason"),
                    "full_logits_required": dataset_row.get("full_logits_required"),
                    "rerun_required": dataset_row.get("rerun_required"),
                    "row_count": metrics.get("row_count", dataset_row.get("row_count")),
                    "positive_count": metrics.get("positive_count", dataset_row.get("positive_count")),
                    "negative_count": metrics.get("negative_count"),
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "threshold": dataset_row.get("threshold"),
                    "predicted_positive_rate": metrics.get("predicted_positive_rate"),
                    "brier_score": metrics.get("brier_score"),
                }
            )
            for bin_field in (PRIMARY_BIN_FIELD, SENSITIVITY_BIN_FIELD):
                for bin_entry in metrics.get(bin_field, []):
                    bin_metrics = bin_entry.get("metrics", {})
                    rows.append(
                        {
                            "method_name": baseline.get("method_name"),
                            "scope": "dataset_bin",
                            "dataset": dataset_row.get("dataset"),
                            "bin_field": bin_field,
                            "bin_id": bin_entry.get("bin_id"),
                            "status": "ok" if bin_metrics.get("unavailable_reason") is None else "unavailable",
                            "unavailable_reason": bin_metrics.get("unavailable_reason"),
                            "full_logits_required": dataset_row.get("full_logits_required"),
                            "rerun_required": dataset_row.get("rerun_required"),
                            "row_count": bin_metrics.get("row_count"),
                            "positive_count": bin_metrics.get("positive_count"),
                            "negative_count": bin_metrics.get("negative_count"),
                            "auroc": bin_metrics.get("auroc"),
                            "auprc": bin_metrics.get("auprc"),
                            "accuracy": bin_metrics.get("accuracy"),
                            "precision": bin_metrics.get("precision"),
                            "recall": bin_metrics.get("recall"),
                            "f1": bin_metrics.get("f1"),
                            "threshold": bin_metrics.get("threshold"),
                            "predicted_positive_rate": bin_metrics.get("predicted_positive_rate"),
                            "brier_score": bin_metrics.get("brier_score"),
                        }
                    )
        for bin_field in (PRIMARY_BIN_FIELD, SENSITIVITY_BIN_FIELD):
            for bin_entry in aggregate.get(bin_field, []):
                bin_metrics = bin_entry.get("metrics", {})
                rows.append(
                    {
                        "method_name": baseline.get("method_name"),
                        "scope": "aggregate_bin",
                        "dataset": "AGGREGATE",
                        "bin_field": bin_field,
                        "bin_id": bin_entry.get("bin_id"),
                        "status": "ok" if bin_metrics.get("unavailable_reason") is None else "unavailable",
                        "unavailable_reason": bin_metrics.get("unavailable_reason"),
                        "full_logits_required": baseline.get("full_logits_required"),
                        "rerun_required": baseline.get("rerun_required"),
                        "row_count": bin_metrics.get("row_count"),
                        "positive_count": bin_metrics.get("positive_count"),
                        "negative_count": bin_metrics.get("negative_count"),
                        "auroc": bin_metrics.get("auroc"),
                        "auprc": bin_metrics.get("auprc"),
                        "accuracy": bin_metrics.get("accuracy"),
                        "precision": bin_metrics.get("precision"),
                        "recall": bin_metrics.get("recall"),
                        "f1": bin_metrics.get("f1"),
                        "threshold": bin_metrics.get("threshold"),
                        "predicted_positive_rate": bin_metrics.get("predicted_positive_rate"),
                        "brier_score": bin_metrics.get("brier_score"),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_feature_importance_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for baseline in summary.get("baselines", []):
        importance = baseline.get("feature_importance") or {}
        aggregate = importance.get("aggregate")
        if not isinstance(aggregate, dict):
            continue
        for feature_name, stats in aggregate.items():
            if not isinstance(stats, dict):
                continue
            rows.append(
                {
                    "method_name": baseline.get("method_name"),
                    "feature_name": feature_name,
                    "coefficient_mean": stats.get("mean"),
                    "coefficient_std": stats.get("std"),
                    "coefficient_min": stats.get("min"),
                    "coefficient_max": stats.get("max"),
                    "fold_count": stats.get("fold_count"),
                }
            )
    _write_csv(
        path,
        rows,
        [
            "method_name",
            "feature_name",
            "coefficient_mean",
            "coefficient_std",
            "coefficient_min",
            "coefficient_max",
            "fold_count",
        ],
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fusion baseline evaluation",
        "",
        "Primary split: deterministic leave-one-dataset-out with annotation-backed `is_hallucination` target.",
        "Archived `TypeLabel` / `label` remain diagnostic metadata only.",
        "",
        "## Aggregate baseline table",
        "",
        "| baseline | status | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for baseline in summary.get("baselines", []):
        aggregate = baseline.get("aggregate", {})
        lines.append(
            "| {name} | {status} | {auroc} | {auprc} | {accuracy} | {precision} | {recall} | {f1} | {reason} |".format(
                name=baseline.get("method_name"),
                status=baseline.get("status"),
                auroc=_format_metric(aggregate.get("auroc")),
                auprc=_format_metric(aggregate.get("auprc")),
                accuracy=_format_metric(aggregate.get("accuracy")),
                precision=_format_metric(aggregate.get("precision")),
                recall=_format_metric(aggregate.get("recall")),
                f1=_format_metric(aggregate.get("f1")),
                reason=baseline.get("unavailable_reason") or "",
            )
        )
    lines.extend(["", "## Aggregate corpus-axis bin metrics", ""])
    for baseline in summary.get("baselines", []):
        lines.append(f"### {baseline.get('method_name')}")
        for bin_field in (PRIMARY_BIN_FIELD, SENSITIVITY_BIN_FIELD):
            lines.append("")
            lines.append(f"#### {bin_field}")
            lines.append("")
            lines.append("| bin | AUROC | AUPRC | F1 | reason |")
            lines.append("| --- | ---: | ---: | ---: | --- |")
            for entry in baseline.get("aggregate", {}).get(bin_field, []):
                metrics = entry.get("metrics", {})
                lines.append(
                    "| {bin_id} | {auroc} | {auprc} | {f1} | {reason} |".format(
                        bin_id=entry.get("bin_id"),
                        auroc=_format_metric(metrics.get("auroc")),
                        auprc=_format_metric(metrics.get("auprc")),
                        f1=_format_metric(metrics.get("f1")),
                        reason=metrics.get("unavailable_reason") or "",
                    )
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _learned_fusion_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    by_name = {baseline.get("method_name"): baseline for baseline in summary.get("baselines", [])}
    without_corpus = by_name.get("global learned fusion without corpus axis")
    with_corpus = by_name.get("global learned fusion with corpus axis")
    if not without_corpus or not with_corpus:
        return {"aggregate": {"status": "unavailable", "reason": "missing_learned_baseline"}, "per_dataset": {}}

    def compare_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for key in ("auroc", "auprc", "accuracy", "precision", "recall", "f1"):
            left_value = _coerce_float(left.get(key))
            right_value = _coerce_float(right.get(key))
            delta[key] = None if left_value is None or right_value is None else right_value - left_value
        return delta

    aggregate_without = without_corpus.get("aggregate", {})
    aggregate_with = with_corpus.get("aggregate", {})
    aggregate = {
        "status": "ok"
        if without_corpus.get("status") == "ok" and with_corpus.get("status") == "ok"
        else "unavailable",
        "reason": None,
        "without_corpus": aggregate_without,
        "with_corpus": aggregate_with,
        "delta": compare_metrics(aggregate_without, aggregate_with),
    }
    if aggregate["status"] != "ok":
        aggregate["reason"] = without_corpus.get("unavailable_reason") or with_corpus.get("unavailable_reason")

    without_lookup = {item.get("dataset"): item for item in without_corpus.get("per_dataset", [])}
    with_lookup = {item.get("dataset"): item for item in with_corpus.get("per_dataset", [])}
    per_dataset: dict[str, Any] = {}
    for dataset_name in sorted(set(without_lookup) | set(with_lookup)):
        left = without_lookup.get(dataset_name)
        right = with_lookup.get(dataset_name)
        if not left or not right:
            per_dataset[dataset_name] = {"status": "unavailable", "reason": "missing_dataset_fold"}
            continue
        left_metrics = left.get("metrics", {})
        right_metrics = right.get("metrics", {})
        status = "ok" if left.get("unavailable_reason") is None and right.get("unavailable_reason") is None else "unavailable"
        per_dataset[dataset_name] = {
            "status": status,
            "reason": left.get("unavailable_reason") or right.get("unavailable_reason"),
            "without_corpus": left_metrics,
            "with_corpus": right_metrics,
            "delta": compare_metrics(left_metrics, right_metrics),
        }
    return {"aggregate": aggregate, "per_dataset": per_dataset}


def run_fusion(features_path: Path, config_path: Path, out_dir: Path) -> dict[str, Any]:
    rows, storage_report = read_feature_rows(features_path)
    if not rows:
        raise ValueError(f"Fusion features artifact is empty: {features_path}")
    config = load_json(config_path)
    baselines = _baseline_specs(config)
    datasets = _ordered_datasets(config, rows)
    run_id = f"fusion-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    baseline_summaries: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for baseline in baselines:
        folds: list[FoldResult] = []
        for dataset_name in datasets:
            test_rows = _rows_for_dataset(rows, dataset_name)
            train_rows = [row for row in rows if _dataset_name(row) != dataset_name]
            folds.append(_evaluate_baseline_on_fold(baseline, train_rows, test_rows, run_id=run_id, config=config))
        baseline_summary = _summarize_baseline(baseline, folds, config)
        baseline_summaries.append(baseline_summary)
        for fold in folds:
            prediction_rows.extend(list(fold.prediction_rows))

    summary = {
        "run_id": run_id,
        "generated_at": _iso_now(),
        "method_name": str(config.get("method_name", "Corpus-Conditioned Hallucination Metric Reliability Study")),
        "features_path": str(features_path),
        "features_storage": storage_report,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "row_count": len(rows),
        "datasets": list(datasets),
        "evaluation": config.get("evaluation", {}),
        "binary_target": {
            **(config.get("binary_target", {}) if isinstance(config.get("binary_target"), dict) else {}),
            "field": "is_hallucination",
            "fallback": "not is_correct",
            "archived_label_not_primary": True,
        },
        "formula_manifest_ref": FORMULA_MANIFEST_REF,
        "dataset_manifest_ref": DATASET_MANIFEST_REF,
        "baselines": baseline_summaries,
    }
    summary["learned_fusion_comparison"] = _learned_fusion_comparison(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    metrics_csv_path = out_dir / "baseline_metrics.csv"
    predictions_path = out_dir / "predictions.jsonl"
    importance_json_path = out_dir / "feature_importance.json"
    importance_csv_path = out_dir / "feature_importance.csv"
    comparison_path = out_dir / "learned_fusion_comparison.json"
    report_path = out_dir / "report.md"

    write_json(summary_path, summary)
    _write_csv(
        metrics_csv_path,
        _flatten_summary_rows(summary),
        [
            "method_name",
            "scope",
            "dataset",
            "bin_field",
            "bin_id",
            "status",
            "unavailable_reason",
            "full_logits_required",
            "rerun_required",
            "row_count",
            "positive_count",
            "negative_count",
            "auroc",
            "auprc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "threshold",
            "predicted_positive_rate",
            "brier_score",
        ],
    )
    _write_predictions(predictions_path, prediction_rows)
    feature_importance_payload = {
        baseline["method_name"]: baseline["feature_importance"]
        for baseline in baseline_summaries
        if baseline.get("feature_importance") is not None
    }
    write_json(importance_json_path, feature_importance_payload)
    _write_feature_importance_csv(importance_csv_path, summary)
    write_json(comparison_path, summary["learned_fusion_comparison"])
    _write_markdown(report_path, summary)

    return {
        "run_id": run_id,
        "row_count": len(rows),
        "datasets": list(datasets),
        "artifacts": {
            "summary": str(summary_path),
            "baseline_metrics_csv": str(metrics_csv_path),
            "predictions": str(predictions_path),
            "feature_importance_json": str(importance_json_path),
            "feature_importance_csv": str(importance_csv_path),
            "learned_fusion_comparison": str(comparison_path),
            "report": str(report_path),
        },
        "required_baselines": [baseline.name for baseline in baselines],
    }
