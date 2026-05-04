"""Leakage-safe fusion baseline evaluation for hallucination detection."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.adapters.corpus_features import load_json, read_feature_rows, write_json
from experiments.domain import TypeLabel

NORMAL_LABEL = TypeLabel.NORMAL.value
FULL_LOGITS_REQUIRED_REASON = "full_logits_required"
RERUN_REQUIRED_REASON = "rerun_required"
MISSING_SIGNAL_REASON = "missing_signal"
PARTIAL_DATASET_AVAILABILITY_REASON = "partial_dataset_availability"
EMPTY_DATASET_REASON = "empty_dataset"
SINGLE_CLASS_REASON = "single_class_subset"

FORMULA_MANIFEST_REF = "experiments/literature/formula_notes.md"
DATASET_MANIFEST_REF = "experiments/manifests/upstream_artifacts_manifest.json"


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


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    kind: str
    signal: str | None = None
    semantic_entropy_weight: float | None = None
    semantic_energy_weight: float | None = None


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


def _dataset_name(row: dict[str, Any]) -> str:
    return str(row.get("dataset", "unknown"))


def _split_id(row: dict[str, Any]) -> str:
    return str(row.get("split_id", "unknown_split"))


def _sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", "unknown_sample"))


def _label(row: dict[str, Any]) -> str:
    return str(row.get("label", ""))


def _is_positive(row: dict[str, Any]) -> int:
    return 0 if _label(row) == NORMAL_LABEL else 1


def _is_energy_available(row: dict[str, Any]) -> bool:
    features = _features(row)
    available = _coerce_bool(features.get("energy_available"))
    return bool(available) and _coerce_float(features.get("semantic_energy_boltzmann")) is not None


def _has_required_signal(row: dict[str, Any], signal_name: str) -> bool:
    return _coerce_float(_features(row).get(signal_name)) is not None


def _rank_average_pairs(scores: list[float]) -> list[tuple[float, float]]:
    ordered = sorted(enumerate(scores), key=lambda item: (item[1], item[0]))
    ranks: list[float] = [0.0] * len(scores)
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
    positive_rank_sum = sum(rank for label, (_, rank) in zip(labels, ranked, strict=True) if label == 1)
    return (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)


def _compute_auprc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(
        zip(scores, labels, range(len(labels)), strict=True),
        key=lambda item: (-item[0], item[2]),
    )
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
    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def _threshold_candidates(scores: list[float]) -> list[float]:
    ordered = sorted(set(scores))
    if not ordered:
        return [0.5]
    candidates = [0.0]
    candidates.extend(ordered)
    candidates.append(1.0)
    return candidates


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


def _label_breakdown(rows: list[dict[str, Any]], scores: list[float], threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(_label(row), []).append((row, score))
    payload: dict[str, Any] = {}
    for label_name, items in sorted(grouped.items()):
        label_scores = [score for _, score in items]
        predicted_positive = sum(1 for score in label_scores if score >= threshold)
        payload[label_name] = {
            "row_count": len(items),
            "score_mean": sum(label_scores) / len(label_scores) if label_scores else None,
            "score_min": min(label_scores) if label_scores else None,
            "score_max": max(label_scores) if label_scores else None,
            "predicted_positive_rate": predicted_positive / len(items) if items else None,
        }
    return payload


def _subset_metric(rows: list[dict[str, Any]], scores: list[float], positive_label: str) -> dict[str, Any]:
    subset_rows: list[dict[str, Any]] = []
    subset_scores: list[float] = []
    labels: list[int] = []
    for row, score in zip(rows, scores, strict=True):
        label_name = _label(row)
        if label_name not in {NORMAL_LABEL, positive_label}:
            continue
        subset_rows.append(row)
        subset_scores.append(score)
        labels.append(1 if label_name == positive_label else 0)
    return {
        "row_count": len(subset_rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "auroc": _compute_auroc(labels, subset_scores) if subset_rows else None,
        "auprc": _compute_auprc(labels, subset_scores) if subset_rows else None,
        "reason": None if len(set(labels)) == 2 else (EMPTY_DATASET_REASON if not labels else SINGLE_CLASS_REASON),
    }


def _evaluate_rows(rows: list[dict[str, Any]], scores: list[float], threshold: float) -> dict[str, Any]:
    labels = [_is_positive(row) for row in rows]
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
        "type_specific": {
            "high_diversity_vs_normal": _subset_metric(rows, scores, TypeLabel.HIGH_DIVERSITY.value),
            "low_diversity_vs_normal": _subset_metric(rows, scores, TypeLabel.LOW_DIVERSITY.value),
        },
        "per_label": _label_breakdown(rows, scores, threshold),
    }


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
        if scale_value == 0.0:
            scale_value = 1.0
        means.append(mean_value)
        scales.append(scale_value)
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


def _extract_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        features = _features(row)
        vector: list[float] = []
        for feature_name in feature_names:
            raw_value = features.get(feature_name)
            if isinstance(raw_value, bool):
                vector.append(1.0 if raw_value else 0.0)
                continue
            value = _coerce_float(raw_value)
            if value is None:
                raise ValueError(f"Missing numeric feature {feature_name!r} for sample {_sample_id(row)!r}")
            vector.append(value)
        matrix.append(vector)
    return matrix


def _feature_available(rows: list[dict[str, Any]], feature_name: str) -> bool:
    for row in rows:
        value = _features(row).get(feature_name)
        if isinstance(value, bool):
            return True
        if _coerce_float(value) is not None:
            return True
    return False


def _resolve_feature_set(
    rows: list[dict[str, Any]],
    requested: list[str],
    *,
    forbidden: set[str] | None = None,
    require_energy: bool = False,
) -> tuple[str, ...]:
    selected: list[str] = []
    forbidden = forbidden or set()
    for feature_name in requested:
        if feature_name in forbidden:
            continue
        if require_energy and feature_name == "semantic_energy_boltzmann" and not all(_is_energy_available(row) for row in rows):
            continue
        if _feature_available(rows, feature_name):
            selected.append(feature_name)
    return tuple(selected)


def _normalize_signal(scores: list[float]) -> tuple[list[float], float, float]:
    if not scores:
        return [], 0.0, 1.0
    minimum = min(scores)
    maximum = max(scores)
    if maximum <= minimum:
        return [0.0 for _ in scores], minimum, maximum
    return ([(score - minimum) / (maximum - minimum) for score in scores], minimum, maximum)


def _build_single_signal_scores(rows: list[dict[str, Any]], signal_name: str) -> list[float]:
    scores: list[float] = []
    for row in rows:
        value = _coerce_float(_features(row).get(signal_name))
        if value is None:
            raise ValueError(f"Missing signal {signal_name!r} for sample {_sample_id(row)!r}")
        scores.append(value)
    return scores


def _build_fixed_linear_scores(
    rows: list[dict[str, Any]],
    *,
    train_rows: list[dict[str, Any]],
    semantic_entropy_weight: float,
    semantic_energy_weight: float,
) -> list[float]:
    train_se = _build_single_signal_scores(train_rows, "semantic_entropy")
    train_energy = _build_single_signal_scores(train_rows, "semantic_energy_boltzmann")
    _, se_min, se_max = _normalize_signal(train_se)
    _, energy_min, energy_max = _normalize_signal(train_energy)
    scores: list[float] = []
    for row in rows:
        semantic_entropy = _coerce_float(_features(row).get("semantic_entropy"))
        semantic_energy = _coerce_float(_features(row).get("semantic_energy_boltzmann"))
        if semantic_entropy is None or semantic_energy is None:
            raise ValueError("Fixed linear baseline received missing SE/Energy signal")
        se_norm = 0.0 if se_max <= se_min else (semantic_entropy - se_min) / (se_max - se_min)
        energy_norm = 0.0 if energy_max <= energy_min else (semantic_energy - energy_min) / (energy_max - energy_min)
        scores.append((semantic_entropy_weight * se_norm) + (semantic_energy_weight * energy_norm))
    return scores


def _build_hard_cascade_scores(rows: list[dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    for row in rows:
        features = _features(row)
        semantic_entropy = _coerce_float(features.get("semantic_entropy"))
        semantic_energy = _coerce_float(features.get("semantic_energy_boltzmann"))
        if semantic_entropy is None or semantic_energy is None:
            raise ValueError("Hard cascade baseline received missing SE/Energy signal")
        if semantic_entropy <= 0.1:
            scores.append(semantic_energy)
        else:
            scores.append(semantic_entropy)
    return scores


def _build_coverage_adaptive_scores(rows: list[dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    for row in rows:
        features = _features(row)
        semantic_entropy = _coerce_float(features.get("semantic_entropy"))
        semantic_energy = _coerce_float(features.get("semantic_energy_boltzmann"))
        coverage = _coerce_float(features.get("coverage_score"))
        if semantic_entropy is None or semantic_energy is None or coverage is None:
            raise ValueError("Coverage-adaptive baseline received missing SE/Energy/Coverage signal")
        energy_weight = max(0.2, min(0.8, 0.8 - (0.6 * coverage)))
        semantic_entropy_weight = 1.0 - energy_weight
        scores.append((semantic_entropy_weight * semantic_entropy) + (energy_weight * semantic_energy))
    return scores


def _unavailable_payload(reason: str) -> dict[str, Any]:
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
        "type_specific": {
            "high_diversity_vs_normal": {
                "row_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "auroc": None,
                "auprc": None,
                "reason": reason,
            },
            "low_diversity_vs_normal": {
                "row_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "auroc": None,
                "auprc": None,
                "reason": reason,
            },
        },
        "per_label": {},
    }


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
) -> list[dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        prediction_rows.append(
            {
                "run_id": run_id,
                "method_name": baseline_name,
                "dataset": _dataset_name(row),
                "split_id": f"leave_one_dataset_out::{test_dataset}",
                "sample_id": _sample_id(row),
                "label": _label(row),
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


def _baseline_specs(config: dict[str, Any]) -> tuple[BaselineSpec, ...]:
    baselines = config.get("baselines")
    if not isinstance(baselines, list):
        raise ValueError("Fusion config is missing a baselines list")
    specs: list[BaselineSpec] = []
    for entry in baselines:
        if not isinstance(entry, dict):
            continue
        specs.append(
            BaselineSpec(
                name=str(entry["name"]),
                kind=str(entry["kind"]),
                signal=entry.get("signal"),
                semantic_entropy_weight=_coerce_float(entry.get("semantic_entropy_weight")),
                semantic_energy_weight=_coerce_float(entry.get("semantic_energy_weight")),
            )
        )
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


def _fit_learned_baseline(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    config: dict[str, Any],
) -> tuple[list[float], list[float], float, LogisticRegressionModel]:
    training = config.get("learned_fusion", {})
    train_matrix = _extract_matrix(train_rows, feature_names)
    test_matrix = _extract_matrix(test_rows, feature_names)
    train_labels = [_is_positive(row) for row in train_rows]
    model = _fit_logistic_regression(
        train_matrix,
        train_labels,
        feature_names=feature_names,
        standardize=bool(training.get("standardize", True)),
        l2_lambda=float(training.get("l2_lambda", 0.1)),
        learning_rate=float(training.get("learning_rate", 0.1)),
        max_iterations=int(training.get("max_iterations", 1200)),
        tolerance=float(training.get("tolerance", 1e-6)),
    )
    train_scores = model.predict_scores(train_matrix)
    test_scores = model.predict_scores(test_matrix)
    threshold, _ = _select_threshold(train_labels, train_scores)
    return train_scores, test_scores, threshold, model


def _coefficient_payload(model: LogisticRegressionModel, fold_name: str) -> dict[str, Any]:
    coefficients = {name: weight for name, weight in zip(model.feature_names, model.weights, strict=True)}
    scaler = None
    if model.scaler is not None:
        scaler = {
            "means": {name: value for name, value in zip(model.feature_names, model.scaler.means, strict=True)},
            "scales": {name: value for name, value in zip(model.feature_names, model.scaler.scales, strict=True)},
        }
    return {
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


def _aggregate_coefficients(per_fold: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not per_fold:
        return None
    grouped: dict[str, list[float]] = {}
    for fold in per_fold:
        coefficients = fold.get("coefficients")
        if not isinstance(coefficients, dict):
            continue
        for feature_name, weight in coefficients.items():
            numeric = _coerce_float(weight)
            if numeric is None:
                continue
            grouped.setdefault(str(feature_name), []).append(numeric)
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
    positive_count = sum(_is_positive(row) for row in test_rows)
    row_count = len(test_rows)

    if not test_rows:
        return FoldResult(
            dataset=dataset_name,
            split_id=split_id,
            row_count=0,
            positive_count=0,
            threshold=None,
            threshold_source=None,
            metrics=_unavailable_payload(EMPTY_DATASET_REASON),
            unavailable_reason=EMPTY_DATASET_REASON,
            full_logits_required=False,
            rerun_required=False,
            train_datasets=train_datasets,
            prediction_rows=(),
            feature_importance=None,
        )

    if baseline.kind in {"energy_only", "fixed_linear", "hard_cascade", "coverage_adaptive"}:
        if not all(_is_energy_available(row) for row in train_rows + test_rows):
            return FoldResult(
                dataset=dataset_name,
                split_id=split_id,
                row_count=row_count,
                positive_count=positive_count,
                threshold=None,
                threshold_source=None,
                metrics=_unavailable_payload(FULL_LOGITS_REQUIRED_REASON),
                unavailable_reason=FULL_LOGITS_REQUIRED_REASON,
                full_logits_required=True,
                rerun_required=True,
                train_datasets=train_datasets,
                prediction_rows=(),
                feature_importance=None,
            )

    train_labels = [_is_positive(row) for row in train_rows]
    if len(set(train_labels)) < 2:
        return FoldResult(
            dataset=dataset_name,
            split_id=split_id,
            row_count=row_count,
            positive_count=positive_count,
            threshold=None,
            threshold_source=None,
            metrics=_unavailable_payload(SINGLE_CLASS_REASON),
            unavailable_reason=SINGLE_CLASS_REASON,
            full_logits_required=False,
            rerun_required=False,
            train_datasets=train_datasets,
            prediction_rows=(),
            feature_importance=None,
        )

    feature_importance = None
    threshold_source = "train_max_f1"
    entropy_weight = baseline.semantic_entropy_weight if baseline.semantic_entropy_weight is not None else 0.0
    energy_weight = baseline.semantic_energy_weight if baseline.semantic_energy_weight is not None else 0.0
    if baseline.kind == "single_signal":
        signal = str(baseline.signal)
        if not all(_has_required_signal(row, signal) for row in train_rows + test_rows):
            return FoldResult(
                dataset=dataset_name,
                split_id=split_id,
                row_count=row_count,
                positive_count=positive_count,
                threshold=None,
                threshold_source=None,
                metrics=_unavailable_payload(MISSING_SIGNAL_REASON),
                unavailable_reason=MISSING_SIGNAL_REASON,
                full_logits_required=False,
                rerun_required=False,
                train_datasets=train_datasets,
                prediction_rows=(),
                feature_importance=None,
            )
        train_scores = _build_single_signal_scores(train_rows, signal)
        test_scores = _build_single_signal_scores(test_rows, signal)
        threshold, _ = _select_threshold(train_labels, train_scores)
    elif baseline.kind == "energy_only":
        train_scores = _build_single_signal_scores(train_rows, "semantic_energy_boltzmann")
        test_scores = _build_single_signal_scores(test_rows, "semantic_energy_boltzmann")
        threshold, _ = _select_threshold(train_labels, train_scores)
    elif baseline.kind == "fixed_linear":
        train_scores = _build_fixed_linear_scores(
            train_rows,
            train_rows=train_rows,
            semantic_entropy_weight=entropy_weight,
            semantic_energy_weight=energy_weight,
        )
        test_scores = _build_fixed_linear_scores(
            test_rows,
            train_rows=train_rows,
            semantic_entropy_weight=entropy_weight,
            semantic_energy_weight=energy_weight,
        )
        threshold, _ = _select_threshold(train_labels, train_scores)
    elif baseline.kind == "hard_cascade":
        train_scores = _build_hard_cascade_scores(train_rows)
        test_scores = _build_hard_cascade_scores(test_rows)
        threshold, _ = _select_threshold(train_labels, train_scores)
    elif baseline.kind == "coverage_adaptive":
        train_scores = _build_coverage_adaptive_scores(train_rows)
        test_scores = _build_coverage_adaptive_scores(test_rows)
        threshold, _ = _select_threshold(train_labels, train_scores)
    elif baseline.kind == "learned_without_corpus":
        requested = list(config.get("learned_fusion", {}).get("feature_sets", {}).get("without_corpus", []))
        forbidden = {str(item) for item in config.get("learned_fusion", {}).get("forbidden_features", [])}
        feature_names = _resolve_feature_set(train_rows + test_rows, requested, forbidden=forbidden)
        if not feature_names:
            return FoldResult(
                dataset=dataset_name,
                split_id=split_id,
                row_count=row_count,
                positive_count=positive_count,
                threshold=None,
                threshold_source=None,
                metrics=_unavailable_payload(MISSING_SIGNAL_REASON),
                unavailable_reason=MISSING_SIGNAL_REASON,
                full_logits_required=False,
                rerun_required=False,
                train_datasets=train_datasets,
                prediction_rows=(),
                feature_importance=None,
            )
        train_scores, test_scores, threshold, model = _fit_learned_baseline(
            train_rows,
            test_rows,
            feature_names=feature_names,
            config=config,
        )
        feature_importance = _coefficient_payload(model, dataset_name)
    elif baseline.kind == "learned_with_corpus":
        requested = list(config.get("learned_fusion", {}).get("feature_sets", {}).get("with_corpus", []))
        forbidden = {str(item) for item in config.get("learned_fusion", {}).get("forbidden_features", [])}
        feature_names = _resolve_feature_set(train_rows + test_rows, requested, forbidden=forbidden)
        if not feature_names:
            return FoldResult(
                dataset=dataset_name,
                split_id=split_id,
                row_count=row_count,
                positive_count=positive_count,
                threshold=None,
                threshold_source=None,
                metrics=_unavailable_payload(MISSING_SIGNAL_REASON),
                unavailable_reason=MISSING_SIGNAL_REASON,
                full_logits_required=False,
                rerun_required=False,
                train_datasets=train_datasets,
                prediction_rows=(),
                feature_importance=None,
            )
        train_scores, test_scores, threshold, model = _fit_learned_baseline(
            train_rows,
            test_rows,
            feature_names=feature_names,
            config=config,
        )
        feature_importance = _coefficient_payload(model, dataset_name)
    else:
        raise ValueError(f"Unsupported baseline kind: {baseline.kind}")

    metrics = _evaluate_rows(test_rows, test_scores, threshold)
    predictions = _prediction_rows(
        test_rows,
        test_scores,
        baseline_name=baseline.name,
        run_id=run_id,
        threshold=threshold,
        threshold_source=threshold_source,
        train_datasets=train_datasets,
        test_dataset=dataset_name,
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
    )


def _summarize_baseline(
    baseline: BaselineSpec,
    folds: list[FoldResult],
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unavailable = [fold for fold in folds if fold.unavailable_reason is not None]
    feature_importance_per_fold = [fold.feature_importance for fold in folds if fold.feature_importance is not None]
    if unavailable:
        reason = (
            FULL_LOGITS_REQUIRED_REASON
            if any(fold.full_logits_required for fold in unavailable)
            else str(unavailable[0].unavailable_reason or MISSING_SIGNAL_REASON)
        )
        per_dataset = []
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
                }
            )
        return {
            "method_name": baseline.name,
            "status": "unavailable",
            "unavailable_reason": reason,
            "full_logits_required": reason == FULL_LOGITS_REQUIRED_REASON,
            "rerun_required": reason == FULL_LOGITS_REQUIRED_REASON,
            "aggregate": _unavailable_payload(reason if len(unavailable) == len(folds) else PARTIAL_DATASET_AVAILABILITY_REASON),
            "per_dataset": per_dataset,
            "feature_importance": {
                "per_fold": feature_importance_per_fold,
                "aggregate": _aggregate_coefficients([item for item in feature_importance_per_fold if item is not None]),
            }
            if feature_importance_per_fold
            else None,
        }

    combined_rows: list[dict[str, Any]] = []
    combined_scores: list[float] = []
    per_dataset = []
    for fold in folds:
        fold_scores = [float(item["prediction_score"]) for item in fold.prediction_rows]
        fold_rows = _rows_for_dataset(all_rows, fold.dataset)
        combined_rows.extend(fold_rows)
        combined_scores.extend(fold_scores)
        per_dataset.append(
            {
                "dataset": fold.dataset,
                "split_id": fold.split_id,
                "method_name": baseline.name,
                "row_count": fold.row_count,
                "positive_count": fold.positive_count,
                "metrics": fold.metrics,
                "unavailable_reason": None,
                "full_logits_required": False,
                "rerun_required": False,
                "threshold": fold.threshold,
                "threshold_source": fold.threshold_source,
                "train_datasets": list(fold.train_datasets),
            }
        )

    combined_threshold = sum(fold.threshold or 0.0 for fold in folds) / len(folds)
    aggregate_metrics = _evaluate_rows(combined_rows, combined_scores, combined_threshold)
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
            "aggregate": _aggregate_coefficients([item for item in feature_importance_per_fold if item is not None]),
        }
        if feature_importance_per_fold
        else None,
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
                "high_diversity_auroc": aggregate.get("type_specific", {}).get("high_diversity_vs_normal", {}).get("auroc"),
                "low_diversity_auroc": aggregate.get("type_specific", {}).get("low_diversity_vs_normal", {}).get("auroc"),
            }
        )
        for per_dataset in baseline.get("per_dataset", []):
            metrics = per_dataset.get("metrics", {})
            rows.append(
                {
                    "method_name": baseline.get("method_name"),
                    "scope": "dataset_holdout",
                    "dataset": per_dataset.get("dataset"),
                    "status": baseline.get("status") if per_dataset.get("unavailable_reason") is None else "unavailable",
                    "unavailable_reason": per_dataset.get("unavailable_reason"),
                    "full_logits_required": per_dataset.get("full_logits_required"),
                    "rerun_required": per_dataset.get("rerun_required"),
                    "row_count": metrics.get("row_count", per_dataset.get("row_count")),
                    "positive_count": metrics.get("positive_count", per_dataset.get("positive_count")),
                    "negative_count": metrics.get("negative_count"),
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "threshold": per_dataset.get("threshold"),
                    "predicted_positive_rate": metrics.get("predicted_positive_rate"),
                    "brier_score": metrics.get("brier_score"),
                    "high_diversity_auroc": metrics.get("type_specific", {}).get("high_diversity_vs_normal", {}).get("auroc"),
                    "low_diversity_auroc": metrics.get("type_specific", {}).get("low_diversity_vs_normal", {}).get("auroc"),
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
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fusion baseline evaluation",
        "",
        "Primary split: deterministic leave-one-dataset-out. This keeps dataset-level visibility front and center and avoids a pooled random split as the headline.",
        "",
        "True Boltzmann Semantic Energy stays explicit. If full logits are unavailable, Energy-dependent baselines remain listed with null metrics and rerun flags instead of quietly using `semantic_energy_proxy`.",
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
    lines.extend(["", "## Learned fusion comparison", ""])
    comparison = summary.get("learned_fusion_comparison", {})
    aggregate = comparison.get("aggregate", {}) if isinstance(comparison, dict) else {}
    lines.append("### aggregate")
    lines.append("")
    if not isinstance(aggregate, dict) or aggregate.get("status") != "ok":
        lines.append(f"Unavailable: {aggregate.get('reason') if isinstance(aggregate, dict) else 'unknown'}")
        lines.append("")
    else:
        lines.append(
            "Observed delta, with corpus minus without corpus: AUROC {auroc}, AUPRC {auprc}, F1 {f1}.".format(
                auroc=_format_metric(aggregate.get("delta", {}).get("auroc")),
                auprc=_format_metric(aggregate.get("delta", {}).get("auprc")),
                f1=_format_metric(aggregate.get("delta", {}).get("f1")),
            )
        )
        lines.append("")
    per_dataset = comparison.get("per_dataset", {}) if isinstance(comparison, dict) else {}
    lines.append("### per_dataset")
    lines.append("")
    lines.append("| dataset | status | ΔAUROC | ΔAUPRC | ΔF1 | reason |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for dataset_name, payload in sorted(per_dataset.items()):
        delta = payload.get("delta", {}) if isinstance(payload, dict) else {}
        lines.append(
            "| {dataset} | {status} | {auroc} | {auprc} | {f1} | {reason} |".format(
                dataset=dataset_name,
                status=payload.get("status") if isinstance(payload, dict) else "unavailable",
                auroc=_format_metric(delta.get("auroc")),
                auprc=_format_metric(delta.get("auprc")),
                f1=_format_metric(delta.get("f1")),
                reason=(payload.get("reason") or "") if isinstance(payload, dict) else "unknown",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _learned_fusion_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    by_name = {baseline.get("method_name"): baseline for baseline in summary.get("baselines", [])}
    without_corpus = by_name.get("learned fusion without corpus")
    with_corpus = by_name.get("learned fusion with corpus")
    if not without_corpus or not with_corpus:
        return {"aggregate": {"status": "unavailable", "reason": "missing_learned_baseline"}}

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

    per_dataset: dict[str, Any] = {}
    without_lookup = {item.get("dataset"): item for item in without_corpus.get("per_dataset", [])}
    with_lookup = {item.get("dataset"): item for item in with_corpus.get("per_dataset", [])}
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
            folds.append(
                _evaluate_baseline_on_fold(
                    baseline,
                    train_rows,
                    test_rows,
                    run_id=run_id,
                    config=config,
                )
            )
        baseline_summary = _summarize_baseline(baseline, folds, rows)
        baseline_summaries.append(baseline_summary)
        for fold in folds:
            prediction_rows.extend(list(fold.prediction_rows))

    summary = {
        "run_id": run_id,
        "generated_at": _iso_now(),
        "method_name": str(config.get("method_name", "Corpus-Grounded Selective Fusion Detector")),
        "features_path": str(features_path),
        "features_storage": storage_report,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "row_count": len(rows),
        "datasets": list(datasets),
        "evaluation": config.get("evaluation", {}),
        "binary_target": config.get("binary_target", {}),
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
            "high_diversity_auroc",
            "low_diversity_auroc",
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
