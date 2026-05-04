"""Robustness, threshold-sensitivity, and selective-risk reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.adapters.corpus_features import load_json, read_feature_rows, write_json
from experiments.application.fusion import (
    DATASET_MANIFEST_REF,
    FORMULA_MANIFEST_REF,
    _coerce_float,
    _compute_auprc,
    _compute_auroc,
    _evaluate_rows,
    _fit_learned_baseline,
    _is_positive,
    _label,
    _ordered_datasets,
    _resolve_feature_set,
    _select_threshold,
)
from experiments.domain import TypeLabel

NORMAL_LABEL = TypeLabel.NORMAL.value
HIGH_DIVERSITY_LABEL = TypeLabel.HIGH_DIVERSITY.value
LOW_DIVERSITY_LABEL = TypeLabel.LOW_DIVERSITY.value
AMBIGUOUS_INCORRECT_LABEL = TypeLabel.AMBIGUOUS_INCORRECT.value

BOOTSTRAP_SEED = 20260504
BOOTSTRAP_ITERATIONS = 2000
DETERMINISTIC_SPLIT_MODULUS = 5
DETERMINISTIC_SPLIT_TEST_BUCKET = 0
LOW_CUTOFFS = (0.05, 0.1, 0.2)
HIGH_CUTOFFS = (0.4, 0.5, 0.6)
SELECTIVE_METHODS = (
    "SE-only",
    "corpus-risk-only",
    "learned fusion without corpus",
    "learned fusion with corpus",
)
BOOTSTRAP_COMPARISONS = (
    ("learned fusion with corpus", "learned fusion without corpus"),
    ("learned fusion with corpus", "SE-only"),
    ("learned fusion without corpus", "SE-only"),
    ("learned fusion with corpus", "corpus-risk-only"),
)
FORBIDDEN_OVERCLAIM_PHRASES = (
    "proven",
    "significant improvement",
    "confirmed",
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


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


def _baseline_lookup(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    baselines = summary.get("baselines", [])
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
        method_name = str(row.get("method_name", "unknown"))
        sample_id = str(row.get("sample_id", "unknown_sample"))
        grouped.setdefault(method_name, {})[sample_id] = row
    return grouped


def _truth_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id", "unknown_sample")): row for row in rows}


def _label_from_cutoffs(row: dict[str, Any], low_cutoff: float, high_cutoff: float) -> str:
    semantic_entropy = _coerce_float((row.get("features") or {}).get("semantic_entropy"))
    if semantic_entropy is None:
        return _label(row)
    if _label(row) == NORMAL_LABEL:
        return NORMAL_LABEL
    if semantic_entropy <= low_cutoff:
        return LOW_DIVERSITY_LABEL
    if semantic_entropy <= high_cutoff:
        return AMBIGUOUS_INCORRECT_LABEL
    return HIGH_DIVERSITY_LABEL


def _metric_from_scores(labels: list[int], scores: list[float], metric_name: str) -> float | None:
    if metric_name == "auroc":
        return _compute_auroc(labels, scores)
    if metric_name == "auprc":
        return _compute_auprc(labels, scores)
    raise ValueError(f"Unsupported metric for bootstrap comparison: {metric_name}")


def _bootstrap_metric_delta(
    labels: list[int],
    candidate_scores: list[float],
    reference_scores: list[float],
    *,
    metric_name: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
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
            "statistically_significant": False,
            "ci_crosses_zero": None,
            "claim_text": "Observed delta unavailable because the resampled target collapses to a single class.",
        }

    observed_delta = observed_candidate - observed_reference
    rng = random.Random(seed)
    deltas: list[float] = []
    sample_count = len(labels)
    for _ in range(iterations):
        indices = [rng.randrange(sample_count) for _ in range(sample_count)]
        boot_labels = [labels[index] for index in indices]
        if sum(boot_labels) == 0 or sum(boot_labels) == len(boot_labels):
            continue
        boot_candidate = [candidate_scores[index] for index in indices]
        boot_reference = [reference_scores[index] for index in indices]
        candidate_metric = _metric_from_scores(boot_labels, boot_candidate, metric_name)
        reference_metric = _metric_from_scores(boot_labels, boot_reference, metric_name)
        if candidate_metric is None or reference_metric is None:
            continue
        deltas.append(candidate_metric - reference_metric)

    ci_lower = _percentile(deltas, 0.025)
    ci_upper = _percentile(deltas, 0.975)
    crosses_zero = None if ci_lower is None or ci_upper is None else ci_lower <= 0.0 <= ci_upper
    statistically_significant = bool(crosses_zero is False)
    if crosses_zero is True:
        claim_text = "Observed delta only. The bootstrap interval crosses zero, so this run does not support a directional claim."
    elif crosses_zero is False:
        direction = "above" if observed_delta > 0 else "below"
        claim_text = f"Observed delta stays {direction} zero in this bootstrap interval."
    else:
        claim_text = "Observed delta only. The bootstrap interval is unavailable."
    return {
        "metric": metric_name,
        "observed_delta": observed_delta,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "bootstrap_mean_delta": None if not deltas else sum(deltas) / len(deltas),
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "valid_iterations": len(deltas),
        "statistically_significant": statistically_significant,
        "ci_crosses_zero": crosses_zero,
        "claim_text": claim_text,
    }


def _build_bootstrap_report(
    truth_rows: list[dict[str, Any]],
    predictions_by_method: dict[str, dict[str, dict[str, Any]]],
    baseline_lookup: dict[str, dict[str, Any]],
    *,
    seed: int,
    iterations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_ids = sorted(str(row.get("sample_id", "unknown_sample")) for row in truth_rows)
    labels = [_is_positive(next(row for row in truth_rows if str(row.get("sample_id")) == sample_id)) for sample_id in sample_ids]
    comparisons: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    truth_lookup = _truth_by_sample(truth_rows)
    for candidate_method, reference_method in BOOTSTRAP_COMPARISONS:
        candidate_summary = baseline_lookup.get(candidate_method, {})
        reference_summary = baseline_lookup.get(reference_method, {})
        candidate_predictions = predictions_by_method.get(candidate_method)
        reference_predictions = predictions_by_method.get(reference_method)
        if (
            candidate_summary.get("status") != "ok"
            or reference_summary.get("status") != "ok"
            or not candidate_predictions
            or not reference_predictions
        ):
            comparisons.append(
                {
                    "candidate_method": candidate_method,
                    "reference_method": reference_method,
                    "status": "unavailable",
                    "reason": "missing_available_predictions",
                    "metrics": [],
                }
            )
            continue

        common_sample_ids = [
            sample_id
            for sample_id in sample_ids
            if sample_id in candidate_predictions and sample_id in reference_predictions and sample_id in truth_lookup
        ]
        candidate_scores = [
            float(candidate_predictions[sample_id]["prediction_score"])
            for sample_id in common_sample_ids
        ]
        reference_scores = [
            float(reference_predictions[sample_id]["prediction_score"])
            for sample_id in common_sample_ids
        ]
        aligned_labels = [_is_positive(truth_lookup[sample_id]) for sample_id in common_sample_ids]
        metric_results: list[dict[str, Any]] = []
        for metric_name in ("auroc", "auprc"):
            result = _bootstrap_metric_delta(
                aligned_labels,
                candidate_scores,
                reference_scores,
                metric_name=metric_name,
                seed=seed + len(metric_results),
                iterations=iterations,
            )
            result.update(
                {
                    "candidate_method": candidate_method,
                    "reference_method": reference_method,
                    "sample_count": len(common_sample_ids),
                }
            )
            metric_results.append(result)
            flat_row = {
                "candidate_method": candidate_method,
                "reference_method": reference_method,
                **result,
            }
            flat_rows.append(flat_row)
            claim_rows.append(
                {
                    "comparison_id": f"{candidate_method}__minus__{reference_method}__{metric_name}".replace(" ", "_"),
                    "candidate_method": candidate_method,
                    "reference_method": reference_method,
                    "metric": metric_name,
                    "observed_delta": result.get("observed_delta"),
                    "ci_95_lower": result.get("ci_95_lower"),
                    "ci_95_upper": result.get("ci_95_upper"),
                    "statistically_significant": result.get("statistically_significant"),
                    "claim_text": result.get("claim_text"),
                }
            )
        comparisons.append(
            {
                "candidate_method": candidate_method,
                "reference_method": reference_method,
                "status": "ok",
                "sample_count": len(common_sample_ids),
                "metrics": metric_results,
            }
        )
    return {
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "comparisons": comparisons,
    }, flat_rows, claim_rows


def _read_baseline_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _coerce_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if value in {"", None}:
            payload[key] = None
            continue
        if key in {"status", "method_name", "scope", "dataset", "unavailable_reason"}:
            payload[key] = value
            continue
        if key in {"full_logits_required", "rerun_required"}:
            payload[key] = str(value).lower() == "true"
            continue
        numeric = _coerce_float(value)
        payload[key] = numeric if numeric is not None else value
    return payload


def _leave_one_dataset_out_report(summary: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    per_dataset: dict[str, dict[str, Any]] = {}
    for raw_row in metrics_rows:
        row = _coerce_metric_row(raw_row)
        method_name = str(row.get("method_name", "unknown"))
        scope = str(row.get("scope", "unknown"))
        dataset_name = str(row.get("dataset", "unknown"))
        if scope == "aggregate":
            aggregate[method_name] = row
        elif scope == "dataset_holdout":
            per_dataset.setdefault(dataset_name, {})[method_name] = row
    return {
        "evaluation": summary.get("evaluation", {}),
        "aggregate": aggregate,
        "per_dataset": per_dataset,
        "learned_fusion_comparison": summary.get("learned_fusion_comparison", {}),
    }


def _score_rows_for_signal(rows: list[dict[str, Any]], signal_name: str) -> list[float] | None:
    scores: list[float] = []
    for row in rows:
        features = row.get("features") or {}
        value = _coerce_float(features.get(signal_name))
        if value is None:
            return None
        scores.append(value)
    return scores


def _deterministic_split_bucket(sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % DETERMINISTIC_SPLIT_MODULUS


def _evaluate_within_dataset_checks(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    datasets: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    training = config.get("learned_fusion", {})
    requested_without = tuple(training.get("feature_sets", {}).get("without_corpus", []))
    requested_with = tuple(training.get("feature_sets", {}).get("with_corpus", []))
    forbidden = tuple(training.get("forbidden_features", []))

    per_dataset: dict[str, Any] = {}
    for dataset_name in datasets:
        dataset_rows = [row for row in rows if str(row.get("dataset", "unknown")) == dataset_name]
        train_rows = [
            row
            for row in dataset_rows
            if _deterministic_split_bucket(str(row.get("sample_id", "unknown_sample"))) != DETERMINISTIC_SPLIT_TEST_BUCKET
        ]
        test_rows = [
            row
            for row in dataset_rows
            if _deterministic_split_bucket(str(row.get("sample_id", "unknown_sample"))) == DETERMINISTIC_SPLIT_TEST_BUCKET
        ]
        dataset_report: dict[str, Any] = {
            "row_count": len(dataset_rows),
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "train_positive_count": sum(_is_positive(row) for row in train_rows),
            "test_positive_count": sum(_is_positive(row) for row in test_rows),
            "grouped_split": {
                "status": "unavailable",
                "reason": "Existing feature rows expose dataset and sample identifiers, but no higher-level group id for a group-preserving within-dataset split.",
            },
            "deterministic_split": {},
        }
        if not train_rows or not test_rows:
            dataset_report["deterministic_split"]["status"] = "unavailable"
            dataset_report["deterministic_split"]["reason"] = "empty_train_or_test_partition"
            per_dataset[dataset_name] = dataset_report
            continue
        if len({ _is_positive(row) for row in train_rows }) < 2 or len({ _is_positive(row) for row in test_rows }) < 2:
            dataset_report["deterministic_split"]["status"] = "unavailable"
            dataset_report["deterministic_split"]["reason"] = "single_class_partition"
            per_dataset[dataset_name] = dataset_report
            continue

        se_train_scores = _score_rows_for_signal(train_rows, "semantic_entropy")
        se_test_scores = _score_rows_for_signal(test_rows, "semantic_entropy")
        if se_train_scores is not None and se_test_scores is not None:
            threshold, _ = _select_threshold([_is_positive(row) for row in train_rows], se_train_scores)
            dataset_report["deterministic_split"]["SE-only"] = {
                "status": "ok",
                "threshold": threshold,
                "metrics": _evaluate_rows(test_rows, se_test_scores, threshold),
            }

        without_features = _resolve_feature_set(
            train_rows,
            list(requested_without),
            forbidden=set(forbidden),
            require_energy=False,
        )
        _, without_test_scores, without_threshold, _ = _fit_learned_baseline(
            train_rows,
            test_rows,
            feature_names=without_features,
            config=config,
        )
        dataset_report["deterministic_split"]["learned fusion without corpus"] = {
            "status": "ok",
            "feature_names": list(without_features),
            "threshold": without_threshold,
            "metrics": _evaluate_rows(test_rows, without_test_scores, without_threshold),
        }

        with_features = _resolve_feature_set(
            train_rows,
            list(requested_with),
            forbidden=set(forbidden),
            require_energy=False,
        )
        _, with_test_scores, with_threshold, _ = _fit_learned_baseline(
            train_rows,
            test_rows,
            feature_names=with_features,
            config=config,
        )
        dataset_report["deterministic_split"]["learned fusion with corpus"] = {
            "status": "ok",
            "feature_names": list(with_features),
            "threshold": with_threshold,
            "metrics": _evaluate_rows(test_rows, with_test_scores, with_threshold),
        }
        per_dataset[dataset_name] = dataset_report

    return {
        "scheme": {
            "type": "deterministic_hash_split",
            "sample_id_hash": "sha256",
            "test_bucket": DETERMINISTIC_SPLIT_TEST_BUCKET,
            "bucket_modulus": DETERMINISTIC_SPLIT_MODULUS,
        },
        "per_dataset": per_dataset,
    }


def _subset_auc(labels: list[int], scores: list[float]) -> dict[str, Any]:
    return {
        "row_count": len(labels),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "auroc": _compute_auroc(labels, scores),
        "auprc": _compute_auprc(labels, scores),
        "reason": None if len(set(labels)) == 2 else ("empty_subset" if not labels else "single_class_subset"),
    }


def _threshold_sensitivity(
    rows: list[dict[str, Any]],
    predictions_by_method: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    truth_lookup = _truth_by_sample(rows)
    method_sample_scores: dict[str, dict[str, float]] = {}
    for method_name in SELECTIVE_METHODS:
        method_rows = predictions_by_method.get(method_name, {})
        method_sample_scores[method_name] = {
            sample_id: float(payload["prediction_score"])
            for sample_id, payload in method_rows.items()
            if sample_id in truth_lookup and "prediction_score" in payload
        }

    evaluations: list[dict[str, Any]] = []
    for low_cutoff in LOW_CUTOFFS:
        for high_cutoff in HIGH_CUTOFFS:
            if low_cutoff >= high_cutoff:
                continue
            label_counts = {
                NORMAL_LABEL: 0,
                HIGH_DIVERSITY_LABEL: 0,
                LOW_DIVERSITY_LABEL: 0,
                AMBIGUOUS_INCORRECT_LABEL: 0,
            }
            relabeled = {
                sample_id: _label_from_cutoffs(row, low_cutoff, high_cutoff)
                for sample_id, row in truth_lookup.items()
            }
            for label_name in relabeled.values():
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
            method_results: dict[str, Any] = {}
            for method_name, sample_scores in method_sample_scores.items():
                high_labels: list[int] = []
                high_scores: list[float] = []
                low_labels: list[int] = []
                low_scores: list[float] = []
                for sample_id, score in sample_scores.items():
                    label_name = relabeled[sample_id]
                    if label_name in {NORMAL_LABEL, HIGH_DIVERSITY_LABEL}:
                        high_labels.append(1 if label_name == HIGH_DIVERSITY_LABEL else 0)
                        high_scores.append(score)
                    if label_name in {NORMAL_LABEL, LOW_DIVERSITY_LABEL}:
                        low_labels.append(1 if label_name == LOW_DIVERSITY_LABEL else 0)
                        low_scores.append(score)
                method_results[method_name] = {
                    "high_diversity_vs_normal": _subset_auc(high_labels, high_scores),
                    "low_diversity_vs_normal": _subset_auc(low_labels, low_scores),
                }
            evaluations.append(
                {
                    "low_cutoff": low_cutoff,
                    "high_cutoff": high_cutoff,
                    "label_counts": label_counts,
                    "methods": method_results,
                }
            )
    return {
        "reference_cutoffs": {"low": 0.1, "high": 0.5},
        "evaluations": evaluations,
    }


def _rank_normalized_confidence(scores: list[float]) -> list[float]:
    if not scores:
        return []
    ordered = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        score = scores[ordered[index]]
        while end < len(ordered) and scores[ordered[end]] == score:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for tied_index in range(index, end):
            ranks[ordered[tied_index]] = average_rank
        index = end
    if len(scores) == 1:
        return [1.0]
    probabilities = [(rank - 1.0) / (len(scores) - 1.0) for rank in ranks]
    return [abs(probability - 0.5) * 2.0 for probability in probabilities]


def _trapezoid_area(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    area = 0.0
    for index in range(1, len(xs)):
        width = xs[index] - xs[index - 1]
        area += width * (ys[index] + ys[index - 1]) / 2.0
    return area


def _selective_risk_report(
    summary: dict[str, Any],
    predictions_by_method: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    baseline_lookup = _baseline_lookup(summary)
    coverage_levels = [round(step / 10.0, 1) for step in range(1, 11)]
    report: dict[str, Any] = {
        "label": "Phillips-inspired evaluation only. This is a selective-prediction framing check, not a probe-paper reproduction.",
        "coverage_levels": coverage_levels,
        "methods": {},
    }
    for method_name in SELECTIVE_METHODS:
        baseline = baseline_lookup.get(method_name, {})
        if baseline.get("status") != "ok":
            report["methods"][method_name] = {
                "status": "unavailable",
                "reason": baseline.get("unavailable_reason") or "missing_baseline",
            }
            continue
        rows_by_sample = predictions_by_method.get(method_name, {})
        ordered_rows = sorted(rows_by_sample.values(), key=lambda row: str(row.get("sample_id", "unknown_sample")))
        scores = [float(row.get("prediction_score", 0.0)) for row in ordered_rows]
        predictions = [bool(row.get("prediction_label", False)) for row in ordered_rows]
        truth = [str(row.get("label", "")) != NORMAL_LABEL for row in ordered_rows]
        correctness = [prediction == actual for prediction, actual in zip(predictions, truth, strict=True)]
        confidence = _rank_normalized_confidence(scores)
        ranked_indices = sorted(range(len(scores)), key=lambda index: (-confidence[index], index))
        coverage_curve: list[dict[str, Any]] = []
        risks: list[float] = []
        coverages: list[float] = []
        for coverage in coverage_levels:
            retain_count = max(1, int(math.ceil(len(scores) * coverage)))
            retained = ranked_indices[:retain_count]
            retained_correct = sum(1 for index in retained if correctness[index])
            accuracy = retained_correct / retain_count
            risk = 1.0 - accuracy
            min_confidence = min(confidence[index] for index in retained)
            coverage_curve.append(
                {
                    "coverage": coverage,
                    "retained_count": retain_count,
                    "risk": risk,
                    "selective_accuracy": accuracy,
                    "minimum_retained_confidence": min_confidence,
                }
            )
            coverages.append(coverage)
            risks.append(risk)
        overall_error_rate = 1.0 - (sum(1 for item in correctness if item) / len(correctness))
        low_confidence_count = max(1, int(math.ceil(len(scores) * 0.2)))
        low_confidence_indices = ranked_indices[-low_confidence_count:]
        low_confidence_error_rate = sum(1 for index in low_confidence_indices if not correctness[index]) / low_confidence_count
        report["methods"][method_name] = {
            "status": "ok",
            "coverage_curve": coverage_curve,
            "aurc": _trapezoid_area(coverages, risks),
            "overall_error_rate": overall_error_rate,
            "tce_like": {
                "name": "low_confidence_error_concentration",
                "description": "Phillips-inspired only. Error concentration in the lowest-confidence 20 percent, not a literal paper TCE reproduction.",
                "low_confidence_fraction": 0.2,
                "low_confidence_error_rate": low_confidence_error_rate,
                "error_concentration_ratio": None
                if overall_error_rate == 0.0
                else low_confidence_error_rate / overall_error_rate,
            },
        }
    return report


def _caveats(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    label_counts = ((summary.get("features_storage") or {}).get("report") or {}).get("label_counts") or {}
    caveats = [
        "Phillips is used only for selective-prediction framing. No probe-style correctness or hidden-representation feature is implemented here.",
        "True Energy robustness remains unavailable because current rows do not carry row-level full logits. semantic_energy_proxy is not treated as thesis-valid Energy evidence.",
    ]
    if label_counts.get(AMBIGUOUS_INCORRECT_LABEL, 0) == 0:
        caveats.append(
            "The current feature table contains no AMBIGUOUS_INCORRECT rows, so gray-zone threshold sensitivity reports absence explicitly instead of fabricating that slice."
        )
    if any((row.get("features") or {}).get("energy_available") is False for row in rows):
        caveats.append(
            "All current rows mark true Boltzmann Energy as unavailable, so Energy-dependent robustness baselines stay null or rerun-required in this report."
        )
    return caveats


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    bootstrap = summary.get("bootstrap", {})
    threshold_sensitivity = summary.get("threshold_sensitivity", {})
    selective = summary.get("selective_risk", {})
    lines = [
        "# Robustness report",
        "",
        "Observed robustness summary for the current fusion outputs. This report uses neutral wording whenever a bootstrap interval crosses zero.",
        "",
        "## Bootstrap deltas",
        "",
        "| candidate | reference | metric | observed_delta | ci_95_lower | ci_95_upper | crosses_zero | statistically_significant |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for comparison in bootstrap.get("comparisons", []):
        for metric in comparison.get("metrics", []):
            lines.append(
                "| {candidate} | {reference} | {metric_name} | {delta} | {lower} | {upper} | {crosses_zero} | {significant} |".format(
                    candidate=comparison.get("candidate_method"),
                    reference=comparison.get("reference_method"),
                    metric_name=metric.get("metric"),
                    delta=_format_metric(metric.get("observed_delta")),
                    lower=_format_metric(metric.get("ci_95_lower")),
                    upper=_format_metric(metric.get("ci_95_upper")),
                    crosses_zero=metric.get("ci_crosses_zero"),
                    significant=metric.get("statistically_significant"),
                )
            )
    lines.extend(["", "## Threshold sensitivity", ""])
    lines.append("Threshold sensitivity uses alternative low/high Semantic Entropy cutoffs around 0.1 and 0.5, then recomputes type-specific subsets.")
    lines.append("")
    lines.append("| low_cutoff | high_cutoff | HIGH_DIVERSITY | LOW_DIVERSITY | AMBIGUOUS_INCORRECT | NORMAL |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in threshold_sensitivity.get("evaluations", []):
        counts = item.get("label_counts", {})
        lines.append(
            "| {low} | {high} | {high_count} | {low_count} | {ambiguous_count} | {normal_count} |".format(
                low=item.get("low_cutoff"),
                high=item.get("high_cutoff"),
                high_count=counts.get(HIGH_DIVERSITY_LABEL),
                low_count=counts.get(LOW_DIVERSITY_LABEL),
                ambiguous_count=counts.get(AMBIGUOUS_INCORRECT_LABEL),
                normal_count=counts.get(NORMAL_LABEL),
            )
        )
    lines.extend(["", "## Selective-risk framing", ""])
    lines.append(str(selective.get("label", "Phillips-inspired selective framing only.")))
    lines.append("")
    lines.append("| method | AURC | overall_error_rate | low_confidence_error_rate | error_concentration_ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for method_name, payload in sorted((selective.get("methods") or {}).items()):
        tce_like = payload.get("tce_like", {}) if isinstance(payload, dict) else {}
        lines.append(
            "| {method} | {aurc} | {overall_error} | {low_confidence_error} | {ratio} |".format(
                method=method_name,
                aurc=_format_metric(payload.get("aurc") if isinstance(payload, dict) else None),
                overall_error=_format_metric(payload.get("overall_error_rate") if isinstance(payload, dict) else None),
                low_confidence_error=_format_metric(tce_like.get("low_confidence_error_rate")),
                ratio=_format_metric(tce_like.get("error_concentration_ratio")),
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


def _flatten_bootstrap_rows(bootstrap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in bootstrap_rows:
        flattened.append(
            {
                "candidate_method": row.get("candidate_method"),
                "reference_method": row.get("reference_method"),
                "metric": row.get("metric"),
                "observed_delta": row.get("observed_delta"),
                "ci_95_lower": row.get("ci_95_lower"),
                "ci_95_upper": row.get("ci_95_upper"),
                "bootstrap_mean_delta": row.get("bootstrap_mean_delta"),
                "valid_iterations": row.get("valid_iterations"),
                "bootstrap_seed": row.get("bootstrap_seed"),
                "bootstrap_iterations": row.get("bootstrap_iterations"),
                "ci_crosses_zero": row.get("ci_crosses_zero"),
                "statistically_significant": row.get("statistically_significant"),
            }
        )
    return flattened


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

    fusion_summary_path = fusion_dir / "summary.json"
    baseline_metrics_path = fusion_dir / "baseline_metrics.csv"
    predictions_path = fusion_dir / "predictions.jsonl"
    fusion_summary = load_json(fusion_summary_path)
    fusion_config = load_json(Path(str(fusion_summary.get("config_path", "experiments/configs/fusion.yaml"))))
    baseline_lookup = _baseline_lookup(fusion_summary)
    metrics_rows = _read_baseline_metrics_csv(baseline_metrics_path)
    predictions = _read_jsonl(predictions_path)
    predictions_by_method = _predictions_by_method(predictions)
    datasets = _ordered_datasets(fusion_config, rows)

    bootstrap, bootstrap_rows, report_claims = _build_bootstrap_report(
        rows,
        predictions_by_method,
        baseline_lookup,
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )
    leave_one_dataset_out = _leave_one_dataset_out_report(fusion_summary, metrics_rows)
    threshold_sensitivity = _threshold_sensitivity(rows, predictions_by_method)
    within_dataset_checks = _evaluate_within_dataset_checks(rows, fusion_config, datasets)
    selective_risk = _selective_risk_report(fusion_summary, predictions_by_method)

    run_id = f"robustness-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    summary = {
        "run_id": run_id,
        "generated_at": _iso_now(),
        "method_name": str(fusion_summary.get("method_name", "Corpus-Grounded Selective Fusion Detector")),
        "features_path": str(features_path),
        "features_storage": storage_report,
        "fusion_dir": str(fusion_dir),
        "fusion_summary_path": str(fusion_summary_path),
        "baseline_metrics_path": str(baseline_metrics_path),
        "predictions_path": str(predictions_path),
        "row_count": len(rows),
        "datasets": datasets,
        "formula_manifest_ref": FORMULA_MANIFEST_REF,
        "dataset_manifest_ref": DATASET_MANIFEST_REF,
        "bootstrap": bootstrap,
        "leave_one_dataset_out": leave_one_dataset_out,
        "threshold_sensitivity": threshold_sensitivity,
        "within_dataset_checks": within_dataset_checks,
        "selective_risk": selective_risk,
        "report_claims": report_claims,
        "caveats": _caveats(rows, fusion_summary),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    bootstrap_csv_path = out_dir / "bootstrap_ci.csv"
    bootstrap_json_path = out_dir / "bootstrap_ci.json"
    lodo_path = out_dir / "leave_one_dataset_out.json"
    threshold_path = out_dir / "threshold_sensitivity.json"
    within_dataset_path = out_dir / "within_dataset_checks.json"
    selective_path = out_dir / "selective_risk.json"
    report_path = out_dir / "report.md"

    write_json(summary_path, summary)
    write_json(bootstrap_json_path, bootstrap)
    _write_csv(
        bootstrap_csv_path,
        _flatten_bootstrap_rows(bootstrap_rows),
        [
            "candidate_method",
            "reference_method",
            "metric",
            "observed_delta",
            "ci_95_lower",
            "ci_95_upper",
            "bootstrap_mean_delta",
            "valid_iterations",
            "bootstrap_seed",
            "bootstrap_iterations",
            "ci_crosses_zero",
            "statistically_significant",
        ],
    )
    write_json(lodo_path, leave_one_dataset_out)
    write_json(threshold_path, threshold_sensitivity)
    write_json(within_dataset_path, within_dataset_checks)
    write_json(selective_path, selective_risk)
    _write_markdown_report(report_path, summary)

    return {
        "run_id": run_id,
        "row_count": len(rows),
        "datasets": datasets,
        "artifacts": {
            "summary": str(summary_path),
            "bootstrap_ci_json": str(bootstrap_json_path),
            "bootstrap_ci_csv": str(bootstrap_csv_path),
            "leave_one_dataset_out": str(lodo_path),
            "threshold_sensitivity": str(threshold_path),
            "within_dataset_checks": str(within_dataset_path),
            "selective_risk": str(selective_path),
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
        items = []
        for key, entry in value.items():
            if "claim" in str(key).lower() or key in {"summary_statement", "headline", "conclusion", "wording"}:
                items.extend(_collect_claim_strings(entry))
        return items
    return []


def validate_report_claims(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    problems: list[str] = []
    claim_entries = summary.get("report_claims", [])
    if not isinstance(claim_entries, list):
        problems.append("report_claims must be a list")
        claim_entries = []

    for index, entry in enumerate(claim_entries):
        if not isinstance(entry, dict):
            problems.append(f"report_claims[{index}] must be an object")
            continue
        claim_text = " ".join(_collect_claim_strings(entry)).lower()
        ci_lower = _coerce_float(entry.get("ci_95_lower"))
        ci_upper = _coerce_float(entry.get("ci_95_upper"))
        statistically_significant = bool(entry.get("statistically_significant", False))
        if ci_lower is None or ci_upper is None:
            continue
        crosses_zero = ci_lower <= 0.0 <= ci_upper
        if crosses_zero and statistically_significant:
            problems.append(
                f"report_claims[{index}] marks statistically_significant=true even though the interval crosses zero"
            )
        if crosses_zero:
            for phrase in FORBIDDEN_OVERCLAIM_PHRASES:
                if phrase in claim_text:
                    problems.append(
                        f"report_claims[{index}] uses forbidden overclaim phrase '{phrase}' while the interval crosses zero"
                    )
        observed_delta = _coerce_float(entry.get("observed_delta"))
        if observed_delta is not None and observed_delta <= 0.0:
            for phrase in ("improvement", "better", "outperforms"):
                if phrase in claim_text:
                    problems.append(
                        f"report_claims[{index}] claims positive direction with non-positive observed delta using phrase '{phrase}'"
                    )

    if problems:
        raise ValueError("Report-claim validation failed:\n- " + "\n- ".join(problems))
    return {
        "summary_path": str(summary_path),
        "checked_claims": len(claim_entries),
        "status": "ok",
        "forbidden_phrases": list(FORBIDDEN_OVERCLAIM_PHRASES),
    }
