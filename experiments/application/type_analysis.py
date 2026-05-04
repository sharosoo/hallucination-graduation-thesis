"""Type-specific signal analysis for hallucination detection features."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.adapters.corpus_features import load_json, read_feature_rows, write_json
from experiments.domain import TypeLabel


NORMAL_LABEL = TypeLabel.NORMAL.value
HIGH_DIVERSITY_LABEL = TypeLabel.HIGH_DIVERSITY.value
LOW_DIVERSITY_LABEL = TypeLabel.LOW_DIVERSITY.value
AMBIGUOUS_INCORRECT_LABEL = TypeLabel.AMBIGUOUS_INCORRECT.value
ZERO_SE_BIN_ID = "se_eq_0"
FULL_LOGITS_REQUIRED_REASON = "full_logits_required"
SINGLE_CLASS_SUBSET_REASON = "single_class_subset"
EMPTY_SUBSET_REASON = "empty_subset"
MISSING_SIGNAL_REASON = "missing_signal"


@dataclass(frozen=True)
class AnalysisSlice:
    name: str
    description: str


@dataclass(frozen=True)
class SignalSpec:
    name: str
    description: str
    feature_key: str


SLICES = (
    AnalysisSlice(
        name="overall",
        description="All rows. Positive class is any non-NORMAL hallucination label.",
    ),
    AnalysisSlice(
        name="high_diversity_vs_normal",
        description="Rows labeled HIGH_DIVERSITY or NORMAL only. Positive class is HIGH_DIVERSITY.",
    ),
    AnalysisSlice(
        name="low_diversity_vs_normal",
        description="Rows labeled LOW_DIVERSITY or NORMAL only. Positive class is LOW_DIVERSITY.",
    ),
    AnalysisSlice(
        name="zero_se_vs_normal",
        description="Exact-zero Semantic Entropy bucket only. Positive class is any non-NORMAL row inside the zero-SE bucket.",
    ),
)

SIGNALS = (
    SignalSpec(
        name="semantic_entropy",
        description="Semantic Entropy score from the feature table.",
        feature_key="semantic_entropy",
    ),
    SignalSpec(
        name="corpus_risk_only",
        description="Corpus-only risk baseline from cached or proxy corpus features.",
        feature_key="corpus_risk_only",
    ),
    SignalSpec(
        name="semantic_energy_boltzmann",
        description="True Boltzmann Semantic Energy. Remains unavailable until full logits are regenerated.",
        feature_key="semantic_energy_boltzmann",
    ),
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _storage_report_for(path: Path) -> dict[str, Any] | None:
    if path.suffix != ".parquet":
        return None
    sidecar = path.with_suffix(path.suffix + ".storage.json")
    if sidecar.exists():
        return load_json(sidecar)
    return None


def _features(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("features")
    return payload if isinstance(payload, dict) else {}


def _label(row: dict[str, Any]) -> str:
    return str(row.get("label", ""))


def _se_bin_id(row: dict[str, Any]) -> str | None:
    payload = _features(row).get("se_bin")
    if isinstance(payload, dict):
        value = payload.get("bin_id")
        return None if value is None else str(value)
    return None


def _is_zero_se_row(row: dict[str, Any]) -> bool:
    bin_id = _se_bin_id(row)
    if bin_id == ZERO_SE_BIN_ID:
        return True
    value = _features(row).get("semantic_entropy")
    return value == 0 or value == 0.0


def _slice_rows(rows: list[dict[str, Any]], slice_name: str) -> list[dict[str, Any]]:
    if slice_name == "overall":
        return list(rows)
    if slice_name == "high_diversity_vs_normal":
        return [row for row in rows if _label(row) in {NORMAL_LABEL, HIGH_DIVERSITY_LABEL}]
    if slice_name == "low_diversity_vs_normal":
        return [row for row in rows if _label(row) in {NORMAL_LABEL, LOW_DIVERSITY_LABEL}]
    if slice_name == "zero_se_vs_normal":
        return [row for row in rows if _is_zero_se_row(row)]
    raise ValueError(f"Unknown analysis slice: {slice_name}")


def _is_positive_label(label_value: str, slice_name: str) -> bool:
    if slice_name == "overall":
        return label_value != NORMAL_LABEL
    if slice_name == "high_diversity_vs_normal":
        return label_value == HIGH_DIVERSITY_LABEL
    if slice_name == "low_diversity_vs_normal":
        return label_value == LOW_DIVERSITY_LABEL
    if slice_name == "zero_se_vs_normal":
        return label_value != NORMAL_LABEL
    raise ValueError(f"Unknown analysis slice: {slice_name}")


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
    positive_rank_sum = sum(rank for (label, (_, rank)) in zip(labels, ranked, strict=True) if label == 1)
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


def _energy_unavailable_reason(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        features = _features(row)
        if features.get("energy_available") is False or features.get("semantic_energy_boltzmann") is None:
            return {
                "reason": FULL_LOGITS_REQUIRED_REASON,
                "full_logits_required": bool(features.get("full_logits_required", True)),
                "rerun_required": bool(features.get("rerun_required", True)),
            }
    return None


def _extract_signal_values(rows: list[dict[str, Any]], signal_name: str) -> tuple[list[float] | None, dict[str, Any] | None]:
    if signal_name == "semantic_energy_boltzmann":
        unavailable = _energy_unavailable_reason(rows)
        if unavailable is not None:
            return None, unavailable

    values: list[float] = []
    for row in rows:
        value = _coerce_float(_features(row).get(signal_name))
        if value is None:
            return None, {"reason": MISSING_SIGNAL_REASON}
        values.append(value)
    return values, None


def _metric_payload(
    labels: list[int],
    scores: list[float] | None,
    *,
    unavailable: dict[str, Any] | None,
) -> dict[str, Any]:
    positives = sum(labels)
    negatives = len(labels) - positives
    base = {
        "auroc": None,
        "auprc": None,
        "auroc_reason": None,
        "auprc_reason": None,
        "full_logits_required": False,
        "rerun_required": False,
    }
    if not labels:
        base["auroc_reason"] = EMPTY_SUBSET_REASON
        base["auprc_reason"] = EMPTY_SUBSET_REASON
        return base
    if unavailable is not None:
        base["auroc_reason"] = str(unavailable.get("reason"))
        base["auprc_reason"] = str(unavailable.get("reason"))
        base["full_logits_required"] = bool(unavailable.get("full_logits_required", False))
        base["rerun_required"] = bool(unavailable.get("rerun_required", False))
        return base
    if positives == 0 or negatives == 0:
        base["auroc_reason"] = SINGLE_CLASS_SUBSET_REASON
        base["auprc_reason"] = SINGLE_CLASS_SUBSET_REASON
        return base
    if scores is None:
        base["auroc_reason"] = MISSING_SIGNAL_REASON
        base["auprc_reason"] = MISSING_SIGNAL_REASON
        return base
    base["auroc"] = _compute_auroc(labels, scores)
    base["auprc"] = _compute_auprc(labels, scores)
    return base


def _dataset_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("dataset", "unknown")) for row in rows})


def _rows_for_dataset(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("dataset", "unknown")) == dataset_name]


def _build_record(dataset_name: str, slice_name: str, signal_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [1 if _is_positive_label(_label(row), slice_name) else 0 for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    scores, unavailable = _extract_signal_values(rows, signal_name)
    metrics = _metric_payload(labels, scores, unavailable=unavailable)
    return {
        "dataset": dataset_name,
        "slice": slice_name,
        "signal": signal_name,
        "row_count": len(rows),
        "positive_count": positives,
        "negative_count": negatives,
        **metrics,
    }


def _analysis_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset_name in _dataset_names(rows) + ["AGGREGATE"]:
        dataset_rows = rows if dataset_name == "AGGREGATE" else _rows_for_dataset(rows, dataset_name)
        for slice_spec in SLICES:
            slice_rows = _slice_rows(dataset_rows, slice_spec.name)
            for signal_spec in SIGNALS:
                records.append(_build_record(dataset_name, slice_spec.name, signal_spec.name, slice_rows))
    return records


def _self_check() -> dict[str, Any]:
    labels = [1, 1, 1]
    scores = [0.4, 0.4, 0.2]
    payload = _metric_payload(labels, scores, unavailable=None)
    return {
        "single_class_subset_case": {
            "labels": labels,
            "scores": scores,
            "auroc": payload["auroc"],
            "auprc": payload["auprc"],
            "auroc_reason": payload["auroc_reason"],
            "auprc_reason": payload["auprc_reason"],
            "passes": payload["auroc_reason"] == SINGLE_CLASS_SUBSET_REASON
            and payload["auprc_reason"] == SINGLE_CLASS_SUBSET_REASON,
        }
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "slice",
        "signal",
        "row_count",
        "positive_count",
        "negative_count",
        "auroc",
        "auprc",
        "auroc_reason",
        "auprc_reason",
        "full_logits_required",
        "rerun_required",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def _format_metric(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = []
    for record in records:
        dataset_name = str(record["dataset"])
        if dataset_name not in datasets:
            datasets.append(dataset_name)

    lines = [
        "# Type-specific signal analysis",
        "",
        "This report keeps unavailable Energy visible as unavailable. `semantic_energy_boltzmann` is never replaced with proxy selected-logit energy.",
        "",
    ]
    for dataset_name in datasets:
        lines.append(f"## {dataset_name}")
        lines.append("")
        dataset_records = [record for record in records if record["dataset"] == dataset_name]
        for slice_spec in SLICES:
            lines.append(f"### {slice_spec.name}")
            lines.append("")
            lines.append(slice_spec.description)
            lines.append("")
            lines.append("| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |")
            lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |")
            for record in dataset_records:
                if record["slice"] != slice_spec.name:
                    continue
                signal_name = str(record["signal"])
                lines.append(
                    "| {signal} | {rows} | {positives} | {negatives} | {auroc} | {auroc_reason} | {auprc} | {auprc_reason} |".format(
                        signal=signal_name,
                        rows=record["row_count"],
                        positives=record["positive_count"],
                        negatives=record["negative_count"],
                        auroc=_format_metric(record["auroc"]),
                        auroc_reason=record["auroc_reason"] or "",
                        auprc=_format_metric(record["auprc"]),
                        auprc_reason=record["auprc_reason"] or "",
                    )
                )
            lines.append("")
        lines.append("")

    lines.append("## Signal notes")
    lines.append("")
    for signal_spec in SIGNALS:
        lines.append(f"- `{signal_spec.name}`: {signal_spec.description}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_type_analysis(features_path: Path, out_dir: Path) -> dict[str, Any]:
    rows, storage_report = read_feature_rows(features_path)
    records = _analysis_records(rows)
    self_check = _self_check()
    run_id = f"type-analysis-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "metrics.csv"
    markdown_path = out_dir / "report.md"

    summary = {
        "run_id": run_id,
        "generated_at": _iso_now(),
        "features_artifact": str(features_path),
        "features_storage": storage_report or _storage_report_for(features_path),
        "row_count": len(rows),
        "datasets": _dataset_names(rows),
        "aggregate_dataset_name": "AGGREGATE",
        "slice_definitions": {slice_spec.name: slice_spec.description for slice_spec in SLICES},
        "signal_definitions": {signal_spec.name: signal_spec.description for signal_spec in SIGNALS},
        "metrics": records,
        "self_check": self_check,
        "artifacts": {
            "summary_json": str(summary_path),
            "metrics_csv": str(csv_path),
            "report_md": str(markdown_path),
        },
        "notes": [
            "Overall uses positive = label != NORMAL.",
            "Type-specific slices compare the relevant incorrect label against NORMAL only.",
            "Zero-SE uses the exact-zero Semantic Entropy bucket (`se_eq_0`).",
            "True Boltzmann Semantic Energy remains null with explicit full-logits rerun requirements until upstream full logits exist.",
        ],
    }

    write_json(summary_path, summary)
    _write_csv(csv_path, records)
    _write_markdown(markdown_path, records)
    return summary
