"""Aggregate metric records for evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricResult:
    """Named aggregate metric with dataset and corpus-axis bin scope."""

    metric_name: str
    metric_value: float
    dataset: str
    split_id: str
    corpus_axis_bin: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    sample_count: int | None = None
    note: str | None = None
