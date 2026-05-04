"""Aggregate metric records for evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass

from .labels import TypeLabel


@dataclass(frozen=True)
class MetricResult:
    """Named aggregate metric with dataset and label scope."""

    metric_name: str
    metric_value: float
    dataset: str
    split_id: str
    label_scope: TypeLabel | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    sample_count: int | None = None
    note: str | None = None
