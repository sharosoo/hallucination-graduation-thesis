"""Evaluation port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from experiments.domain import ExperimentManifest, FeatureVector, MetricResult


class EvaluatorPort(ABC):
    """Produces aggregate evaluation outputs from scored rows."""

    @abstractmethod
    def evaluate(
        self,
        manifest: ExperimentManifest,
        feature_vectors: Sequence[FeatureVector],
        scores: Sequence[float],
    ) -> Sequence[MetricResult]:
        """Return aggregate metric outputs."""
