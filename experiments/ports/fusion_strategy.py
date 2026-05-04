"""Fusion strategy port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from experiments.domain import FeatureVector


class FusionStrategyPort(ABC):
    """Scores hallucination risk from typed feature vectors."""

    @abstractmethod
    def score(self, feature_vector: FeatureVector) -> float:
        """Score a single feature vector."""

    @abstractmethod
    def score_batch(self, feature_vectors: Sequence[FeatureVector]) -> Sequence[float]:
        """Score a batch of feature vectors."""
