"""Artifact persistence port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from experiments.domain import ExperimentManifest, FeatureVector, MetricResult


class ArtifactStorePort(ABC):
    """Persists manifests, feature rows, and metrics without fixing storage backends."""

    @abstractmethod
    def write_manifest(self, manifest: ExperimentManifest) -> str:
        """Persist and reference a manifest."""

    @abstractmethod
    def write_feature_vectors(self, feature_vectors: Sequence[FeatureVector]) -> str:
        """Persist row-level feature vectors."""

    @abstractmethod
    def write_metrics(self, metrics: Sequence[MetricResult]) -> str:
        """Persist aggregate metrics."""
