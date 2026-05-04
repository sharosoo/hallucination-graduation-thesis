"""Abstract ports for the experiments hexagonal architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .artifact_store import ArtifactStorePort
from .corpus_stats import CorpusStatsPort
from .dataset_loader import DatasetLoaderPort
from .evaluator import EvaluatorPort
from .feature_extractor import FeatureExtractorPort
from .fusion_strategy import FusionStrategyPort


class ModelLogitsPort(ABC):
    """Abstract interface for row-level prompt generation with preserved logits."""

    @abstractmethod
    def build_artifact(self, *, out_path: str, prompt_rows_path: str | None = None) -> dict[str, Any]:
        """Generate or load prompt rows and materialize a self-contained logits artifact."""

__all__ = [
    "ArtifactStorePort",
    "CorpusStatsPort",
    "DatasetLoaderPort",
    "EvaluatorPort",
    "FeatureExtractorPort",
    "FusionStrategyPort",
    "ModelLogitsPort",
]
