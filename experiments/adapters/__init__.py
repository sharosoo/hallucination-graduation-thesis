"""Concrete placeholder adapters for the experiments package."""

from .corpus_features import CachedOrProxyCorpusAdapter, read_feature_rows, write_feature_artifact
from .energy_features import EnergyFeatureAdapter
from .placeholders import (
    InMemoryArtifactStore,
    PlaceholderCorpusStatsProvider,
    PlaceholderDatasetLoader,
    PlaceholderEvaluator,
    PlaceholderFeatureExtractor,
    PlaceholderFusionStrategy,
)

__all__ = [
    "CachedOrProxyCorpusAdapter",
    "EnergyFeatureAdapter",
    "InMemoryArtifactStore",
    "PlaceholderCorpusStatsProvider",
    "PlaceholderDatasetLoader",
    "PlaceholderEvaluator",
    "PlaceholderFeatureExtractor",
    "PlaceholderFusionStrategy",
    "read_feature_rows",
    "write_feature_artifact",
]
