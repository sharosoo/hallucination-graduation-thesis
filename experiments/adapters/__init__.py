"""Concrete placeholder adapters for the experiments package."""

from .corpus_counts import build_corpus_count_backend
from .corpus_features import CorpusFeatureAdapter, read_feature_rows, write_feature_artifact
from .correctness_dataset import build_annotation_correctness_dataset, candidate_label_row_to_json  # type: ignore[reportMissingImports]
from .energy_features import EnergyFeatureUnavailableError, build_energy_rows_from_candidate_scores
from .hf_datasets import HuggingFaceDatasetLoader, candidate_row_to_json, prompt_group_to_json, prompt_row_to_json
from .placeholders import (
    InMemoryArtifactStore,
    PlaceholderCorpusStatsProvider,
    PlaceholderDatasetLoader,
    PlaceholderEvaluator,
    PlaceholderFeatureExtractor,
    PlaceholderFusionStrategy,
)

__all__ = [
    "CorpusFeatureAdapter",
    "build_corpus_count_backend",
    "build_annotation_correctness_dataset",
    "candidate_label_row_to_json",
    "EnergyFeatureUnavailableError",
    "HuggingFaceDatasetLoader",
    "InMemoryArtifactStore",
    "PlaceholderCorpusStatsProvider",
    "PlaceholderDatasetLoader",
    "PlaceholderEvaluator",
    "PlaceholderFeatureExtractor",
    "PlaceholderFusionStrategy",
    "read_feature_rows",
    "candidate_row_to_json",
    "prompt_group_to_json",
    "prompt_row_to_json",
    "build_energy_rows_from_candidate_scores",
    "write_feature_artifact",
]
