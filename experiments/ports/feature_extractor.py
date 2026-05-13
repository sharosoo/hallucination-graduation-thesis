"""Feature extraction port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from experiments.domain import (
    AnalysisBin,
    CorrectnessJudgment,
    CorpusStats,
    EnergyResult,
    FeatureVector,
    ModelResponse,
    QuestionExample,
    SemanticEntropyResult,
)


class FeatureExtractorPort(ABC):
    """Builds trainable row-level features from typed domain inputs."""

    @abstractmethod
    def build_feature_vector(
        self,
        *,
        run_id: str,
        example: QuestionExample,
        response: ModelResponse,
        correctness: CorrectnessJudgment,
        semantic_entropy: SemanticEntropyResult,
        energy: EnergyResult,
        corpus_stats: CorpusStats,
        label: bool,
        analysis_bin: AnalysisBin | None = None,
    ) -> FeatureVector:
        """Build a typed feature vector for one sample."""
