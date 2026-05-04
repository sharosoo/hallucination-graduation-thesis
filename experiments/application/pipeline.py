"""Use-case orchestration placeholders for experiment execution."""

from __future__ import annotations

from collections.abc import Sequence

from experiments.domain import (
    AnalysisBin,
    CorrectnessJudgment,
    EnergyResult,
    ExperimentManifest,
    FeatureVector,
    MetricResult,
    ModelResponse,
    QuestionExample,
    SemanticEntropyResult,
    TypeLabel,
)
from experiments.ports import (
    ArtifactStorePort,
    CorpusStatsPort,
    DatasetLoaderPort,
    EvaluatorPort,
    FeatureExtractorPort,
    FusionStrategyPort,
)


class ExperimentPipelineService:
    """Coordinates ports while keeping CLIs thin and adapters replaceable."""

    def __init__(
        self,
        *,
        dataset_loader: DatasetLoaderPort,
        corpus_stats_provider: CorpusStatsPort,
        feature_extractor: FeatureExtractorPort,
        fusion_strategy: FusionStrategyPort,
        evaluator: EvaluatorPort,
        artifact_store: ArtifactStorePort,
    ) -> None:
        self._dataset_loader = dataset_loader
        self._corpus_stats_provider = corpus_stats_provider
        self._feature_extractor = feature_extractor
        self._fusion_strategy = fusion_strategy
        self._evaluator = evaluator
        self._artifact_store = artifact_store

    def load_examples(self, dataset_name: str, split_id: str) -> Sequence[QuestionExample]:
        """Load typed examples through the configured dataset port."""

        return self._dataset_loader.load_examples(dataset_name, split_id)

    def build_feature_vector(
        self,
        *,
        run_id: str,
        example: QuestionExample,
        response: ModelResponse,
        correctness: CorrectnessJudgment,
        semantic_entropy: SemanticEntropyResult,
        energy: EnergyResult,
        label: TypeLabel,
        analysis_bin: AnalysisBin | None = None,
    ) -> FeatureVector:
        """Build one feature vector from already-produced typed signals."""

        corpus_stats = self._corpus_stats_provider.get_corpus_stats(example, response)
        return self._feature_extractor.build_feature_vector(
            run_id=run_id,
            example=example,
            response=response,
            correctness=correctness,
            semantic_entropy=semantic_entropy,
            energy=energy,
            corpus_stats=corpus_stats,
            label=label,
            analysis_bin=analysis_bin,
        )

    def score_and_evaluate(
        self,
        manifest: ExperimentManifest,
        feature_vectors: Sequence[FeatureVector],
    ) -> tuple[Sequence[float], Sequence[MetricResult]]:
        """Score rows, evaluate aggregates, and persist both artifact families."""

        scores = self._fusion_strategy.score_batch(feature_vectors)
        metrics = self._evaluator.evaluate(manifest, feature_vectors, scores)
        self._artifact_store.write_feature_vectors(feature_vectors)
        self._artifact_store.write_metrics(metrics)
        return scores, metrics

    def persist_manifest(self, manifest: ExperimentManifest) -> str:
        """Persist a run manifest through the artifact store port."""

        return self._artifact_store.write_manifest(manifest)
