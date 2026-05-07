"""Concrete placeholders that make architecture wiring importable without real backends."""

from __future__ import annotations

from collections.abc import Sequence

from experiments.domain import (
    AnalysisBin,
    CorrectnessJudgment,
    CorpusStats,
    EnergyResult,
    ExperimentManifest,
    FeatureProvenance,
    FeatureRole,
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


class PlaceholderDatasetLoader(DatasetLoaderPort):
    """Stub dataset loader for future task implementations."""

    def load_examples(self, dataset_name: str, split_id: str) -> Sequence[QuestionExample]:
        raise NotImplementedError(
            f"Dataset loading is not implemented yet for dataset={dataset_name!r}, split={split_id!r}."
        )


class PlaceholderCorpusStatsProvider(CorpusStatsPort):
    """Stub corpus-stat adapter for future task implementations."""

    def get_corpus_stats(self, example: QuestionExample, response: ModelResponse) -> CorpusStats:
        raise NotImplementedError(
            f"Corpus statistics are not implemented yet for sample={example.sample_id!r}."
        )


class PlaceholderFeatureExtractor(FeatureExtractorPort):
    """Minimal typed feature builder used only as an architecture placeholder."""

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
        label: TypeLabel,
        analysis_bin: AnalysisBin | None = None,
    ) -> FeatureVector:
        return FeatureVector(
            run_id=run_id,
            dataset=example.dataset,
            split_id=example.split_id,
            sample_id=example.sample_id,
            label=label,
            semantic_entropy=semantic_entropy.semantic_entropy,
            cluster_count=semantic_entropy.cluster_count,
            semantic_energy=energy.energy_value,
            energy_kind=energy.energy_kind,
            logit_variance=energy.logit_variance,
            confidence_margin=energy.confidence_margin,
            entity_frequency=(
                sum(corpus_stats.entity_frequencies) / len(corpus_stats.entity_frequencies)
                if corpus_stats.entity_frequencies
                else None
            ),
            entity_frequency_mean=(
                sum(corpus_stats.entity_frequencies) / len(corpus_stats.entity_frequencies)
                if corpus_stats.entity_frequencies
                else None
            ),
            entity_frequency_min=min(corpus_stats.entity_frequencies) if corpus_stats.entity_frequencies else None,
            entity_pair_cooccurrence=(
                sum(corpus_stats.entity_pair_cooccurrence) / len(corpus_stats.entity_pair_cooccurrence)
                if corpus_stats.entity_pair_cooccurrence
                else None
            ),
            low_frequency_entity_flag=any(corpus_stats.low_frequency_entity_flags)
            if corpus_stats.low_frequency_entity_flags
            else None,
            zero_cooccurrence_flag=all(value == 0 for value in corpus_stats.entity_pair_cooccurrence)
            if corpus_stats.entity_pair_cooccurrence
            else None,
            coverage_score=corpus_stats.coverage_score,
            corpus_source=corpus_stats.corpus_source,
            corpus_risk_only=None,
            corpus_status=corpus_stats.status,
            se_bin=analysis_bin,
            provenance=(
                FeatureProvenance(
                    feature_name="label",
                    role=FeatureRole.LABEL_ONLY,
                    source=correctness.judge_name,
                    source_artifact_path=None,
                    depends_on_correctness=True,
                    trainable=False,
                    note="Operational label is required on every row but is not a trainable feature.",
                ),
                FeatureProvenance(
                    feature_name="semantic_entropy",
                    role=FeatureRole.TRAINABLE,
                    source="semantic_entropy",
                    source_artifact_path=None,
                    depends_on_correctness=False,
                    trainable=True,
                ),
                FeatureProvenance(
                    feature_name="se_bin",
                    role=FeatureRole.ANALYSIS_ONLY,
                    source="analysis_configuration",
                    source_artifact_path=None,
                    depends_on_correctness=False,
                    trainable=False,
                    note="SE bins are analysis metadata only and do not replace TypeLabel.",
                ),
            ),
        )


class PlaceholderFusionStrategy(FusionStrategyPort):
    """Stub fusion strategy for future tasks."""

    def score(self, feature_vector: FeatureVector) -> float:
        raise NotImplementedError(
            f"Fusion scoring is not implemented yet for sample={feature_vector.sample_id!r}."
        )

    def score_batch(self, feature_vectors: Sequence[FeatureVector]) -> Sequence[float]:
        raise NotImplementedError(
            f"Fusion batch scoring is not implemented yet for {len(feature_vectors)} samples."
        )


class PlaceholderEvaluator(EvaluatorPort):
    """Stub evaluator for future tasks."""

    def evaluate(
        self,
        manifest: ExperimentManifest,
        feature_vectors: Sequence[FeatureVector],
        scores: Sequence[float],
    ) -> Sequence[MetricResult]:
        raise NotImplementedError(
            f"Evaluation is not implemented yet for run={manifest.run_id!r} with {len(feature_vectors)} rows."
        )


class InMemoryArtifactStore(ArtifactStorePort):
    """Minimal in-memory artifact store for contract wiring tests."""

    def __init__(self) -> None:
        self.manifests: list[ExperimentManifest] = []
        self.feature_vectors: list[FeatureVector] = []
        self.metrics: list[MetricResult] = []

    def write_manifest(self, manifest: ExperimentManifest) -> str:
        self.manifests.append(manifest)
        return f"memory://manifest/{manifest.run_id}"

    def write_feature_vectors(self, feature_vectors: Sequence[FeatureVector]) -> str:
        self.feature_vectors.extend(feature_vectors)
        return f"memory://features/{len(self.feature_vectors)}"

    def write_metrics(self, metrics: Sequence[MetricResult]) -> str:
        self.metrics.extend(metrics)
        return f"memory://metrics/{len(self.metrics)}"
