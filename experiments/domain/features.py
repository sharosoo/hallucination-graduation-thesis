"""Typed feature records for fusion-ready experiment rows."""

from __future__ import annotations

from dataclasses import dataclass

from .labels import EnergyComputationKind, FeatureRole


@dataclass(frozen=True)
class FeatureProvenance:
    """Documents whether a feature is trainable, label-only, or analysis-only."""

    feature_name: str
    role: FeatureRole
    source: str
    source_artifact_path: str | None = None
    depends_on_correctness: bool = False
    trainable: bool = False
    note: str | None = None


@dataclass(frozen=True)
class AnalysisBin:
    """Analysis-only bin metadata that never replaces the operational label."""

    scheme_name: str
    bin_id: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    includes_upper_bound: bool = True
    note: str = "Analysis-only bin. Supervised target is annotation-backed is_hallucination."


@dataclass(frozen=True)
class FeatureVector:
    """Row-level feature vector with explicit label and SE-bin analysis metadata."""

    run_id: str
    dataset: str
    split_id: str
    sample_id: str
    is_hallucination: bool
    semantic_entropy: float
    cluster_count: int
    semantic_energy: float | None = None
    energy_kind: EnergyComputationKind | None = None
    logit_variance: float | None = None
    confidence_margin: float | None = None
    entity_frequency: float | None = None
    entity_frequency_mean: float | None = None
    entity_frequency_min: float | None = None
    entity_pair_cooccurrence: float | None = None
    low_frequency_entity_flag: bool | None = None
    zero_cooccurrence_flag: bool | None = None
    coverage_score: float | None = None
    corpus_source: str | None = None
    corpus_risk_only: float | None = None
    corpus_status: str | None = None
    se_bin: AnalysisBin | None = None
    provenance: tuple[FeatureProvenance, ...] = ()
