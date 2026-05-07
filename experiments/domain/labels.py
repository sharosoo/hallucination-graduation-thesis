"""Core enums for experiment labels and feature provenance."""

from __future__ import annotations

from enum import Enum


class FeatureRole(Enum):
    """How a field is allowed to participate in training or analysis."""

    TRAINABLE = "trainable"
    LABEL_ONLY = "label_only"
    CORRECTNESS_DERIVED = "correctness_derived"
    ANALYSIS_ONLY = "analysis_only"
    EXTERNAL_CORPUS = "external_corpus"


class EnergyComputationKind(Enum):
    """Semantic Energy calculation mode."""

    TRUE_BOLTZMANN = "true_boltzmann"
    SAMPLED_RESPONSE_CLUSTER = "sampled_response_cluster"
    CANDIDATE_BOLTZMANN_DIAGNOSTIC = "candidate_boltzmann_diagnostic"
    PROXY_SELECTED_LOGIT = "proxy_selected_logit"
