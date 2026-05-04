"""Core enums for experiment labels and feature provenance."""

from __future__ import annotations

from enum import Enum


class TypeLabel(Enum):
    """Operational row-level sample label."""

    NORMAL = "NORMAL"
    HIGH_DIVERSITY = "HIGH_DIVERSITY"
    LOW_DIVERSITY = "LOW_DIVERSITY"
    AMBIGUOUS_INCORRECT = "AMBIGUOUS_INCORRECT"


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
    PROXY_SELECTED_LOGIT = "proxy_selected_logit"
