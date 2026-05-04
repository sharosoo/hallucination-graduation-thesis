"""Primary domain records for dataset rows and derived signals."""

from __future__ import annotations

from dataclasses import dataclass, field

from .labels import EnergyComputationKind


KeyValueMetadata = tuple[tuple[str, str], ...]
EntityPair = tuple[str, str]
TokenIdSequence = tuple[int, ...]
NumericVector = tuple[float, ...]
NumericMatrix = tuple[NumericVector, ...]


@dataclass(frozen=True)
class PromptRow:
    """A self-contained prompt row ready for local generation/export."""

    dataset: str
    split_id: str
    sample_id: str
    prompt: str
    question: str | None = None
    context: str | None = None
    metadata: KeyValueMetadata = ()


@dataclass(frozen=True)
class QuestionExample:
    """A dataset example before model-specific feature extraction."""

    dataset: str
    split_id: str
    sample_id: str
    question: str
    context: str | None = None
    gold_answers: tuple[str, ...] = ()
    candidate_answers: tuple[str, ...] = ()
    metadata: KeyValueMetadata = ()


@dataclass(frozen=True)
class ModelResponse:
    """Generated or loaded model response payload."""

    sample_id: str
    response_text: str
    probability: float = 1.0
    log_probability: float | None = None
    token_logits: tuple[float, ...] = ()
    decoded_tokens: tuple[str, ...] = ()
    generated_token_ids: TokenIdSequence = ()
    selected_token_ids: TokenIdSequence = ()
    selected_token_logits: NumericVector = ()
    full_logits: NumericMatrix = ()
    logsumexp_values: NumericVector = ()
    full_vocabulary_logits: bool = False
    metadata: KeyValueMetadata = ()


@dataclass(frozen=True)
class CorrectnessJudgment:
    """Row-level correctness signal used for label assignment only."""

    sample_id: str
    is_correct: bool
    judge_name: str
    rationale: str | None = None
    evidence: KeyValueMetadata = ()


@dataclass(frozen=True)
class SemanticEntropyResult:
    """Semantic Entropy outcome for a response set."""

    sample_id: str
    semantic_entropy: float
    cluster_count: int
    cluster_probabilities: tuple[float, ...] = ()
    cluster_representatives: tuple[str, ...] = ()
    provenance_note: str = "paper-derived semantic entropy result"


@dataclass(frozen=True)
class EnergyResult:
    """Semantic Energy outcome with room for multiple computation modes."""

    sample_id: str
    energy_value: float
    energy_kind: EnergyComputationKind
    temperature: float | None = None
    selected_token_logit: float | None = None
    logit_variance: float | None = None
    confidence_margin: float | None = None
    provenance_note: str = "paper-derived energy or explicit proxy"


@dataclass(frozen=True)
class CorpusStats:
    """Corpus-grounded statistics for a sample response."""

    sample_id: str
    entity_terms: tuple[str, ...] = ()
    entity_frequencies: tuple[int, ...] = ()
    entity_pair_terms: tuple[EntityPair, ...] = ()
    entity_pair_cooccurrence: tuple[int, ...] = ()
    low_frequency_entity_flags: tuple[bool, ...] = ()
    coverage_score: float = 0.0
    corpus_source: str = ""
    status: str = "unknown"
    corpus_provenance: str = ""
    snapshot_reference: str = ""
    metadata: KeyValueMetadata = field(default_factory=tuple)
