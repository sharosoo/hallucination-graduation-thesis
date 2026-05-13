"""Primary domain records for dataset rows and derived signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .labels import EnergyComputationKind


KeyValueMetadata = tuple[tuple[str, str], ...]
EntityPair = tuple[str, str]
TokenIdSequence = tuple[int, ...]
NumericVector = tuple[float, ...]
NumericMatrix = tuple[NumericVector, ...]


def _metadata_tuple(metadata: Mapping[str, object] | None) -> KeyValueMetadata:
    if metadata is None:
        return ()
    return tuple((str(key), str(value)) for key, value in metadata.items() if value is not None)


def _metadata_dict(metadata: KeyValueMetadata) -> dict[str, str]:
    return {key: value for key, value in metadata}


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_text(value: object, *, field_name: str) -> str:
    text = _clean_text(value)
    if text is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


@dataclass(frozen=True)
class PromptGroup:
    """Prompt-level paired-candidate record used at the dataset boundary."""

    dataset: str
    split_id: str
    source_row_id: str
    prompt_id: str
    pair_id: str
    question: str
    prompt: str
    context: str | None = None
    prompt_hash: str = ""
    label_source: str = ""
    metadata: KeyValueMetadata = ()


@dataclass(frozen=True)
class CandidateRow:
    """Typed candidate row bound to a prompt group and annotation-backed role."""

    prompt_id: str
    candidate_id: str
    pair_id: str
    source_row_id: str
    dataset: str
    split_id: str
    candidate_text: str
    candidate_role: str
    is_correct: bool
    label_source: str
    question: str
    prompt: str
    context: str | None = None
    metadata: KeyValueMetadata = ()

    @classmethod
    def from_prompt_group(
        cls,
        prompt_group: PromptGroup,
        *,
        candidate_text: str,
        candidate_role: str,
        is_correct: bool,
        label_source: str,
        metadata: Mapping[str, object] | None = None,
    ) -> CandidateRow:
        normalized_role = _require_text(candidate_role, field_name="candidate_role")
        normalized_candidate_text = _require_text(candidate_text, field_name="candidate_text")
        return cls(
            prompt_id=prompt_group.prompt_id,
            candidate_id=f"{prompt_group.pair_id}:{normalized_role}",
            pair_id=prompt_group.pair_id,
            source_row_id=prompt_group.source_row_id,
            dataset=prompt_group.dataset,
            split_id=prompt_group.split_id,
            candidate_text=normalized_candidate_text,
            candidate_role=normalized_role,
            is_correct=is_correct,
            label_source=_require_text(label_source, field_name="label_source"),
            question=prompt_group.question,
            prompt=prompt_group.prompt,
            context=prompt_group.context,
            metadata=_metadata_tuple(metadata),
        )

    def to_row(self) -> dict[str, str | bool | None | dict[str, str]]:
        return {
            "prompt_id": self.prompt_id,
            "candidate_id": self.candidate_id,
            "pair_id": self.pair_id,
            "source_row_id": self.source_row_id,
            "dataset": self.dataset,
            "split_id": self.split_id,
            "question": self.question,
            "prompt": self.prompt,
            "context": self.context,
            "candidate_text": self.candidate_text,
            "candidate_role": self.candidate_role,
            "is_correct": self.is_correct,
            "label_source": self.label_source,
            "metadata": _metadata_dict(self.metadata),
        }


@dataclass(frozen=True)
class CandidateLabelRow:
    """Annotation-backed candidate label artifact row."""

    prompt_id: str
    candidate_id: str
    pair_id: str
    candidate_role: str
    candidate_text: str
    is_correct: bool
    label_source: str
    source_row_id: str
    dataset: str
    split_id: str


@dataclass(frozen=True)
class TeacherForcedTokenScore:
    """Per-token teacher-forced scoring details for one candidate answer."""

    prompt_id: str
    candidate_id: str
    candidate_token_position: int
    token_id: int
    selected_token_logit: float
    logsumexp: float
    full_logits: NumericVector = ()
    decoded_token: str | None = None


@dataclass(frozen=True)
class TeacherForcedCandidateScore:
    """Aggregate teacher-forced score record for one fixed candidate."""

    prompt_id: str
    candidate_id: str
    candidate_token_count: int
    candidate_token_start: int
    candidate_token_end: int
    selected_token_logit_sum: float
    selected_token_logit_mean: float
    sequence_log_probability: float | None = None
    average_log_probability: float | None = None
    token_scores: tuple[TeacherForcedTokenScore, ...] = ()


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
