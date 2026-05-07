"""Repo-owned prompt-level NLI likelihood Semantic Entropy features from free-sample artifacts."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

from experiments.adapters.corpus_features import load_json, write_feature_artifact
from experiments.domain import FeatureRole
from experiments.scripts.stage_control import (
    GENERATION_FREE_SAMPLE_SCHEMA_VERSION,
    SEMANTIC_ENTROPY_SCHEMA_VERSION,
    add_schema_version,
)

FORMULA_MANIFEST_REF = "experiments/literature/formula_notes.md"
FREE_SAMPLE_ARTIFACT_TYPE = "free_sample_rows"
FREE_SAMPLE_SOURCE_KIND = "free_sample"
EXPECTED_FREE_SAMPLE_COUNT = 10
DEFAULT_NLI_MODEL_NAME = "microsoft/deberta-large-mnli"
DIAGNOSTIC_FIXTURE_NLI_MODEL_REF = "diagnostic_fixture_exact_match_v1"
SEMANTIC_CLUSTERER_NAME = "strict_bidirectional_nli_equivalence_v1"
TOKEN_LOGPROB_TOLERANCE = 1e-6


class SemanticEntropyInputError(ValueError):
    """Raised when Semantic Entropy input is invalid."""


class SemanticEntropyDependencyError(RuntimeError):
    """Raised when required NLI dependencies are unavailable for a real run."""


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(raw: dict[str, object], field_name: str, sample_position: int) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise SemanticEntropyInputError(f"sample {sample_position} must include non-empty {field_name!r}")
    return value


def _required_float_vector(raw: dict[str, object], field_name: str, sample_position: int) -> tuple[float, ...]:
    value = raw.get(field_name)
    if not isinstance(value, list) or not value:
        raise SemanticEntropyInputError(f"sample {sample_position} must include non-empty list {field_name!r}")
    result: list[float] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, int | float) or isinstance(item, bool) or not math.isfinite(float(item)):
            raise SemanticEntropyInputError(
                f"sample {sample_position} field {field_name!r} item {item_index} must be a finite number"
            )
        result.append(float(item))
    return tuple(result)


def _required_int_vector(raw: dict[str, object], field_name: str, sample_position: int) -> tuple[int, ...]:
    value = raw.get(field_name)
    if not isinstance(value, list) or not value:
        raise SemanticEntropyInputError(f"sample {sample_position} must include non-empty list {field_name!r}")
    result: list[int] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise SemanticEntropyInputError(
                f"sample {sample_position} field {field_name!r} item {item_index} must be an integer"
            )
        result.append(item)
    return tuple(result)


def _required_text_vector(raw: dict[str, object], field_name: str, sample_position: int) -> tuple[str, ...]:
    value = raw.get(field_name)
    if not isinstance(value, list) or not value:
        raise SemanticEntropyInputError(f"sample {sample_position} must include non-empty list {field_name!r}")
    result: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise SemanticEntropyInputError(
                f"sample {sample_position} field {field_name!r} item {item_index} must be a string"
            )
        result.append(item)
    return tuple(result)


def _logsumexp(values: tuple[float, ...] | list[float]) -> float:
    if not values:
        raise SemanticEntropyInputError("logsumexp requires at least one value")
    max_value = max(values)
    return float(max_value + math.log(sum(math.exp(item - max_value) for item in values)))


def normalize_semantic_response(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


@dataclass(frozen=True)
class FreeSample:
    dataset: str
    split_id: str
    prompt_id: str
    pair_id: str | None
    prompt: str | None
    question: str | None
    context: str | None
    sample_index: int
    response_text: str
    selected_token_logits: tuple[float, ...]
    logsumexp: tuple[float, ...]
    generated_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    generated_tokens: tuple[str, ...]
    sequence_log_probability: float | None
    metadata: dict[str, object]

    @classmethod
    def from_raw(cls, raw: object, sample_position: int) -> Self:
        if not isinstance(raw, dict):
            raise SemanticEntropyInputError(f"sample {sample_position} must be an object")
        if raw.get("source_kind") != FREE_SAMPLE_SOURCE_KIND:
            raise SemanticEntropyInputError(
                f"sample {sample_position} must set source_kind={FREE_SAMPLE_SOURCE_KIND!r}"
            )
        prompt_id = _required_text(raw, "prompt_id", sample_position)
        response_text = _required_text(raw, "response_text", sample_position)
        sample_index = raw.get("sample_index")
        if not isinstance(sample_index, int) or isinstance(sample_index, bool):
            raise SemanticEntropyInputError(f"sample {sample_position} must include integer 'sample_index'")
        selected_token_logits = _required_float_vector(raw, "selected_token_logits", sample_position)
        logsumexp_values = _required_float_vector(raw, "logsumexp", sample_position)
        generated_token_ids = _required_int_vector(raw, "generated_token_ids", sample_position)
        selected_token_ids = _required_int_vector(raw, "selected_token_ids", sample_position)
        generated_tokens = _required_text_vector(raw, "generated_tokens", sample_position)
        token_count = len(selected_token_logits)
        for field_name, field_value in (
            ("logsumexp", logsumexp_values),
            ("generated_token_ids", generated_token_ids),
            ("selected_token_ids", selected_token_ids),
            ("generated_tokens", generated_tokens),
        ):
            if len(field_value) != token_count:
                raise SemanticEntropyInputError(
                    f"sample {sample_position} field {field_name!r} must have length {token_count}; got {len(field_value)}"
                )
        raw_sequence_log_probability = raw.get("sequence_log_probability")
        sequence_log_probability = None
        if raw_sequence_log_probability is not None:
            if not isinstance(raw_sequence_log_probability, int | float) or isinstance(raw_sequence_log_probability, bool):
                raise SemanticEntropyInputError(
                    f"sample {sample_position} field 'sequence_log_probability' must be numeric when present"
                )
            sequence_log_probability = float(raw_sequence_log_probability)
        metadata = raw.get("metadata")
        return cls(
            dataset=str(raw.get("dataset", "unknown")),
            split_id=str(raw.get("split_id", "unknown")),
            prompt_id=prompt_id,
            pair_id=_optional_text(raw.get("pair_id")),
            prompt=_optional_text(raw.get("prompt")),
            question=_optional_text(raw.get("question")),
            context=_optional_text(raw.get("context")),
            sample_index=sample_index,
            response_text=response_text,
            selected_token_logits=selected_token_logits,
            logsumexp=logsumexp_values,
            generated_token_ids=generated_token_ids,
            selected_token_ids=selected_token_ids,
            generated_tokens=generated_tokens,
            sequence_log_probability=sequence_log_probability,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    @property
    def token_log_likelihoods(self) -> tuple[float, ...]:
        return tuple(logit - partition for logit, partition in zip(self.selected_token_logits, self.logsumexp, strict=True))

    @property
    def mean_token_log_likelihood(self) -> float:
        token_log_likelihoods = self.token_log_likelihoods
        return float(sum(token_log_likelihoods) / len(token_log_likelihoods))

    @property
    def recomputed_sequence_log_probability(self) -> float:
        return float(sum(self.token_log_likelihoods))

    @property
    def explicit_sequence_log_probability_is_consistent(self) -> bool:
        if self.sequence_log_probability is None:
            return False
        return math.isclose(
            self.sequence_log_probability,
            self.recomputed_sequence_log_probability,
            rel_tol=TOKEN_LOGPROB_TOLERANCE,
            abs_tol=TOKEN_LOGPROB_TOLERANCE,
        )

    @property
    def is_fixture_sample(self) -> bool:
        return bool(self.metadata.get("fixture") is True)

    def sample_ref(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "sample_index": self.sample_index,
            "response_text": self.response_text,
        }


@dataclass(frozen=True)
class FreeSampleArtifact:
    path: Path
    samples: tuple[FreeSample, ...]
    dataset_manifest_ref: str | None
    formula_manifest_ref: str | None
    source_run_id: str | None
    source_model_name: str | None
    source_tokenizer_name: str | None
    source_created_at: str | None
    source_logits_schema_version: str | None
    sample_count_per_prompt: int
    prompt_group_count: int
    fixture_mode: bool

    @classmethod
    def from_path(cls, path: Path) -> Self:
        raw_artifact = load_json(path)
        if not isinstance(raw_artifact, dict):
            raise SemanticEntropyInputError(f"free-sample artifact must be a JSON object: {path}")
        artifact_type = raw_artifact.get("artifact_type")
        if artifact_type != FREE_SAMPLE_ARTIFACT_TYPE:
            raise SemanticEntropyInputError(
                "Semantic Entropy now accepts only free-sample artifacts "
                f"with artifact_type={FREE_SAMPLE_ARTIFACT_TYPE!r}; got {artifact_type!r}"
            )
        schema_version = raw_artifact.get("schema_version")
        if schema_version != GENERATION_FREE_SAMPLE_SCHEMA_VERSION:
            raise SemanticEntropyInputError(
                f"free-sample artifact schema_version must be {GENERATION_FREE_SAMPLE_SCHEMA_VERSION!r}; got {schema_version!r}"
            )
        raw_samples = raw_artifact.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise SemanticEntropyInputError(f"free-sample artifact must contain a non-empty samples array: {path}")
        raw_sample_count_per_prompt = raw_artifact.get("sample_count_per_prompt")
        if raw_sample_count_per_prompt != EXPECTED_FREE_SAMPLE_COUNT:
            raise SemanticEntropyInputError(
                f"sample_count_per_prompt must equal {EXPECTED_FREE_SAMPLE_COUNT}; got {raw_sample_count_per_prompt!r}"
            )
        raw_prompt_group_count = raw_artifact.get("prompt_group_count")
        if not isinstance(raw_prompt_group_count, int) or isinstance(raw_prompt_group_count, bool) or raw_prompt_group_count <= 0:
            raise SemanticEntropyInputError("prompt_group_count must be a positive integer")
        samples = tuple(FreeSample.from_raw(raw_sample, index) for index, raw_sample in enumerate(raw_samples))
        _validate_prompt_sample_counts(samples, raw_prompt_group_count)
        return cls(
            path=path,
            samples=samples,
            dataset_manifest_ref=_optional_text(raw_artifact.get("dataset_manifest_ref")),
            formula_manifest_ref=_optional_text(raw_artifact.get("formula_manifest_ref")),
            source_run_id=_optional_text(raw_artifact.get("run_id")),
            source_model_name=_optional_text(raw_artifact.get("model_name")),
            source_tokenizer_name=_optional_text(raw_artifact.get("tokenizer_name")),
            source_created_at=_optional_text(raw_artifact.get("created_at")),
            source_logits_schema_version=_optional_text(raw_artifact.get("logits_schema_version")),
            sample_count_per_prompt=raw_sample_count_per_prompt,
            prompt_group_count=raw_prompt_group_count,
            fixture_mode=bool(raw_artifact.get("fixture_mode") is True),
        )


def _validate_prompt_sample_counts(samples: tuple[FreeSample, ...], prompt_group_count: int) -> None:
    prompt_ids = {sample.prompt_id for sample in samples}
    if len(prompt_ids) != prompt_group_count:
        raise SemanticEntropyInputError(
            f"prompt_group_count={prompt_group_count} does not match observed distinct prompt coverage count {len(prompt_ids)}"
        )
    expected_total_samples = prompt_group_count * EXPECTED_FREE_SAMPLE_COUNT
    if len(samples) != expected_total_samples:
        raise SemanticEntropyInputError(
            f"free-sample artifact has {len(samples)} rows; expected {expected_total_samples} from prompt_group_count={prompt_group_count} "
            f"and sample_count_per_prompt={EXPECTED_FREE_SAMPLE_COUNT}"
        )
    expected_indexes = tuple(range(EXPECTED_FREE_SAMPLE_COUNT))
    samples_by_prompt: dict[str, list[FreeSample]] = defaultdict(list)
    for sample in samples:
        samples_by_prompt[sample.prompt_id].append(sample)
    for prompt_id, prompt_samples in sorted(samples_by_prompt.items()):
        observed_indexes = sorted(sample.sample_index for sample in prompt_samples)
        if len(prompt_samples) != EXPECTED_FREE_SAMPLE_COUNT:
            raise SemanticEntropyInputError(
                f"prompt_id {prompt_id!r} has {len(prompt_samples)} free samples; expected {EXPECTED_FREE_SAMPLE_COUNT}"
            )
        if tuple(observed_indexes) != expected_indexes:
            raise SemanticEntropyInputError(
                f"prompt_id {prompt_id!r} must cover exact free-sample sample_index set {list(expected_indexes)}; "
                f"coverage={observed_indexes}"
            )


@dataclass(frozen=True)
class SampleLogLikelihoodRecord:
    sample_index: int
    response_text: str
    token_log_likelihoods: tuple[float, ...]
    sequence_log_probability: float
    explicit_sequence_log_probability: float | None
    explicit_sequence_log_probability_consistent: bool
    sample_log_likelihood: float
    likelihood_source: str

    @classmethod
    def from_sample(cls, sample: FreeSample) -> Self:
        explicit_consistent = sample.explicit_sequence_log_probability_is_consistent
        likelihood_source = (
            "mean_token_log_likelihood_from_generation_tokens"
            if sample.sequence_log_probability is None
            else "mean_token_log_likelihood_with_consistent_explicit_sequence_log_probability"
            if explicit_consistent
            else "mean_token_log_likelihood_with_inconsistent_explicit_sequence_log_probability_ignored"
        )
        return cls(
            sample_index=sample.sample_index,
            response_text=sample.response_text,
            token_log_likelihoods=sample.token_log_likelihoods,
            sequence_log_probability=sample.recomputed_sequence_log_probability,
            explicit_sequence_log_probability=sample.sequence_log_probability,
            explicit_sequence_log_probability_consistent=explicit_consistent,
            sample_log_likelihood=sample.mean_token_log_likelihood,
            likelihood_source=likelihood_source,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "response_text": self.response_text,
            "token_log_likelihoods": list(self.token_log_likelihoods),
            "sequence_log_probability": self.sequence_log_probability,
            "explicit_sequence_log_probability": self.explicit_sequence_log_probability,
            "explicit_sequence_log_probability_consistent": self.explicit_sequence_log_probability_consistent,
            "sample_log_likelihood": self.sample_log_likelihood,
            "likelihood_source": self.likelihood_source,
        }


@dataclass(frozen=True)
class EntailmentDecision:
    premise_sample_index: int
    hypothesis_sample_index: int
    premise_text: str
    hypothesis_text: str
    label: str
    entails: bool
    mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticCluster:
    cluster_id: str
    representative_sample_index: int
    representative_response_text: str
    member_sample_indexes: tuple[int, ...]
    member_response_texts: tuple[str, ...]
    count: int
    cluster_log_likelihood: float
    cluster_log_probability: float
    cluster_probability: float
    sample_refs: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "representative_sample_index": self.representative_sample_index,
            "representative_response_text": self.representative_response_text,
            "member_sample_indexes": list(self.member_sample_indexes),
            "member_response_texts": list(self.member_response_texts),
            "count": self.count,
            "cluster_log_likelihood": self.cluster_log_likelihood,
            "cluster_log_probability": self.cluster_log_probability,
            "cluster_probability": self.cluster_probability,
            "sample_refs": list(self.sample_refs),
        }


class _EntailmentModel:
    def model_ref(self) -> str:
        raise NotImplementedError

    def mode(self) -> str:
        raise NotImplementedError

    def entails(self, premise: FreeSample, hypothesis: FreeSample) -> EntailmentDecision:
        raise NotImplementedError

    def batch_entails(
        self, pairs: list[tuple[FreeSample, FreeSample]]
    ) -> list[EntailmentDecision]:
        # Default fallback: serial. Concrete models override to do a single
        # batched forward across all pairs.
        return [self.entails(premise, hypothesis) for premise, hypothesis in pairs]


class _FixtureEntailmentModel(_EntailmentModel):
    def model_ref(self) -> str:
        return DIAGNOSTIC_FIXTURE_NLI_MODEL_REF

    def mode(self) -> str:
        return "diagnostic_fixture_exact_match"

    def entails(self, premise: FreeSample, hypothesis: FreeSample) -> EntailmentDecision:
        normalized_premise = normalize_semantic_response(premise.response_text)
        normalized_hypothesis = normalize_semantic_response(hypothesis.response_text)
        entails = normalized_premise == normalized_hypothesis
        return EntailmentDecision(
            premise_sample_index=premise.sample_index,
            hypothesis_sample_index=hypothesis.sample_index,
            premise_text=premise.response_text,
            hypothesis_text=hypothesis.response_text,
            label="entailment" if entails else "not_entailment",
            entails=entails,
            mode=self.mode(),
        )


class _TransformersEntailmentModel(_EntailmentModel):
    DEFAULT_BATCH_SIZE = 64

    def __init__(self, model_name: str) -> None:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise SemanticEntropyDependencyError(
                "Missing optional dependency 'torch'. Install the repo-managed generation stack with "
                "`uv sync --group generation` before running thesis-valid Semantic Entropy."
            ) from exc
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise SemanticEntropyDependencyError(
                "Missing optional dependency 'transformers'. Install the repo-managed generation stack with "
                "`uv sync --group generation` before running thesis-valid Semantic Entropy."
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - dependency/runtime branch
            raise SemanticEntropyDependencyError(
                f"Unable to load NLI model {model_name!r}. Download the model locally or provide a fixture-mode artifact for offline verification."
            ) from exc
        self._torch = torch
        # Move to GPU when available so batched forwards saturate the device.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_fp16 = self._device.type == "cuda"
        if self._use_fp16:
            self._model = self._model.to(self._device).half()
        else:
            self._model = self._model.to(self._device)
        self._model.eval()
        self._model_name = model_name
        self._entailment_labels = self._resolve_entailment_labels()

    def _resolve_entailment_labels(self) -> set[str]:
        config_labels = getattr(self._model.config, "id2label", {})
        resolved: set[str] = set()
        if isinstance(config_labels, dict):
            for label in config_labels.values():
                normalized = str(label).strip().lower()
                if "entail" in normalized:
                    resolved.add(normalized)
        if not resolved:
            raise SemanticEntropyDependencyError(
                f"NLI model {self._model_name!r} does not expose an entailment label through config.id2label"
            )
        return resolved

    def model_ref(self) -> str:
        return self._model_name

    def mode(self) -> str:
        return "transformers_mnli_argmax"

    def entails(self, premise: FreeSample, hypothesis: FreeSample) -> EntailmentDecision:
        # Delegate to batched path so single-pair callers also get the GPU forward
        # benefit. The batched implementation handles both n=1 and n>1.
        return self.batch_entails([(premise, hypothesis)])[0]

    def batch_entails(
        self, pairs: list[tuple[FreeSample, FreeSample]]
    ) -> list[EntailmentDecision]:
        if not pairs:
            return []
        decisions: list[EntailmentDecision] = []
        id2label = self._model.config.id2label
        for chunk_start in range(0, len(pairs), self.DEFAULT_BATCH_SIZE):
            chunk = pairs[chunk_start : chunk_start + self.DEFAULT_BATCH_SIZE]
            premises = [premise.response_text for premise, _ in chunk]
            hypotheses = [hypothesis.response_text for _, hypothesis in chunk]
            inputs = self._tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.no_grad():
                if self._use_fp16:
                    with self._torch.autocast(device_type=self._device.type, dtype=self._torch.float16):
                        logits = self._model(**inputs).logits
                else:
                    logits = self._model(**inputs).logits
            label_indexes = logits.argmax(dim=-1).tolist()
            for (premise, hypothesis), label_index in zip(chunk, label_indexes, strict=True):
                label = str(id2label[int(label_index)]).strip().lower()
                decisions.append(
                    EntailmentDecision(
                        premise_sample_index=premise.sample_index,
                        hypothesis_sample_index=hypothesis.sample_index,
                        premise_text=premise.response_text,
                        hypothesis_text=hypothesis.response_text,
                        label=label,
                        entails=label in self._entailment_labels,
                        mode=self.mode(),
                    )
                )
        return decisions


def _build_entailment_model(
    free_sample_artifact: FreeSampleArtifact,
    *,
    nli_model_name: str,
) -> _EntailmentModel:
    if free_sample_artifact.fixture_mode or all(sample.is_fixture_sample for sample in free_sample_artifact.samples):
        return _FixtureEntailmentModel()
    return _TransformersEntailmentModel(nli_model_name)


def _count_entropy_from_keys(cluster_keys: tuple[str, ...]) -> float:
    if not cluster_keys:
        return 0.0
    counts = Counter(cluster_keys)
    total = len(cluster_keys)
    if len(counts) <= 1:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


@dataclass(frozen=True)
class SemanticClusterResult:
    entropy: float
    cluster_count: int
    discrete_cluster_entropy: float
    clusters: tuple[SemanticCluster, ...]
    sample_log_likelihoods: tuple[SampleLogLikelihoodRecord, ...]
    pairwise_entailment_decisions: tuple[EntailmentDecision, ...]
    sample_count: int
    nli_model_ref: str
    nli_decision_mode: str
    normalized_string_entropy_diagnostic: float
    normalized_string_cluster_count_diagnostic: int

    @classmethod
    def from_samples(cls, samples: tuple[FreeSample, ...], entailment_model: _EntailmentModel) -> Self:
        ordered_samples = tuple(sorted(samples, key=lambda item: item.sample_index))
        sample_log_likelihoods = tuple(SampleLogLikelihoodRecord.from_sample(sample) for sample in ordered_samples)
        sample_likelihood_map = {
            record.sample_index: record.sample_log_likelihood for record in sample_log_likelihoods
        }
        cluster_samples: list[list[FreeSample]] = []
        decisions_by_key: dict[tuple[int, int], EntailmentDecision] = {}
        decision_order: list[tuple[int, int]] = []

        # Pre-compute every directed pairwise NLI decision in a single batched
        # forward so the greedy clustering loop becomes pure cache lookups. For
        # N=10 free samples this is 90 directed pairs per prompt.
        sample_pairs: list[tuple[FreeSample, FreeSample]] = [
            (premise, hypothesis)
            for premise in ordered_samples
            for hypothesis in ordered_samples
            if premise.sample_index != hypothesis.sample_index
        ]
        if sample_pairs:
            batched_decisions = entailment_model.batch_entails(sample_pairs)
            for (premise, hypothesis), decision in zip(sample_pairs, batched_decisions, strict=True):
                key = (premise.sample_index, hypothesis.sample_index)
                if key not in decisions_by_key:
                    decisions_by_key[key] = decision
                    decision_order.append(key)

        def get_decision(premise: FreeSample, hypothesis: FreeSample) -> EntailmentDecision:
            key = (premise.sample_index, hypothesis.sample_index)
            cached = decisions_by_key.get(key)
            if cached is not None:
                return cached
            decision = entailment_model.entails(premise, hypothesis)
            decisions_by_key[key] = decision
            decision_order.append(key)
            return decision

        for sample in ordered_samples:
            assigned = False
            for cluster in cluster_samples:
                representative = cluster[0]
                forward = get_decision(representative, sample)
                backward = get_decision(sample, representative)
                if forward.entails and backward.entails:
                    cluster.append(sample)
                    assigned = True
                    break
            if not assigned:
                cluster_samples.append([sample])

        cluster_log_likelihood_values: list[float] = []
        for cluster in cluster_samples:
            member_log_likelihoods = tuple(sample_likelihood_map[sample.sample_index] for sample in cluster)
            cluster_log_likelihood_values.append(_logsumexp(member_log_likelihoods))
        total_log_likelihood = _logsumexp(cluster_log_likelihood_values)

        clusters: list[SemanticCluster] = []
        for cluster_index, cluster in enumerate(cluster_samples):
            cluster_log_likelihood = cluster_log_likelihood_values[cluster_index]
            cluster_log_probability = float(cluster_log_likelihood - total_log_likelihood)
            cluster_probability = float(math.exp(cluster_log_probability))
            representative = cluster[0]
            clusters.append(
                SemanticCluster(
                    cluster_id=f"cluster_{cluster_index:03d}",
                    representative_sample_index=representative.sample_index,
                    representative_response_text=representative.response_text,
                    member_sample_indexes=tuple(sample.sample_index for sample in cluster),
                    member_response_texts=tuple(sample.response_text for sample in cluster),
                    count=len(cluster),
                    cluster_log_likelihood=cluster_log_likelihood,
                    cluster_log_probability=cluster_log_probability,
                    cluster_probability=cluster_probability,
                    sample_refs=tuple(sample.sample_ref() for sample in cluster),
                )
            )

        entropy = float(
            -sum(cluster.cluster_probability * cluster.cluster_log_probability for cluster in clusters)
        ) if clusters else 0.0
        normalized_keys = tuple(normalize_semantic_response(sample.response_text) for sample in ordered_samples)
        return cls(
            entropy=entropy,
            cluster_count=len(clusters),
            discrete_cluster_entropy=_count_entropy_from_keys(
                tuple(
                    cluster.cluster_id
                    for cluster in clusters
                    for _sample_index in cluster.member_sample_indexes
                )
            ),
            clusters=tuple(clusters),
            sample_log_likelihoods=sample_log_likelihoods,
            pairwise_entailment_decisions=tuple(decisions_by_key[key] for key in decision_order),
            sample_count=len(ordered_samples),
            nli_model_ref=entailment_model.model_ref(),
            nli_decision_mode=entailment_model.mode(),
            normalized_string_entropy_diagnostic=_count_entropy_from_keys(normalized_keys),
            normalized_string_cluster_count_diagnostic=len(set(normalized_keys)),
        )

    def to_features(self) -> dict[str, object]:
        cluster_payloads = [cluster.to_dict() for cluster in self.clusters]
        return {
            "semantic_entropy_nli_likelihood": self.entropy,
            "semantic_entropy_cluster_count": self.cluster_count,
            "semantic_entropy_discrete_cluster_entropy": self.discrete_cluster_entropy,
            "semantic_entropy": self.entropy,
            "cluster_count": self.cluster_count,
            "semantic_clusters": cluster_payloads,
            "cluster_log_likelihoods": [
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_log_likelihood": cluster.cluster_log_likelihood,
                    "cluster_log_probability": cluster.cluster_log_probability,
                    "cluster_probability": cluster.cluster_probability,
                    "member_sample_indexes": list(cluster.member_sample_indexes),
                }
                for cluster in self.clusters
            ],
            "sample_log_likelihoods": [record.to_dict() for record in self.sample_log_likelihoods],
            "pairwise_entailment_decisions": [decision.to_dict() for decision in self.pairwise_entailment_decisions],
            "nli_model_ref": self.nli_model_ref,
            "nli_decision_mode": self.nli_decision_mode,
            "nli_cache_refs": [],
            "semantic_clusterer": SEMANTIC_CLUSTERER_NAME,
            "free_sample_count": self.sample_count,
            "normalized_string_entropy_diagnostic": self.normalized_string_entropy_diagnostic,
            "normalized_string_cluster_count_diagnostic": self.normalized_string_cluster_count_diagnostic,
        }


@dataclass(frozen=True)
class FeatureProvenanceRecord:
    feature_name: str
    role: FeatureRole
    source: str
    source_artifact_path: str
    depends_on_correctness: bool
    trainable: bool
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["role"] = self.role.value
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class SemanticEntropyRow:
    run_id: str
    prompt_id: str
    dataset: str
    split_id: str
    pair_id: str | None
    question: str | None
    context: str | None
    prompt: str | None
    cluster_result: SemanticClusterResult
    source_free_sample_path: str
    dataset_manifest_ref: str | None
    formula_manifest_ref: str | None
    source_free_sample_run_id: str | None
    source_model_name: str | None
    source_tokenizer_name: str | None
    source_logits_schema_version: str | None
    source_created_at: str | None

    @classmethod
    def from_prompt_samples(
        cls,
        *,
        run_id: str,
        prompt_id: str,
        samples: tuple[FreeSample, ...],
        source_free_sample_path: str,
        dataset_manifest_ref: str | None,
        formula_manifest_ref: str | None,
        source_free_sample_run_id: str | None,
        source_model_name: str | None,
        source_tokenizer_name: str | None,
        source_logits_schema_version: str | None,
        source_created_at: str | None,
        entailment_model: _EntailmentModel,
    ) -> Self:
        ordered_samples = tuple(sorted(samples, key=lambda item: item.sample_index))
        first_sample = ordered_samples[0]
        return cls(
            run_id=run_id,
            prompt_id=prompt_id,
            dataset=first_sample.dataset,
            split_id=first_sample.split_id,
            pair_id=first_sample.pair_id,
            question=first_sample.question,
            context=first_sample.context,
            prompt=first_sample.prompt,
            cluster_result=SemanticClusterResult.from_samples(ordered_samples, entailment_model),
            source_free_sample_path=source_free_sample_path,
            dataset_manifest_ref=dataset_manifest_ref,
            formula_manifest_ref=formula_manifest_ref or FORMULA_MANIFEST_REF,
            source_free_sample_run_id=source_free_sample_run_id,
            source_model_name=source_model_name,
            source_tokenizer_name=source_tokenizer_name,
            source_logits_schema_version=source_logits_schema_version,
            source_created_at=source_created_at,
        )

    def to_row(self) -> dict[str, object]:
        features = self.cluster_result.to_features()
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "split_id": self.split_id,
            "prompt_id": self.prompt_id,
            "pair_id": self.pair_id,
            "question": self.question,
            "context": self.context,
            "prompt": self.prompt,
            "semantic_entropy_nli_likelihood": self.cluster_result.entropy,
            "semantic_entropy_cluster_count": self.cluster_result.cluster_count,
            "semantic_entropy_discrete_cluster_entropy": self.cluster_result.discrete_cluster_entropy,
            "semantic_entropy": self.cluster_result.entropy,
            "cluster_count": self.cluster_result.cluster_count,
            "semantic_clusters": features["semantic_clusters"],
            "cluster_log_likelihoods": features["cluster_log_likelihoods"],
            "sample_log_likelihoods": features["sample_log_likelihoods"],
            "pairwise_entailment_decisions": features["pairwise_entailment_decisions"],
            "nli_model_ref": self.cluster_result.nli_model_ref,
            "nli_decision_mode": self.cluster_result.nli_decision_mode,
            "nli_cache_refs": [],
            "normalized_string_entropy_diagnostic": self.cluster_result.normalized_string_entropy_diagnostic,
            "normalized_string_cluster_count_diagnostic": self.cluster_result.normalized_string_cluster_count_diagnostic,
            "features": features,
            "feature_provenance": [entry.to_dict() for entry in self.provenance()],
            "formula_manifest_ref": self.formula_manifest_ref,
            "dataset_manifest_ref": self.dataset_manifest_ref,
            "source_free_sample_path": self.source_free_sample_path,
            "source_free_sample_run_id": self.source_free_sample_run_id,
            "source_model_name": self.source_model_name,
            "source_tokenizer_name": self.source_tokenizer_name,
            "source_logits_schema_version": self.source_logits_schema_version,
            "source_created_at": self.source_created_at,
        }

    def provenance(self) -> tuple[FeatureProvenanceRecord, ...]:
        source_note = (
            "Prompt-level N=10 answer-only free samples clustered by strict bidirectional NLI entailment. "
            "Cluster masses use log-sum-exp over mean token log-likelihoods derived from selected_token_logits - logsumexp."
        )
        return (
            FeatureProvenanceRecord(
                feature_name="semantic_entropy_nli_likelihood",
                role=FeatureRole.TRAINABLE,
                source=source_note,
                source_artifact_path=self.source_free_sample_path,
                depends_on_correctness=False,
                trainable=True,
                note="Paper-faithful Semantic Entropy estimator over normalized semantic cluster likelihood masses.",
            ),
            FeatureProvenanceRecord(
                feature_name="semantic_entropy_cluster_count",
                role=FeatureRole.TRAINABLE,
                source=source_note,
                source_artifact_path=self.source_free_sample_path,
                depends_on_correctness=False,
                trainable=True,
            ),
            FeatureProvenanceRecord(
                feature_name="semantic_entropy_discrete_cluster_entropy",
                role=FeatureRole.TRAINABLE,
                source="Count-based entropy over the same strict bidirectional NLI clusters.",
                source_artifact_path=self.source_free_sample_path,
                depends_on_correctness=False,
                trainable=True,
            ),
            FeatureProvenanceRecord(
                feature_name="semantic_entropy",
                role=FeatureRole.TRAINABLE,
                source="Legacy alias of semantic_entropy_nli_likelihood for downstream compatibility.",
                source_artifact_path=self.source_free_sample_path,
                depends_on_correctness=False,
                trainable=True,
            ),
            FeatureProvenanceRecord(
                feature_name="cluster_count",
                role=FeatureRole.TRAINABLE,
                source="Legacy alias of semantic_entropy_cluster_count for downstream compatibility.",
                source_artifact_path=self.source_free_sample_path,
                depends_on_correctness=False,
                trainable=True,
            ),
        )


@dataclass(frozen=True)
class SemanticEntropyReport:
    schema_version: str
    run_id: str
    row_count: int
    source_free_sample_path: str
    source_free_sample_run_id: str | None
    semantic_clusterer: str
    nli_model_ref: str
    prompt_count: int
    sample_count: int
    sample_count_per_prompt: int
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticEntropyArtifact:
    rows: tuple[SemanticEntropyRow, ...]
    report: SemanticEntropyReport

    @classmethod
    def from_free_samples(
        cls,
        free_sample_artifact: FreeSampleArtifact,
        *,
        nli_model_name: str = DEFAULT_NLI_MODEL_NAME,
    ) -> Self:
        run_id = f"semantic-entropy-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        entailment_model = _build_entailment_model(free_sample_artifact, nli_model_name=nli_model_name)
        samples_by_prompt: dict[str, list[FreeSample]] = defaultdict(list)
        for sample in free_sample_artifact.samples:
            samples_by_prompt[sample.prompt_id].append(sample)
        rows = tuple(
            SemanticEntropyRow.from_prompt_samples(
                run_id=run_id,
                prompt_id=prompt_id,
                samples=tuple(samples_by_prompt[prompt_id]),
                source_free_sample_path=str(free_sample_artifact.path),
                dataset_manifest_ref=free_sample_artifact.dataset_manifest_ref,
                formula_manifest_ref=free_sample_artifact.formula_manifest_ref,
                source_free_sample_run_id=free_sample_artifact.source_run_id,
                source_model_name=free_sample_artifact.source_model_name,
                source_tokenizer_name=free_sample_artifact.source_tokenizer_name,
                source_logits_schema_version=free_sample_artifact.source_logits_schema_version,
                source_created_at=free_sample_artifact.source_created_at,
                entailment_model=entailment_model,
            )
            for prompt_id in sorted(samples_by_prompt)
        )
        report = SemanticEntropyReport(
            schema_version=SEMANTIC_ENTROPY_SCHEMA_VERSION,
            run_id=run_id,
            row_count=len(rows),
            source_free_sample_path=str(free_sample_artifact.path),
            source_free_sample_run_id=free_sample_artifact.source_run_id,
            semantic_clusterer=SEMANTIC_CLUSTERER_NAME,
            nli_model_ref=entailment_model.model_ref(),
            prompt_count=len(rows),
            sample_count=len(free_sample_artifact.samples),
            sample_count_per_prompt=free_sample_artifact.sample_count_per_prompt,
            note=(
                "Semantic Entropy is prompt-level only: exactly N=10 answer-only free samples are clustered by strict "
                "bidirectional NLI equivalence in sample_index order, then aggregated with likelihood mass over mean token log-likelihoods."
            ),
        )
        return cls(rows=rows, report=report)

    def row_payloads(self) -> list[dict[str, object]]:
        return [row.to_row() for row in self.rows]


def build_semantic_entropy_artifact(
    free_samples_path: Path,
    *,
    nli_model_name: str = DEFAULT_NLI_MODEL_NAME,
) -> SemanticEntropyArtifact:
    return SemanticEntropyArtifact.from_free_samples(
        FreeSampleArtifact.from_path(free_samples_path),
        nli_model_name=nli_model_name,
    )


def write_semantic_entropy_artifact(
    free_samples_path: Path,
    out_path: Path,
    *,
    nli_model_name: str = DEFAULT_NLI_MODEL_NAME,
) -> dict[str, object]:
    artifact = build_semantic_entropy_artifact(free_samples_path, nli_model_name=nli_model_name)
    report = artifact.report.to_dict()
    storage = write_feature_artifact(
        out_path,
        add_schema_version(artifact.row_payloads(), SEMANTIC_ENTROPY_SCHEMA_VERSION),
        report,
        schema_version=SEMANTIC_ENTROPY_SCHEMA_VERSION,
    )
    return {"artifact": artifact, "report": report, "storage": storage}
