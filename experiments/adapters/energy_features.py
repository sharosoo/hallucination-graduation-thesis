"""Semantic Energy features from sampled responses plus candidate diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from experiments.adapters.corpus_features import load_json, read_feature_rows
from experiments.adapters.semantic_entropy_features import EXPECTED_FREE_SAMPLE_COUNT, FreeSample, FreeSampleArtifact
from experiments.domain import EnergyComputationKind, FeatureRole
from experiments.scripts.stage_control import SEMANTIC_ENTROPY_SCHEMA_VERSION

FORMULA_MANIFEST_REF = "experiments/literature/formula_notes.md"
SAMPLED_RESPONSE_ENERGY_KIND = "sampled_response_cluster"
SAMPLE_ENERGY_FORMULA = "sample_energy = mean(-selected_token_logits)"
CLUSTER_ENERGY_AGGREGATION = "cluster_energy = mean(member sample_energy); uncertainty = sum(cluster_probability * cluster_energy)"


class EnergyFeatureUnavailableError(RuntimeError):
    """Raised when an input artifact cannot support energy feature extraction."""


@dataclass(frozen=True)
class CandidateScoreWindow:
    dataset: str
    split_id: str
    prompt_id: str
    pair_id: str
    candidate_id: str
    candidate_role: str | None
    is_correct: bool | None
    label_source: str | None
    candidate_text: str | None
    candidate_token_count: int
    candidate_token_start: int
    candidate_token_end: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SampleEnergyRecord:
    dataset: str
    split_id: str
    prompt_id: str
    sample_index: int
    response_text: str
    selected_token_logits: tuple[float, ...]
    logsumexp: tuple[float, ...]
    selected_token_logprobs: tuple[float, ...]
    selected_token_probabilities: tuple[float, ...]
    token_energies: tuple[float, ...]
    sample_energy: float
    generated_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    generated_tokens: tuple[str, ...]
    model_name: str | None
    tokenizer_name: str | None
    logits_schema_version: str | None
    answer_only_protocol: dict[str, Any] | None

    @classmethod
    def from_sample(
        cls,
        sample: FreeSample,
        *,
        model_name: str | None,
        tokenizer_name: str | None,
        logits_schema_version: str | None,
        answer_only_protocol: dict[str, Any] | None,
    ) -> "SampleEnergyRecord":
        selected_token_logprobs = tuple(
            float(logit - partition) for logit, partition in zip(sample.selected_token_logits, sample.logsumexp, strict=True)
        )
        selected_token_probabilities = tuple(float(math.exp(logprob)) for logprob in selected_token_logprobs)
        token_energies = tuple(float(-logit) for logit in sample.selected_token_logits)
        return cls(
            dataset=sample.dataset,
            split_id=sample.split_id,
            prompt_id=sample.prompt_id,
            sample_index=sample.sample_index,
            response_text=sample.response_text,
            selected_token_logits=sample.selected_token_logits,
            logsumexp=sample.logsumexp,
            selected_token_logprobs=selected_token_logprobs,
            selected_token_probabilities=selected_token_probabilities,
            token_energies=token_energies,
            sample_energy=float(sum(token_energies) / len(token_energies)),
            generated_token_ids=sample.generated_token_ids,
            selected_token_ids=sample.selected_token_ids,
            generated_tokens=sample.generated_tokens,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            logits_schema_version=logits_schema_version,
            answer_only_protocol=answer_only_protocol,
        )

    def to_audit_dict(self, *, cluster_id: str | None) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "sample_index": self.sample_index,
            "cluster_id": cluster_id,
            "response_text": self.response_text,
            "sample_energy": self.sample_energy,
            "token_energies": list(self.token_energies),
            "selected_token_logits": list(self.selected_token_logits),
            "logsumexp": list(self.logsumexp),
            "selected_token_logprobs": list(self.selected_token_logprobs),
            "selected_token_probabilities": list(self.selected_token_probabilities),
            "generated_token_ids": list(self.generated_token_ids),
            "selected_token_ids": list(self.selected_token_ids),
            "generated_tokens": list(self.generated_tokens),
            "token_count": len(self.selected_token_logits),
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "logits_schema_version": self.logits_schema_version,
            "answer_only_protocol": self.answer_only_protocol,
        }


@dataclass(frozen=True)
class SemanticEnergyClusterRecord:
    cluster_id: str
    representative_sample_index: int
    member_sample_indexes: tuple[int, ...]
    cluster_probability: float
    member_sample_energies: tuple[float, ...]
    cluster_energy: float

    @classmethod
    def from_raw(
        cls,
        raw_cluster: dict[str, Any],
        *,
        sample_energies_by_index: dict[int, SampleEnergyRecord],
        prompt_id: str,
    ) -> "SemanticEnergyClusterRecord":
        cluster_id = raw_cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            raise ValueError(f"semantic entropy prompt {prompt_id!r} cluster must include non-empty cluster_id")
        representative_sample_index = raw_cluster.get("representative_sample_index")
        if not isinstance(representative_sample_index, int) or isinstance(representative_sample_index, bool):
            raise ValueError(f"semantic entropy prompt {prompt_id!r} cluster {cluster_id!r} must include integer representative_sample_index")
        raw_member_indexes = raw_cluster.get("member_sample_indexes")
        if not isinstance(raw_member_indexes, list) or not raw_member_indexes:
            raise ValueError(f"semantic entropy prompt {prompt_id!r} cluster {cluster_id!r} must include member_sample_indexes")
        member_sample_indexes: list[int] = []
        for member_index in raw_member_indexes:
            if not isinstance(member_index, int) or isinstance(member_index, bool):
                raise ValueError(f"semantic entropy prompt {prompt_id!r} cluster {cluster_id!r} has non-integer member sample index")
            if member_index not in sample_energies_by_index:
                raise ValueError(
                    f"semantic entropy prompt {prompt_id!r} cluster {cluster_id!r} references missing free-sample index {member_index}"
                )
            member_sample_indexes.append(member_index)
        if representative_sample_index not in member_sample_indexes:
            raise ValueError(f"semantic entropy prompt {prompt_id!r} cluster {cluster_id!r} representative is not a member")
        raw_cluster_probability = raw_cluster.get("cluster_probability")
        if not isinstance(raw_cluster_probability, int | float) or isinstance(raw_cluster_probability, bool):
            raise ValueError(f"semantic entropy prompt {prompt_id!r} cluster {cluster_id!r} must include numeric cluster_probability")
        cluster_probability = float(raw_cluster_probability)
        if not math.isfinite(cluster_probability):
            raise ValueError(f"semantic entropy prompt {prompt_id!r} cluster {cluster_id!r} must include numeric cluster_probability")
        member_sample_energies = tuple(sample_energies_by_index[index].sample_energy for index in member_sample_indexes)
        return cls(
            cluster_id=cluster_id,
            representative_sample_index=representative_sample_index,
            member_sample_indexes=tuple(member_sample_indexes),
            cluster_probability=cluster_probability,
            member_sample_energies=member_sample_energies,
            cluster_energy=float(sum(member_sample_energies) / len(member_sample_energies)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "representative_sample_index": self.representative_sample_index,
            "member_sample_indexes": list(self.member_sample_indexes),
            "cluster_probability": self.cluster_probability,
            "member_sample_energies": list(self.member_sample_energies),
            "cluster_energy": self.cluster_energy,
            "aggregation": "arithmetic_mean_member_sample_energy",
        }


@dataclass(frozen=True)
class PromptSemanticEnergyRecord:
    prompt_id: str
    sample_records: tuple[SampleEnergyRecord, ...]
    cluster_records: tuple[SemanticEnergyClusterRecord, ...]
    semantic_clusterer: str
    nli_model_ref: str
    nli_decision_mode: str | None
    semantic_entropy_source_free_sample_path: str
    source_semantic_entropy_run_id: str | None

    @property
    def sample_energy(self) -> float:
        return float(sum(record.sample_energy for record in self.sample_records) / len(self.sample_records))

    @property
    def cluster_uncertainty(self) -> float:
        return float(sum(record.cluster_probability * record.cluster_energy for record in self.cluster_records))

    @property
    def sample_cluster_ids(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for cluster in self.cluster_records:
            for sample_index in cluster.member_sample_indexes:
                mapping[sample_index] = cluster.cluster_id
        return mapping

    def feature_payload(self) -> dict[str, Any]:
        sample_cluster_ids = self.sample_cluster_ids
        return {
            "semantic_energy_cluster_uncertainty": self.cluster_uncertainty,
            "semantic_energy_sample_energy": self.sample_energy,
            "semantic_energy_cluster_ids": [record.cluster_id for record in self.cluster_records],
            "semantic_energy_cluster_energies": [record.to_dict() for record in self.cluster_records],
            "semantic_energy_sample_energies": [
                {"sample_index": record.sample_index, "cluster_id": sample_cluster_ids.get(record.sample_index), "sample_energy": record.sample_energy}
                for record in self.sample_records
            ],
            "semantic_energy_sample_cluster_assignments": [
                {"sample_index": sample_index, "cluster_id": cluster_id}
                for sample_index, cluster_id in sorted(sample_cluster_ids.items())
            ],
            "semantic_energy_token_provenance": [
                record.to_audit_dict(cluster_id=sample_cluster_ids.get(record.sample_index)) for record in self.sample_records
            ],
        }


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _numeric_list(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    values: list[float] = []
    for item in value:
        if not _is_number(item):
            return None
        values.append(float(item))
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _confidence_margin(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    top_1 = float("-inf")
    top_2 = float("-inf")
    for value in values:
        if value > top_1:
            top_2 = top_1
            top_1 = value
        elif value > top_2:
            top_2 = value
    return top_1 - top_2


def _required_vector_size(storage: dict[str, Any]) -> int:
    vector_size = storage.get("vector_size")
    if not isinstance(vector_size, int) or isinstance(vector_size, bool) or vector_size <= 1:
        raise ValueError("candidate-score full_logits_storage must include integer vector_size > 1")
    return vector_size


def _required_text(row: dict[str, Any], field_name: str, *, label: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must include non-empty {field_name!r}")
    return value


def _required_int(row: dict[str, Any], field_name: str, *, label: str) -> int:
    value = row.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must include integer {field_name!r}")
    return int(value)


def _optional_text(row: dict[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(row: dict[str, Any], field_name: str) -> bool | None:
    value = row.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name!r} must be boolean when present")
    return value


def _optional_source_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _same_source_path(left: object, right: Path) -> bool:
    if not isinstance(left, str) or not left.strip():
        return False
    left_path = Path(left)
    if str(left_path) == str(right):
        return True
    try:
        return left_path.resolve() == right.resolve()
    except OSError:
        return False


def _load_free_sample_protocols(source_path: Path) -> dict[tuple[str, int], dict[str, Any] | None]:
    payload = load_json(source_path)
    if not isinstance(payload, dict):
        raise ValueError(f"free-sample artifact must decode to an object: {source_path}")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("free-sample artifact must contain samples for Energy token-provenance extraction")
    protocols: dict[tuple[str, int], dict[str, Any] | None] = {}
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"free-sample sample {index} must be an object")
        prompt_id = raw_sample.get("prompt_id")
        sample_index = raw_sample.get("sample_index")
        if not isinstance(prompt_id, str) or not isinstance(sample_index, int) or isinstance(sample_index, bool):
            raise ValueError(f"free-sample sample {index} must include prompt_id and integer sample_index")
        answer_only_protocol = raw_sample.get("answer_only_protocol")
        protocols[(prompt_id, sample_index)] = dict(answer_only_protocol) if isinstance(answer_only_protocol, dict) else None
    return protocols


def _sample_records_by_prompt(free_samples_path: Path) -> tuple[dict[str, dict[int, SampleEnergyRecord]], FreeSampleArtifact]:
    artifact = FreeSampleArtifact.from_path(free_samples_path)
    if not artifact.source_model_name or not artifact.source_tokenizer_name:
        raise ValueError("free-sample artifact must include model_name and tokenizer_name for Semantic Energy provenance")
    protocols = _load_free_sample_protocols(free_samples_path)
    records_by_prompt: dict[str, dict[int, SampleEnergyRecord]] = defaultdict(dict)
    for sample in artifact.samples:
        protocol = protocols.get((sample.prompt_id, sample.sample_index))
        record = SampleEnergyRecord.from_sample(
            sample,
            model_name=artifact.source_model_name,
            tokenizer_name=artifact.source_tokenizer_name,
            logits_schema_version=artifact.source_logits_schema_version,
            answer_only_protocol=protocol,
        )
        records_by_prompt[sample.prompt_id][sample.sample_index] = record
    return records_by_prompt, artifact


def _validate_semantic_entropy_rows(
    semantic_entropy_path: Path,
    sample_records_by_prompt: dict[str, dict[int, SampleEnergyRecord]],
    free_samples_path: Path,
) -> dict[str, PromptSemanticEnergyRecord]:
    rows, _storage_report = read_feature_rows(semantic_entropy_path)
    if not rows:
        raise ValueError("semantic entropy artifact must contain at least one prompt row")
    records_by_prompt: dict[str, PromptSemanticEnergyRecord] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"semantic entropy row {row_index} must be an object")
        if row.get("schema_version") != SEMANTIC_ENTROPY_SCHEMA_VERSION:
            raise ValueError(
                f"semantic entropy row {row_index} schema_version must be {SEMANTIC_ENTROPY_SCHEMA_VERSION!r}; got {row.get('schema_version')!r}"
            )
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"semantic entropy row {row_index} must include non-empty prompt_id")
        if prompt_id in records_by_prompt:
            raise ValueError(f"semantic entropy artifact duplicates prompt_id {prompt_id!r}")
        sample_records_for_prompt = sample_records_by_prompt.get(prompt_id)
        if sample_records_for_prompt is None:
            raise ValueError(f"semantic entropy prompt {prompt_id!r} has no matching free samples")
        observed_indexes = tuple(sorted(sample_records_for_prompt))
        expected_indexes = tuple(range(EXPECTED_FREE_SAMPLE_COUNT))
        if observed_indexes != expected_indexes:
            raise ValueError(f"prompt {prompt_id!r} free samples must cover exact sample indexes {list(expected_indexes)}")
        source_free_sample_path = row.get("source_free_sample_path")
        if not _same_source_path(source_free_sample_path, free_samples_path):
            raise ValueError(f"semantic entropy row {row_index} source_free_sample_path does not match requested free-sample input")
        semantic_clusters = row.get("semantic_clusters")
        if not isinstance(semantic_clusters, list) or not semantic_clusters:
            raise ValueError(f"semantic entropy prompt {prompt_id!r} must include non-empty semantic_clusters list")
        nli_model_ref = row.get("nli_model_ref")
        if not isinstance(nli_model_ref, str) or not nli_model_ref.strip():
            raise ValueError(f"semantic entropy prompt {prompt_id!r} must include nli_model_ref")
        semantic_clusterer = row.get("features", {}).get("semantic_clusterer") if isinstance(row.get("features"), dict) else None
        if not isinstance(semantic_clusterer, str) or not semantic_clusterer.strip():
            semantic_clusterer = _optional_source_text(row.get("semantic_clusterer"))
        if not semantic_clusterer:
            raise ValueError(f"semantic entropy prompt {prompt_id!r} must include semantic_clusterer provenance")
        cluster_records = tuple(
            SemanticEnergyClusterRecord.from_raw(
                raw_cluster,
                sample_energies_by_index=sample_records_for_prompt,
                prompt_id=prompt_id,
            )
            for raw_cluster in semantic_clusters
            if isinstance(raw_cluster, dict)
        )
        if len(cluster_records) != len(semantic_clusters):
            raise ValueError(f"semantic entropy prompt {prompt_id!r} contains a non-object semantic cluster")
        assigned_indexes = sorted(index for cluster in cluster_records for index in cluster.member_sample_indexes)
        if tuple(assigned_indexes) != expected_indexes:
            raise ValueError(
                f"semantic entropy prompt {prompt_id!r} clusters must partition sample indexes {list(expected_indexes)}; got {assigned_indexes}"
            )
        probability_sum = sum(record.cluster_probability for record in cluster_records)
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"semantic entropy prompt {prompt_id!r} cluster probabilities must sum to 1.0; got {probability_sum}")
        records_by_prompt[prompt_id] = PromptSemanticEnergyRecord(
            prompt_id=prompt_id,
            sample_records=tuple(sample_records_for_prompt[index] for index in expected_indexes),
            cluster_records=cluster_records,
            semantic_clusterer=semantic_clusterer,
            nli_model_ref=nli_model_ref,
            nli_decision_mode=_optional_source_text(row.get("nli_decision_mode")),
            semantic_entropy_source_free_sample_path=str(source_free_sample_path),
            source_semantic_entropy_run_id=_optional_source_text(row.get("run_id")),
        )
    missing_prompts = set(sample_records_by_prompt) - set(records_by_prompt)
    extra_prompts = set(records_by_prompt) - set(sample_records_by_prompt)
    if missing_prompts or extra_prompts:
        raise ValueError(
            "semantic entropy prompt coverage must exactly match free-sample prompts; "
            f"missing={sorted(missing_prompts)[:5]}, extra={sorted(extra_prompts)[:5]}"
        )
    return records_by_prompt


def _validate_candidate_score_artifact(payload: dict[str, Any], source_path: Path) -> tuple[list[CandidateScoreWindow], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if payload.get("artifact_type") != "teacher_forced_candidate_scores":
        raise ValueError("candidate-score artifact must set artifact_type='teacher_forced_candidate_scores'")
    if payload.get("candidate_scoring_mode") != "teacher_forced":
        raise ValueError("candidate-score artifact must set candidate_scoring_mode='teacher_forced'")

    candidate_rows = payload.get("candidate_score_rows")
    token_rows = payload.get("token_score_rows")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        raise ValueError("candidate-score artifact must contain non-empty 'candidate_score_rows'")
    if not isinstance(token_rows, list) or not token_rows:
        raise ValueError("candidate-score artifact must contain non-empty 'token_score_rows'")

    token_rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for index, raw_row in enumerate(token_rows):
        label = f"token_score_rows[{index}]"
        if not isinstance(raw_row, dict):
            raise ValueError(f"{label} must be an object")
        candidate_id = _required_text(raw_row, "candidate_id", label=label)
        _required_text(raw_row, "prompt_id", label=label)
        _required_int(raw_row, "candidate_token_position", label=label)
        if not _is_number(raw_row.get("selected_token_logit")):
            raise ValueError(f"{label} must include numeric 'selected_token_logit'")
        if not _is_number(raw_row.get("logsumexp")):
            raise ValueError(f"{label} must include numeric 'logsumexp'")
        full_logits = raw_row.get("full_logits")
        full_logits_ref = raw_row.get("full_logits_ref")
        if full_logits is not None and full_logits != [] and _numeric_list(full_logits) is None:
            raise ValueError(f"{label} full_logits must be a numeric list when present")
        if full_logits in (None, []) and payload.get("has_full_logits") is True and not isinstance(full_logits_ref, dict):
            raise ValueError(f"{label} must include inline full_logits or a full_logits_ref when has_full_logits=true")
        token_rows_by_candidate.setdefault(candidate_id, []).append(raw_row)

    candidate_windows: list[CandidateScoreWindow] = []
    candidate_ids_seen: set[str] = set()
    ignored_prefix_token_rows = 0
    ignored_suffix_token_rows = 0
    candidate_token_rows_used = 0
    token_score_count = 0

    for index, raw_row in enumerate(candidate_rows):
        label = f"candidate_score_rows[{index}]"
        if not isinstance(raw_row, dict):
            raise ValueError(f"{label} must be an object")
        candidate_id = _required_text(raw_row, "candidate_id", label=label)
        if candidate_id in candidate_ids_seen:
            raise ValueError(f"{label} duplicates candidate_id {candidate_id!r}")
        candidate_ids_seen.add(candidate_id)

        candidate_token_count = _required_int(raw_row, "candidate_token_count", label=label)
        candidate_token_start = _required_int(raw_row, "candidate_token_start", label=label)
        candidate_token_end = _required_int(raw_row, "candidate_token_end", label=label)
        if candidate_token_count <= 0:
            raise ValueError(f"{label} candidate_token_count must be > 0")
        if candidate_token_end < candidate_token_start:
            raise ValueError(f"{label} candidate_token_end must be >= candidate_token_start")
        if candidate_token_end - candidate_token_start + 1 != candidate_token_count:
            raise ValueError(f"{label} token boundary width must match candidate_token_count")

        related_token_rows = token_rows_by_candidate.get(candidate_id, [])
        if not related_token_rows:
            raise ValueError(f"{label} has no token_score_rows for candidate_id {candidate_id!r}")

        in_window = [
            token_row
            for token_row in related_token_rows
            if candidate_token_start <= int(token_row["candidate_token_position"]) <= candidate_token_end
        ]
        if len(in_window) != candidate_token_count:
            raise ValueError(
                f"{label} candidate-window token row count {len(in_window)} does not match candidate_token_count {candidate_token_count}"
            )
        positions = sorted(int(token_row["candidate_token_position"]) for token_row in in_window)
        expected_positions = list(range(candidate_token_start, candidate_token_end + 1))
        if positions != expected_positions:
            raise ValueError(f"{label} token positions do not match candidate_token_start/end")

        ignored_prefix_token_rows += sum(1 for token_row in related_token_rows if int(token_row["candidate_token_position"]) < candidate_token_start)
        ignored_suffix_token_rows += sum(1 for token_row in related_token_rows if int(token_row["candidate_token_position"]) > candidate_token_end)
        candidate_token_rows_used += len(in_window)
        token_score_count += len(related_token_rows)

        raw_metadata = raw_row.get("metadata")
        metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        candidate_windows.append(
            CandidateScoreWindow(
                dataset=_required_text(raw_row, "dataset", label=label),
                split_id=_required_text(raw_row, "split_id", label=label),
                prompt_id=_required_text(raw_row, "prompt_id", label=label),
                pair_id=_required_text(raw_row, "pair_id", label=label),
                candidate_id=candidate_id,
                candidate_role=_optional_text(raw_row, "candidate_role"),
                is_correct=_optional_bool(raw_row, "is_correct"),
                label_source=_optional_text(raw_row, "label_source"),
                candidate_text=_optional_text(raw_row, "candidate_text"),
                candidate_token_count=candidate_token_count,
                candidate_token_start=candidate_token_start,
                candidate_token_end=candidate_token_end,
                metadata=metadata,
            )
        )

    artifact_summary = {
        "source_artifact_path": str(source_path),
        "source_run_id": payload.get("run_id"),
        "source_artifact_type": payload.get("artifact_type"),
        "source_candidate_count": len(candidate_windows),
        "source_token_score_count": token_score_count,
        "candidate_token_rows_used": candidate_token_rows_used,
        "ignored_prompt_prefix_token_rows": ignored_prefix_token_rows,
        "ignored_post_candidate_token_rows": ignored_suffix_token_rows,
        "prompt_prefix_scoring_excluded": bool(payload.get("prompt_prefix_scoring_excluded") is True),
        "has_full_logits": bool(payload.get("has_full_logits", False)),
        "formula_manifest_ref": payload.get("formula_manifest_ref", FORMULA_MANIFEST_REF),
        "dataset_manifest_ref": payload.get("dataset_manifest_ref"),
        "logits_schema_version": payload.get("logits_schema_version"),
        "model_name": payload.get("model_name"),
        "tokenizer_name": payload.get("tokenizer_name"),
    }
    return candidate_windows, token_rows_by_candidate, artifact_summary


def _resolve_artifact_sidecar_path(source_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return source_path.parent / path


def _load_candidate_full_logits_from_parquet(payload: dict[str, Any], source_path: Path) -> dict[tuple[str, int], list[float]]:
    storage = payload.get("full_logits_storage")
    if not isinstance(storage, dict) or storage.get("format") != "parquet":
        return {}
    parquet_path = _resolve_artifact_sidecar_path(source_path, storage.get("path"))
    if parquet_path is None or not parquet_path.exists():
        raise ValueError(f"candidate-score full_logits parquet sidecar is missing: {storage.get('path')!r}")
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        raise EnergyFeatureUnavailableError(
            "Missing optional dependency 'pyarrow'. Install with `uv sync --group generation` to read full-logits parquet shards."
        ) from exc
    rows = pq.read_table(parquet_path).to_pylist()
    hydrated: dict[tuple[str, int], list[float]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"full_logits parquet row {index} must be an object")
        candidate_id = row.get("candidate_id")
        candidate_token_offset = row.get("candidate_token_offset")
        logits = _numeric_list(row.get("full_logits"))
        if not isinstance(candidate_id, str) or not isinstance(candidate_token_offset, int) or logits is None:
            raise ValueError(f"full_logits parquet row {index} has invalid candidate_id/offset/logits fields")
        hydrated[(candidate_id, candidate_token_offset)] = logits
    return hydrated


def _candidate_full_logit_diagnostics(
    payload: dict[str, Any],
    source_path: Path,
    candidate_windows: list[CandidateScoreWindow],
    token_rows_by_candidate: dict[str, list[dict[str, Any]]],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    candidate_window_by_id = {candidate.candidate_id: candidate for candidate in candidate_windows}
    expected_keys: set[tuple[str, int]] = set()
    for candidate_id, token_rows in token_rows_by_candidate.items():
        diagnostics[candidate_id] = {"variances": [], "margins": [], "count": 0}
        candidate_window = candidate_window_by_id.get(candidate_id)
        if candidate_window is None:
            continue
        for token_row in token_rows:
            candidate_token_position = token_row.get("candidate_token_position")
            if not isinstance(candidate_token_position, int):
                continue
            if not candidate_window.candidate_token_start <= candidate_token_position <= candidate_window.candidate_token_end:
                continue
            offset = token_row.get("candidate_token_offset")
            if isinstance(offset, int):
                expected_keys.add((candidate_id, offset))
            inline_logits = _numeric_list(token_row.get("full_logits"))
            if inline_logits:
                variance = _variance(inline_logits)
                margin = _confidence_margin(inline_logits)
                diagnostics[candidate_id]["count"] += 1
                if variance is not None:
                    diagnostics[candidate_id]["variances"].append(variance)
                if margin is not None:
                    diagnostics[candidate_id]["margins"].append(margin)

    storage = payload.get("full_logits_storage")
    if not isinstance(storage, dict) or storage.get("format") != "parquet":
        return diagnostics
    parquet_path = _resolve_artifact_sidecar_path(source_path, storage.get("path"))
    if parquet_path is None or not parquet_path.exists():
        raise ValueError(f"candidate-score full_logits parquet sidecar is missing: {storage.get('path')!r}")
    try:
        import numpy as np  # type: ignore
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        raise EnergyFeatureUnavailableError(
            "Missing optional dependency 'pyarrow' or 'numpy'. Install with `uv sync --group generation` to read full-logits parquet shards."
        ) from exc

    seen_keys: set[tuple[str, int]] = set()
    parquet_file = pq.ParquetFile(parquet_path)
    vector_size = _required_vector_size(storage)
    total_rows = len(expected_keys)
    completed_rows = 0
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "full_logits_diagnostics",
                "completed": completed_rows,
                "total": total_rows,
                "percent": 0.0 if total_rows else 100.0,
                "message": "reading candidate full-logits parquet with vectorized variance/margin diagnostics",
            }
        )
    for batch in parquet_file.iter_batches(batch_size=128, columns=["candidate_id", "candidate_token_offset", "full_logits"], use_threads=True):
        candidate_ids = batch.column("candidate_id").to_pylist()
        token_offsets_array = batch.column("candidate_token_offset")
        token_offsets = token_offsets_array.to_numpy(zero_copy_only=False)
        logits_array = batch.column("full_logits")
        if isinstance(logits_array, pa.ChunkedArray):
            logits_array = logits_array.combine_chunks()
        if logits_array.null_count:
            raise ValueError("full_logits parquet contains null full_logits rows")
        if pa.types.is_fixed_size_list(logits_array.type):
            if logits_array.type.list_size != vector_size:
                raise ValueError(f"full_logits fixed-size list width {logits_array.type.list_size} does not match vector_size {vector_size}")
            values = logits_array.values.to_numpy(zero_copy_only=False)
        elif pa.types.is_list(logits_array.type) or pa.types.is_large_list(logits_array.type):
            lengths = pc.list_value_length(logits_array).to_numpy(zero_copy_only=False)
            if not np.all(lengths == vector_size):
                raise ValueError("full_logits parquet contains rows whose vector length does not match full_logits_storage.vector_size")
            offsets = logits_array.offsets.to_numpy(zero_copy_only=False)
            child_start = int(offsets[0])
            child_end = int(offsets[-1])
            values = logits_array.values.slice(child_start, child_end - child_start).to_numpy(zero_copy_only=False)
        else:
            raise ValueError(f"full_logits parquet column must be a list/fixed-size-list array, got {logits_array.type}")
        if values.size != len(candidate_ids) * vector_size:
            raise ValueError("full_logits parquet batch values cannot be reshaped to row_count x vector_size")
        matrix = values.reshape(len(candidate_ids), vector_size)
        variances = np.var(matrix, axis=1, dtype=np.float64)
        top_two = np.partition(matrix, -2, axis=1)[:, -2:]
        margins = top_two.max(axis=1) - top_two.min(axis=1)

        batch_completed = 0
        for row_index, candidate_id in enumerate(candidate_ids):
            candidate_token_offset = int(token_offsets[row_index])
            if not isinstance(candidate_id, str):
                raise ValueError("full_logits parquet row has invalid candidate_id field")
            key = (candidate_id, candidate_token_offset)
            if key not in expected_keys:
                continue
            if key in seen_keys:
                raise ValueError(f"full_logits parquet duplicates candidate token key: {key!r}")
            seen_keys.add(key)
            entry = diagnostics.setdefault(candidate_id, {"variances": [], "margins": [], "count": 0})
            entry["count"] += 1
            entry["variances"].append(float(variances[row_index]))
            entry["margins"].append(float(margins[row_index]))
            batch_completed += 1
        completed_rows += batch_completed
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "full_logits_diagnostics",
                    "completed": completed_rows,
                    "total": total_rows,
                    "percent": round(completed_rows / total_rows * 100, 4) if total_rows else 100.0,
                    "message": "reading candidate full-logits parquet with vectorized variance/margin diagnostics",
                }
            )
    missing_keys = expected_keys - seen_keys
    if expected_keys and missing_keys:
        preview = sorted(repr(key) for key in missing_keys)[:5]
        raise ValueError(f"full_logits parquet is missing {len(missing_keys)} candidate token keys: {preview}")
    return diagnostics


def _build_feature_provenance(*, source_artifact_path: str) -> list[dict[str, Any]]:
    return [
        {
            "feature_name": "energy_kind",
            "role": FeatureRole.ANALYSIS_ONLY.value,
            "source": "teacher_forced_candidate_scores candidate token window",
            "source_artifact_path": source_artifact_path,
            "depends_on_correctness": False,
            "trainable": False,
            "note": "Branch marker only; Task 10 uses teacher-forced candidate token rows, not free-sample generation rows.",
        },
        {
            "feature_name": "semantic_energy_boltzmann_diagnostic",
            "role": FeatureRole.TRAINABLE.value,
            "source": "per-token mean of -logsumexp over candidate answer token positions only",
            "source_artifact_path": source_artifact_path,
            "depends_on_correctness": False,
            "trainable": True,
            "note": "Candidate-level Boltzmann-style diagnostic only; not paper-faithful Semantic Energy without multi-generation semantic clustering and cluster-level aggregation.",
        },
        {
            "feature_name": "mean_negative_log_probability",
            "role": FeatureRole.TRAINABLE.value,
            "source": "per-token mean of (logsumexp - selected_token_logit) over candidate answer token positions only",
            "source_artifact_path": source_artifact_path,
            "depends_on_correctness": False,
            "trainable": True,
            "note": "This is the mean negative log probability of the fixed candidate under teacher forcing.",
        },
        {
            "feature_name": "logit_variance",
            "role": FeatureRole.TRAINABLE.value,
            "source": "per-token mean variance(full_logits) over candidate answer token positions when full_logits are present",
            "source_artifact_path": source_artifact_path,
            "depends_on_correctness": False,
            "trainable": True,
            "note": "Null only when full logits are unavailable for a token row.",
        },
        {
            "feature_name": "confidence_margin",
            "role": FeatureRole.TRAINABLE.value,
            "source": "per-token mean top1-minus-top2 full-logit margin over candidate answer token positions when full_logits are present",
            "source_artifact_path": source_artifact_path,
            "depends_on_correctness": False,
            "trainable": True,
            "note": "Null only when full logits are unavailable for a token row.",
        },
    ]


def build_energy_rows_from_candidate_scores(
    candidate_scores_path: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute candidate-level energy/logit diagnostics from teacher-forced candidate-score artifacts."""

    payload = load_json(candidate_scores_path)
    if not isinstance(payload, dict):
        raise ValueError(f"candidate-score artifact must decode to an object: {candidate_scores_path}")

    candidate_windows, token_rows_by_candidate, artifact_summary = _validate_candidate_score_artifact(payload, candidate_scores_path)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "validate_candidate_scores",
                "completed": len(candidate_windows),
                "total": len(candidate_windows),
                "percent": 100.0,
                "message": "validated teacher-forced candidate score rows",
            }
        )
    full_logit_diagnostics = _candidate_full_logit_diagnostics(
        payload, candidate_scores_path, candidate_windows, token_rows_by_candidate, progress_callback=progress_callback
    )
    if artifact_summary["has_full_logits"] is not True:
        raise ValueError("candidate-score artifact must set has_full_logits=true for candidate-level Energy/logit diagnostics")
    run_id = f"energy-from-candidate-scores-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    rows: list[dict[str, Any]] = []

    total_candidates = len(candidate_windows)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "build_energy_rows",
                "completed": 0,
                "total": total_candidates,
                "percent": 0.0 if total_candidates else 100.0,
                "message": "building candidate-level energy feature rows",
            }
        )
    for index, candidate in enumerate(candidate_windows, start=1):
        source_token_rows = token_rows_by_candidate[candidate.candidate_id]
        window_token_rows = sorted(
            [
                token_row
                for token_row in source_token_rows
                if candidate.candidate_token_start <= int(token_row["candidate_token_position"]) <= candidate.candidate_token_end
            ],
            key=lambda token_row: int(token_row["candidate_token_position"]),
        )
        token_energies = [-float(token_row["logsumexp"]) for token_row in window_token_rows]
        negative_log_probabilities = [
            float(token_row["logsumexp"]) - float(token_row["selected_token_logit"]) for token_row in window_token_rows
        ]

        diagnostics = full_logit_diagnostics.get(candidate.candidate_id, {"variances": [], "margins": [], "count": 0})
        per_token_variances = diagnostics["variances"]
        per_token_margins = diagnostics["margins"]
        tokens_with_full_logits = int(diagnostics["count"])
        if tokens_with_full_logits != candidate.candidate_token_count:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} has full-logits diagnostics for {tokens_with_full_logits} token(s), "
                f"expected {candidate.candidate_token_count} candidate answer token(s)"
            )

        features = {
            "semantic_entropy": None,
            "cluster_count": None,
            "semantic_energy_boltzmann": _mean(token_energies),
            "semantic_energy_boltzmann_diagnostic": _mean(token_energies),
            "semantic_energy_proxy": None,
            "energy_kind": EnergyComputationKind.CANDIDATE_BOLTZMANN_DIAGNOSTIC.value,
            "mean_negative_log_probability": _mean(negative_log_probabilities),
            "logit_variance": _mean(per_token_variances),
            "confidence_margin": _mean(per_token_margins),
            "diagnostic_only": True,
            "not_for_thesis_claims": True,
            "not_for_validation": False,
        }
        energy_provenance = {
            "source": "teacher_forced_candidate_scores",
            "source_artifact_has_logits": True,
            "source_artifact_has_full_logits": artifact_summary["has_full_logits"],
            "requires_rerun_for_true_boltzmann": True,
            "rerun_required_for_true_boltzmann": True,
            "requires_paper_faithful_semantic_energy": True,
            "diagnostic_only": True,
            "not_for_thesis_claims": True,
            "not_for_validation": False,
            "candidate_scoring_mode": "teacher_forced",
            "token_position_source": "candidate answer token positions only",
            "candidate_token_start": candidate.candidate_token_start,
            "candidate_token_end": candidate.candidate_token_end,
            "candidate_token_count": candidate.candidate_token_count,
            "prompt_prefix_scoring_excluded": True,
            "prompt_prefix_rows_ignored": sum(
                1 for token_row in source_token_rows if int(token_row["candidate_token_position"]) < candidate.candidate_token_start
            ),
            "post_candidate_rows_ignored": sum(
                1 for token_row in source_token_rows if int(token_row["candidate_token_position"]) > candidate.candidate_token_end
            ),
            "mean_normalization": "per-token arithmetic mean over candidate answer tokens",
            "formula_manifest_ref": artifact_summary["formula_manifest_ref"],
            "dataset_manifest_ref": artifact_summary["dataset_manifest_ref"],
            "logits_schema_version": artifact_summary["logits_schema_version"],
            "model_name": artifact_summary["model_name"],
            "tokenizer_name": artifact_summary["tokenizer_name"],
            "note": (
                "semantic_energy_boltzmann_diagnostic is the arithmetic mean of -logsumexp(logits) over teacher-forced candidate "
                "answer tokens only. It is a candidate-level diagnostic, not paper-faithful Semantic Energy; paper-faithful "
                "Semantic Energy requires multi-generation semantic clustering and cluster-level energy aggregation."
            ),
            "tokens_with_full_logits": tokens_with_full_logits,
        }
        rows.append(
            {
                "run_id": run_id,
                "dataset": candidate.dataset,
                "dataset_id": candidate.dataset,
                "split_id": candidate.split_id,
                "candidate_id": candidate.candidate_id,
                "prompt_id": candidate.prompt_id,
                "pair_id": candidate.pair_id,
                "sample_id": candidate.candidate_id,
                "candidate_role": candidate.candidate_role,
                "is_correct": candidate.is_correct,
                "label": None,
                "label_status": "not_assigned_by_energy_features",
                "label_source": candidate.label_source,
                "candidate_text": candidate.candidate_text,
                "candidate_token_count": candidate.candidate_token_count,
                "candidate_token_start": candidate.candidate_token_start,
                "candidate_token_end": candidate.candidate_token_end,
                "source_artifact_path": str(candidate_scores_path),
                "artifact_id": str(payload.get("run_id") or Path(candidate_scores_path).name),
                "features": features,
                "energy_provenance": energy_provenance,
                "feature_provenance": _build_feature_provenance(source_artifact_path=str(candidate_scores_path)),
                "formula_manifest_ref": artifact_summary["formula_manifest_ref"],
                "dataset_manifest_ref": artifact_summary["dataset_manifest_ref"],
                "metadata": candidate.metadata,
            }
        )
        if progress_callback is not None and (index == total_candidates or index % 256 == 0):
            progress_callback(
                {
                    "phase": "build_energy_rows",
                    "completed": index,
                    "total": total_candidates,
                    "percent": round(index / total_candidates * 100, 4) if total_candidates else 100.0,
                    "message": "building candidate-level energy feature rows",
                }
            )

    report = {
        "run_id": run_id,
        "row_count": len(rows),
        "source_candidate_scores_path": str(candidate_scores_path),
        "source_artifact_type": payload.get("artifact_type"),
        "candidate_scoring_mode": payload.get("candidate_scoring_mode"),
        "prompt_prefix_scoring_excluded": True,
        "true_boltzmann_available": False,
        "candidate_boltzmann_diagnostic_available": True,
        "rerun_required_for_true_boltzmann": True,
        "requires_paper_faithful_semantic_energy": True,
        "energy_kind_counts": {EnergyComputationKind.CANDIDATE_BOLTZMANN_DIAGNOSTIC.value: len(rows)},
        "selected_artifacts": [
            {
                "artifact_id": payload.get("run_id", Path(candidate_scores_path).name),
                "dataset_id": "paired_candidate_teacher_forced_scores",
                "dataset_name": "paired_candidate_teacher_forced_scores",
                "split_id": "mixed",
                "has_logits": True,
                "has_full_logits": artifact_summary["has_full_logits"],
                "path": str(candidate_scores_path),
            }
        ],
        "source_candidate_count": artifact_summary["source_candidate_count"],
        "source_token_score_count": artifact_summary["source_token_score_count"],
        "candidate_token_rows_used": artifact_summary["candidate_token_rows_used"],
        "ignored_prompt_prefix_token_rows": artifact_summary["ignored_prompt_prefix_token_rows"],
        "ignored_post_candidate_token_rows": artifact_summary["ignored_post_candidate_token_rows"],
        "note": "Task 10 energy rows are candidate-level only and are derived from teacher-forced candidate answer token windows.",
    }
    return rows, report


def _sampled_energy_feature_provenance(
    *,
    free_samples_path: str,
    semantic_entropy_path: str,
) -> list[dict[str, Any]]:
    shared_source = "N=10 free-sample generated responses joined to Task 4 semantic_clusters by (prompt_id, sample_index)"
    return [
        {
            "feature_name": "semantic_energy_cluster_uncertainty",
            "role": FeatureRole.TRAINABLE.value,
            "source": shared_source,
            "source_artifact_path": semantic_entropy_path,
            "depends_on_correctness": False,
            "trainable": True,
            "note": (
                "Paper-faithful sampled-response Semantic Energy uncertainty. Lower raw Energy means more reliable; "
                f"{CLUSTER_ENERGY_AGGREGATION}."
            ),
        },
        {
            "feature_name": "semantic_energy_sample_energy",
            "role": FeatureRole.TRAINABLE.value,
            "source": "selected-token raw logits from N=10 generated free samples",
            "source_artifact_path": free_samples_path,
            "depends_on_correctness": False,
            "trainable": True,
            "note": f"Prompt-level arithmetic mean of response energies where {SAMPLE_ENERGY_FORMULA}.",
        },
        {
            "feature_name": "semantic_energy_boltzmann",
            "role": FeatureRole.TRAINABLE.value,
            "source": "legacy compatibility alias of semantic_energy_cluster_uncertainty",
            "source_artifact_path": semantic_entropy_path,
            "depends_on_correctness": False,
            "trainable": True,
            "note": "Compatibility alias only; candidate teacher-forced -logsumexp remains semantic_energy_boltzmann_diagnostic.",
        },
    ]


def build_energy_rows_from_generation_artifacts(
    *,
    candidate_scores_path: Path,
    free_samples_path: Path,
    semantic_entropy_path: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute sampled-response Semantic Energy while preserving candidate diagnostics."""

    candidate_rows, candidate_report = build_energy_rows_from_candidate_scores(candidate_scores_path, progress_callback=progress_callback)
    sample_records_by_prompt, free_sample_artifact = _sample_records_by_prompt(free_samples_path)
    prompt_energy_records = _validate_semantic_entropy_rows(semantic_entropy_path, sample_records_by_prompt, free_samples_path)
    candidate_prompt_ids = {str(row.get("prompt_id")) for row in candidate_rows}
    if candidate_prompt_ids != set(prompt_energy_records):
        missing_prompts = candidate_prompt_ids - set(prompt_energy_records)
        extra_prompts = set(prompt_energy_records) - candidate_prompt_ids
        raise ValueError(
            "Energy input prompt coverage must match across candidate scores, free samples, and semantic entropy; "
            f"missing={sorted(missing_prompts)[:5]}, extra={sorted(extra_prompts)[:5]}"
        )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "build_sampled_response_energy",
                "completed": 0,
                "total": len(candidate_rows),
                "percent": 0.0 if candidate_rows else 100.0,
                "message": "joining free-sample response energies to Task 4 semantic clusters",
            }
        )

    run_id = f"semantic-energy-sampled-response-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    sampled_feature_provenance = _sampled_energy_feature_provenance(
        free_samples_path=str(free_samples_path),
        semantic_entropy_path=str(semantic_entropy_path),
    )
    updated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_rows, start=1):
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str):
            raise ValueError(f"candidate Energy row {index - 1} must include string prompt_id")
        prompt_energy = prompt_energy_records[prompt_id]
        sampled_features = prompt_energy.feature_payload()
        raw_features = row.get("features")
        features: dict[str, Any] = dict(cast(dict[str, Any], raw_features)) if isinstance(raw_features, dict) else {}
        candidate_diagnostic = features.get("semantic_energy_boltzmann_diagnostic")
        features.update(sampled_features)
        features.update(
            {
                "semantic_energy_boltzmann": sampled_features["semantic_energy_cluster_uncertainty"],
                "semantic_energy_boltzmann_legacy_alias_of": "semantic_energy_cluster_uncertainty",
                "semantic_energy_boltzmann_diagnostic": candidate_diagnostic,
                "energy_kind": SAMPLED_RESPONSE_ENERGY_KIND,
                "paper_faithful_energy_available": True,
                "candidate_diagnostics_available": True,
                "candidate_diagnostics_are_paper_faithful_energy": False,
                "diagnostic_only": False,
                "not_for_thesis_claims": False,
                "not_for_validation": False,
            }
        )
        raw_original_provenance = row.get("energy_provenance")
        original_provenance: dict[str, Any] = (
            dict(cast(dict[str, Any], raw_original_provenance)) if isinstance(raw_original_provenance, dict) else {}
        )
        energy_provenance = {
            **original_provenance,
            "source": "free_sample_rows_and_semantic_entropy_clusters",
            "candidate_diagnostic_source": "teacher_forced_candidate_scores",
            "source_candidate_scores_path": str(candidate_scores_path),
            "source_free_sample_path": str(free_samples_path),
            "source_semantic_entropy_path": str(semantic_entropy_path),
            "semantic_entropy_source_free_sample_path": prompt_energy.semantic_entropy_source_free_sample_path,
            "paper_faithful_energy_available": True,
            "energy_kind": SAMPLED_RESPONSE_ENERGY_KIND,
            "energy_granularity": "prompt_level_broadcast_to_candidate_rows",
            "sample_energy_formula": SAMPLE_ENERGY_FORMULA,
            "sample_energy_aggregation": "arithmetic_mean_over_10_response_sample_energies",
            "cluster_energy_aggregation": CLUSTER_ENERGY_AGGREGATION,
            "token_position_source": "candidate answer token positions only for diagnostics; free-sample generated answer token positions for paper-faithful energy",
            "selected_token_logit_source": "free_samples.samples[].selected_token_logits raw generated-token logits",
            "selected_token_logprob_source": "selected_token_logits - logsumexp for each generated token",
            "selected_token_probability_source": "exp(selected_token_logits - logsumexp) for each generated token",
            "free_sample_count_per_prompt": EXPECTED_FREE_SAMPLE_COUNT,
            "semantic_clusterer": prompt_energy.semantic_clusterer,
            "nli_model_ref": prompt_energy.nli_model_ref,
            "nli_decision_mode": prompt_energy.nli_decision_mode,
            "model_name": free_sample_artifact.source_model_name,
            "tokenizer_name": free_sample_artifact.source_tokenizer_name,
            "free_sample_logits_schema_version": free_sample_artifact.source_logits_schema_version,
            "truncation_end_token_policy": [
                record.answer_only_protocol for record in prompt_energy.sample_records if record.answer_only_protocol is not None
            ],
            "sign_convention": "lower raw semantic_energy_cluster_uncertainty and semantic_energy_sample_energy mean more reliable",
            "candidate_diagnostics_note": (
                "semantic_energy_boltzmann_diagnostic, mean_negative_log_probability, logit_variance, and confidence_margin "
                "remain teacher-forced candidate diagnostics only."
            ),
            "diagnostic_only": False,
            "not_for_thesis_claims": False,
            "requires_paper_faithful_semantic_energy": False,
            "requires_rerun_for_true_boltzmann": False,
        }
        raw_feature_provenance = row.get("feature_provenance")
        feature_provenance: list[Any] = list(raw_feature_provenance) if isinstance(raw_feature_provenance, list) else []
        updated_row = {
            **row,
            "run_id": run_id,
            "source_free_sample_path": str(free_samples_path),
            "source_semantic_entropy_path": str(semantic_entropy_path),
            "features": features,
            "energy_provenance": energy_provenance,
            "feature_provenance": feature_provenance + sampled_feature_provenance,
        }
        updated_rows.append(updated_row)
        if progress_callback is not None and (index == len(candidate_rows) or index % 256 == 0):
            progress_callback(
                {
                    "phase": "build_sampled_response_energy",
                    "completed": index,
                    "total": len(candidate_rows),
                    "percent": round(index / len(candidate_rows) * 100, 4) if candidate_rows else 100.0,
                    "message": "joining free-sample response energies to Task 4 semantic clusters",
                }
            )

    report = {
        **candidate_report,
        "run_id": run_id,
        "row_count": len(updated_rows),
        "source_candidate_scores_path": str(candidate_scores_path),
        "source_free_sample_path": str(free_samples_path),
        "source_semantic_entropy_path": str(semantic_entropy_path),
        "source_free_sample_run_id": free_sample_artifact.source_run_id,
        "semantic_energy_kind": SAMPLED_RESPONSE_ENERGY_KIND,
        "paper_faithful_energy_available": True,
        "true_boltzmann_available": False,
        "candidate_boltzmann_diagnostic_available": True,
        "rerun_required_for_true_boltzmann": False,
        "requires_paper_faithful_semantic_energy": False,
        "energy_granularity": "prompt_level_broadcast_to_candidate_rows",
        "sample_energy_formula": SAMPLE_ENERGY_FORMULA,
        "cluster_energy_aggregation": CLUSTER_ENERGY_AGGREGATION,
        "free_sample_count_per_prompt": EXPECTED_FREE_SAMPLE_COUNT,
        "prompt_count": len(prompt_energy_records),
        "energy_kind_counts": {SAMPLED_RESPONSE_ENERGY_KIND: len(updated_rows)},
        "note": (
            "Paper-faithful Semantic Energy is computed from N=10 sampled responses using selected-token raw logits and "
            "Task 4 semantic clusters. Candidate teacher-forced Energy fields remain diagnostics only."
        ),
    }
    return updated_rows, report
