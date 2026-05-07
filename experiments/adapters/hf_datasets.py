"""Hugging Face dataset materialization for paired prompt/candidate records."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from typing import Any

from experiments.domain import CandidateRow, PromptGroup, PromptRow, QuestionExample
from experiments.ports import DatasetLoaderPort


class DatasetMaterializationError(RuntimeError):
    """Raised when a configured dataset cannot be materialized."""


_ACTIVE_PAIRED_DATASETS = frozenset({"TruthfulQA", "HaluEval-QA"})
_UNSUPPORTED_THESIS_DATASETS = frozenset({"TriviaQA", "Natural Questions", "HotpotQA", "FEVER", "BioASQ"})


def _json_metadata_tuple(payload: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((key, json.dumps(payload[key], ensure_ascii=False, sort_keys=True, default=str)) for key in sorted(payload))


def _decode_metadata(metadata: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in metadata:
        try:
            payload[key] = json.loads(value)
        except json.JSONDecodeError:
            payload[key] = value
    return payload


def metadata_dict(row: PromptRow | QuestionExample | PromptGroup | CandidateRow) -> dict[str, Any]:
    return _decode_metadata(row.metadata)


def _stable_slug(value: str) -> str:
    safe = "-".join(value.strip().lower().split())
    return "".join(character for character in safe if character.isalnum() or character == "-")


def _stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    slug = "-".join(_stable_slug(part) for part in parts if part)
    return f"{slug}-{digest}" if slug else digest


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def stable_sample_id(dataset_id: str, split_id: str, source_id: str | int) -> str:
    raw = f"{dataset_id}:{split_id}:{source_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    safe_source = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(source_id))[:40].strip("_")
    return f"{dataset_id}-{safe_source or 'row'}-{digest}"


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _require_text(value: object, *, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise DatasetMaterializationError(f"{field_name} must be a non-empty string")
    return text


def _require_raw_non_empty_text(value: object, *, field_name: str) -> str:
    if value is None:
        raise DatasetMaterializationError(f"{field_name} must be a non-empty string")
    text = str(value)
    if not text.strip():
        raise DatasetMaterializationError(f"{field_name} must be a non-empty string")
    return text


def _string_candidates(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return (cleaned,) if cleaned else ()
    if not isinstance(value, list | tuple):
        return ()
    cleaned_values: list[str] = []
    for item in value:
        cleaned = _clean_text(item)
        if cleaned:
            cleaned_values.append(cleaned)
    return tuple(cleaned_values)


def _sorted_unique_candidates(value: object) -> tuple[str, ...]:
    return tuple(sorted(set(_string_candidates(value))))


def _pair_prompt(question: str, context: str | None) -> str:
    if context:
        return (
            f"Context: {context}\n\nQuestion: {question}\n"
            "Return only the shortest final answer span. Do not explain or write a full sentence.\nAnswer:"
        )
    return f"Question: {question}\nReturn only the shortest final answer span. Do not explain or write a full sentence.\nAnswer:"


def _stable_candidate_index(*, seed: int, prompt_identity: str, role: str, candidates: tuple[str, ...]) -> int:
    if not candidates:
        raise DatasetMaterializationError(f"{role} candidates must contain at least one non-empty string")
    selector = f"{seed}:{prompt_identity}:{role}"
    digest = hashlib.sha256(selector.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % len(candidates)


def _dataset_name(dataset_config: dict[str, Any]) -> str:
    return str(dataset_config["name"])


def _dataset_source_id(raw_row: dict[str, Any], source_index: int) -> str:
    for key in ("id", "question_id", "source_id"):
        value = raw_row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(source_index)


class HuggingFaceDatasetLoader(DatasetLoaderPort):
    """Concrete HF dataset loader and paired prompt/candidate materializer."""

    def load_examples(self, dataset_name: str, split_id: str) -> tuple[QuestionExample, ...]:
        raise NotImplementedError("Use materialize_paired_rows with dataset registry entries for reproducible loading.")

    def materialize_paired_rows(
        self,
        dataset_config: dict[str, Any],
        *,
        target_count: int,
    ) -> tuple[tuple[PromptGroup, ...], tuple[CandidateRow, ...], dict[str, Any]]:
        dataset, hf_id, hf_config, split, seed = self._load_dataset(dataset_config)

        available_count = len(dataset)
        selected_indices = self._selected_indices(available_count=available_count, seed=seed, target_count=target_count)

        prompt_groups: list[PromptGroup] = []
        candidate_rows: list[CandidateRow] = []
        skipped: list[dict[str, Any]] = []
        materialized_source_rows: list[int] = []

        for source_index in selected_indices:
            raw_row = dict(dataset[source_index])
            try:
                prompt_group, paired_candidates = self._row_to_paired_records(dataset_config, raw_row, source_index)
            except DatasetMaterializationError as exc:
                skipped.append({"source_index": source_index, "reason": str(exc)})
                continue

            prompt_groups.append(prompt_group)
            candidate_rows.extend(paired_candidates)
            materialized_source_rows.append(source_index)

        report = self._materialization_report(
            dataset_config,
            dataset=dataset,
            hf_id=hf_id,
            hf_config=hf_config,
            split=split,
            seed=seed,
            target_count=target_count,
            available_count=available_count,
            selected_indices=selected_indices,
            materialized_source_rows=materialized_source_rows,
            skipped=skipped,
            prompt_group_count=len(prompt_groups),
            candidate_row_count=len(candidate_rows),
        )
        return tuple(prompt_groups), tuple(candidate_rows), report

    def materialize_prompt_rows(self, dataset_config: dict[str, Any], *, target_count: int) -> tuple[tuple[PromptRow, ...], dict[str, Any]]:
        prompt_groups, _, paired_report = self.materialize_paired_rows(dataset_config, target_count=target_count)
        prompt_rows = tuple(
            PromptRow(
                dataset=prompt_group.dataset,
                split_id=prompt_group.split_id,
                sample_id=prompt_group.prompt_id,
                prompt=prompt_group.prompt,
                question=prompt_group.question,
                context=prompt_group.context,
                metadata=prompt_group.metadata,
            )
            for prompt_group in prompt_groups
        )
        report = dict(paired_report)
        report["prompt_row_count"] = len(prompt_rows)
        return prompt_rows, report

    def _load_dataset(self, dataset_config: dict[str, Any]) -> tuple[Any, str, Any, str, int]:
        try:
            from datasets import load_dataset  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency branch
            raise DatasetMaterializationError(
                "Missing optional dependency 'datasets'. Run `uv sync --group generation` first."
            ) from exc

        name = _dataset_name(dataset_config)
        if name in _UNSUPPORTED_THESIS_DATASETS:
            raise DatasetMaterializationError(
                f"dataset {name!r} is excluded from thesis paired materialization because it lacks clean dataset-provided candidate pairs"
            )
        if name not in _ACTIVE_PAIRED_DATASETS:
            raise DatasetMaterializationError(f"unsupported dataset mapping: {name}")

        hf_id = str(dataset_config.get("hf_id") or dataset_config.get("source_id") or "")
        hf_config = dataset_config.get("config")
        split = str(dataset_config["split"])
        seed = int(dataset_config["seed"])
        if not hf_id:
            raise DatasetMaterializationError(f"dataset {name!r} has no hf_id/source_id")

        if hf_config is None:
            dataset = load_dataset(hf_id, split=split)
        else:
            dataset = load_dataset(hf_id, str(hf_config), split=split)
        return dataset, hf_id, hf_config, split, seed

    def _selected_indices(self, *, available_count: int, seed: int, target_count: int) -> list[int]:
        selected_count = min(target_count, available_count)
        indices = list(range(available_count))
        random.Random(seed).shuffle(indices)
        return sorted(indices[:selected_count])

    def _row_to_paired_records(
        self,
        dataset_config: dict[str, Any],
        raw_row: dict[str, Any],
        source_index: int,
    ) -> tuple[PromptGroup, tuple[CandidateRow, CandidateRow]]:
        name = _dataset_name(dataset_config)
        if name == "TruthfulQA":
            return self._truthfulqa_pair(dataset_config, raw_row, source_index)
        if name == "HaluEval-QA":
            return self._halueval_pair(dataset_config, raw_row, source_index)
        if name in _UNSUPPORTED_THESIS_DATASETS:
            raise DatasetMaterializationError(
                f"dataset {name!r} is excluded from thesis paired materialization because it lacks clean dataset-provided candidate pairs"
            )
        raise DatasetMaterializationError(f"unsupported dataset mapping: {name}")

    def _truthfulqa_pair(
        self,
        dataset_config: dict[str, Any],
        raw_row: dict[str, Any],
        source_index: int,
    ) -> tuple[PromptGroup, tuple[CandidateRow, CandidateRow]]:
        question = _require_text(raw_row.get("question"), field_name="question")
        sorted_correct = _sorted_unique_candidates(raw_row.get("correct_answers"))
        sorted_incorrect = _sorted_unique_candidates(raw_row.get("incorrect_answers"))
        if not sorted_correct:
            raise DatasetMaterializationError("correct_answers must contain at least one non-empty string")
        if not sorted_incorrect:
            raise DatasetMaterializationError("incorrect_answers must contain at least one non-empty string")
        overlapping_candidates = sorted(set(sorted_correct) & set(sorted_incorrect))
        if overlapping_candidates:
            raise DatasetMaterializationError(
                f"correct and incorrect candidate pools overlap after cleaning: {overlapping_candidates}"
            )
        prompt_identity = _prompt_hash(question)
        seed = int(dataset_config["seed"])
        correct_index = _stable_candidate_index(
            seed=seed,
            prompt_identity=prompt_identity,
            role="correct",
            candidates=sorted_correct,
        )
        incorrect_index = _stable_candidate_index(
            seed=seed,
            prompt_identity=prompt_identity,
            role="incorrect",
            candidates=sorted_incorrect,
        )
        selected_correct = sorted_correct[correct_index]
        selected_incorrect = sorted_incorrect[incorrect_index]
        if selected_correct == selected_incorrect:
            raise DatasetMaterializationError(
                "selected correct and incorrect candidates are identical after deterministic TruthfulQA pairing"
            )

        prompt_group, paired_candidates = PromptGroup.from_raw_truthfulqa(
            {
                **raw_row,
                "correct_answers": list(sorted_correct),
                "incorrect_answers": list(sorted_incorrect),
            },
            split_id=str(dataset_config["split_id"]),
            source_row_id=_dataset_source_id(raw_row, source_index),
            correct_candidate_index=correct_index,
            incorrect_candidate_index=incorrect_index,
        )
        if paired_candidates[0].candidate_text != selected_correct:
            raise DatasetMaterializationError("TruthfulQA correct candidate text does not match deterministic sorted pool selection")
        if paired_candidates[1].candidate_text != selected_incorrect:
            raise DatasetMaterializationError("TruthfulQA incorrect candidate text does not match deterministic sorted pool selection")

        prompt_metadata = metadata_dict(prompt_group) | {
            "source_index": source_index,
            "hf_id": dataset_config.get("hf_id"),
            "hf_config": dataset_config.get("config"),
            "hf_split": dataset_config.get("split"),
            "dataset_id": dataset_config.get("split_id"),
            "role": dataset_config.get("role"),
            "raw_keys": sorted(raw_row.keys()),
            "prompt_hash": prompt_group.prompt_hash,
            "selection_seed": seed,
            "correct_answers": [selected_correct],
            "incorrect_answers": [selected_incorrect],
            "correct_candidate_pool": list(sorted_correct),
            "incorrect_candidate_pool": list(sorted_incorrect),
            "correct_candidate_index": correct_index,
            "incorrect_candidate_index": incorrect_index,
        }
        prompt_group = PromptGroup(
            dataset=prompt_group.dataset,
            split_id=prompt_group.split_id,
            source_row_id=prompt_group.source_row_id,
            prompt_id=prompt_group.prompt_id,
            pair_id=prompt_group.pair_id,
            question=prompt_group.question,
            prompt=prompt_group.prompt,
            context=prompt_group.context,
            prompt_hash=prompt_group.prompt_hash,
            label_source=prompt_group.label_source,
            metadata=_json_metadata_tuple(prompt_metadata),
        )

        correct_candidate = CandidateRow(
            prompt_id=prompt_group.prompt_id,
            candidate_id=f"{prompt_group.pair_id}:right",
            pair_id=prompt_group.pair_id,
            source_row_id=prompt_group.source_row_id,
            dataset=prompt_group.dataset,
            split_id=prompt_group.split_id,
            candidate_text=selected_correct,
            candidate_role="right",
            is_correct=True,
            label_source="truthfulqa_correct_answers",
            question=prompt_group.question,
            prompt=prompt_group.prompt,
            context=prompt_group.context,
            metadata=_json_metadata_tuple({
                "source_index": source_index,
                "dataset_id": dataset_config.get("split_id"),
                "selection_seed": seed,
                "candidate_pool_index": correct_index,
                "candidate_pool_size": len(sorted_correct),
            }),
        )
        incorrect_candidate = CandidateRow(
            prompt_id=prompt_group.prompt_id,
            candidate_id=f"{prompt_group.pair_id}:hallucinated",
            pair_id=prompt_group.pair_id,
            source_row_id=prompt_group.source_row_id,
            dataset=prompt_group.dataset,
            split_id=prompt_group.split_id,
            candidate_text=selected_incorrect,
            candidate_role="hallucinated",
            is_correct=False,
            label_source="truthfulqa_incorrect_answers",
            question=prompt_group.question,
            prompt=prompt_group.prompt,
            context=prompt_group.context,
            metadata=_json_metadata_tuple({
                "source_index": source_index,
                "dataset_id": dataset_config.get("split_id"),
                "selection_seed": seed,
                "candidate_pool_index": incorrect_index,
                "candidate_pool_size": len(sorted_incorrect),
            }),
        )
        return prompt_group, (correct_candidate, incorrect_candidate)

    def _halueval_pair(
        self,
        dataset_config: dict[str, Any],
        raw_row: dict[str, Any],
        source_index: int,
    ) -> tuple[PromptGroup, tuple[CandidateRow, CandidateRow]]:
        question = _require_text(raw_row.get("question"), field_name="question")
        context = _clean_text(raw_row.get("knowledge")) or None
        prompt = _pair_prompt(question, context)
        source_row_id = _dataset_source_id(raw_row, source_index)
        prompt_id = _stable_id("HaluEval-QA", str(dataset_config["split_id"]), source_row_id, question, context or "")
        pair_id = f"{prompt_id}:pair"
        prompt_group = PromptGroup(
            dataset="HaluEval-QA",
            split_id=str(dataset_config["split_id"]),
            source_row_id=source_row_id,
            prompt_id=prompt_id,
            pair_id=pair_id,
            question=question,
            prompt=prompt,
            context=context,
            prompt_hash=hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            label_source="halueval_annotation",
            metadata=_json_metadata_tuple(
                {
                    "source_index": source_index,
                    "hf_id": dataset_config.get("hf_id"),
                    "hf_config": dataset_config.get("config"),
                    "hf_split": dataset_config.get("split"),
                    "dataset_id": dataset_config.get("split_id"),
                    "role": dataset_config.get("role"),
                    "raw_keys": sorted(raw_row.keys()),
                    "paired_answer_available": True,
                    "right_answer": _require_raw_non_empty_text(raw_row.get("right_answer"), field_name="right_answer"),
                    "hallucinated_answer": _require_raw_non_empty_text(raw_row.get("hallucinated_answer"), field_name="hallucinated_answer"),
                }
            ),
        )
        return prompt_group, (
            CandidateRow(
                prompt_id=prompt_group.prompt_id,
                candidate_id=f"{prompt_group.pair_id}:right",
                pair_id=prompt_group.pair_id,
                source_row_id=prompt_group.source_row_id,
                dataset=prompt_group.dataset,
                split_id=prompt_group.split_id,
                candidate_text=_require_raw_non_empty_text(raw_row.get("right_answer"), field_name="right_answer"),
                candidate_role="right",
                is_correct=True,
                label_source="halueval_annotation",
                question=prompt_group.question,
                prompt=prompt_group.prompt,
                context=prompt_group.context,
                metadata=_json_metadata_tuple({"source_index": source_index, "dataset_id": dataset_config.get("split_id")}),
            ),
            CandidateRow(
                prompt_id=prompt_group.prompt_id,
                candidate_id=f"{prompt_group.pair_id}:hallucinated",
                pair_id=prompt_group.pair_id,
                source_row_id=prompt_group.source_row_id,
                dataset=prompt_group.dataset,
                split_id=prompt_group.split_id,
                candidate_text=_require_raw_non_empty_text(raw_row.get("hallucinated_answer"), field_name="hallucinated_answer"),
                candidate_role="hallucinated",
                is_correct=False,
                label_source="halueval_annotation",
                question=prompt_group.question,
                prompt=prompt_group.prompt,
                context=prompt_group.context,
                metadata=_json_metadata_tuple({"source_index": source_index, "dataset_id": dataset_config.get("split_id")}),
            ),
        )

    def _materialization_report(
        self,
        dataset_config: dict[str, Any],
        *,
        dataset: Any,
        hf_id: str,
        hf_config: Any,
        split: str,
        seed: int,
        target_count: int,
        available_count: int,
        selected_indices: list[int],
        materialized_source_rows: list[int],
        skipped: list[dict[str, Any]],
        prompt_group_count: int,
        candidate_row_count: int,
    ) -> dict[str, Any]:
        return {
            "dataset": _dataset_name(dataset_config),
            "dataset_id": dataset_config.get("split_id", _dataset_name(dataset_config)),
            "hf_id": hf_id,
            "hf_config": hf_config,
            "split": split,
            "split_id": dataset_config["split_id"],
            "seed": seed,
            "target_count": target_count,
            "available_count": available_count,
            "selected_source_row_count": len(selected_indices),
            "selected_source_rows": selected_indices,
            "materialized_source_row_count": len(materialized_source_rows),
            "materialized_source_rows": materialized_source_rows,
            "prompt_group_count": prompt_group_count,
            "candidate_row_count": candidate_row_count,
            "skipped_rows": skipped,
            "features": list(getattr(dataset, "features", {}).keys()),
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "cache_files": getattr(dataset, "cache_files", []),
        }


def prompt_group_to_json(prompt_group: PromptGroup) -> dict[str, Any]:
    payload = asdict(prompt_group)
    payload["metadata"] = metadata_dict(prompt_group)
    payload["dataset_id"] = payload["metadata"].get("dataset_id", prompt_group.split_id)
    return payload


def candidate_row_to_json(candidate_row: CandidateRow) -> dict[str, Any]:
    payload = asdict(candidate_row)
    payload["metadata"] = metadata_dict(candidate_row)
    payload["dataset_id"] = payload["metadata"].get("dataset_id", candidate_row.split_id)
    return payload


def prompt_row_to_json(row: PromptRow) -> dict[str, Any]:
    payload = asdict(row)
    metadata = metadata_dict(row)
    payload["metadata"] = metadata
    payload["dataset_id"] = metadata.get("dataset_id", row.split_id)
    gold_answers = metadata.get("correct_answers") or metadata.get("answers") or metadata.get("right_answer") or metadata.get("answer") or metadata.get("ideal_answer")
    payload["gold_answers"] = gold_answers if isinstance(gold_answers, list) else ([] if gold_answers is None else [gold_answers])
    return payload
