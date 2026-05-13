"""Self-contained local generation adapter with split free-sampling and teacher-forced scoring."""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from experiments.domain import (
    CandidateRow,
    PromptGroup,
    PromptRow,
    TeacherForcedCandidateScore,
    TeacherForcedTokenScore,
)
from experiments.ports import ModelLogitsPort
from experiments.scripts.stage_control import (
    GENERATION_CANDIDATE_SCORE_SCHEMA_VERSION,
    GENERATION_FREE_SAMPLE_SCHEMA_VERSION,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMULA_MANIFEST_REF = "experiments/literature/formula_notes.md"
DEFAULT_DATASET_MANIFEST_REF = "experiments/configs/datasets_se.yaml"
DEFAULT_LOGITS_SCHEMA_VERSION = "generation_logits_v1"
FREE_SAMPLE_COUNT = 10
FULL_LOGITS_DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
}
DEFAULT_DISK_RESERVE_GIB = 100.0
DEFAULT_ANSWER_ONLY_PROMPT_SUFFIX = "Answer:"
DEFAULT_ANSWER_ONLY_FORBIDDEN_PATTERNS = (
    r"\b(?:steam|stream)\s+of\s+consciousness\b",
    r"\bstep\s*1\b",
    r"\bstep[- ]by[- ]step\b",
    r"\blet['’]?s\s+(?:think|do|give|answer|work)\b",
    r"\bto answer (?:the|this)\b",
    r"\bavailable choices\b",
    r"\bsingle-select problem\b",
    r"\bwhich one of the following\b",
)


class GenerationConfigError(RuntimeError):
    """Raised when the generation configuration or input rows are invalid."""


class GenerationDependencyError(RuntimeError):
    """Raised when optional ML dependencies are unavailable."""


class ModelLoadError(RuntimeError):
    """Raised when the configured model/tokenizer cannot be loaded."""


class GenerationValidationError(RuntimeError):
    """Raised when a generation artifact violates the required schema."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    write_json_atomic(path, payload)


def _atomic_write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        write_json(temp_path, payload)
        return temp_path
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_json_path(out_path: str | Path) -> Path:
    path = Path(out_path)
    if path.suffix:
        return path
    return path.with_suffix(".json")


def _config_generation_section(config: dict[str, Any]) -> dict[str, Any]:
    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise GenerationConfigError("generation config is missing the top-level 'generation' object")
    return generation


def _answer_only_policy(generation_config: dict[str, Any]) -> dict[str, Any]:
    raw_policy = generation_config.get("answer_only")
    if not isinstance(raw_policy, dict):
        return {"enabled": False}
    policy = dict(raw_policy)
    policy["enabled"] = bool(policy.get("enabled", False))
    return policy


def _answer_only_enabled(generation_config: dict[str, Any]) -> bool:
    return bool(_answer_only_policy(generation_config).get("enabled", False))


def _answer_only_prompt_suffix(generation_config: dict[str, Any]) -> str:
    policy = _answer_only_policy(generation_config)
    suffix = str(policy.get("prompt_suffix") or DEFAULT_ANSWER_ONLY_PROMPT_SUFFIX)
    if not suffix:
        raise GenerationConfigError("generation.answer_only.prompt_suffix must be non-empty when answer_only is enabled")
    return suffix


def _answer_only_forbidden_patterns(generation_config: dict[str, Any]) -> tuple[str, ...]:
    policy = _answer_only_policy(generation_config)
    raw_patterns = policy.get("forbidden_patterns")
    if raw_patterns is None:
        return DEFAULT_ANSWER_ONLY_FORBIDDEN_PATTERNS
    if not isinstance(raw_patterns, list) or not all(isinstance(item, str) and item for item in raw_patterns):
        raise GenerationConfigError("generation.answer_only.forbidden_patterns must be a list of non-empty regex strings")
    return tuple(raw_patterns)


def _answer_only_stop_on_newline(generation_config: dict[str, Any]) -> bool:
    policy = _answer_only_policy(generation_config)
    return bool(policy.get("stop_on_newline", True))


def _answer_only_stop_on_punctuation(generation_config: dict[str, Any]) -> bool:
    policy = _answer_only_policy(generation_config)
    return bool(policy.get("stop_on_punctuation", True))


def _answer_only_fail_on_max_new_tokens(generation_config: dict[str, Any]) -> bool:
    policy = _answer_only_policy(generation_config)
    return bool(policy.get("fail_on_max_new_tokens", True))


def _answer_only_max_answer_tokens(generation_config: dict[str, Any]) -> int | None:
    policy = _answer_only_policy(generation_config)
    raw_value = policy.get("max_answer_tokens")
    if raw_value is None:
        return None
    if not isinstance(raw_value, int) or raw_value <= 0:
        raise GenerationConfigError("generation.answer_only.max_answer_tokens must be a positive integer when provided")
    return raw_value


def _answer_only_max_invalid_attempts(generation_config: dict[str, Any]) -> int:
    policy = _answer_only_policy(generation_config)
    raw_value = policy.get("max_invalid_attempts", 1)
    if not isinstance(raw_value, int) or raw_value <= 0:
        raise GenerationConfigError("generation.answer_only.max_invalid_attempts must be a positive integer")
    return raw_value


def _load_generation_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if not isinstance(config, dict):
        raise GenerationConfigError(f"generation config must decode to an object: {path}")
    return config


def _seed_random_generators(seed: int) -> None:
    random.seed(seed)


def _resolve_runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency branch
        raise GenerationDependencyError(
            "Missing optional dependency 'torch'. Install the repo-managed generation stack with "
            "`uv sync --group generation` before running repo-owned generation."
        ) from exc

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency branch
        raise GenerationDependencyError(
            "Missing optional dependency 'transformers'. Install the repo-managed generation stack with "
            "`uv sync --group generation` before running repo-owned generation."
        ) from exc

    return torch, AutoModelForCausalLM, AutoTokenizer


def _resolve_device(torch_module: Any, requested_device: str | None) -> Any:
    device_name = requested_device or "cpu"
    if device_name.startswith("cuda") and not torch_module.cuda.is_available():
        raise GenerationDependencyError(
            f"Generation requested device {device_name!r}, but CUDA is unavailable. "
            "Set runtime.device to 'cpu' or install a CUDA-enabled PyTorch build."
        )
    return torch_module.device(device_name)


def _token_to_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def _metadata_to_tuples(metadata: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not isinstance(metadata, dict):
        return ()
    entries: list[tuple[str, str]] = []
    for key in sorted(metadata):
        entries.append((str(key), json.dumps(metadata[key], ensure_ascii=False, sort_keys=True)))
    return tuple(entries)


def _metadata_dict(metadata: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in metadata:
        try:
            payload[key] = json.loads(value)
        except json.JSONDecodeError:
            payload[key] = value
    return payload


def _logsumexp_python(values: list[float]) -> float:
    if not values:
        return float("nan")
    max_value = max(values)
    return float(max_value + math.log(sum(math.exp(item - max_value) for item in values)))


def _answer_span_text(tokenizer: Any, token_ids: list[int]) -> str:
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded.split("\n", 1)[0].strip()


def _contains_answer_terminal_punctuation(token_text: str) -> bool:
    return any(mark in token_text for mark in (".", "?", "!", "。", "؟", "！", "？"))


def _answer_only_protocol_row(
    *,
    generation_config: dict[str, Any],
    finish_reason: str,
    raw_response_text: str,
) -> dict[str, Any]:
    return {
        "enabled": _answer_only_enabled(generation_config),
        "prompt_suffix": _answer_only_prompt_suffix(generation_config),
        "stop_on_newline": _answer_only_stop_on_newline(generation_config),
        "stop_on_punctuation": _answer_only_stop_on_punctuation(generation_config),
        "fail_on_max_new_tokens": _answer_only_fail_on_max_new_tokens(generation_config),
        "max_answer_tokens": _answer_only_max_answer_tokens(generation_config),
        "max_invalid_attempts": _answer_only_max_invalid_attempts(generation_config),
        "forbidden_patterns": list(_answer_only_forbidden_patterns(generation_config)),
        "finish_reason": finish_reason,
        "raw_response_text": raw_response_text,
    }


def _answer_only_validation_problems(sample: dict[str, Any], generation_config: dict[str, Any], *, label: str) -> list[str]:
    if not _answer_only_enabled(generation_config):
        return []
    problems: list[str] = []
    prompt = sample.get("prompt")
    response_text = sample.get("response_text")
    protocol = sample.get("answer_only_protocol")
    if not isinstance(prompt, str) or not prompt.rstrip().endswith(_answer_only_prompt_suffix(generation_config)):
        problems.append(f"{label} prompt must end with {_answer_only_prompt_suffix(generation_config)!r}")
    if not isinstance(response_text, str) or not response_text.strip():
        problems.append(f"{label} answer-only response_text must be non-empty")
    elif "\n" in response_text:
        problems.append(f"{label} answer-only response_text must not contain newlines")
    if not isinstance(protocol, dict) or protocol.get("enabled") is not True:
        problems.append(f"{label} must include answer_only_protocol.enabled=true")
        return problems
    finish_reason = protocol.get("finish_reason")
    if finish_reason not in {"eos", "newline", "punctuation", "max_new_tokens"}:
        problems.append(f"{label} answer_only_protocol.finish_reason is invalid: {finish_reason!r}")
    if _answer_only_fail_on_max_new_tokens(generation_config) and finish_reason == "max_new_tokens":
        problems.append(f"{label} reached max_new_tokens under answer-only protocol")
    max_answer_tokens = _answer_only_max_answer_tokens(generation_config)
    generated_token_ids = sample.get("generated_token_ids")
    if max_answer_tokens is not None and isinstance(generated_token_ids, list) and len(generated_token_ids) > max_answer_tokens:
        problems.append(
            f"{label} answer span has {len(generated_token_ids)} generated tokens; max_answer_tokens={max_answer_tokens}"
        )
    raw_response_text = protocol.get("raw_response_text")
    if not isinstance(raw_response_text, str):
        problems.append(f"{label} answer_only_protocol.raw_response_text must be a string")
    for pattern in _answer_only_forbidden_patterns(generation_config):
        if isinstance(response_text, str) and re.search(pattern, response_text, flags=re.IGNORECASE):
            problems.append(f"{label} answer-only response_text matched forbidden pattern {pattern!r}")
        if isinstance(raw_response_text, str) and re.search(pattern, raw_response_text, flags=re.IGNORECASE):
            problems.append(f"{label} raw_response_text matched forbidden pattern {pattern!r}")
    return problems


def _required_text(entry: dict[str, Any], field_name: str, *, label: str) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise GenerationConfigError(f"{label} must define a non-empty {field_name!r}")
    return value


def _optional_text(entry: dict[str, Any], field_name: str) -> str | None:
    value = entry.get(field_name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_bool(entry: dict[str, Any], field_name: str, *, label: str) -> bool:
    value = entry.get(field_name)
    if not isinstance(value, bool):
        raise GenerationConfigError(f"{label} must define boolean {field_name!r}")
    return value


def _load_json_records(path: Path, *, label: str, collection_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        raise GenerationConfigError(f"{label} path does not exist: {path}")
    if path.suffix == ".jsonl":
        jsonl_rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise GenerationConfigError(f"{label} JSONL line {line_number} must be an object")
                jsonl_rows.append(payload)
        return jsonl_rows

    payload = load_json(path)
    if isinstance(payload, list):
        rows: list[Any] | None = payload
    elif isinstance(payload, dict):
        rows = None
        for key in collection_keys:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise GenerationConfigError(f"{label} file {path} must contain one of {collection_keys!r}")
    else:
        raise GenerationConfigError(f"{label} file {path} must decode to a list or object")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise GenerationConfigError(f"{label} row {index} in {path} must be an object")
        normalized.append(row)
    return normalized


def _normalize_prompt_entry(entry: dict[str, Any], index: int) -> PromptRow:
    dataset = str(entry.get("dataset") or "unknown_dataset")
    split_id = str(entry.get("split_id") or "unknown_split")
    sample_id = str(entry.get("sample_id") or f"sample_{index:04d}")
    prompt_value = entry.get("prompt")
    question = entry.get("question")
    context = entry.get("context")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        question_text = question.strip() if isinstance(question, str) else ""
        context_text = context.strip() if isinstance(context, str) else ""
        if question_text:
            if context_text:
                prompt_value = (
                    f"Context: {context_text}\nQuestion: {question_text}\n"
                    "Return only the shortest final answer span. Do not explain or write a full sentence.\nAnswer:"
                )
            else:
                prompt_value = (
                    f"Question: {question_text}\n"
                    "Return only the shortest final answer span. Do not explain or write a full sentence.\nAnswer:"
                )
        else:
            raise GenerationConfigError(
                f"prompt row {sample_id!r} must define a non-empty 'prompt' or at least a 'question' field"
            )
    return PromptRow(
        dataset=dataset,
        split_id=split_id,
        sample_id=sample_id,
        prompt=prompt_value,
        question=question if isinstance(question, str) else None,
        context=context if isinstance(context, str) else None,
        metadata=_metadata_to_tuples(entry.get("metadata") if isinstance(entry.get("metadata"), dict) else None),
    )


def _load_prompt_rows_from_path(path: Path) -> tuple[PromptRow, ...]:
    raw_rows = _load_json_records(path, label="prompt rows", collection_keys=("prompt_rows", "samples"))
    return tuple(_normalize_prompt_entry(entry, index) for index, entry in enumerate(raw_rows))


def load_prompt_rows(config: dict[str, Any], prompt_rows_path: str | None = None) -> tuple[PromptRow, ...]:
    override_path = Path(prompt_rows_path).resolve() if prompt_rows_path else None
    if override_path is not None:
        rows = _load_prompt_rows_from_path(override_path)
    else:
        configured_path = config.get("prompt_rows_path")
        if isinstance(configured_path, str) and configured_path.strip():
            rows = _load_prompt_rows_from_path((ROOT / configured_path).resolve())
        else:
            prompt_rows = config.get("prompt_rows")
            if not isinstance(prompt_rows, list) or not prompt_rows:
                raise GenerationConfigError(
                    "generation config must define a non-empty 'prompt_rows' list or a 'prompt_rows_path'"
                )
            rows = tuple(_normalize_prompt_entry(entry, index) for index, entry in enumerate(prompt_rows) if isinstance(entry, dict))
    if not rows:
        raise GenerationConfigError("no prompt rows were loaded")
    return rows


def _normalize_prompt_group_entry(entry: dict[str, Any], index: int) -> PromptGroup:
    prompt = _required_text(entry, "prompt", label=f"prompt group {index}")
    prompt_id = _required_text(entry, "prompt_id", label=f"prompt group {index}")
    question = _required_text(entry, "question", label=f"prompt group {index}")
    dataset = str(entry.get("dataset") or "unknown_dataset")
    split_id = str(entry.get("split_id") or "unknown_split")
    source_row_id = str(entry.get("source_row_id") or prompt_id)
    pair_id = str(entry.get("pair_id") or f"{prompt_id}:pair")
    prompt_hash = str(entry.get("prompt_hash") or sha1(prompt.encode("utf-8")).hexdigest())
    label_source = str(entry.get("label_source") or "dataset_annotation")
    return PromptGroup(
        dataset=dataset,
        split_id=split_id,
        source_row_id=source_row_id,
        prompt_id=prompt_id,
        pair_id=pair_id,
        question=question,
        prompt=prompt,
        context=_optional_text(entry, "context"),
        prompt_hash=prompt_hash,
        label_source=label_source,
        metadata=_metadata_to_tuples(entry.get("metadata") if isinstance(entry.get("metadata"), dict) else None),
    )


def load_prompt_groups(path: str | Path) -> tuple[PromptGroup, ...]:
    rows = _load_json_records(Path(path).resolve(), label="prompt groups", collection_keys=("prompt_groups", "rows"))
    prompt_groups = tuple(_normalize_prompt_group_entry(entry, index) for index, entry in enumerate(rows))
    if not prompt_groups:
        raise GenerationConfigError("no prompt groups were loaded")
    return prompt_groups


def _normalize_candidate_entry(entry: dict[str, Any], index: int) -> CandidateRow:
    return CandidateRow(
        prompt_id=_required_text(entry, "prompt_id", label=f"candidate row {index}"),
        candidate_id=_required_text(entry, "candidate_id", label=f"candidate row {index}"),
        pair_id=_required_text(entry, "pair_id", label=f"candidate row {index}"),
        source_row_id=_required_text(entry, "source_row_id", label=f"candidate row {index}"),
        dataset=_required_text(entry, "dataset", label=f"candidate row {index}"),
        split_id=_required_text(entry, "split_id", label=f"candidate row {index}"),
        candidate_text=_required_text(entry, "candidate_text", label=f"candidate row {index}"),
        candidate_role=_required_text(entry, "candidate_role", label=f"candidate row {index}"),
        is_correct=_required_bool(entry, "is_correct", label=f"candidate row {index}"),
        label_source=_required_text(entry, "label_source", label=f"candidate row {index}"),
        question=_required_text(entry, "question", label=f"candidate row {index}"),
        prompt=_required_text(entry, "prompt", label=f"candidate row {index}"),
        context=_optional_text(entry, "context"),
        metadata=_metadata_to_tuples(entry.get("metadata") if isinstance(entry.get("metadata"), dict) else None),
    )


def load_candidate_rows(path: str | Path) -> tuple[CandidateRow, ...]:
    rows = _load_json_records(Path(path).resolve(), label="candidate rows", collection_keys=("candidate_rows", "rows"))
    candidates = tuple(_normalize_candidate_entry(entry, index) for index, entry in enumerate(rows))
    if not candidates:
        raise GenerationConfigError("no candidate rows were loaded")
    return candidates


def _existing_disk_path(path: Path) -> Path:
    probe = path.parent if path.suffix else path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe


def _runtime_budget_bytes(config: dict[str, Any]) -> int | None:
    runtime_config = _runtime_config(config)
    raw_budget = runtime_config.get("max_full_logits_parquet_gib", runtime_config.get("max_full_logits_json_gib"))
    if raw_budget is None:
        return None
    if not isinstance(raw_budget, (int, float)) or isinstance(raw_budget, bool) or raw_budget <= 0:
        raise GenerationConfigError("runtime.max_full_logits_parquet_gib must be a positive number when provided")
    return int(float(raw_budget) * (1024**3))


def _runtime_disk_reserve_bytes(config: dict[str, Any]) -> int:
    runtime_config = _runtime_config(config)
    raw_reserve = runtime_config.get("min_full_logits_disk_reserve_gib", DEFAULT_DISK_RESERVE_GIB)
    if not isinstance(raw_reserve, (int, float)) or isinstance(raw_reserve, bool) or raw_reserve < 0:
        raise GenerationConfigError("runtime.min_full_logits_disk_reserve_gib must be a non-negative number when provided")
    return int(float(raw_reserve) * (1024**3))


def _runtime_batch_size(config: dict[str, Any], field_name: str) -> int:
    runtime_config = _runtime_config(config)
    raw_value = runtime_config.get(field_name, 1)
    if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value <= 0:
        raise GenerationConfigError(f"runtime.{field_name} must be a positive integer")
    return raw_value


def _runtime_full_logits_dtype(config: dict[str, Any]) -> str:
    runtime_config = _runtime_config(config)
    raw_dtype = runtime_config.get("full_logits_dtype", "float32")
    if not isinstance(raw_dtype, str) or raw_dtype not in FULL_LOGITS_DTYPE_BYTES:
        raise GenerationConfigError(
            f"runtime.full_logits_dtype must be one of {sorted(FULL_LOGITS_DTYPE_BYTES)}; got {raw_dtype!r}"
        )
    return raw_dtype


def _full_logits_bytes_per_value(config: dict[str, Any]) -> int:
    return FULL_LOGITS_DTYPE_BYTES[_runtime_full_logits_dtype(config)]


def _free_sample_token_limit(config: dict[str, Any]) -> int:
    generation_config = _config_generation_section(config)
    max_new_tokens = int(generation_config.get("max_new_tokens", 1) or 1)
    if _answer_only_enabled(generation_config):
        max_answer_tokens = _answer_only_max_answer_tokens(generation_config)
        if max_answer_tokens is not None:
            return min(max_new_tokens, max_answer_tokens)
    return max_new_tokens


def _chunks(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _estimate_full_logits_bytes(*, token_positions: int, vocab_size: int, bytes_per_value: int) -> int:
    return int(token_positions * vocab_size * bytes_per_value)


def _enforce_full_logits_budget(
    *,
    config: dict[str, Any],
    out_path: str | Path,
    artifact_type: str,
    token_positions: int,
    vocab_size: int,
) -> None:
    storage_dtype = _runtime_full_logits_dtype(config)
    estimated_bytes = _estimate_full_logits_bytes(
        token_positions=token_positions,
        vocab_size=vocab_size,
        bytes_per_value=_full_logits_bytes_per_value(config),
    )
    disk_path = _existing_disk_path(Path(out_path))
    free_bytes = shutil.disk_usage(disk_path).free
    configured_budget = _runtime_budget_bytes(config)
    disk_reserve = _runtime_disk_reserve_bytes(config)
    disk_budget = max(0, free_bytes - disk_reserve)
    effective_budget = min(disk_budget, configured_budget) if configured_budget is not None else disk_budget
    if estimated_bytes <= effective_budget:
        return
    estimated_gib = estimated_bytes / (1024**3)
    free_gib = free_bytes / (1024**3)
    reserve_gib = disk_reserve / (1024**3)
    disk_budget_gib = disk_budget / (1024**3)
    configured_note = f", configured budget={configured_budget / (1024**3):.1f} GiB" if configured_budget is not None else ""
    raise GenerationConfigError(
        f"Refusing to write {artifact_type} because full-vocabulary logits parquet shards are estimated at {estimated_gib:.1f} GiB "
        f"({token_positions} token positions x vocab {vocab_size}) but {disk_path} has {free_gib:.1f} GiB free"
        f", reserve={reserve_gib:.1f} GiB, disk budget after reserve={disk_budget_gib:.1f} GiB"
        f"{configured_note}, sidecar dtype={storage_dtype}. Qwen2.5 is supported, but thesis-valid full logits need more disk capacity "
        "or a smaller preflight subset before a full execute run."
    )


def _parquet_path_for_artifact(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".full_logits.parquet")


def checkpoint_root_for_artifact(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".checkpoint")


def _phase_checkpoint_root(path: Path, *, artifact_type: str) -> Path:
    if artifact_type == "free_sample_rows":
        return checkpoint_root_for_artifact(path) / "free_sample_rows"
    if artifact_type == "teacher_forced_candidate_scores":
        return checkpoint_root_for_artifact(path) / "candidate_scores"
    raise GenerationValidationError(f"unsupported checkpoint artifact_type {artifact_type!r}")


def _safe_key_part(value: str) -> str:
    digest = sha1(value.encode("utf-8")).hexdigest()[:16]
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    return f"{cleaned[:48]}-{digest}"


def _json_fingerprint(payload: object) -> str:
    return sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _rows_fingerprint(rows: tuple[Any, ...]) -> str:
    return _json_fingerprint([asdict(row) for row in rows])


def _checkpoint_metadata(
    *,
    config: dict[str, Any],
    runtime: "_LiveModelRuntime",
    artifact_type: str,
    input_fingerprint: str,
) -> dict[str, Any]:
    generation_config = dict(runtime.generation_config)
    if artifact_type == "free_sample_rows":
        generation_config["do_sample"] = True
    schema_version = (
        GENERATION_FREE_SAMPLE_SCHEMA_VERSION
        if artifact_type == "free_sample_rows"
        else GENERATION_CANDIDATE_SCORE_SCHEMA_VERSION
        if artifact_type == "teacher_forced_candidate_scores"
        else None
    )
    return {
        "checkpoint_schema_version": "generation_checkpoint_v1",
        "schema_version": schema_version,
        "artifact_type": artifact_type,
        "model_name": runtime.model_name,
        "tokenizer_name": runtime.tokenizer_name,
        "generation_config": generation_config,
        "logits_schema_version": str(config.get("logits_schema_version") or DEFAULT_LOGITS_SCHEMA_VERSION),
        "input_fingerprint": input_fingerprint,
    }


def _metadata_matches(payload: dict[str, Any], metadata: dict[str, Any]) -> bool:
    for key, expected_value in metadata.items():
        if payload.get(key) != expected_value:
            return False
    return True


# Fields that materially determine the SEMANTICS of stored samples (which model,
# which tokenizer, which schema, which input set). Mismatches here mean the
# checkpoint shard was produced under a different data contract and must not be
# reused as-is. Other generation-config knobs (batch sizes via runtime, length
# caps, retry budgets) only constrain how NEW samples can look; existing valid
# shards stay valid even if those knobs changed.
ESSENTIAL_METADATA_FIELDS: tuple[str, ...] = (
    "checkpoint_schema_version",
    "schema_version",
    "artifact_type",
    "model_name",
    "tokenizer_name",
    "logits_schema_version",
    "input_fingerprint",
)


def _is_metadata_compatible(payload: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Return True if the shard payload matches the run's essential metadata.

    Lets callers preserve previously generated checkpoint shards across runtime/
    generation-config tweaks (e.g. batch size or length cap changes) while still
    rejecting shards produced under a different model, tokenizer, or schema.
    """
    for key in ESSENTIAL_METADATA_FIELDS:
        if key in metadata and payload.get(key) != metadata[key]:
            return False
    return True


def _sample_seed(config: dict[str, Any], *, prompt_id: str, sample_index: int) -> int:
    base_seed = int(_config_generation_section(config).get("seed", 13))
    digest = sha1(f"{base_seed}:{prompt_id}:{sample_index}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _atomic_promote_dir(temp_dir: Path, complete_dir: Path) -> None:
    if complete_dir.exists():
        shutil.rmtree(complete_dir)
    temp_dir.replace(complete_dir)


def _copy_parquet_rows(source_path: Path, writer: "_FullLogitsParquetWriter") -> None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        raise GenerationDependencyError(
            "Missing optional dependency 'pyarrow'. Install with `uv sync --group generation` before merging checkpoint shards."
        ) from exc
    parquet_file = pq.ParquetFile(source_path)
    for batch in parquet_file.iter_batches(batch_size=16):
        rows = batch.to_pylist()
        if not all(isinstance(row, dict) for row in rows):
            raise GenerationValidationError(f"checkpoint parquet rows must decode to objects: {source_path}")
        writer.write_rows(rows)


def _rewrite_free_sample_ref(sample: dict[str, Any], parquet_path: Path) -> dict[str, Any]:
    rewritten = dict(sample)
    rewritten["full_logits_ref"] = {
        "format": "parquet",
        "path": str(parquet_path),
        "key_fields": ["prompt_id", "sample_index", "token_offset"],
    }
    return rewritten


def _rewrite_candidate_token_ref(token_row: dict[str, Any], parquet_path: Path) -> dict[str, Any]:
    rewritten = dict(token_row)
    rewritten["full_logits_ref"] = {
        "format": "parquet",
        "path": str(parquet_path),
        "key_fields": ["candidate_id", "candidate_token_offset"],
    }
    return rewritten


def _validate_final_sidecar_is_same_stem(payload: dict[str, Any], source_path: Path) -> None:
    storage = payload.get("full_logits_storage")
    if not isinstance(storage, dict) or storage.get("format") != "parquet":
        return
    resolved_sidecar = _resolve_sidecar_path(source_path, storage.get("path"))
    expected_sidecar = _parquet_path_for_artifact(source_path)
    if resolved_sidecar is None or resolved_sidecar.resolve() != expected_sidecar.resolve():
        raise GenerationValidationError(
            f"final full_logits_storage.path must point to the same-stem sidecar {expected_sidecar}; got {storage.get('path')!r}"
        )


def _write_validated_generation_artifact(path: Path, payload: dict[str, Any]) -> None:
    temp_path = _atomic_write_json(path, payload)
    try:
        _validate_generation_artifact_payload(payload, path)
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _resolve_sidecar_path(source_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return source_path.parent / path


def _resolve_checkpoint_sidecar_path(source_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute():
        raise GenerationValidationError(f"checkpoint sidecar path must be relative to the shard directory: {raw_path!r}")
    resolved = (source_path.parent / path).resolve()
    shard_root = source_path.parent.resolve()
    try:
        resolved.relative_to(shard_root)
    except ValueError as exc:
        raise GenerationValidationError(f"checkpoint sidecar path escapes the shard directory: {raw_path!r}") from exc
    return resolved


def _validate_full_logits_sidecar(
    *,
    payload: dict[str, Any],
    source_path: Path,
    artifact_type: str,
    expected_keys: set[tuple[Any, ...]],
    key_fields: tuple[str, ...],
    checkpoint_sidecar: bool = False,
) -> None:
    storage = payload.get("full_logits_storage")
    if not isinstance(storage, dict) or storage.get("format") != "parquet":
        raise GenerationValidationError("full_logits_storage must describe a parquet sidecar when full_logits_ref is used")
    sidecar_path = (
        _resolve_checkpoint_sidecar_path(source_path, storage.get("path"))
        if checkpoint_sidecar
        else _resolve_sidecar_path(source_path, storage.get("path"))
    )
    if sidecar_path is None or not sidecar_path.exists():
        raise GenerationValidationError(f"full logits parquet sidecar is missing: {storage.get('path')!r}")
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        raise GenerationValidationError(
            "Missing optional dependency 'pyarrow'. Install with `uv sync --group generation` before validating full-logits sidecars."
        ) from exc

    try:
        parquet_file = pq.ParquetFile(sidecar_path)
    except Exception as exc:
        raise GenerationValidationError(f"full logits parquet sidecar is unreadable: {sidecar_path}: {exc}") from exc

    metadata = parquet_file.metadata
    if metadata is None:
        raise GenerationValidationError(f"full logits parquet sidecar has no metadata: {sidecar_path}")
    actual_row_count = int(metadata.num_rows)
    expected_row_count = len(expected_keys)
    storage_row_count = storage.get("row_count")
    if isinstance(storage_row_count, int) and storage_row_count != actual_row_count:
        raise GenerationValidationError(
            f"full_logits_storage.row_count={storage_row_count} does not match parquet row count {actual_row_count}"
        )
    if actual_row_count != expected_row_count:
        raise GenerationValidationError(
            f"{artifact_type} sidecar row count {actual_row_count} does not match expected token row count {expected_row_count}"
        )

    storage_vector_size = storage.get("vector_size")
    if not isinstance(storage_vector_size, int) or storage_vector_size <= 0:
        raise GenerationValidationError("full_logits_storage.vector_size must be a positive integer")
    storage_dtype = storage.get("dtype")
    if not isinstance(storage_dtype, str) or storage_dtype not in FULL_LOGITS_DTYPE_BYTES:
        raise GenerationValidationError(
            f"full_logits_storage.dtype must be one of {sorted(FULL_LOGITS_DTYPE_BYTES)}; got {storage_dtype!r}"
        )

    required_columns = tuple(key_fields) + ("full_logits",)
    parquet_columns = set(parquet_file.schema_arrow.names)
    missing_columns = set(required_columns) - parquet_columns
    if missing_columns:
        raise GenerationValidationError(
            f"full logits parquet sidecar is missing required columns: {sorted(missing_columns)}"
        )
    full_logits_field = parquet_file.schema_arrow.field("full_logits")
    full_logits_type = full_logits_field.type
    if pa.types.is_fixed_size_list(full_logits_type) or pa.types.is_list(full_logits_type) or pa.types.is_large_list(full_logits_type):
        value_type = full_logits_type.value_type
    else:
        raise GenerationValidationError(f"full logits parquet full_logits column must be a list type; got {full_logits_type}")
    expected_value_type = pa.float16() if storage_dtype == "float16" else pa.float32()
    if value_type != expected_value_type:
        raise GenerationValidationError(
            f"full logits parquet value dtype {value_type} does not match full_logits_storage.dtype={storage_dtype!r}"
        )

    observed_keys: set[tuple[Any, ...]] = set()
    # Final live sidecars can be tens or hundreds of GiB because each row carries
    # a full-vocabulary logits vector.  Integrity here is a structural check:
    # metadata row count, required schema columns, duplicate/missing/unexpected
    # token keys, and writer-recorded vector_size.  Decoding full_logits into
    # Python lists during final validation turns atomic JSON promotion into an
    # O(rows * vocab) CPU/IO bottleneck, so validation reads only key columns.
    validation_columns = tuple(key_fields)
    for batch in parquet_file.iter_batches(batch_size=256, columns=list(validation_columns)):
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                raise GenerationValidationError("full logits parquet rows must decode to objects")
            key = tuple(row.get(field_name) for field_name in key_fields)
            if key not in expected_keys:
                raise GenerationValidationError(f"full logits parquet contains unexpected key {key!r}")
            if key in observed_keys:
                raise GenerationValidationError(f"full logits parquet duplicates key {key!r}")
            observed_keys.add(key)

    missing_keys = expected_keys - observed_keys
    if missing_keys:
        preview = sorted(repr(key) for key in missing_keys)[:5]
        raise GenerationValidationError(f"full logits parquet is missing {len(missing_keys)} expected keys: {preview}")


class _FullLogitsParquetWriter:
    def __init__(self, path: Path, *, artifact_type: str, storage_dtype: str) -> None:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ModuleNotFoundError as exc:
            raise GenerationDependencyError(
                "Missing optional dependency 'pyarrow'. Install with `uv sync --group generation` before writing full-logits parquet shards."
            ) from exc
        self.pa = pa
        self.pq = pq
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_type = artifact_type
        if artifact_type not in {"free_sample_rows", "teacher_forced_candidate_scores"}:
            raise GenerationValidationError(f"unsupported artifact_type for full logits parquet: {artifact_type!r}")
        self.schema: Any | None = None
        self.writer: Any | None = None
        self.row_count = 0
        self.vector_size: int | None = None
        if storage_dtype not in FULL_LOGITS_DTYPE_BYTES:
            raise GenerationValidationError(
                f"unsupported full logits storage dtype {storage_dtype!r}; expected one of {sorted(FULL_LOGITS_DTYPE_BYTES)}"
            )
        self.storage_dtype = storage_dtype

    def _value_type(self) -> Any:
        return self.pa.float16() if self.storage_dtype == "float16" else self.pa.float32()

    def _schema_for_rows(self) -> Any:
        if self.artifact_type == "free_sample_rows":
            return self.pa.schema(
                [
                    ("prompt_id", self.pa.string()),
                    ("sample_index", self.pa.int64()),
                    ("token_offset", self.pa.int64()),
                    ("full_logits", self.pa.list_(self._value_type())),
                ]
            )
        return self.pa.schema(
            [
                ("candidate_id", self.pa.string()),
                ("candidate_token_offset", self.pa.int64()),
                ("candidate_token_position", self.pa.int64()),
                ("full_logits", self.pa.list_(self._value_type())),
            ]
        )

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        for index, row in enumerate(rows):
            logits = row.get("full_logits")
            if not isinstance(logits, list):
                raise GenerationValidationError(f"full logits row {index} for {self.artifact_type} must contain a list")
            if self.vector_size is None:
                self.vector_size = len(logits)
            elif len(logits) != self.vector_size:
                raise GenerationValidationError(
                    f"full logits row {index} for {self.artifact_type} has vector length {len(logits)}; "
                    f"expected {self.vector_size}"
                )
        if self.writer is None:
            self.schema = self._schema_for_rows()
            self.writer = self.pq.ParquetWriter(self.path, self.schema, compression="zstd")
        assert self.schema is not None
        assert self.writer is not None
        table = self.pa.Table.from_pylist(rows, schema=self.schema)
        self.writer.write_table(table)
        self.row_count += len(rows)

    def close(self) -> dict[str, Any]:
        if self.writer is None:
            raise GenerationValidationError("full logits parquet writer found no logits rows to materialize")
        self.writer.close()
        if self.row_count <= 0:
            raise GenerationValidationError("full logits parquet writer found no logits rows to materialize")
        return {
            "format": "parquet",
            "path": str(self.path),
            "compression": "zstd",
            "dtype": self.storage_dtype,
            "row_count": self.row_count,
            "vector_size": self.vector_size,
        }


def _write_full_logits_parquet(path: Path, artifact: dict[str, Any]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        raise GenerationDependencyError(
            "Missing optional dependency 'pyarrow'. Install with `uv sync --group generation` before writing full-logits parquet shards."
        ) from exc

    parquet_path = _parquet_path_for_artifact(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    raw_storage = artifact.get("full_logits_storage")
    storage_dtype = raw_storage.get("dtype") if isinstance(raw_storage, dict) else "float32"
    if storage_dtype not in FULL_LOGITS_DTYPE_BYTES:
        storage_dtype = "float32"
    value_type = pa.float16() if storage_dtype == "float16" else pa.float32()
    rows: list[dict[str, Any]] = []
    artifact_type = artifact.get("artifact_type")
    if artifact_type == "free_sample_rows":
        for sample in artifact.get("samples", []):
            if not isinstance(sample, dict):
                continue
            full_logits = sample.pop("full_logits", None)
            if not isinstance(full_logits, list):
                continue
            sample["full_logits_ref"] = {
                "format": "parquet",
                "path": str(parquet_path),
                "key_fields": ["prompt_id", "sample_index", "token_offset"],
            }
            for token_offset, logits in enumerate(full_logits):
                rows.append(
                    {
                        "prompt_id": sample.get("prompt_id"),
                        "sample_index": sample.get("sample_index"),
                        "token_offset": token_offset,
                        "full_logits": [float(value) for value in logits],
                    }
                )
    elif artifact_type == "teacher_forced_candidate_scores":
        for token_row in artifact.get("token_score_rows", []):
            if not isinstance(token_row, dict):
                continue
            full_logits = token_row.pop("full_logits", None)
            if not isinstance(full_logits, list):
                continue
            token_row["full_logits_ref"] = {
                "format": "parquet",
                "path": str(parquet_path),
                "key_fields": ["candidate_id", "candidate_token_offset"],
            }
            rows.append(
                {
                    "candidate_id": token_row.get("candidate_id"),
                    "candidate_token_offset": token_row.get("candidate_token_offset"),
                    "candidate_token_position": token_row.get("candidate_token_position"),
                    "full_logits": [float(value) for value in full_logits],
                }
            )
    if not rows:
        raise GenerationValidationError("full logits parquet writer found no logits rows to materialize")
    if artifact_type == "free_sample_rows":
        schema = pa.schema(
            [
                ("prompt_id", pa.string()),
                ("sample_index", pa.int64()),
                ("token_offset", pa.int64()),
                ("full_logits", pa.list_(value_type)),
            ]
        )
    elif artifact_type == "teacher_forced_candidate_scores":
        schema = pa.schema(
            [
                ("candidate_id", pa.string()),
                ("candidate_token_offset", pa.int64()),
                ("candidate_token_position", pa.int64()),
                ("full_logits", pa.list_(value_type)),
            ]
        )
    else:
        raise GenerationValidationError(f"unsupported artifact_type for full logits parquet: {artifact_type!r}")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, parquet_path, compression="zstd")
    first_logits = rows[0].get("full_logits")
    vector_size = len(first_logits) if isinstance(first_logits, list) else None
    artifact["full_logits_storage"] = {
        "format": "parquet",
        "path": str(parquet_path),
        "compression": "zstd",
        "dtype": storage_dtype,
        "row_count": len(rows),
        "vector_size": vector_size,
    }


def _prompt_group_from_prompt_row(prompt_row: PromptRow) -> PromptGroup:
    prompt_hash = sha1(prompt_row.prompt.encode("utf-8")).hexdigest()
    return PromptGroup(
        dataset=prompt_row.dataset,
        split_id=prompt_row.split_id,
        source_row_id=prompt_row.sample_id,
        prompt_id=prompt_row.sample_id,
        pair_id=f"{prompt_row.sample_id}:pair",
        question=prompt_row.question or prompt_row.prompt,
        prompt=prompt_row.prompt,
        context=prompt_row.context,
        prompt_hash=prompt_hash,
        label_source="legacy_prompt_row",
        metadata=prompt_row.metadata,
    )


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    raw_model_config = config.get("model")
    return raw_model_config if isinstance(raw_model_config, dict) else {}


def _runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    raw_runtime_config = config.get("runtime")
    return raw_runtime_config if isinstance(raw_runtime_config, dict) else {}


def _provenance(config: dict[str, Any]) -> dict[str, Any]:
    model_config = _model_config(config)
    return {
        "model_name": str(model_config.get("model_name") or "fixture-local-model"),
        "tokenizer_name": str(model_config.get("tokenizer_name") or model_config.get("model_name") or "fixture-local-model"),
        "generation_config": _config_generation_section(config),
        "logits_schema_version": str(config.get("logits_schema_version") or DEFAULT_LOGITS_SCHEMA_VERSION),
        "formula_manifest_ref": str(config.get("formula_manifest_ref") or DEFAULT_FORMULA_MANIFEST_REF),
        "dataset_manifest_ref": str(config.get("dataset_manifest_ref") or DEFAULT_DATASET_MANIFEST_REF),
    }


def _base_artifact(config: dict[str, Any], *, artifact_type: str, fixture_mode: bool, has_full_logits: bool) -> dict[str, Any]:
    payload = _provenance(config)
    schema_version = (
        GENERATION_FREE_SAMPLE_SCHEMA_VERSION
        if artifact_type == "free_sample_rows"
        else GENERATION_CANDIDATE_SCORE_SCHEMA_VERSION
        if artifact_type == "teacher_forced_candidate_scores"
        else None
    )
    payload.update(
        {
            "run_id": f"{artifact_type}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "schema_version": schema_version,
            "artifact_type": artifact_type,
            "created_at": _now_iso(),
            "has_logits": True,
            "has_full_logits": has_full_logits,
            "full_vocabulary_logits": has_full_logits,
            "fixture_mode": fixture_mode,
        }
    )
    return payload


def _fixture_free_sample_row(
    prompt_group: PromptGroup,
    *,
    sample_index: int,
    include_full_logits: bool,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    token_names = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta")
    first_token_id = sample_index % len(token_names)
    second_token_id = (sample_index + 1) % len(token_names)
    full_logits = (
        tuple(float(6 - abs(first_token_id - vocab_id)) for vocab_id in range(len(token_names))),
        tuple(float(5 - abs(second_token_id - vocab_id)) for vocab_id in range(len(token_names))),
    )
    generated_token_ids = (first_token_id, second_token_id)
    generated_tokens = (token_names[first_token_id], token_names[second_token_id])
    row = {
        "dataset": prompt_group.dataset,
        "split_id": prompt_group.split_id,
        "prompt_id": prompt_group.prompt_id,
        "pair_id": prompt_group.pair_id,
        "prompt": prompt_group.prompt,
        "question": prompt_group.question,
        "context": prompt_group.context,
        "sample_index": sample_index,
        "response_text": " ".join(generated_tokens),
        "generated_token_ids": list(generated_token_ids),
        "selected_token_ids": list(generated_token_ids),
        "selected_token_logits": [full_logits[0][first_token_id], full_logits[1][second_token_id]],
        "generated_tokens": list(generated_tokens),
        "logsumexp": [_logsumexp_python(list(step)) for step in full_logits],
        "full_vocabulary_logits": include_full_logits,
        "source_kind": "free_sample",
        "answer_only_protocol": _answer_only_protocol_row(
            generation_config=generation_config,
            finish_reason="eos",
            raw_response_text=" ".join(generated_tokens),
        ) if _answer_only_enabled(generation_config) else {"enabled": False},
        "metadata": _metadata_dict(prompt_group.metadata) | {"fixture": True},
    }
    if include_full_logits:
        row["full_logits"] = [list(step) for step in full_logits]
    return row


def build_free_sample_fixture_artifact(
    config: dict[str, Any],
    prompt_groups: tuple[PromptGroup, ...],
    *,
    variant: str,
) -> dict[str, Any]:
    include_full_logits = variant == "full_logits"
    artifact = _base_artifact(
        config,
        artifact_type="free_sample_rows",
        fixture_mode=True,
        has_full_logits=include_full_logits,
    )
    sampling_config = dict(_config_generation_section(config))
    sampling_config["do_sample"] = True
    artifact["generation_config"] = sampling_config
    samples: list[dict[str, Any]] = []
    for prompt_group in prompt_groups:
        for sample_index in range(FREE_SAMPLE_COUNT):
            samples.append(
                _fixture_free_sample_row(
                    prompt_group,
                    sample_index=sample_index,
                    include_full_logits=include_full_logits,
                    generation_config=sampling_config,
                )
            )
    artifact.update(
        {
            "sample_count_per_prompt": FREE_SAMPLE_COUNT,
            "prompt_group_count": len(prompt_groups),
            "samples": samples,
        }
    )
    return artifact


def _fixture_candidate_token_scores(
    candidate_row: CandidateRow,
    *,
    include_full_logits: bool,
) -> TeacherForcedCandidateScore:
    prompt_prefix_token_count = max(1, len(candidate_row.prompt.split()))
    candidate_pieces = tuple(piece for piece in candidate_row.candidate_text.split() if piece) or (candidate_row.candidate_text,)
    token_scores: list[TeacherForcedTokenScore] = []
    for offset, piece in enumerate(candidate_pieces):
        absolute_position = prompt_prefix_token_count + offset
        selected_token_logit = float(1.5 + offset)
        logits_vector = tuple(float(selected_token_logit - abs(vocab_id - offset)) for vocab_id in range(4))
        token_scores.append(
            TeacherForcedTokenScore(
                prompt_id=candidate_row.prompt_id,
                candidate_id=candidate_row.candidate_id,
                candidate_token_position=absolute_position,
                token_id=offset,
                selected_token_logit=selected_token_logit,
                logsumexp=_logsumexp_python(list(logits_vector)),
                full_logits=logits_vector if include_full_logits else (),
                decoded_token=piece,
            )
        )
    selected_token_logit_sum = float(sum(score.selected_token_logit for score in token_scores))
    sequence_log_probability = float(sum(score.selected_token_logit - score.logsumexp for score in token_scores))
    candidate_token_count = len(token_scores)
    return TeacherForcedCandidateScore(
        prompt_id=candidate_row.prompt_id,
        candidate_id=candidate_row.candidate_id,
        candidate_token_count=candidate_token_count,
        candidate_token_start=prompt_prefix_token_count,
        candidate_token_end=prompt_prefix_token_count + candidate_token_count - 1,
        selected_token_logit_sum=selected_token_logit_sum,
        selected_token_logit_mean=selected_token_logit_sum / candidate_token_count,
        sequence_log_probability=sequence_log_probability,
        average_log_probability=sequence_log_probability / candidate_token_count,
        token_scores=tuple(token_scores),
    )


def _candidate_score_row(candidate_row: CandidateRow, score: TeacherForcedCandidateScore) -> dict[str, Any]:
    return {
        "dataset": candidate_row.dataset,
        "split_id": candidate_row.split_id,
        "pair_id": candidate_row.pair_id,
        "prompt_id": score.prompt_id,
        "candidate_id": score.candidate_id,
        "candidate_role": candidate_row.candidate_role,
        "candidate_text": candidate_row.candidate_text,
        "is_correct": candidate_row.is_correct,
        "label_source": candidate_row.label_source,
        "candidate_token_count": score.candidate_token_count,
        "candidate_token_start": score.candidate_token_start,
        "candidate_token_end": score.candidate_token_end,
        "selected_token_logit_sum": score.selected_token_logit_sum,
        "selected_token_logit_mean": score.selected_token_logit_mean,
        "sequence_log_probability": score.sequence_log_probability,
        "average_log_probability": score.average_log_probability,
        "metadata": _metadata_dict(candidate_row.metadata),
    }


def _token_score_row(score: TeacherForcedTokenScore, *, candidate_token_start: int) -> dict[str, Any]:
    payload = asdict(score)
    payload["candidate_token_offset"] = score.candidate_token_position - candidate_token_start
    payload["sequence_token_position"] = score.candidate_token_position
    payload["full_logits"] = list(score.full_logits)
    return payload


def build_candidate_score_fixture_artifact(
    config: dict[str, Any],
    candidate_rows: tuple[CandidateRow, ...],
    *,
    variant: str,
) -> dict[str, Any]:
    include_full_logits = variant == "full_logits"
    artifact = _base_artifact(
        config,
        artifact_type="teacher_forced_candidate_scores",
        fixture_mode=True,
        has_full_logits=include_full_logits,
    )
    candidate_score_rows: list[dict[str, Any]] = []
    token_score_rows: list[dict[str, Any]] = []
    for candidate_row in candidate_rows:
        score = _fixture_candidate_token_scores(candidate_row, include_full_logits=include_full_logits)
        candidate_score_rows.append(_candidate_score_row(candidate_row, score))
        token_score_rows.extend(
            _token_score_row(token_score, candidate_token_start=score.candidate_token_start)
            for token_score in score.token_scores
        )
    artifact.update(
        {
            "candidate_count": len(candidate_rows),
            "candidate_score_rows": candidate_score_rows,
            "token_score_rows": token_score_rows,
            "candidate_scoring_mode": "teacher_forced",
            "prompt_prefix_scoring_excluded": True,
            "free_sample_count": FREE_SAMPLE_COUNT,
        }
    )
    return artifact


def _select_token_id(torch_module: Any, step_logits: Any, generation_config: dict[str, Any]) -> int:
    temperature = float(generation_config.get("temperature", 1.0) or 1.0)
    do_sample = bool(generation_config.get("do_sample", False))
    top_k = int(generation_config.get("top_k", 0) or 0)
    if not do_sample:
        return int(torch_module.argmax(step_logits, dim=-1).item())

    logits = step_logits
    if temperature <= 0:
        raise GenerationConfigError("generation.temperature must be > 0 when do_sample=true")
    logits = logits / temperature
    if top_k > 0:
        values, indices = torch_module.topk(logits, k=min(top_k, int(logits.shape[-1])), dim=-1)
        probabilities = torch_module.softmax(values, dim=-1)
        sample_index = int(torch_module.multinomial(probabilities, num_samples=1).item())
        return int(indices[0, sample_index].item())
    probabilities = torch_module.softmax(logits, dim=-1)
    return int(torch_module.multinomial(probabilities, num_samples=1).item())


def _select_token_id_from_logits(
    torch_module: Any,
    logits: Any,
    generation_config: dict[str, Any],
    *,
    generator: Any | None = None,
) -> int:
    temperature = float(generation_config.get("temperature", 1.0) or 1.0)
    do_sample = bool(generation_config.get("do_sample", False))
    top_k = int(generation_config.get("top_k", 0) or 0)
    if not do_sample:
        return int(torch_module.argmax(logits, dim=-1).item())
    if temperature <= 0:
        raise GenerationConfigError("generation.temperature must be > 0 when do_sample=true")
    scaled_logits = logits / temperature
    if top_k > 0:
        values, indices = torch_module.topk(scaled_logits, k=min(top_k, int(scaled_logits.shape[-1])), dim=-1)
        probabilities = torch_module.softmax(values, dim=-1)
        sample_index = int(torch_module.multinomial(probabilities, num_samples=1, generator=generator).item())
        return int(indices[sample_index].item())
    probabilities = torch_module.softmax(scaled_logits, dim=-1)
    return int(torch_module.multinomial(probabilities, num_samples=1, generator=generator).item())


class _LiveModelRuntime:
    def __init__(self, config: dict[str, Any]) -> None:
        torch_module, auto_model_cls, auto_tokenizer_cls = _resolve_runtime_modules()
        runtime_config = _runtime_config(config)
        model_config = _model_config(config)
        generation_config = _config_generation_section(config)

        model_name = str(model_config.get("model_name") or "").strip()
        tokenizer_name = str(model_config.get("tokenizer_name") or model_name).strip()
        if not model_name:
            raise GenerationConfigError("generation config must define model.model_name")
        if not tokenizer_name:
            raise GenerationConfigError("generation config must define model.tokenizer_name or model.model_name")

        local_files_only = bool(runtime_config.get("local_files_only", True))
        trust_remote_code = bool(runtime_config.get("trust_remote_code", False))
        device = _resolve_device(torch_module, str(runtime_config.get("device") or "cpu"))
        seed = int(generation_config.get("seed", 13))
        _seed_random_generators(seed)
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():  # pragma: no cover - optional device branch
            torch_module.cuda.manual_seed_all(seed)

        from_pretrained_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        revision = model_config.get("revision")
        if isinstance(revision, str) and revision.strip():
            from_pretrained_kwargs["revision"] = revision

        try:
            tokenizer = auto_tokenizer_cls.from_pretrained(tokenizer_name, **from_pretrained_kwargs)
        except Exception as exc:  # pragma: no cover - optional dependency branch
            hint = (
                f"Tokenizer {tokenizer_name!r} is unavailable locally. Download it first or set runtime.local_files_only=false. "
                f"Example: `uv run python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained({tokenizer_name!r})\"`"
            )
            raise ModelLoadError(hint) from exc

        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = auto_model_cls.from_pretrained(model_name, **from_pretrained_kwargs)
        except Exception as exc:  # pragma: no cover - optional dependency branch
            hint = (
                f"Model {model_name!r} is unavailable locally. Download it first or set runtime.local_files_only=false. "
                f"Example: `uv run python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained({model_name!r})\"`"
            )
            raise ModelLoadError(hint) from exc

        model.to(device)
        model.eval()

        self.torch = torch_module
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.generation_config = generation_config

    def reseed(self, seed: int) -> None:
        _seed_random_generators(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():  # pragma: no cover - optional device branch
            self.torch.cuda.manual_seed_all(seed)

    def _model_forward(self, **kwargs: Any) -> Any:
        try:
            return self.model(**kwargs)
        except TypeError:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("logits_to_keep", None)
            return self.model(**fallback_kwargs)

    def _sample_generator(self, seed: int) -> Any:
        try:
            generator = self.torch.Generator(device=self.device)
        except TypeError:
            generator = self.torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _batched_tokenize(self, texts: list[str], *, padding_side: str) -> tuple[Any, Any]:
        previous_padding_side = getattr(self.tokenizer, "padding_side", None)
        if previous_padding_side is not None:
            self.tokenizer.padding_side = padding_side
        try:
            encoded = self.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        finally:
            if previous_padding_side is not None:
                self.tokenizer.padding_side = previous_padding_side
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def _encode_ids(self, text: str) -> Any:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[-1]) <= 0:
            raise GenerationConfigError("tokenization produced zero tokens for a required text field")
        return input_ids.to(self.device)

    def generate_free_sample_row(self, prompt_group: PromptGroup, *, sample_index: int) -> dict[str, Any]:
        sampling_config = dict(self.generation_config)
        sampling_config["do_sample"] = True
        max_new_tokens = int(sampling_config.get("max_new_tokens", 1) or 1)
        stop_on_eos = bool(sampling_config.get("stop_on_eos", True))
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        answer_only = _answer_only_enabled(sampling_config)
        stop_on_newline = _answer_only_stop_on_newline(sampling_config)
        stop_on_punctuation = _answer_only_stop_on_punctuation(sampling_config)

        current_input_ids = self._encode_ids(prompt_group.prompt)
        generated_token_ids: list[int] = []
        selected_token_logits: list[float] = []
        logsumexp_values: list[float] = []
        full_logits: list[list[float]] = []
        generated_tokens: list[str] = []
        finish_reason = "max_new_tokens"

        for _step in range(max_new_tokens):
            with self.torch.no_grad():
                outputs = self.model(input_ids=current_input_ids)
            step_logits = outputs.logits[:, -1, :]
            next_token_id = _select_token_id(self.torch, step_logits, sampling_config)
            token_text = _token_to_text(self.tokenizer, next_token_id)
            generated_token_ids.append(next_token_id)
            selected_token_logits.append(float(step_logits[0, next_token_id].item()))
            logsumexp_values.append(float(self.torch.logsumexp(step_logits, dim=-1).item()))
            full_logits.append([float(value) for value in step_logits[0].detach().cpu().tolist()])
            generated_tokens.append(token_text)
            next_token_tensor = self.torch.tensor([[next_token_id]], device=self.device, dtype=current_input_ids.dtype)
            current_input_ids = self.torch.cat([current_input_ids, next_token_tensor], dim=-1)
            if stop_on_eos and eos_token_id is not None and next_token_id == int(eos_token_id):
                finish_reason = "eos"
                break
            if answer_only and stop_on_newline and "\n" in token_text:
                finish_reason = "newline"
                break
            if answer_only and stop_on_punctuation and _contains_answer_terminal_punctuation(token_text):
                finish_reason = "punctuation"
                break
        raw_response_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        response_text = _answer_span_text(self.tokenizer, generated_token_ids) if answer_only else raw_response_text

        return {
            "dataset": prompt_group.dataset,
            "split_id": prompt_group.split_id,
            "prompt_id": prompt_group.prompt_id,
            "pair_id": prompt_group.pair_id,
            "prompt": prompt_group.prompt,
            "question": prompt_group.question,
            "context": prompt_group.context,
            "sample_index": sample_index,
            "response_text": response_text,
            "generated_token_ids": generated_token_ids,
            "selected_token_ids": generated_token_ids,
            "selected_token_logits": selected_token_logits,
            "generated_tokens": generated_tokens,
            "full_logits": full_logits,
            "logsumexp": logsumexp_values,
            "full_vocabulary_logits": True,
            "source_kind": "free_sample",
            "answer_only_protocol": _answer_only_protocol_row(
                generation_config=sampling_config,
                finish_reason=finish_reason,
                raw_response_text=raw_response_text,
            ) if answer_only else {"enabled": False},
            "metadata": _metadata_dict(prompt_group.metadata),
        }

    def teacher_forced_candidate_score(self, candidate_row: CandidateRow) -> TeacherForcedCandidateScore:
        prompt_ids = self._encode_ids(candidate_row.prompt)
        candidate_ids = self._encode_ids(candidate_row.candidate_text)
        prompt_length = int(prompt_ids.shape[-1])
        candidate_length = int(candidate_ids.shape[-1])
        combined_ids = self.torch.cat([prompt_ids, candidate_ids], dim=-1)
        with self.torch.no_grad():
            outputs = self.model(input_ids=combined_ids)
        token_scores: list[TeacherForcedTokenScore] = []
        for offset in range(candidate_length):
            absolute_position = prompt_length + offset
            prediction_index = absolute_position - 1
            token_id = int(candidate_ids[0, offset].item())
            step_logits = outputs.logits[0, prediction_index, :]
            full_logits = tuple(float(value) for value in step_logits.detach().cpu().tolist())
            token_scores.append(
                TeacherForcedTokenScore(
                    prompt_id=candidate_row.prompt_id,
                    candidate_id=candidate_row.candidate_id,
                    candidate_token_position=absolute_position,
                    token_id=token_id,
                    selected_token_logit=float(step_logits[token_id].item()),
                    logsumexp=float(self.torch.logsumexp(step_logits, dim=-1).item()),
                    full_logits=full_logits,
                    decoded_token=_token_to_text(self.tokenizer, token_id),
                )
            )
        selected_token_logit_sum = float(sum(score.selected_token_logit for score in token_scores))
        sequence_log_probability = float(sum(score.selected_token_logit - score.logsumexp for score in token_scores))
        return TeacherForcedCandidateScore(
            prompt_id=candidate_row.prompt_id,
            candidate_id=candidate_row.candidate_id,
            candidate_token_count=candidate_length,
            candidate_token_start=prompt_length,
            candidate_token_end=prompt_length + candidate_length - 1,
            selected_token_logit_sum=selected_token_logit_sum,
            selected_token_logit_mean=selected_token_logit_sum / candidate_length,
            sequence_log_probability=sequence_log_probability,
            average_log_probability=sequence_log_probability / candidate_length,
            token_scores=tuple(token_scores),
        )

    def write_free_sample_row(
        self,
        prompt_group: PromptGroup,
        *,
        sample_index: int,
        writer: _FullLogitsParquetWriter,
        parquet_path: Path,
    ) -> dict[str, Any]:
        sampling_config = dict(self.generation_config)
        sampling_config["do_sample"] = True
        max_new_tokens = int(sampling_config.get("max_new_tokens", 1) or 1)
        stop_on_eos = bool(sampling_config.get("stop_on_eos", True))
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        answer_only = _answer_only_enabled(sampling_config)
        stop_on_newline = _answer_only_stop_on_newline(sampling_config)
        stop_on_punctuation = _answer_only_stop_on_punctuation(sampling_config)

        current_input_ids = self._encode_ids(prompt_group.prompt)
        generated_token_ids: list[int] = []
        selected_token_logits: list[float] = []
        logsumexp_values: list[float] = []
        generated_tokens: list[str] = []
        finish_reason = "max_new_tokens"

        for token_offset in range(max_new_tokens):
            with self.torch.no_grad():
                outputs = self.model(input_ids=current_input_ids)
            step_logits = outputs.logits[:, -1, :]
            next_token_id = _select_token_id(self.torch, step_logits, sampling_config)
            token_text = _token_to_text(self.tokenizer, next_token_id)
            generated_token_ids.append(next_token_id)
            selected_token_logits.append(float(step_logits[0, next_token_id].item()))
            logsumexp_values.append(float(self.torch.logsumexp(step_logits, dim=-1).item()))
            generated_tokens.append(token_text)
            writer.write_rows(
                [
                    {
                        "prompt_id": prompt_group.prompt_id,
                        "sample_index": sample_index,
                        "token_offset": token_offset,
                        "full_logits": [float(value) for value in step_logits[0].detach().cpu().tolist()],
                    }
                ]
            )
            next_token_tensor = self.torch.tensor([[next_token_id]], device=self.device, dtype=current_input_ids.dtype)
            current_input_ids = self.torch.cat([current_input_ids, next_token_tensor], dim=-1)
            if stop_on_eos and eos_token_id is not None and next_token_id == int(eos_token_id):
                finish_reason = "eos"
                break
            if answer_only and stop_on_newline and "\n" in token_text:
                finish_reason = "newline"
                break
            if answer_only and stop_on_punctuation and _contains_answer_terminal_punctuation(token_text):
                finish_reason = "punctuation"
                break
        raw_response_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        response_text = _answer_span_text(self.tokenizer, generated_token_ids) if answer_only else raw_response_text

        return {
            "dataset": prompt_group.dataset,
            "split_id": prompt_group.split_id,
            "prompt_id": prompt_group.prompt_id,
            "pair_id": prompt_group.pair_id,
            "prompt": prompt_group.prompt,
            "question": prompt_group.question,
            "context": prompt_group.context,
            "sample_index": sample_index,
            "response_text": response_text,
            "generated_token_ids": generated_token_ids,
            "selected_token_ids": generated_token_ids,
            "selected_token_logits": selected_token_logits,
            "generated_tokens": generated_tokens,
            "logsumexp": logsumexp_values,
            "full_vocabulary_logits": True,
            "source_kind": "free_sample",
            "full_logits_ref": {
                "format": "parquet",
                "path": str(parquet_path),
                "key_fields": ["prompt_id", "sample_index", "token_offset"],
            },
            "answer_only_protocol": _answer_only_protocol_row(
                generation_config=sampling_config,
                finish_reason=finish_reason,
                raw_response_text=raw_response_text,
            ) if answer_only else {"enabled": False},
            "metadata": _metadata_dict(prompt_group.metadata),
        }

    def write_free_sample_rows_batch(
        self,
        sample_specs: list[tuple[PromptGroup, int, _FullLogitsParquetWriter, Path, int]],
    ) -> list[dict[str, Any]]:
        if not sample_specs:
            return []
        sampling_config = dict(self.generation_config)
        sampling_config["do_sample"] = True
        max_new_tokens = int(sampling_config.get("max_new_tokens", 1) or 1)
        stop_on_eos = bool(sampling_config.get("stop_on_eos", True))
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        answer_only = _answer_only_enabled(sampling_config)
        stop_on_newline = _answer_only_stop_on_newline(sampling_config)
        stop_on_punctuation = _answer_only_stop_on_punctuation(sampling_config)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        prompt_groups = [spec[0] for spec in sample_specs]
        sample_indices = [spec[1] for spec in sample_specs]
        writers = [spec[2] for spec in sample_specs]
        parquet_paths = [spec[3] for spec in sample_specs]
        generators = [self._sample_generator(spec[4]) for spec in sample_specs]
        current_input_ids, attention_mask = self._batched_tokenize(
            [prompt_group.prompt for prompt_group in prompt_groups], padding_side="left"
        )
        active = [True for _ in sample_specs]
        generated_token_ids: list[list[int]] = [[] for _ in sample_specs]
        selected_token_logits: list[list[float]] = [[] for _ in sample_specs]
        logsumexp_values: list[list[float]] = [[] for _ in sample_specs]
        generated_tokens: list[list[str]] = [[] for _ in sample_specs]
        finish_reasons = ["max_new_tokens" for _ in sample_specs]

        for token_offset in range(max_new_tokens):
            if not any(active):
                break
            with self.torch.no_grad():
                outputs = self._model_forward(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=1,
                )
            step_logits = outputs.logits[:, -1, :]
            next_token_ids: list[int] = []
            next_attention_values: list[int] = []
            for row_index, is_active in enumerate(active):
                if not is_active:
                    next_token_ids.append(int(pad_token_id))
                    next_attention_values.append(0)
                    continue
                row_logits = step_logits[row_index]
                next_token_id = _select_token_id_from_logits(
                    self.torch,
                    row_logits,
                    sampling_config,
                    generator=generators[row_index],
                )
                token_text = _token_to_text(self.tokenizer, next_token_id)
                generated_token_ids[row_index].append(next_token_id)
                selected_token_logits[row_index].append(float(row_logits[next_token_id].item()))
                logsumexp_values[row_index].append(float(self.torch.logsumexp(row_logits, dim=-1).item()))
                generated_tokens[row_index].append(token_text)
                writers[row_index].write_rows(
                    [
                        {
                            "prompt_id": prompt_groups[row_index].prompt_id,
                            "sample_index": sample_indices[row_index],
                            "token_offset": token_offset,
                            "full_logits": [float(value) for value in row_logits.detach().cpu().tolist()],
                        }
                    ]
                )
                next_token_ids.append(next_token_id)
                next_attention_values.append(1)
                if stop_on_eos and eos_token_id is not None and next_token_id == int(eos_token_id):
                    finish_reasons[row_index] = "eos"
                    active[row_index] = False
                if answer_only and stop_on_newline and "\n" in token_text:
                    finish_reasons[row_index] = "newline"
                    active[row_index] = False
                if answer_only and stop_on_punctuation and _contains_answer_terminal_punctuation(token_text):
                    finish_reasons[row_index] = "punctuation"
                    active[row_index] = False
            next_token_tensor = self.torch.tensor([next_token_ids], device=self.device, dtype=current_input_ids.dtype).T
            next_attention_tensor = self.torch.tensor([next_attention_values], device=self.device, dtype=attention_mask.dtype).T
            current_input_ids = self.torch.cat([current_input_ids, next_token_tensor], dim=-1)
            attention_mask = self.torch.cat([attention_mask, next_attention_tensor], dim=-1)

        samples: list[dict[str, Any]] = []
        for row_index, prompt_group in enumerate(prompt_groups):
            raw_response_text = self.tokenizer.decode(
                generated_token_ids[row_index],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            response_text = (
                _answer_span_text(self.tokenizer, generated_token_ids[row_index])
                if answer_only
                else raw_response_text
            )
            samples.append(
                {
                    "dataset": prompt_group.dataset,
                    "split_id": prompt_group.split_id,
                    "prompt_id": prompt_group.prompt_id,
                    "pair_id": prompt_group.pair_id,
                    "prompt": prompt_group.prompt,
                    "question": prompt_group.question,
                    "context": prompt_group.context,
                    "sample_index": sample_indices[row_index],
                    "response_text": response_text,
                    "generated_token_ids": generated_token_ids[row_index],
                    "selected_token_ids": generated_token_ids[row_index],
                    "selected_token_logits": selected_token_logits[row_index],
                    "generated_tokens": generated_tokens[row_index],
                    "logsumexp": logsumexp_values[row_index],
                    "full_vocabulary_logits": True,
                    "source_kind": "free_sample",
                    "full_logits_ref": {
                        "format": "parquet",
                        "path": str(parquet_paths[row_index]),
                        "key_fields": ["prompt_id", "sample_index", "token_offset"],
                    },
                    "answer_only_protocol": _answer_only_protocol_row(
                        generation_config=sampling_config,
                        finish_reason=finish_reasons[row_index],
                        raw_response_text=raw_response_text,
                    ) if answer_only else {"enabled": False},
                    "metadata": _metadata_dict(prompt_group.metadata),
                }
            )
        return samples

    def write_teacher_forced_candidate_score(
        self,
        candidate_row: CandidateRow,
        *,
        writer: _FullLogitsParquetWriter,
        parquet_path: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        prompt_ids = self._encode_ids(candidate_row.prompt)
        candidate_ids = self._encode_ids(candidate_row.candidate_text)
        prompt_length = int(prompt_ids.shape[-1])
        candidate_length = int(candidate_ids.shape[-1])
        combined_ids = self.torch.cat([prompt_ids, candidate_ids], dim=-1)
        with self.torch.no_grad():
            outputs = self.model(input_ids=combined_ids)

        token_rows: list[dict[str, Any]] = []
        selected_token_logit_sum = 0.0
        sequence_log_probability = 0.0
        for offset in range(candidate_length):
            absolute_position = prompt_length + offset
            prediction_index = absolute_position - 1
            token_id = int(candidate_ids[0, offset].item())
            step_logits = outputs.logits[0, prediction_index, :]
            selected_token_logit = float(step_logits[token_id].item())
            logsumexp = float(self.torch.logsumexp(step_logits, dim=-1).item())
            selected_token_logit_sum += selected_token_logit
            sequence_log_probability += selected_token_logit - logsumexp
            writer.write_rows(
                [
                    {
                        "candidate_id": candidate_row.candidate_id,
                        "candidate_token_offset": offset,
                        "candidate_token_position": absolute_position,
                        "full_logits": [float(value) for value in step_logits.detach().cpu().tolist()],
                    }
                ]
            )
            token_rows.append(
                {
                    "prompt_id": candidate_row.prompt_id,
                    "candidate_id": candidate_row.candidate_id,
                    "candidate_token_position": absolute_position,
                    "token_id": token_id,
                    "selected_token_logit": selected_token_logit,
                    "logsumexp": logsumexp,
                    "decoded_token": _token_to_text(self.tokenizer, token_id),
                    "candidate_token_offset": offset,
                    "sequence_token_position": absolute_position,
                    "full_logits_ref": {
                        "format": "parquet",
                        "path": str(parquet_path),
                        "key_fields": ["candidate_id", "candidate_token_offset"],
                    },
                }
            )

        candidate_score_row = {
            "dataset": candidate_row.dataset,
            "split_id": candidate_row.split_id,
            "pair_id": candidate_row.pair_id,
            "prompt_id": candidate_row.prompt_id,
            "candidate_id": candidate_row.candidate_id,
            "candidate_role": candidate_row.candidate_role,
            "candidate_text": candidate_row.candidate_text,
            "is_correct": candidate_row.is_correct,
            "label_source": candidate_row.label_source,
            "candidate_token_count": candidate_length,
            "candidate_token_start": prompt_length,
            "candidate_token_end": prompt_length + candidate_length - 1,
            "selected_token_logit_sum": selected_token_logit_sum,
            "selected_token_logit_mean": selected_token_logit_sum / candidate_length,
            "sequence_log_probability": sequence_log_probability,
            "average_log_probability": sequence_log_probability / candidate_length,
            "metadata": _metadata_dict(candidate_row.metadata),
        }
        return candidate_score_row, token_rows

    def write_teacher_forced_candidate_scores_batch(
        self,
        candidate_specs: list[tuple[CandidateRow, _FullLogitsParquetWriter, Path]],
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        if not candidate_specs:
            return []
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        encoded_rows: list[tuple[CandidateRow, Any, Any, int, int, _FullLogitsParquetWriter, Path]] = []
        combined_sequences: list[Any] = []
        for candidate_row, writer, parquet_path in candidate_specs:
            prompt_ids = self._encode_ids(candidate_row.prompt).squeeze(0)
            candidate_ids = self._encode_ids(candidate_row.candidate_text).squeeze(0)
            prompt_length = int(prompt_ids.shape[-1])
            candidate_length = int(candidate_ids.shape[-1])
            combined_ids = self.torch.cat([prompt_ids, candidate_ids], dim=-1)
            encoded_rows.append((candidate_row, combined_ids, candidate_ids, prompt_length, candidate_length, writer, parquet_path))
            combined_sequences.append(combined_ids)
        max_length = max(int(sequence.shape[-1]) for sequence in combined_sequences)
        input_ids = self.torch.full(
            (len(combined_sequences), max_length),
            int(pad_token_id),
            device=self.device,
            dtype=combined_sequences[0].dtype,
        )
        attention_mask = self.torch.zeros((len(combined_sequences), max_length), device=self.device, dtype=self.torch.long)
        for row_index, sequence in enumerate(combined_sequences):
            sequence_length = int(sequence.shape[-1])
            input_ids[row_index, :sequence_length] = sequence
            attention_mask[row_index, :sequence_length] = 1

        with self.torch.no_grad():
            outputs = self._model_forward(input_ids=input_ids, attention_mask=attention_mask)

        results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for row_index, (candidate_row, _combined_ids, candidate_ids, prompt_length, candidate_length, writer, parquet_path) in enumerate(
            encoded_rows
        ):
            token_rows: list[dict[str, Any]] = []
            selected_token_logit_sum = 0.0
            sequence_log_probability = 0.0
            for offset in range(candidate_length):
                absolute_position = prompt_length + offset
                prediction_index = absolute_position - 1
                token_id = int(candidate_ids[offset].item())
                step_logits = outputs.logits[row_index, prediction_index, :]
                selected_token_logit = float(step_logits[token_id].item())
                logsumexp = float(self.torch.logsumexp(step_logits, dim=-1).item())
                selected_token_logit_sum += selected_token_logit
                sequence_log_probability += selected_token_logit - logsumexp
                writer.write_rows(
                    [
                        {
                            "candidate_id": candidate_row.candidate_id,
                            "candidate_token_offset": offset,
                            "candidate_token_position": absolute_position,
                            "full_logits": [float(value) for value in step_logits.detach().cpu().tolist()],
                        }
                    ]
                )
                token_rows.append(
                    {
                        "prompt_id": candidate_row.prompt_id,
                        "candidate_id": candidate_row.candidate_id,
                        "candidate_token_position": absolute_position,
                        "token_id": token_id,
                        "selected_token_logit": selected_token_logit,
                        "logsumexp": logsumexp,
                        "decoded_token": _token_to_text(self.tokenizer, token_id),
                        "candidate_token_offset": offset,
                        "sequence_token_position": absolute_position,
                        "full_logits_ref": {
                            "format": "parquet",
                            "path": str(parquet_path),
                            "key_fields": ["candidate_id", "candidate_token_offset"],
                        },
                    }
                )
            candidate_score_row = {
                "dataset": candidate_row.dataset,
                "split_id": candidate_row.split_id,
                "pair_id": candidate_row.pair_id,
                "prompt_id": candidate_row.prompt_id,
                "candidate_id": candidate_row.candidate_id,
                "candidate_role": candidate_row.candidate_role,
                "candidate_text": candidate_row.candidate_text,
                "is_correct": candidate_row.is_correct,
                "label_source": candidate_row.label_source,
                "candidate_token_count": candidate_length,
                "candidate_token_start": prompt_length,
                "candidate_token_end": prompt_length + candidate_length - 1,
                "selected_token_logit_sum": selected_token_logit_sum,
                "selected_token_logit_mean": selected_token_logit_sum / candidate_length,
                "sequence_log_probability": sequence_log_probability,
                "average_log_probability": sequence_log_probability / candidate_length,
                "metadata": _metadata_dict(candidate_row.metadata),
            }
            results.append((candidate_score_row, token_rows))
        return results

    def vocab_size(self) -> int:
        tokenizer_vocab = len(self.tokenizer)
        model_vocab = int(getattr(getattr(self.model, "config", None), "vocab_size", tokenizer_vocab) or tokenizer_vocab)
        return max(int(tokenizer_vocab), model_vocab)

    def candidate_token_count(self, candidate_row: CandidateRow) -> int:
        return int(self._encode_ids(candidate_row.candidate_text).shape[-1])


def _build_live_free_sample_artifact(config: dict[str, Any], prompt_groups: tuple[PromptGroup, ...]) -> dict[str, Any]:
    runtime = _LiveModelRuntime(config)
    artifact = _base_artifact(
        config,
        artifact_type="free_sample_rows",
        fixture_mode=False,
        has_full_logits=True,
    )
    artifact["model_name"] = runtime.model_name
    artifact["tokenizer_name"] = runtime.tokenizer_name
    sampling_config = dict(runtime.generation_config)
    sampling_config["do_sample"] = True
    artifact["generation_config"] = sampling_config
    samples: list[dict[str, Any]] = []
    for prompt_group in prompt_groups:
        for sample_index in range(FREE_SAMPLE_COUNT):
            samples.append(runtime.generate_free_sample_row(prompt_group, sample_index=sample_index))
    artifact.update(
        {
            "sample_count_per_prompt": FREE_SAMPLE_COUNT,
            "prompt_group_count": len(prompt_groups),
            "samples": samples,
        }
    )
    return artifact


def _build_live_candidate_score_artifact(config: dict[str, Any], candidate_rows: tuple[CandidateRow, ...]) -> dict[str, Any]:
    runtime = _LiveModelRuntime(config)
    artifact = _base_artifact(
        config,
        artifact_type="teacher_forced_candidate_scores",
        fixture_mode=False,
        has_full_logits=True,
    )
    artifact["model_name"] = runtime.model_name
    artifact["tokenizer_name"] = runtime.tokenizer_name
    candidate_score_rows: list[dict[str, Any]] = []
    token_score_rows: list[dict[str, Any]] = []
    for candidate_row in candidate_rows:
        score = runtime.teacher_forced_candidate_score(candidate_row)
        candidate_score_rows.append(_candidate_score_row(candidate_row, score))
        token_score_rows.extend(
            _token_score_row(token_score, candidate_token_start=score.candidate_token_start)
            for token_score in score.token_scores
        )
    artifact.update(
        {
            "candidate_count": len(candidate_rows),
            "candidate_score_rows": candidate_score_rows,
            "token_score_rows": token_score_rows,
            "candidate_scoring_mode": "teacher_forced",
            "prompt_prefix_scoring_excluded": True,
            "free_sample_count": FREE_SAMPLE_COUNT,
        }
    )
    return artifact


def _is_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)


def _is_numeric_matrix(value: Any) -> bool:
    return isinstance(value, list) and all(_is_numeric_list(item) for item in value)


def expected_free_sample_indexes() -> tuple[int, ...]:
    return tuple(range(FREE_SAMPLE_COUNT))


def free_sample_index_coverage(payload: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return {}
    coverage: dict[str, set[int]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        prompt_id = sample.get("prompt_id")
        sample_index = sample.get("sample_index")
        if isinstance(prompt_id, str) and prompt_id.strip() and isinstance(sample_index, int) and not isinstance(sample_index, bool):
            coverage.setdefault(prompt_id, set()).add(sample_index)
    return {prompt_id: tuple(sorted(indexes)) for prompt_id, indexes in sorted(coverage.items())}


def _validate_free_sample_payload(payload: dict[str, Any]) -> None:
    problems: list[str] = []
    for field_name in (
        "run_id",
        "schema_version",
        "created_at",
        "model_name",
        "tokenizer_name",
        "generation_config",
        "logits_schema_version",
        "formula_manifest_ref",
        "dataset_manifest_ref",
        "sample_count_per_prompt",
        "prompt_group_count",
        "samples",
    ):
        if field_name not in payload:
            problems.append(f"missing top-level field {field_name}")
    if payload.get("artifact_type") != "free_sample_rows":
        problems.append("free-sample artifact must set artifact_type='free_sample_rows'")
    if payload.get("schema_version") != GENERATION_FREE_SAMPLE_SCHEMA_VERSION:
        problems.append(
            f"free-sample artifact schema_version must be {GENERATION_FREE_SAMPLE_SCHEMA_VERSION!r}; got {payload.get('schema_version')!r}"
        )
    if payload.get("has_logits") is not True:
        problems.append("artifact must set has_logits=true")
    if payload.get("sample_count_per_prompt") != FREE_SAMPLE_COUNT:
        problems.append(f"sample_count_per_prompt must equal {FREE_SAMPLE_COUNT}")
    generation_config = payload.get("generation_config")
    if not isinstance(generation_config, dict) or generation_config.get("do_sample") is not True:
        problems.append("free-sample artifact generation_config must record do_sample=true")

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        problems.append("free-sample artifact must contain a non-empty 'samples' list")
    if isinstance(samples, list):
        prompt_counts: dict[str, int] = {}
        prompt_sample_indices: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                problems.append(f"sample {index} must be an object")
                continue
            for field_name in (
                "dataset",
                "split_id",
                "prompt_id",
                "pair_id",
                "prompt",
                "sample_index",
                "response_text",
                "generated_token_ids",
                "selected_token_ids",
                "selected_token_logits",
                "generated_tokens",
                "logsumexp",
                "full_vocabulary_logits",
                "source_kind",
            ):
                if field_name not in sample:
                    problems.append(f"sample {index} missing field {field_name}")
            for field_name in ("dataset", "split_id", "prompt_id", "pair_id", "prompt", "response_text", "source_kind"):
                value = sample.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    problems.append(f"sample {index} must include non-empty string {field_name!r}")
            prompt_id = sample.get("prompt_id")
            if isinstance(prompt_id, str) and prompt_id.strip():
                prompt_counts[prompt_id] = prompt_counts.get(prompt_id, 0) + 1
            sample_index = sample.get("sample_index")
            if not isinstance(sample_index, int) or isinstance(sample_index, bool):
                problems.append(f"sample {index} must include integer 'sample_index'")
            elif isinstance(prompt_id, str) and prompt_id.strip():
                prompt_sample_indices.setdefault(prompt_id, []).append(sample_index)
            if not _is_numeric_list(sample.get("generated_token_ids")):
                problems.append(f"sample {index} must include numeric 'generated_token_ids'")
            if not _is_numeric_list(sample.get("selected_token_ids")):
                problems.append(f"sample {index} must include numeric 'selected_token_ids'")
            if not _is_numeric_list(sample.get("selected_token_logits")):
                problems.append(f"sample {index} must include numeric 'selected_token_logits'")
            if not _is_numeric_list(sample.get("logsumexp")):
                problems.append(f"sample {index} must include numeric 'logsumexp'")
            if not isinstance(sample.get("generated_tokens"), list) or not all(
                isinstance(item, str) for item in sample.get("generated_tokens", [])
            ):
                problems.append(f"sample {index} must include string 'generated_tokens'")
            if payload.get("has_full_logits") is True:
                has_inline_logits = _is_numeric_matrix(sample.get("full_logits"))
                has_parquet_ref = isinstance(sample.get("full_logits_ref"), dict) and isinstance(payload.get("full_logits_storage"), dict)
                if not has_inline_logits and not has_parquet_ref:
                    problems.append(
                        f"sample {index} must include numeric 'full_logits' or a parquet 'full_logits_ref' when has_full_logits=true"
                    )
            if sample.get("source_kind") != "free_sample":
                problems.append(f"sample {index} must set source_kind='free_sample'")
            if isinstance(generation_config, dict):
                problems.extend(_answer_only_validation_problems(sample, generation_config, label=f"sample {index}"))
        prompt_group_count = payload.get("prompt_group_count")
        if not isinstance(prompt_group_count, int) or isinstance(prompt_group_count, bool) or prompt_group_count <= 0:
            problems.append("prompt_group_count must be a positive integer")
        elif len(prompt_sample_indices) != prompt_group_count:
            problems.append(
                f"prompt_group_count={prompt_group_count} does not match observed distinct prompt coverage count {len(prompt_sample_indices)}"
            )
        expected_total_samples = prompt_group_count * FREE_SAMPLE_COUNT if isinstance(prompt_group_count, int) and not isinstance(prompt_group_count, bool) and prompt_group_count > 0 else None
        if expected_total_samples is not None and len(samples) != expected_total_samples:
            problems.append(
                f"free-sample artifact has {len(samples)} rows; expected {expected_total_samples} from prompt_group_count={prompt_group_count} and sample_count_per_prompt={FREE_SAMPLE_COUNT}"
            )
        expected_indexes = set(expected_free_sample_indexes())
        for prompt_id, count in prompt_counts.items():
            if count != FREE_SAMPLE_COUNT:
                problems.append(f"prompt_id {prompt_id!r} has {count} free samples; expected {FREE_SAMPLE_COUNT}")
        for prompt_id, observed_indexes in sorted(prompt_sample_indices.items()):
            coverage = sorted(observed_indexes)
            duplicate_indexes = sorted({sample_index for sample_index in coverage if observed_indexes.count(sample_index) > 1})
            coverage_set = set(coverage)
            missing_indexes = sorted(expected_indexes - coverage_set)
            unexpected_indexes = sorted(coverage_set - expected_indexes)
            if duplicate_indexes:
                problems.append(
                    f"prompt_id {prompt_id!r} duplicates free-sample sample_index values {duplicate_indexes}; coverage={coverage}"
                )
            if missing_indexes or unexpected_indexes or coverage_set != expected_indexes:
                detail_parts: list[str] = [f"coverage={coverage}"]
                if missing_indexes:
                    detail_parts.append(f"missing={missing_indexes}")
                if unexpected_indexes:
                    detail_parts.append(f"unexpected={unexpected_indexes}")
                problems.append(
                    f"prompt_id {prompt_id!r} must cover exact free-sample sample_index set {list(expected_free_sample_indexes())}; "
                    + ", ".join(detail_parts)
                )
    if problems:
        raise GenerationValidationError("Generation validation failed:\n- " + "\n- ".join(problems))


def _validate_candidate_score_payload(payload: dict[str, Any]) -> None:
    problems: list[str] = []
    for field_name in (
        "run_id",
        "schema_version",
        "created_at",
        "model_name",
        "tokenizer_name",
        "generation_config",
        "logits_schema_version",
        "formula_manifest_ref",
        "dataset_manifest_ref",
        "candidate_count",
        "candidate_score_rows",
        "token_score_rows",
    ):
        if field_name not in payload:
            problems.append(f"missing top-level field {field_name}")
    if payload.get("artifact_type") != "teacher_forced_candidate_scores":
        problems.append("candidate-score artifact must set artifact_type='teacher_forced_candidate_scores'")
    if payload.get("schema_version") != GENERATION_CANDIDATE_SCORE_SCHEMA_VERSION:
        problems.append(
            "candidate-score artifact schema_version must be "
            f"{GENERATION_CANDIDATE_SCORE_SCHEMA_VERSION!r}; got {payload.get('schema_version')!r}"
        )
    if payload.get("candidate_scoring_mode") != "teacher_forced":
        problems.append("candidate-score artifact must set candidate_scoring_mode='teacher_forced'")
    if payload.get("prompt_prefix_scoring_excluded") is not True:
        problems.append("candidate-score artifact must set prompt_prefix_scoring_excluded=true")

    candidate_rows = payload.get("candidate_score_rows")
    token_rows = payload.get("token_score_rows")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        problems.append("candidate-score artifact must contain non-empty 'candidate_score_rows'")
    if not isinstance(token_rows, list) or not token_rows:
        problems.append("candidate-score artifact must contain non-empty 'token_score_rows'")
    token_rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    if isinstance(token_rows, list):
        for index, row in enumerate(token_rows):
            if not isinstance(row, dict):
                problems.append(f"token_score_rows[{index}] must be an object")
                continue
            candidate_id = row.get("candidate_id")
            if isinstance(candidate_id, str):
                token_rows_by_candidate.setdefault(candidate_id, []).append(row)
            if not isinstance(row.get("candidate_token_position"), int):
                problems.append(f"token_score_rows[{index}] must include integer 'candidate_token_position'")
            if not isinstance(row.get("candidate_token_offset"), int):
                problems.append(f"token_score_rows[{index}] must include integer 'candidate_token_offset'")
            if not isinstance(row.get("sequence_token_position"), int):
                problems.append(f"token_score_rows[{index}] must include integer 'sequence_token_position'")
            if not isinstance(row.get("selected_token_logit"), (int, float)):
                problems.append(f"token_score_rows[{index}] must include numeric 'selected_token_logit'")
            if not isinstance(row.get("logsumexp"), (int, float)):
                problems.append(f"token_score_rows[{index}] must include numeric 'logsumexp'")
            if payload.get("has_full_logits") is True:
                has_inline_logits = _is_numeric_list(row.get("full_logits"))
                has_parquet_ref = isinstance(row.get("full_logits_ref"), dict) and isinstance(payload.get("full_logits_storage"), dict)
                if not has_inline_logits and not has_parquet_ref:
                    problems.append(
                        f"token_score_rows[{index}] must include numeric 'full_logits' or a parquet 'full_logits_ref' when has_full_logits=true"
                    )

    if isinstance(candidate_rows, list):
        for index, row in enumerate(candidate_rows):
            if not isinstance(row, dict):
                problems.append(f"candidate_score_rows[{index}] must be an object")
                continue
            for field_name in (
                "prompt_id",
                "candidate_id",
                "candidate_token_count",
                "candidate_token_start",
                "candidate_token_end",
                "selected_token_logit_sum",
                "selected_token_logit_mean",
            ):
                if field_name not in row:
                    problems.append(f"candidate_score_rows[{index}] missing field {field_name}")
            candidate_id = row.get("candidate_id")
            related_token_rows = token_rows_by_candidate.get(candidate_id, []) if isinstance(candidate_id, str) else []
            expected_count = row.get("candidate_token_count")
            token_start = row.get("candidate_token_start")
            token_end = row.get("candidate_token_end")
            if isinstance(expected_count, int) and len(related_token_rows) != expected_count:
                problems.append(
                    f"candidate_score_rows[{index}] token row count {len(related_token_rows)} does not match candidate_token_count {expected_count}"
                )
            if isinstance(expected_count, int) and expected_count <= 0:
                problems.append(f"candidate_score_rows[{index}] candidate_token_count must be > 0")
            if isinstance(token_start, int) and isinstance(token_end, int) and isinstance(expected_count, int):
                if token_end - token_start + 1 != expected_count:
                    problems.append(
                        f"candidate_score_rows[{index}] token boundary width does not match candidate_token_count"
                    )
                if related_token_rows:
                    positions = sorted(int(token_row["candidate_token_position"]) for token_row in related_token_rows)
                    if positions[0] != token_start or positions[-1] != token_end:
                        problems.append(
                            f"candidate_score_rows[{index}] token positions do not match candidate_token_start/end"
                        )
    if problems:
        raise GenerationValidationError("Generation validation failed:\n- " + "\n- ".join(problems))


def validate_generation_payload(payload: dict[str, Any]) -> None:
    artifact_type = payload.get("artifact_type")
    if artifact_type == "free_sample_rows":
        _validate_free_sample_payload(payload)
        return
    if artifact_type == "teacher_forced_candidate_scores":
        _validate_candidate_score_payload(payload)
        return
    raise GenerationValidationError(f"unsupported generation artifact_type {artifact_type!r}")


def _validate_generation_artifact_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    validate_generation_payload(payload)
    _validate_final_sidecar_is_same_stem(payload, path)
    artifact_type = payload.get("artifact_type")
    if artifact_type == "free_sample_rows":
        expected_keys: set[tuple[Any, ...]] = set()
        samples = payload.get("samples")
        if isinstance(samples, list):
            for sample in samples:
                if not isinstance(sample, dict) or not isinstance(sample.get("full_logits_ref"), dict):
                    continue
                prompt_id = sample.get("prompt_id")
                sample_index = sample.get("sample_index")
                logsumexp = sample.get("logsumexp")
                if isinstance(prompt_id, str) and isinstance(sample_index, int) and isinstance(logsumexp, list):
                    for token_offset in range(len(logsumexp)):
                        expected_keys.add((prompt_id, sample_index, token_offset))
        if expected_keys:
            _validate_full_logits_sidecar(
                payload=payload,
                source_path=path,
                artifact_type="free_sample_rows",
                expected_keys=expected_keys,
                key_fields=("prompt_id", "sample_index", "token_offset"),
            )
    elif artifact_type == "teacher_forced_candidate_scores":
        expected_keys = set()
        token_rows = payload.get("token_score_rows")
        if isinstance(token_rows, list):
            for token_row in token_rows:
                if not isinstance(token_row, dict) or not isinstance(token_row.get("full_logits_ref"), dict):
                    continue
                candidate_id = token_row.get("candidate_id")
                candidate_token_offset = token_row.get("candidate_token_offset")
                if isinstance(candidate_id, str) and isinstance(candidate_token_offset, int):
                    expected_keys.add((candidate_id, candidate_token_offset))
        if expected_keys:
            _validate_full_logits_sidecar(
                payload=payload,
                source_path=path,
                artifact_type="teacher_forced_candidate_scores",
                expected_keys=expected_keys,
                key_fields=("candidate_id", "candidate_token_offset"),
            )
    return payload


def validate_generation_artifact(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise GenerationValidationError(f"artifact must decode to an object: {path}")
    return _validate_generation_artifact_payload(payload, path)


def _validate_free_sample_checkpoint(payload: dict[str, Any], source_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if payload.get("checkpoint_artifact_type") != "free_sample_rows_checkpoint_shard":
        raise GenerationValidationError(f"unsupported free-sample checkpoint shard: {source_path}")
    if not _is_metadata_compatible(payload, metadata):
        raise GenerationValidationError(f"free-sample checkpoint shard metadata mismatch on essential fields (model/tokenizer/schema/fingerprint): {source_path}")
    sample = payload.get("sample")
    if not isinstance(sample, dict):
        raise GenerationValidationError(f"free-sample checkpoint shard must contain one sample: {source_path}")
    prompt_id = sample.get("prompt_id")
    sample_index = sample.get("sample_index")
    logsumexp = sample.get("logsumexp")
    if not isinstance(prompt_id, str) or not isinstance(sample_index, int) or not isinstance(logsumexp, list) or not logsumexp:
        raise GenerationValidationError(f"free-sample checkpoint shard has invalid sample key/token metadata: {source_path}")
    if payload.get("prompt_id") != prompt_id or payload.get("sample_index") != sample_index:
        raise GenerationValidationError(f"free-sample checkpoint shard key does not match sample row: {source_path}")
    generation_config = payload.get("generation_config")
    if isinstance(generation_config, dict):
        answer_only_problems = _answer_only_validation_problems(sample, generation_config, label=f"checkpoint {source_path}")
        if answer_only_problems:
            raise GenerationValidationError("Free-sample checkpoint answer-only validation failed:\n- " + "\n- ".join(answer_only_problems))
    expected_keys = {(prompt_id, sample_index, token_offset) for token_offset in range(len(logsumexp))}
    _validate_full_logits_sidecar(
        payload=payload,
        source_path=source_path,
        artifact_type="free_sample_rows_checkpoint_shard",
        expected_keys=expected_keys,
        key_fields=("prompt_id", "sample_index", "token_offset"),
        checkpoint_sidecar=True,
    )
    return sample


def _validate_candidate_checkpoint(
    payload: dict[str, Any], source_path: Path, metadata: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if payload.get("checkpoint_artifact_type") != "teacher_forced_candidate_scores_checkpoint_shard":
        raise GenerationValidationError(f"unsupported candidate checkpoint shard: {source_path}")
    if not _is_metadata_compatible(payload, metadata):
        raise GenerationValidationError(f"candidate checkpoint shard metadata mismatch on essential fields (model/tokenizer/schema/fingerprint): {source_path}")
    candidate_score_row = payload.get("candidate_score_row")
    token_rows = payload.get("token_rows")
    if not isinstance(candidate_score_row, dict) or not isinstance(token_rows, list) or not token_rows:
        raise GenerationValidationError(f"candidate checkpoint shard must contain one candidate row and token rows: {source_path}")
    if not all(isinstance(token_row, dict) for token_row in token_rows):
        raise GenerationValidationError(f"candidate checkpoint token rows must be objects: {source_path}")
    candidate_id = candidate_score_row.get("candidate_id")
    expected_count = candidate_score_row.get("candidate_token_count")
    if not isinstance(candidate_id, str) or not isinstance(expected_count, int) or expected_count <= 0:
        raise GenerationValidationError(f"candidate checkpoint shard has invalid candidate metadata: {source_path}")
    if payload.get("candidate_id") != candidate_id or len(token_rows) != expected_count:
        raise GenerationValidationError(f"candidate checkpoint shard key/count mismatch: {source_path}")
    expected_keys: set[tuple[Any, ...]] = set()
    for token_row in token_rows:
        token_candidate_id = token_row.get("candidate_id")
        candidate_token_offset = token_row.get("candidate_token_offset")
        if token_candidate_id != candidate_id or not isinstance(candidate_token_offset, int):
            raise GenerationValidationError(f"candidate checkpoint token row has invalid key: {source_path}")
        expected_keys.add((candidate_id, candidate_token_offset))
    if len(expected_keys) != expected_count:
        raise GenerationValidationError(f"candidate checkpoint token rows contain duplicate offsets: {source_path}")
    _validate_full_logits_sidecar(
        payload=payload,
        source_path=source_path,
        artifact_type="teacher_forced_candidate_scores_checkpoint_shard",
        expected_keys=expected_keys,
        key_fields=("candidate_id", "candidate_token_offset"),
        checkpoint_sidecar=True,
    )
    return candidate_score_row, cast(list[dict[str, Any]], token_rows)


def _load_free_sample_checkpoints(
    checkpoint_root: Path, metadata: dict[str, Any]
) -> dict[tuple[str, int], tuple[dict[str, Any], Path]]:
    completed: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
    if not checkpoint_root.exists():
        return completed
    skipped: list[tuple[str, str]] = []
    for shard_dir in sorted(checkpoint_root.iterdir()):
        if not shard_dir.is_dir():
            continue
        if shard_dir.name.startswith(".tmp-"):
            # Partial-write directories from interrupted runs have no shard.json yet;
            # safe to remove since they cannot be reused.
            shutil.rmtree(shard_dir)
            continue
        shard_json = shard_dir / "shard.json"
        try:
            payload = load_json(shard_json)
            if not isinstance(payload, dict):
                raise GenerationValidationError(f"checkpoint shard must decode to an object: {shard_json}")
            sample = _validate_free_sample_checkpoint(payload, shard_json, metadata)
            key = (str(sample["prompt_id"]), int(sample["sample_index"]))
            if key in completed:
                raise GenerationValidationError(f"duplicate free-sample checkpoint key {key!r}")
            storage = payload.get("full_logits_storage")
            if not isinstance(storage, dict):
                raise GenerationValidationError(f"checkpoint shard is missing full_logits_storage: {shard_json}")
            sidecar_path = _resolve_checkpoint_sidecar_path(shard_json, storage.get("path"))
            if sidecar_path is None:
                raise GenerationValidationError(f"checkpoint shard sidecar path is invalid: {shard_json}")
            completed[key] = (sample, sidecar_path)
        except (OSError, json.JSONDecodeError, GenerationValidationError) as exc:
            # Non-destructive policy: do NOT delete the shard. The orchestrator
            # rebuilds the missing (prompt_id, sample_index) pair from scratch
            # while preserving the on-disk shard for human inspection.
            skipped.append((shard_dir.name, str(exc)))
    if skipped:
        head = ", ".join(name for name, _ in skipped[:3])
        print(
            f"[checkpoint] preserved {len(skipped)} unreadable free-sample shard(s); first: {head}",
            file=sys.stderr,
        )
    return completed


def _load_candidate_checkpoints(
    checkpoint_root: Path, metadata: dict[str, Any]
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]], Path]]:
    completed: dict[str, tuple[dict[str, Any], list[dict[str, Any]], Path]] = {}
    if not checkpoint_root.exists():
        return completed
    skipped: list[tuple[str, str]] = []
    for shard_dir in sorted(checkpoint_root.iterdir()):
        if not shard_dir.is_dir():
            continue
        if shard_dir.name.startswith(".tmp-"):
            shutil.rmtree(shard_dir)
            continue
        shard_json = shard_dir / "shard.json"
        try:
            payload = load_json(shard_json)
            if not isinstance(payload, dict):
                raise GenerationValidationError(f"checkpoint shard must decode to an object: {shard_json}")
            candidate_score_row, token_rows = _validate_candidate_checkpoint(payload, shard_json, metadata)
            candidate_id = str(candidate_score_row["candidate_id"])
            if candidate_id in completed:
                raise GenerationValidationError(f"duplicate candidate checkpoint key {candidate_id!r}")
            storage = payload.get("full_logits_storage")
            if not isinstance(storage, dict):
                raise GenerationValidationError(f"checkpoint shard is missing full_logits_storage: {shard_json}")
            sidecar_path = _resolve_checkpoint_sidecar_path(shard_json, storage.get("path"))
            if sidecar_path is None:
                raise GenerationValidationError(f"checkpoint shard sidecar path is invalid: {shard_json}")
            completed[candidate_id] = (candidate_score_row, token_rows, sidecar_path)
        except (OSError, json.JSONDecodeError, GenerationValidationError) as exc:
            skipped.append((shard_dir.name, str(exc)))
    if skipped:
        head = ", ".join(name for name, _ in skipped[:3])
        print(
            f"[checkpoint] preserved {len(skipped)} unreadable candidate shard(s); first: {head}",
            file=sys.stderr,
        )
    return completed


def _write_free_sample_checkpoint(
    *,
    checkpoint_root: Path,
    metadata: dict[str, Any],
    sample: dict[str, Any],
    sidecar_path: Path,
    storage: dict[str, Any],
) -> None:
    prompt_id = str(sample["prompt_id"])
    sample_index = int(sample["sample_index"])
    complete_dir = checkpoint_root / f"prompt-{_safe_key_part(prompt_id)}-sample-{sample_index:02d}"
    temp_dir = checkpoint_root / f".tmp-{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        target_sidecar = temp_dir / "full_logits.parquet"
        shutil.move(str(sidecar_path), target_sidecar)
        shard_json = temp_dir / "shard.json"
        payload = dict(metadata)
        payload.update(
            {
                "checkpoint_artifact_type": "free_sample_rows_checkpoint_shard",
                "created_at": _now_iso(),
                "prompt_id": prompt_id,
                "sample_index": sample_index,
                "sample": _rewrite_free_sample_ref(sample, Path("full_logits.parquet")),
                "full_logits_storage": dict(storage, path="full_logits.parquet"),
            }
        )
        write_json(shard_json, payload)
        _validate_free_sample_checkpoint(payload, shard_json, metadata)
        _atomic_promote_dir(temp_dir, complete_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def _write_candidate_checkpoint(
    *,
    checkpoint_root: Path,
    metadata: dict[str, Any],
    candidate_score_row: dict[str, Any],
    token_rows: list[dict[str, Any]],
    sidecar_path: Path,
    storage: dict[str, Any],
) -> None:
    candidate_id = str(candidate_score_row["candidate_id"])
    complete_dir = checkpoint_root / f"candidate-{_safe_key_part(candidate_id)}"
    temp_dir = checkpoint_root / f".tmp-{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        target_sidecar = temp_dir / "full_logits.parquet"
        shutil.move(str(sidecar_path), target_sidecar)
        shard_json = temp_dir / "shard.json"
        payload = dict(metadata)
        payload.update(
            {
                "checkpoint_artifact_type": "teacher_forced_candidate_scores_checkpoint_shard",
                "created_at": _now_iso(),
                "candidate_id": candidate_id,
                "candidate_score_row": candidate_score_row,
                "token_rows": [_rewrite_candidate_token_ref(token_row, Path("full_logits.parquet")) for token_row in token_rows],
                "full_logits_storage": dict(storage, path="full_logits.parquet"),
            }
        )
        write_json(shard_json, payload)
        _validate_candidate_checkpoint(payload, shard_json, metadata)
        _atomic_promote_dir(temp_dir, complete_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


class LocalModelGenerationAdapter(ModelLogitsPort):
    """Generates split prompt-level free samples and candidate-level teacher-forced logits artifacts."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = _load_generation_config(config_path)

    def build_artifact(self, *, out_path: str, prompt_rows_path: str | None = None) -> dict[str, Any]:
        prompt_rows = load_prompt_rows(self.config, prompt_rows_path=prompt_rows_path)
        prompt_groups = tuple(_prompt_group_from_prompt_row(prompt_row) for prompt_row in prompt_rows)
        artifact = _build_live_free_sample_artifact(self.config, prompt_groups)
        validate_generation_payload(artifact)
        materialized_path = _artifact_json_path(out_path)
        write_json(materialized_path, artifact)
        return {
            "artifact_path": str(materialized_path),
            "prompt_group_count": len(prompt_groups),
            "sample_count": len(artifact["samples"]),
            "fixture_mode": False,
            "artifact_type": artifact["artifact_type"],
        }

    def write_fixture(self, *, out_path: str, variant: str = "full_logits", prompt_rows_path: str | None = None) -> dict[str, Any]:
        if variant not in {"full_logits", "missing_full_logits"}:
            raise GenerationConfigError("fixture variant must be 'full_logits' or 'missing_full_logits'")
        prompt_rows = load_prompt_rows(self.config, prompt_rows_path=prompt_rows_path)
        prompt_groups = tuple(_prompt_group_from_prompt_row(prompt_row) for prompt_row in prompt_rows)
        artifact = build_free_sample_fixture_artifact(self.config, prompt_groups, variant=variant)
        validate_generation_payload(artifact)
        materialized_path = _artifact_json_path(out_path)
        write_json(materialized_path, artifact)
        return {
            "artifact_path": str(materialized_path),
            "prompt_group_count": len(prompt_groups),
            "sample_count": len(artifact["samples"]),
            "fixture_mode": True,
            "fixture_variant": variant,
            "artifact_type": artifact["artifact_type"],
        }

    def build_free_sample_artifact(self, *, out_path: str, prompt_groups_path: str, resume: bool = False) -> dict[str, Any]:
        prompt_groups = load_prompt_groups(prompt_groups_path)
        prompt_ids = [prompt_group.prompt_id for prompt_group in prompt_groups]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise GenerationConfigError("prompt groups must have unique prompt_id values for checkpoint resume")
        runtime = _LiveModelRuntime(self.config)
        free_sample_token_limit = _free_sample_token_limit(self.config)
        materialized_path = _artifact_json_path(out_path)
        _enforce_full_logits_budget(
            config=self.config,
            out_path=out_path,
            artifact_type="free_sample_rows",
            token_positions=len(prompt_groups) * FREE_SAMPLE_COUNT * free_sample_token_limit,
            vocab_size=runtime.vocab_size(),
        )
        artifact = _base_artifact(
            self.config,
            artifact_type="free_sample_rows",
            fixture_mode=False,
            has_full_logits=True,
        )
        artifact["model_name"] = runtime.model_name
        artifact["tokenizer_name"] = runtime.tokenizer_name
        storage_dtype = _runtime_full_logits_dtype(self.config)
        sampling_config = dict(runtime.generation_config)
        sampling_config["do_sample"] = True
        artifact["generation_config"] = sampling_config
        checkpoint_root = _phase_checkpoint_root(materialized_path, artifact_type="free_sample_rows")
        if checkpoint_root.exists() and not resume:
            raise GenerationConfigError(f"refusing to overwrite existing free-sample checkpoint root {checkpoint_root}; use --resume or --force")
        checkpoint_metadata = _checkpoint_metadata(
            config=self.config,
            runtime=runtime,
            artifact_type="free_sample_rows",
            input_fingerprint=_rows_fingerprint(prompt_groups),
        )
        completed = _load_free_sample_checkpoints(checkpoint_root, checkpoint_metadata) if resume else {}
        expected_sample_keys = {(prompt_group.prompt_id, sample_index) for prompt_group in prompt_groups for sample_index in range(FREE_SAMPLE_COUNT)}
        unexpected_keys = set(completed) - expected_sample_keys
        if unexpected_keys:
            for prompt_id, sample_index in unexpected_keys:
                stale_dir = checkpoint_root / f"prompt-{_safe_key_part(prompt_id)}-sample-{sample_index:02d}"
                if stale_dir.exists():
                    shutil.rmtree(stale_dir)
            completed = {key: value for key, value in completed.items() if key in expected_sample_keys}
        missing_sample_specs = [
            (prompt_group, sample_index, 0)
            for prompt_group in prompt_groups
            for sample_index in range(FREE_SAMPLE_COUNT)
            if (prompt_group.prompt_id, sample_index) not in completed
        ]
        free_sample_batch_size = _runtime_batch_size(self.config, "free_sample_batch_size")
        max_invalid_attempts = (
            _answer_only_max_invalid_attempts(sampling_config) if _answer_only_enabled(sampling_config) else 1
        )
        invalid_attempt_count = 0
        while missing_sample_specs:
            missing_chunk = missing_sample_specs[:free_sample_batch_size]
            missing_sample_specs = missing_sample_specs[free_sample_batch_size:]
            writers: list[_FullLogitsParquetWriter] = []
            shard_parquet_paths: list[Path] = []
            batch_specs: list[tuple[PromptGroup, int, _FullLogitsParquetWriter, Path, int]] = []
            try:
                for prompt_group, sample_index, attempt_index in missing_chunk:
                    shard_parquet_path = checkpoint_root / f".tmp-logits-{uuid4().hex}.parquet"
                    writer = _FullLogitsParquetWriter(
                        shard_parquet_path,
                        artifact_type="free_sample_rows",
                        storage_dtype=storage_dtype,
                    )
                    writers.append(writer)
                    shard_parquet_paths.append(shard_parquet_path)
                    seed = _sample_seed(self.config, prompt_id=prompt_group.prompt_id, sample_index=sample_index) + (
                        attempt_index * 1_000_003
                    )
                    batch_specs.append(
                        (
                            prompt_group,
                            sample_index,
                            writer,
                            shard_parquet_path,
                            seed,
                        )
                    )
                samples_batch = runtime.write_free_sample_rows_batch(batch_specs)
                for (prompt_group, sample_index, attempt_index), sample, writer, shard_parquet_path in zip(
                    missing_chunk, samples_batch, writers, shard_parquet_paths, strict=True
                ):
                    storage = writer.close()
                    protocol = sample.get("answer_only_protocol")
                    if isinstance(protocol, dict):
                        protocol["attempt_index"] = attempt_index
                    answer_only_problems = _answer_only_validation_problems(
                        sample,
                        sampling_config,
                        label=f"prompt_id={prompt_group.prompt_id!r} sample_index={sample_index} attempt={attempt_index}",
                    )
                    if answer_only_problems:
                        invalid_attempt_count += 1
                        if shard_parquet_path.exists():
                            shard_parquet_path.unlink()
                        if attempt_index + 1 < max_invalid_attempts:
                            missing_sample_specs.append((prompt_group, sample_index, attempt_index + 1))
                            continue
                        raise GenerationValidationError(
                            "Free-sample answer-only bounded resampling exhausted:\n- " + "\n- ".join(answer_only_problems)
                        )
                    _write_free_sample_checkpoint(
                        checkpoint_root=checkpoint_root,
                        metadata=checkpoint_metadata,
                        sample=sample,
                        sidecar_path=shard_parquet_path,
                        storage=storage,
                    )
            except Exception:
                for writer in writers:
                    try:
                        if writer.writer is not None:
                            writer.writer.close()
                    except Exception:
                        pass
                for shard_parquet_path in shard_parquet_paths:
                    if shard_parquet_path.exists():
                        shard_parquet_path.unlink()
                raise
        completed = _load_free_sample_checkpoints(checkpoint_root, checkpoint_metadata)
        missing_keys = expected_sample_keys - set(completed)
        if missing_keys:
            preview = sorted(repr(key) for key in missing_keys)[:5]
            raise GenerationValidationError(f"free-sample checkpoints are missing {len(missing_keys)} expected shards: {preview}")
        parquet_path = _parquet_path_for_artifact(materialized_path)
        temp_parquet_path = parquet_path.with_name(f".{parquet_path.name}.tmp-{uuid4().hex}")
        writer = _FullLogitsParquetWriter(
            temp_parquet_path,
            artifact_type="free_sample_rows",
            storage_dtype=storage_dtype,
        )
        samples: list[dict[str, Any]] = []
        try:
            for prompt_group in prompt_groups:
                for sample_index in range(FREE_SAMPLE_COUNT):
                    sample, shard_sidecar_path = completed[(prompt_group.prompt_id, sample_index)]
                    _copy_parquet_rows(shard_sidecar_path, writer)
                    samples.append(_rewrite_free_sample_ref(sample, parquet_path))
            storage = writer.close()
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            temp_parquet_path.replace(parquet_path)
            artifact["full_logits_storage"] = dict(storage, path=str(parquet_path))
        except Exception:
            if writer.writer is not None:
                writer.writer.close()
            if temp_parquet_path.exists():
                temp_parquet_path.unlink()
            raise
        artifact.update(
            {
                "sample_count_per_prompt": FREE_SAMPLE_COUNT,
                "prompt_group_count": len(prompt_groups),
                "answer_only_invalid_attempt_count": invalid_attempt_count,
                "answer_only_max_invalid_attempts": max_invalid_attempts,
                "samples": samples,
            }
        )
        validate_generation_payload(artifact)
        _write_validated_generation_artifact(materialized_path, artifact)
        return {
            "artifact_path": str(materialized_path),
            "prompt_group_count": len(prompt_groups),
            "sample_count": len(artifact["samples"]),
            "fixture_mode": False,
            "artifact_type": artifact["artifact_type"],
        }

    def write_free_sample_fixture(self, *, out_path: str, prompt_groups_path: str, variant: str = "full_logits") -> dict[str, Any]:
        if variant not in {"full_logits", "missing_full_logits"}:
            raise GenerationConfigError("fixture variant must be 'full_logits' or 'missing_full_logits'")
        prompt_groups = load_prompt_groups(prompt_groups_path)
        artifact = build_free_sample_fixture_artifact(self.config, prompt_groups, variant=variant)
        validate_generation_payload(artifact)
        materialized_path = _artifact_json_path(out_path)
        write_json(materialized_path, artifact)
        return {
            "artifact_path": str(materialized_path),
            "prompt_group_count": len(prompt_groups),
            "sample_count": len(artifact["samples"]),
            "fixture_mode": True,
            "fixture_variant": variant,
            "artifact_type": artifact["artifact_type"],
        }

    def build_candidate_score_artifact(self, *, out_path: str, candidates_path: str, resume: bool = False) -> dict[str, Any]:
        candidate_rows = load_candidate_rows(candidates_path)
        candidate_ids = [candidate_row.candidate_id for candidate_row in candidate_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise GenerationConfigError("candidate rows must have unique candidate_id values for checkpoint resume")
        runtime = _LiveModelRuntime(self.config)
        token_positions = sum(runtime.candidate_token_count(candidate_row) for candidate_row in candidate_rows)
        materialized_path = _artifact_json_path(out_path)
        _enforce_full_logits_budget(
            config=self.config,
            out_path=out_path,
            artifact_type="teacher_forced_candidate_scores",
            token_positions=token_positions,
            vocab_size=runtime.vocab_size(),
        )
        artifact = _base_artifact(
            self.config,
            artifact_type="teacher_forced_candidate_scores",
            fixture_mode=False,
            has_full_logits=True,
        )
        artifact["model_name"] = runtime.model_name
        artifact["tokenizer_name"] = runtime.tokenizer_name
        storage_dtype = _runtime_full_logits_dtype(self.config)
        checkpoint_root = _phase_checkpoint_root(materialized_path, artifact_type="teacher_forced_candidate_scores")
        if checkpoint_root.exists() and not resume:
            raise GenerationConfigError(f"refusing to overwrite existing candidate-score checkpoint root {checkpoint_root}; use --resume or --force")
        checkpoint_metadata = _checkpoint_metadata(
            config=self.config,
            runtime=runtime,
            artifact_type="teacher_forced_candidate_scores",
            input_fingerprint=_rows_fingerprint(candidate_rows),
        )
        completed = _load_candidate_checkpoints(checkpoint_root, checkpoint_metadata) if resume else {}
        expected_candidate_ids = {candidate_row.candidate_id for candidate_row in candidate_rows}
        unexpected_candidate_ids = set(completed) - expected_candidate_ids
        if unexpected_candidate_ids:
            for candidate_id in unexpected_candidate_ids:
                stale_dir = checkpoint_root / f"candidate-{_safe_key_part(candidate_id)}"
                if stale_dir.exists():
                    shutil.rmtree(stale_dir)
            completed = {key: value for key, value in completed.items() if key in expected_candidate_ids}
        missing_candidate_rows = [candidate_row for candidate_row in candidate_rows if candidate_row.candidate_id not in completed]
        candidate_score_batch_size = _runtime_batch_size(self.config, "candidate_score_batch_size")
        for missing_chunk in _chunks(missing_candidate_rows, candidate_score_batch_size):
            writers: list[_FullLogitsParquetWriter] = []
            shard_parquet_paths: list[Path] = []
            batch_specs: list[tuple[CandidateRow, _FullLogitsParquetWriter, Path]] = []
            try:
                for candidate_row in missing_chunk:
                    shard_parquet_path = checkpoint_root / f".tmp-logits-{uuid4().hex}.parquet"
                    writer = _FullLogitsParquetWriter(
                        shard_parquet_path,
                        artifact_type="teacher_forced_candidate_scores",
                        storage_dtype=storage_dtype,
                    )
                    writers.append(writer)
                    shard_parquet_paths.append(shard_parquet_path)
                    batch_specs.append((candidate_row, writer, shard_parquet_path))
                score_batches = runtime.write_teacher_forced_candidate_scores_batch(batch_specs)
                for (candidate_score_row, token_rows), writer, shard_parquet_path in zip(
                    score_batches, writers, shard_parquet_paths, strict=True
                ):
                    storage = writer.close()
                    _write_candidate_checkpoint(
                        checkpoint_root=checkpoint_root,
                        metadata=checkpoint_metadata,
                        candidate_score_row=candidate_score_row,
                        token_rows=token_rows,
                        sidecar_path=shard_parquet_path,
                        storage=storage,
                    )
            except Exception:
                for writer in writers:
                    try:
                        if writer.writer is not None:
                            writer.writer.close()
                    except Exception:
                        pass
                for shard_parquet_path in shard_parquet_paths:
                    if shard_parquet_path.exists():
                        shard_parquet_path.unlink()
                raise
        completed = _load_candidate_checkpoints(checkpoint_root, checkpoint_metadata)
        missing_candidate_ids = expected_candidate_ids - set(completed)
        if missing_candidate_ids:
            preview = sorted(missing_candidate_ids)[:5]
            raise GenerationValidationError(
                f"candidate-score checkpoints are missing {len(missing_candidate_ids)} expected shards: {preview}"
            )
        parquet_path = _parquet_path_for_artifact(materialized_path)
        temp_parquet_path = parquet_path.with_name(f".{parquet_path.name}.tmp-{uuid4().hex}")
        writer = _FullLogitsParquetWriter(
            temp_parquet_path,
            artifact_type="teacher_forced_candidate_scores",
            storage_dtype=storage_dtype,
        )
        candidate_score_rows: list[dict[str, Any]] = []
        token_score_rows: list[dict[str, Any]] = []
        try:
            for candidate_row in candidate_rows:
                candidate_score_row, token_rows, shard_sidecar_path = completed[candidate_row.candidate_id]
                _copy_parquet_rows(shard_sidecar_path, writer)
                candidate_score_rows.append(candidate_score_row)
                token_score_rows.extend(_rewrite_candidate_token_ref(token_row, parquet_path) for token_row in token_rows)
            storage = writer.close()
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            temp_parquet_path.replace(parquet_path)
            artifact["full_logits_storage"] = dict(storage, path=str(parquet_path))
        except Exception:
            if writer.writer is not None:
                writer.writer.close()
            if temp_parquet_path.exists():
                temp_parquet_path.unlink()
            raise
        artifact.update(
            {
                "candidate_count": len(candidate_rows),
                "candidate_score_rows": candidate_score_rows,
                "token_score_rows": token_score_rows,
                "candidate_scoring_mode": "teacher_forced",
                "prompt_prefix_scoring_excluded": True,
                "free_sample_count": FREE_SAMPLE_COUNT,
            }
        )
        validate_generation_payload(artifact)
        _write_validated_generation_artifact(materialized_path, artifact)
        return {
            "artifact_path": str(materialized_path),
            "candidate_count": len(artifact["candidate_score_rows"]),
            "token_score_count": len(artifact["token_score_rows"]),
            "fixture_mode": False,
            "artifact_type": artifact["artifact_type"],
        }

    def write_candidate_score_fixture(self, *, out_path: str, candidates_path: str, variant: str = "full_logits") -> dict[str, Any]:
        if variant not in {"full_logits", "missing_full_logits"}:
            raise GenerationConfigError("fixture variant must be 'full_logits' or 'missing_full_logits'")
        candidate_rows = load_candidate_rows(candidates_path)
        artifact = build_candidate_score_fixture_artifact(self.config, candidate_rows, variant=variant)
        validate_generation_payload(artifact)
        materialized_path = _artifact_json_path(out_path)
        write_json(materialized_path, artifact)
        return {
            "artifact_path": str(materialized_path),
            "candidate_count": len(artifact["candidate_score_rows"]),
            "token_score_count": len(artifact["token_score_rows"]),
            "fixture_mode": True,
            "fixture_variant": variant,
            "artifact_type": artifact["artifact_type"],
        }
