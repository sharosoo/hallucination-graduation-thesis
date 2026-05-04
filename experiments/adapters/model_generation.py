"""Self-contained local generation/export adapter with row-level full logits preservation."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from experiments.domain import ModelResponse, PromptRow
from experiments.ports import ModelLogitsPort

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMULA_MANIFEST_REF = "experiments/literature/formula_notes.md"
DEFAULT_DATASET_MANIFEST_REF = "experiments/configs/datasets.yaml"
DEFAULT_LOGITS_SCHEMA_VERSION = "generation_logits_v1"


class GenerationConfigError(RuntimeError):
    """Raised when the generation configuration or prompt rows are invalid."""


class GenerationDependencyError(RuntimeError):
    """Raised when optional ML dependencies are unavailable."""


class ModelLoadError(RuntimeError):
    """Raised when the configured model/tokenizer cannot be loaded."""


class GenerationValidationError(RuntimeError):
    """Raised when a logits artifact violates the required schema."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_to_tuples(metadata: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not isinstance(metadata, dict):
        return ()
    entries: list[tuple[str, str]] = []
    for key in sorted(metadata):
        entries.append((str(key), json.dumps(metadata[key], ensure_ascii=False, sort_keys=True)))
    return tuple(entries)


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
                prompt_value = f"Context: {context_text}\nQuestion: {question_text}\nAnswer:"
            else:
                prompt_value = f"Question: {question_text}\nAnswer:"
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
    if not path.exists():
        raise GenerationConfigError(f"prompt rows path does not exist: {path}")
    if path.suffix == ".jsonl":
        rows: list[PromptRow] = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise GenerationConfigError(f"prompt rows JSONL line {index + 1} must be an object")
                rows.append(_normalize_prompt_entry(payload, len(rows)))
        return tuple(rows)

    payload = load_json(path)
    if isinstance(payload, dict):
        if isinstance(payload.get("prompt_rows"), list):
            raw_rows = payload["prompt_rows"]
        elif isinstance(payload.get("samples"), list):
            raw_rows = payload["samples"]
        else:
            raise GenerationConfigError(f"prompt row file {path} must contain 'prompt_rows' or 'samples'")
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raise GenerationConfigError(f"prompt row file {path} must decode to a list or object")

    rows = []
    for index, entry in enumerate(raw_rows):
        if not isinstance(entry, dict):
            raise GenerationConfigError(f"prompt row {index} in {path} must be an object")
        rows.append(_normalize_prompt_entry(entry, index))
    return tuple(rows)


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
            rows = []
            for index, entry in enumerate(prompt_rows):
                if not isinstance(entry, dict):
                    raise GenerationConfigError(f"prompt_rows[{index}] must be an object")
                rows.append(_normalize_prompt_entry(entry, index))
            rows = tuple(rows)
    if not rows:
        raise GenerationConfigError("no prompt rows were loaded")
    return rows


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


def _fixture_response(sample_id: str) -> ModelResponse:
    full_logits = (
        (3.0, 1.0, -1.0, -2.0),
        (0.0, 2.5, -0.5, -1.0),
    )
    selected_token_ids = (0, 1)
    selected_token_logits = tuple(full_logits[index][token_id] for index, token_id in enumerate(selected_token_ids))
    return ModelResponse(
        sample_id=sample_id,
        response_text="Alpha Beta",
        generated_token_ids=selected_token_ids,
        selected_token_ids=selected_token_ids,
        selected_token_logits=selected_token_logits,
        full_logits=full_logits,
        logsumexp_values=tuple(_logsumexp_python(list(step)) for step in full_logits),
        decoded_tokens=("Alpha", "Beta"),
        full_vocabulary_logits=True,
        metadata=(("fixture", "true"),),
    )


def _fixture_sample(prompt_row: PromptRow, *, include_full_logits: bool) -> dict[str, Any]:
    response = _fixture_response(prompt_row.sample_id)
    sample = {
        "dataset": prompt_row.dataset,
        "split_id": prompt_row.split_id,
        "sample_id": prompt_row.sample_id,
        "prompt": prompt_row.prompt,
        "question": prompt_row.question,
        "context": prompt_row.context,
        "response_text": response.response_text,
        "generated_token_ids": list(response.generated_token_ids),
        "selected_token_ids": list(response.selected_token_ids),
        "selected_token_logits": list(response.selected_token_logits),
        "generated_tokens": list(response.decoded_tokens),
        "logsumexp": list(response.logsumexp_values),
        "full_vocabulary_logits": include_full_logits,
        "metadata": _metadata_dict(prompt_row.metadata) | {"fixture": True},
    }
    if include_full_logits:
        sample["full_logits"] = [list(step) for step in response.full_logits]
    return sample


def build_fixture_artifact(config: dict[str, Any], prompt_rows: tuple[PromptRow, ...], variant: str) -> dict[str, Any]:
    generation = _config_generation_section(config)
    raw_model_config = config.get("model")
    model_config: dict[str, Any] = raw_model_config if isinstance(raw_model_config, dict) else {}
    include_full_logits = variant == "full_logits"
    created_at = _now_iso()
    artifact = {
        "run_id": f"fixture-generation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "artifact_type": "generation_rows",
        "created_at": created_at,
        "model_name": str(model_config.get("model_name") or "fixture-local-model"),
        "tokenizer_name": str(model_config.get("tokenizer_name") or model_config.get("model_name") or "fixture-local-model"),
        "generation_config": generation,
        "logits_schema_version": str(config.get("logits_schema_version") or DEFAULT_LOGITS_SCHEMA_VERSION),
        "formula_manifest_ref": str(config.get("formula_manifest_ref") or DEFAULT_FORMULA_MANIFEST_REF),
        "dataset_manifest_ref": str(config.get("dataset_manifest_ref") or DEFAULT_DATASET_MANIFEST_REF),
        "has_logits": include_full_logits,
        "has_full_logits": include_full_logits,
        "full_vocabulary_logits": include_full_logits,
        "fixture_mode": True,
        "samples": [_fixture_sample(prompt_row, include_full_logits=include_full_logits) for prompt_row in prompt_rows],
    }
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


def _run_local_generation(config: dict[str, Any], prompt_rows: tuple[PromptRow, ...]) -> dict[str, Any]:
    torch_module, auto_model_cls, auto_tokenizer_cls = _resolve_runtime_modules()
    raw_runtime_config = config.get("runtime")
    runtime_config: dict[str, Any] = raw_runtime_config if isinstance(raw_runtime_config, dict) else {}
    raw_model_config = config.get("model")
    model_config: dict[str, Any] = raw_model_config if isinstance(raw_model_config, dict) else {}
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

    max_new_tokens = int(generation_config.get("max_new_tokens", 1) or 1)
    stop_on_eos = bool(generation_config.get("stop_on_eos", True))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    created_at = _now_iso()
    samples: list[dict[str, Any]] = []

    for prompt_row in prompt_rows:
        encoded = tokenizer(prompt_row.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        generated_token_ids: list[int] = []
        selected_token_logits: list[float] = []
        logsumexp_values: list[float] = []
        full_logits: list[list[float]] = []
        generated_tokens: list[str] = []

        current_input_ids = input_ids
        current_attention_mask = attention_mask
        for _step in range(max_new_tokens):
            with torch_module.no_grad():
                outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
            step_logits = outputs.logits[:, -1, :]
            next_token_id = _select_token_id(torch_module, step_logits, generation_config)
            generated_token_ids.append(next_token_id)
            selected_token_logits.append(float(step_logits[0, next_token_id].item()))
            logsumexp_values.append(float(torch_module.logsumexp(step_logits, dim=-1).item()))
            full_logits.append([float(value) for value in step_logits[0].detach().cpu().tolist()])
            generated_tokens.append(_token_to_text(tokenizer, next_token_id))

            next_token_tensor = torch_module.tensor([[next_token_id]], device=device, dtype=current_input_ids.dtype)
            current_input_ids = torch_module.cat([current_input_ids, next_token_tensor], dim=-1)
            if current_attention_mask is not None:
                current_attention_mask = torch_module.cat(
                    [current_attention_mask, torch_module.ones((1, 1), device=device, dtype=current_attention_mask.dtype)],
                    dim=-1,
                )
            if stop_on_eos and eos_token_id is not None and next_token_id == int(eos_token_id):
                break

        response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = ModelResponse(
            sample_id=prompt_row.sample_id,
            response_text=response_text,
            generated_token_ids=tuple(generated_token_ids),
            selected_token_ids=tuple(generated_token_ids),
            selected_token_logits=tuple(selected_token_logits),
            full_logits=tuple(tuple(step) for step in full_logits),
            logsumexp_values=tuple(logsumexp_values),
            decoded_tokens=tuple(generated_tokens),
            full_vocabulary_logits=True,
        )
        samples.append(
            {
                "dataset": prompt_row.dataset,
                "split_id": prompt_row.split_id,
                "sample_id": prompt_row.sample_id,
                "prompt": prompt_row.prompt,
                "question": prompt_row.question,
                "context": prompt_row.context,
                "response_text": response.response_text,
                "generated_token_ids": list(response.generated_token_ids),
                "selected_token_ids": list(response.selected_token_ids),
                "selected_token_logits": list(response.selected_token_logits),
                "generated_tokens": list(response.decoded_tokens),
                "full_logits": [list(step) for step in response.full_logits],
                "logsumexp": list(response.logsumexp_values),
                "full_vocabulary_logits": True,
                "metadata": _metadata_dict(prompt_row.metadata),
            }
        )

    return {
        "run_id": f"generation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "artifact_type": "generation_rows",
        "created_at": created_at,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "generation_config": generation_config,
        "logits_schema_version": str(config.get("logits_schema_version") or DEFAULT_LOGITS_SCHEMA_VERSION),
        "formula_manifest_ref": str(config.get("formula_manifest_ref") or DEFAULT_FORMULA_MANIFEST_REF),
        "dataset_manifest_ref": str(config.get("dataset_manifest_ref") or DEFAULT_DATASET_MANIFEST_REF),
        "has_logits": True,
        "has_full_logits": True,
        "full_vocabulary_logits": True,
        "fixture_mode": False,
        "samples": samples,
    }


def _is_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)


def _is_numeric_matrix(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_numeric_list(item) for item in value)


def validate_generation_payload(payload: dict[str, Any]) -> None:
    problems: list[str] = []
    required_top_level = (
        "run_id",
        "created_at",
        "model_name",
        "tokenizer_name",
        "generation_config",
        "logits_schema_version",
        "formula_manifest_ref",
        "dataset_manifest_ref",
        "samples",
    )
    for field_name in required_top_level:
        if field_name not in payload:
            problems.append(f"missing top-level field {field_name}")

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        problems.append("artifact must contain a non-empty 'samples' list")
    if payload.get("has_logits") is not True:
        problems.append("artifact must set has_logits=true")
    if payload.get("has_full_logits") is not True:
        problems.append("artifact must set has_full_logits=true")
    if payload.get("full_vocabulary_logits") is not True:
        problems.append("artifact must set full_vocabulary_logits=true")
    if payload.get("logits_schema_version") != DEFAULT_LOGITS_SCHEMA_VERSION:
        problems.append(
            f"logits_schema_version must equal {DEFAULT_LOGITS_SCHEMA_VERSION!r}; got {payload.get('logits_schema_version')!r}"
        )

    if isinstance(samples, list):
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                problems.append(f"sample {index} must be an object")
                continue
            for field_name in (
                "dataset",
                "split_id",
                "sample_id",
                "prompt",
                "response_text",
                "generated_token_ids",
                "selected_token_ids",
                "selected_token_logits",
                "generated_tokens",
                "logsumexp",
                "full_vocabulary_logits",
            ):
                if field_name not in sample:
                    problems.append(f"sample {index} missing field {field_name}")
            raw_full_logits = sample.get("full_logits")
            if not _is_numeric_matrix(raw_full_logits):
                problems.append(
                    f"sample {index} must include nested numeric 'full_logits' rows for manifest full-logits detection"
                )
                continue
            full_logits = cast(list[list[float]], raw_full_logits)
            if not _is_numeric_list(sample.get("logsumexp")):
                problems.append(f"sample {index} must include numeric 'logsumexp' values")
            if not _is_numeric_list(sample.get("generated_token_ids")):
                problems.append(f"sample {index} must include numeric 'generated_token_ids'")
            if not _is_numeric_list(sample.get("selected_token_ids")):
                problems.append(f"sample {index} must include numeric 'selected_token_ids'")
            if not _is_numeric_list(sample.get("selected_token_logits")):
                problems.append(f"sample {index} must include numeric 'selected_token_logits'")
            if not isinstance(sample.get("generated_tokens"), list) or not all(
                isinstance(item, str) for item in sample.get("generated_tokens", [])
            ):
                problems.append(f"sample {index} must include string 'generated_tokens'")
            if sample.get("full_vocabulary_logits") is not True:
                problems.append(f"sample {index} must set full_vocabulary_logits=true")
            step_count = len(full_logits)
            for aligned_name in ("generated_token_ids", "selected_token_ids", "selected_token_logits", "generated_tokens", "logsumexp"):
                value = sample.get(aligned_name)
                if isinstance(value, list):
                    value_list = value
                else:
                    value_list = None
                if value_list is not None and len(value_list) != step_count:
                    problems.append(
                        f"sample {index} field {aligned_name} length {len(value_list)} does not match full_logits step count {step_count}"
                    )

    if problems:
        raise GenerationValidationError("Generation logits validation failed:\n- " + "\n- ".join(problems))


def validate_generation_artifact(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise GenerationValidationError(f"artifact must decode to an object: {path}")
    validate_generation_payload(payload)
    return payload


class LocalModelGenerationAdapter(ModelLogitsPort):
    """Generates repo-owned row artifacts with preserved row-level full logits."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = _load_generation_config(config_path)

    def build_artifact(self, *, out_path: str, prompt_rows_path: str | None = None) -> dict[str, Any]:
        prompt_rows = load_prompt_rows(self.config, prompt_rows_path=prompt_rows_path)
        artifact = _run_local_generation(self.config, prompt_rows)
        validate_generation_payload(artifact)
        materialized_path = _artifact_json_path(out_path)
        write_json(materialized_path, artifact)
        return {
            "artifact_path": str(materialized_path),
            "row_count": len(artifact["samples"]),
            "model_name": artifact["model_name"],
            "tokenizer_name": artifact["tokenizer_name"],
            "fixture_mode": False,
        }

    def write_fixture(self, *, out_path: str, variant: str = "full_logits", prompt_rows_path: str | None = None) -> dict[str, Any]:
        if variant not in {"full_logits", "missing_full_logits"}:
            raise GenerationConfigError("fixture variant must be 'full_logits' or 'missing_full_logits'")
        prompt_rows = load_prompt_rows(self.config, prompt_rows_path=prompt_rows_path)
        artifact = build_fixture_artifact(self.config, prompt_rows, variant=variant)
        materialized_path = _artifact_json_path(out_path)
        write_json(materialized_path, artifact)
        return {
            "artifact_path": str(materialized_path),
            "row_count": len(artifact["samples"]),
            "model_name": artifact["model_name"],
            "tokenizer_name": artifact["tokenizer_name"],
            "fixture_mode": True,
            "fixture_variant": variant,
        }
