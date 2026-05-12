"""HuggingFace dataset loader for single-candidate (SE 5-dataset) track.

기존 hf_datasets.py 의 paired (right + hallucinated) 가정을 우회하고, 5 SE
datasets (TriviaQA / SQuAD-1.1 / BioASQ / NQ-Open / SVAMP) 의 single
ground-truth answer 만 candidate 로 산출한다.

Generation-level evaluation 전용 — candidate-level 환각 라벨이 없음 (post-hoc
NLI 매칭으로 is_correct 산출).

Prompt template (Farquhar 2024 sentence-length 그대로):
  Answer the following question in a single brief but complete sentence.
  Question: {question}
  Answer:

Context (passage) 는 의도적으로 누락 (Farquhar paper §Methods).

산출 schema 는 paired track 과 동일 (prompt_groups.jsonl + candidate_rows.jsonl)
하되 candidate_role 은 항상 "right" 만, candidate_rows_per_prompt = 1.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable

from experiments.adapters.hf_datasets import (
    _clean_text,
    _prompt_hash,
    _require_text,
    _stable_id,
    _stable_slug,
    stable_sample_id,
)


SE_SENTENCE_LENGTH_PROMPT_TEMPLATE = (
    "Answer the following question in a single brief but complete sentence.\n"
    "Question: {question}\n"
    "Answer:"
)


def build_se_prompt(question: str) -> str:
    return SE_SENTENCE_LENGTH_PROMPT_TEMPLATE.format(question=question.strip())


# ---------- per-dataset row mappers ----------

def _map_triviaqa(row: dict[str, Any]) -> tuple[str, str, list[str], dict[str, Any]]:
    question = _require_text(row.get("question"), field_name="question")
    answer_obj = row.get("answer") or {}
    if not isinstance(answer_obj, dict):
        raise ValueError("TriviaQA: 'answer' must be dict")
    primary = _require_text(answer_obj.get("value"), field_name="answer.value")
    aliases_raw = answer_obj.get("aliases") or []
    aliases = sorted({_clean_text(a) for a in aliases_raw if isinstance(a, str) and a.strip()})
    if primary not in aliases:
        aliases = [primary] + aliases
    metadata = {
        "question_id": str(row.get("question_id") or ""),
        "question_source": str(row.get("question_source") or ""),
        "alias_count": len(aliases),
    }
    return question, primary, aliases, metadata


def _map_squad(row: dict[str, Any]) -> tuple[str, str, list[str], dict[str, Any]]:
    question = _require_text(row.get("question"), field_name="question")
    ans = row.get("answers") or {}
    texts = ans.get("text") or []
    texts_clean = [_clean_text(t) for t in texts if isinstance(t, str) and t.strip()]
    if not texts_clean:
        raise ValueError("SQuAD: empty answers.text")
    primary = texts_clean[0]
    aliases = sorted(set(texts_clean))
    if primary not in aliases:
        aliases = [primary] + aliases
    metadata = {
        "squad_id": str(row.get("id") or ""),
        "title": str(row.get("title") or ""),
        "alias_count": len(aliases),
    }
    return question, primary, aliases, metadata


def _map_nq_open(row: dict[str, Any]) -> tuple[str, str, list[str], dict[str, Any]]:
    question = _require_text(row.get("question"), field_name="question")
    answers = row.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    texts_clean = [_clean_text(t) for t in answers if isinstance(t, str) and t.strip()]
    if not texts_clean:
        raise ValueError("NQ-Open: empty answer list")
    primary = texts_clean[0]
    aliases = sorted(set(texts_clean))
    if primary not in aliases:
        aliases = [primary] + aliases
    metadata = {"alias_count": len(aliases)}
    return question, primary, aliases, metadata


def _map_svamp(row: dict[str, Any]) -> tuple[str, str, list[str], dict[str, Any]]:
    body = _clean_text(row.get("Body") or "")
    q = _clean_text(row.get("Question") or "")
    question = (body + " " + q).strip() if body else q
    if not question:
        raise ValueError("SVAMP: empty question")
    ans_raw = row.get("Answer")
    primary = _clean_text(str(ans_raw))
    if not primary:
        raise ValueError("SVAMP: empty Answer")
    aliases = [primary]
    metadata = {
        "svamp_id": str(row.get("ID") or ""),
        "type": str(row.get("Type") or ""),
        "equation": str(row.get("Equation") or ""),
        "alias_count": 1,
    }
    return question, primary, aliases, metadata


_BIOASQ_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*<context>", re.DOTALL)


def _map_bioasq(row: dict[str, Any]) -> tuple[str, str, list[str], dict[str, Any]]:
    question = _require_text(row.get("question"), field_name="question")
    text = row.get("text") or ""
    if not isinstance(text, str):
        raise ValueError("BioASQ: text field must be str")
    m = _BIOASQ_ANSWER_RE.search(text)
    if m:
        primary = _clean_text(m.group(1))
    else:
        # fallback: first 50 chars (defensive)
        primary = _clean_text(text[:80])
    if not primary:
        raise ValueError("BioASQ: empty parsed answer")
    aliases = [primary]
    metadata = {
        "bioasq_text_len": len(text),
        "parsed_with_tag": bool(m),
        "alias_count": 1,
    }
    return question, primary, aliases, metadata


_DATASET_MAPPERS = {
    "TriviaQA": _map_triviaqa,
    "SQuAD-1.1": _map_squad,
    "NQ-Open": _map_nq_open,
    "SVAMP": _map_svamp,
    "BioASQ": _map_bioasq,
}


# ---------- materialization ----------

@dataclass(frozen=True)
class SinglePromptRecord:
    """One prompt + one right candidate row (post-hoc generation evaluation)."""
    dataset: str
    split_id: str
    source_index: int
    source_row_id: str
    prompt_id: str
    pair_id: str
    candidate_id: str
    question: str
    prompt: str
    candidate_text: str
    candidate_role: str
    aliases: tuple[str, ...]
    metadata: dict[str, Any]


def materialize_se_dataset(
    *,
    dataset_name: str,
    hf_id: str,
    config: str | None,
    split: str,
    split_id: str,
    target_sample_count: int,
    seed: int,
) -> list[SinglePromptRecord]:
    if dataset_name not in _DATASET_MAPPERS:
        raise ValueError(f"Unknown SE dataset: {dataset_name!r}")
    mapper = _DATASET_MAPPERS[dataset_name]

    from datasets import load_dataset
    kwargs: dict[str, Any] = {"split": split}
    if config:
        kwargs["name"] = config
    ds = load_dataset(hf_id, **kwargs)
    n_total = len(ds)

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    selected = indices[: min(target_sample_count, n_total)]

    out: list[SinglePromptRecord] = []
    for source_index in selected:
        try:
            row = ds[int(source_index)]
            question, primary, aliases, ds_metadata = mapper(row)
        except Exception as exc:  # pragma: no cover - per-row guard
            # skip malformed rows
            continue

        prompt = build_se_prompt(question)
        source_row_id = str(row.get("id") or row.get("question_id") or row.get("ID") or source_index)
        # source_index 를 항상 포함해 (dataset, source_index) 단위 uniqueness 보장.
        # TriviaQA 등 일부 dataset 은 question_id 중복 row 가 있어 source_row_id 만으로는
        # 같은 prompt_id 가 산출될 수 있음.
        prompt_identity = f"{dataset_name}:{split_id}:{source_index}:{source_row_id}:{question[:80]}"
        prompt_id = _stable_id(dataset_name, split_id, str(source_index), source_row_id, _prompt_hash(prompt_identity))
        pair_id = f"{prompt_id}:pair"
        candidate_id = f"{pair_id}:right"

        full_metadata = {
            **ds_metadata,
            "dataset_id": dataset_name,
            "split_id": split_id,
            "source_index": int(source_index),
            "source_row_id": source_row_id,
            "alias_list": list(aliases),
            "prompt_template": "se_sentence_length_v1",
            "candidate_role": "right",
            "candidate_pair_policy": "single_candidate_ground_truth",
        }

        out.append(SinglePromptRecord(
            dataset=dataset_name,
            split_id=split_id,
            source_index=int(source_index),
            source_row_id=source_row_id,
            prompt_id=prompt_id,
            pair_id=pair_id,
            candidate_id=candidate_id,
            question=question,
            prompt=prompt,
            candidate_text=primary,
            candidate_role="right",
            aliases=tuple(aliases),
            metadata=full_metadata,
        ))
    return out


def record_to_prompt_group(rec: SinglePromptRecord) -> dict[str, Any]:
    """For prompt_groups.jsonl emission."""
    return {
        "dataset": rec.dataset,
        "split_id": rec.split_id,
        "prompt_id": rec.prompt_id,
        "pair_id": rec.pair_id,
        "question": rec.question,
        "prompt": rec.prompt,
        "context": "",
        "metadata": rec.metadata,
    }


def record_to_candidate_row(rec: SinglePromptRecord) -> dict[str, Any]:
    """For candidate_rows.jsonl emission."""
    return {
        "candidate_id": rec.candidate_id,
        "candidate_role": rec.candidate_role,
        "candidate_text": rec.candidate_text,
        "context": "",
        "dataset": rec.dataset,
        "dataset_id": rec.dataset,
        "is_correct": True,
        "label_source": "dataset_provided_ground_truth",
        "metadata": rec.metadata,
        "pair_id": rec.pair_id,
        "prompt": rec.prompt,
        "prompt_id": rec.prompt_id,
        "question": rec.question,
        "source_row_id": rec.source_row_id,
        "split_id": rec.split_id,
    }
