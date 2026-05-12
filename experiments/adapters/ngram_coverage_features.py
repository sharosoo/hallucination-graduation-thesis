"""N-gram corpus coverage features per candidate.

답변 token 들의 n-gram (3-gram, 5-gram) 이 corpus 에 얼마나 자주 등장하는지
Infini-gram count() 로 산출. entity-level corpus signal 의 한계 (multi-hop fact
가 corpus 에 직접 없음) 를 보완 — phrase-level coverage 가 LM 학습 단위와 직접
연결.

산출 신호 (per (prompt_id, candidate_role)):
  ans_ngram_3_count      : 답변 안 3-gram 수
  ans_ngram_3_zero_count : count==0 인 3-gram 수
  ans_ngram_3_min        : 모든 3-gram 의 최소 corpus count
  ans_ngram_3_mean       : 평균
  ans_ngram_3_axis       : log(1+mean) / log(1+1e8)
  같은 식으로 5-gram
  ans_ngram_token_count  : 답변 token 수 (분모)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


_TOKEN_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    """간단한 whitespace token. Infini-gram 의 OLMo tokenizer 와 다르지만
    n-gram coverage 는 phrase-level approximation."""
    return _TOKEN_RE.findall((text or "").strip())


def ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _log_normalize(value: float, ceiling: float = 1e8) -> float:
    return math.log(1 + max(0.0, value)) / math.log(1 + ceiling)


@dataclass(frozen=True)
class NGramCoverageRecord:
    prompt_id: str
    candidate_id: str
    candidate_role: str
    n_tokens: int
    per_n_stats: dict[int, dict[str, Any]] = field(default_factory=dict)


def compute_ngram_coverage(
    *,
    prompt_id: str,
    candidate_id: str,
    candidate_role: str,
    candidate_text: str,
    backend,
    n_values: tuple[int, ...] = (3, 5),
    log_axis_ceiling: float = 1e8,
) -> NGramCoverageRecord:
    tokens = tokenize(candidate_text)
    per_n: dict[int, dict[str, Any]] = {}
    for n in n_values:
        grams = ngrams(tokens, n)
        if not grams:
            per_n[n] = {
                "count": 0,
                "zero_count": 0,
                "min": 0,
                "mean": 0.0,
                "axis": 0.0,
            }
            continue
        counts: list[int] = []
        for g in grams:
            res = backend.count_entity(g)
            if res.raw_count is not None:
                counts.append(int(res.raw_count))
            else:
                counts.append(0)
        if not counts:
            per_n[n] = {"count": len(grams), "zero_count": len(grams), "min": 0, "mean": 0.0, "axis": 0.0}
            continue
        mn = min(counts)
        mean = sum(counts) / len(counts)
        per_n[n] = {
            "count": len(counts),
            "zero_count": int(sum(1 for c in counts if c == 0)),
            "min": int(mn),
            "mean": float(mean),
            "axis": _log_normalize(mean, ceiling=log_axis_ceiling),
        }
    return NGramCoverageRecord(
        prompt_id=prompt_id,
        candidate_id=candidate_id,
        candidate_role=candidate_role,
        n_tokens=len(tokens),
        per_n_stats=per_n,
    )


def record_to_row(rec: NGramCoverageRecord) -> dict[str, Any]:
    row = {
        "prompt_id": rec.prompt_id,
        "candidate_id": rec.candidate_id,
        "candidate_role": rec.candidate_role,
        "n_tokens": rec.n_tokens,
    }
    for n, stats in rec.per_n_stats.items():
        row[f"ans_ngram_{n}_count"] = stats["count"]
        row[f"ans_ngram_{n}_zero_count"] = stats["zero_count"]
        row[f"ans_ngram_{n}_min"] = stats["min"]
        row[f"ans_ngram_{n}_mean"] = stats["mean"]
        row[f"ans_ngram_{n}_axis"] = stats["axis"]
    return row
