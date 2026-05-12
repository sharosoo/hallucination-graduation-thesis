"""Question-Answer entity bridge co-occurrence features.

기존 ``entity_pair_cooccurrence`` 가 한 candidate 답변 안의 entity-pair 만
측정하는 한계를 보완한다. 본 어댑터는 (question entity, answer entity) 의
corpus 공동 등장 빈도를 산출한다 — fact (subject, object) 쌍의 corpus 학습
정도를 직접 측정.

데이터셋 형식 차이를 다음과 같이 통합 처리한다:
  - HaluEval-QA right: 짧은 single entity 답 ("Delhi") → q_E × {right answer}
  - HaluEval-QA hallu: sentence-level 답 ("based in Mumbai") + 질문 entity 재사용
                       → q_E × (a_E - q_E)  (질문 entity 제외)
  - TruthfulQA right/hallu: sentence 답 → q_E × (a_E - q_E)

산출 신호 (per (prompt_id, candidate_role)):
  qa_bridge_pair_count       : 산출에 사용된 (q_e, a_e) pair 수
  qa_bridge_min              : pair 들의 최소 co-occurrence
  qa_bridge_mean             : pair 들의 평균 co-occurrence
  qa_bridge_axis             : log(1+min) / log(1+1e5)  ∈ [0,1]
  qa_bridge_zero_flag        : pair 중 하나라도 0 cooc 이면 1 (epistemic gap)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QABridgeRecord:
    prompt_id: str
    candidate_id: str
    candidate_role: str
    question_entities: tuple[str, ...]
    candidate_entities: tuple[str, ...]
    candidate_entities_after_q_subtract: tuple[str, ...]
    pairs_used: tuple[tuple[str, str], ...]
    pair_counts: tuple[int, ...]
    qa_bridge_pair_count: int
    qa_bridge_min: int
    qa_bridge_mean: float
    qa_bridge_axis: float
    qa_bridge_zero_flag: int
    backend_provenance: dict[str, Any] = field(default_factory=dict)


def _log_normalize(value: int, ceiling: float = 1e5) -> float:
    """entity_pair_cooccurrence_axis 와 같은 log-normalization."""
    return math.log(1 + max(0, value)) / math.log(1 + ceiling)


def compute_qa_bridge(
    *,
    prompt_id: str,
    candidate_id: str,
    candidate_role: str,
    question_entities: list[str],
    candidate_entities: list[str],
    backend,
    exclude_question_entities: bool = True,
    log_axis_ceiling: float = 1e5,
) -> QABridgeRecord:
    """질문 entity × 답변 entity 의 corpus pair count 를 집계.

    Parameters
    ----------
    candidate_role : "right" or "hallucinated"
    exclude_question_entities : True 면 a_E 에서 q_E 와 lowercase 일치하는
        entity 를 제외 (질문을 그대로 paraphrase 한 hallu 답이 question entity
        를 재사용해 신호를 오염하는 것을 막음).
    """
    q_E = [e for e in question_entities if e and e.strip()]
    a_E = [e for e in candidate_entities if e and e.strip()]

    if exclude_question_entities:
        q_lower = {e.lower().strip() for e in q_E}
        a_E_filtered = [e for e in a_E if e.lower().strip() not in q_lower]
    else:
        a_E_filtered = list(a_E)

    pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for q in q_E:
        for a in a_E_filtered:
            if q.lower().strip() == a.lower().strip():
                continue
            key = (q.lower().strip(), a.lower().strip())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append((q, a))

    counts: list[int] = []
    provenances: list[dict[str, Any]] = []
    for (q, a) in pairs:
        res = backend.count_pair(q, a)
        if res.raw_count is not None:
            counts.append(int(res.raw_count))
        provenances.append({
            "query": res.provenance.query,
            "status": res.provenance.status,
            "approximate": res.provenance.approximate,
        })

    if not counts:
        return QABridgeRecord(
            prompt_id=prompt_id,
            candidate_id=candidate_id,
            candidate_role=candidate_role,
            question_entities=tuple(q_E),
            candidate_entities=tuple(a_E),
            candidate_entities_after_q_subtract=tuple(a_E_filtered),
            pairs_used=tuple(pairs),
            pair_counts=tuple(),
            qa_bridge_pair_count=0,
            qa_bridge_min=0,
            qa_bridge_mean=0.0,
            qa_bridge_axis=0.0,
            qa_bridge_zero_flag=1,
            backend_provenance={"per_pair": provenances},
        )

    mn = min(counts)
    mean = sum(counts) / len(counts)
    axis = _log_normalize(mn, ceiling=log_axis_ceiling)
    zero_flag = int(any(c == 0 for c in counts))

    return QABridgeRecord(
        prompt_id=prompt_id,
        candidate_id=candidate_id,
        candidate_role=candidate_role,
        question_entities=tuple(q_E),
        candidate_entities=tuple(a_E),
        candidate_entities_after_q_subtract=tuple(a_E_filtered),
        pairs_used=tuple(pairs),
        pair_counts=tuple(counts),
        qa_bridge_pair_count=int(len(counts)),
        qa_bridge_min=int(mn),
        qa_bridge_mean=float(mean),
        qa_bridge_axis=float(axis),
        qa_bridge_zero_flag=zero_flag,
        backend_provenance={"per_pair": provenances},
    )


def record_to_row(rec: QABridgeRecord) -> dict[str, Any]:
    """Flat row dict for parquet writing."""
    return {
        "prompt_id": rec.prompt_id,
        "candidate_id": rec.candidate_id,
        "candidate_role": rec.candidate_role,
        "n_question_entities": len(rec.question_entities),
        "n_candidate_entities": len(rec.candidate_entities),
        "n_candidate_entities_after_q_subtract": len(rec.candidate_entities_after_q_subtract),
        "qa_bridge_pair_count": rec.qa_bridge_pair_count,
        "qa_bridge_min": rec.qa_bridge_min,
        "qa_bridge_mean": rec.qa_bridge_mean,
        "qa_bridge_axis": rec.qa_bridge_axis,
        "qa_bridge_zero_flag": rec.qa_bridge_zero_flag,
        "question_entities": list(rec.question_entities),
        "candidate_entities": list(rec.candidate_entities),
        "pairs_used": [list(p) for p in rec.pairs_used],
        "pair_counts": list(rec.pair_counts),
    }
