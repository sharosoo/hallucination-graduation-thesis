"""Generation-level correctness labeling (Phase 3, SE 5-dataset).

각 free-sample 답변과 dataset 정답 후보(best_answer / right_answer /
correct_answers / correct_candidate_pool) 사이의 NLI 양방향 entailment 매칭
으로 per-sample is_correct 라벨을 생성한다 (Farquhar/Ma 호환 단위).

매칭 모드:
  - "nli_bidirectional_max_entail" (default): deberta-large-mnli 의 양방향 entailment
    확률 max > threshold (default 0.5) 면 매칭. paraphrase 정답을 token-overlap 보다
    잘 잡는다.
  - "token_overlap": 단순 토큰 자카드 + 부분 문자열 fallback. 빠른 sanity check 용
    (NLI 디펜던시가 없을 때 fallback).

산출물:
  - generation_correctness.parquet: prompt_id, dataset, sample_index, response_text,
                                    n_candidates, nli_max_prob, is_correct,
                                    match_method, token_overlap_match
  - generation_correctness.audit.json: model, threshold, per-dataset is_correct rate
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_NLI_MODEL_NAME = "microsoft/deberta-large-mnli"
DEFAULT_NLI_THRESHOLD = 0.5

MATCH_METHOD_NLI = "nli_bidirectional_max_entail"
MATCH_METHOD_TOKEN = "token_overlap"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _norm_lower(text: str) -> str:
    return _norm(text).lower()


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", _norm_lower(text)) if t}


def overlap_match(sample: str, refs: list[str]) -> bool:
    """단순 token-overlap 매칭 (jaccard >= 0.5 또는 substring)."""
    s = _tokenize(sample)
    if not s:
        return False
    s_low = _norm_lower(sample)
    for ref in refs:
        r_low = _norm_lower(ref)
        if not r_low:
            continue
        if r_low in s_low or s_low in r_low:
            return True
        r = _tokenize(ref)
        if not r:
            continue
        jaccard = len(s & r) / max(1, len(s | r))
        if jaccard >= 0.5:
            return True
    return False


def extract_candidates(metadata: dict, dataset: str | None = None) -> list[str]:
    """정답 후보 텍스트 모음 (dedup, 순서 보존)."""
    cands: list[str] = []
    for key in ("right_answer", "best_answer"):
        v = metadata.get(key)
        if isinstance(v, str) and v.strip():
            cands.append(_norm(v))
    for key in ("correct_answers", "correct_candidate_pool"):
        v = metadata.get(key)
        if isinstance(v, list):
            cands.extend(_norm(x) for x in v if isinstance(x, str) and x.strip())
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        k = c.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


@dataclass(frozen=True)
class PromptSampleGroup:
    prompt_id: str
    dataset: str
    samples: tuple[str, ...]
    candidates: tuple[str, ...]


def group_free_samples(rows: Iterable[dict]) -> list[PromptSampleGroup]:
    """free_sample_rows.json 의 samples 리스트를 prompt 단위로 묶음."""
    by_pid: dict[str, dict] = {}
    for s in rows:
        pid = s["prompt_id"]
        if pid not in by_pid:
            by_pid[pid] = {
                "dataset": s.get("dataset", ""),
                "samples": [],
                "candidates": extract_candidates(s.get("metadata") or {}, s.get("dataset")),
            }
        by_pid[pid]["samples"].append(_norm(s.get("response_text", "")))
    return [
        PromptSampleGroup(
            prompt_id=pid,
            dataset=info["dataset"],
            samples=tuple(info["samples"]),
            candidates=tuple(info["candidates"]),
        )
        for pid, info in by_pid.items()
    ]


# ---------- NLI accuracy ----------

def _import_nli_deps():
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing optional dependency for NLI accuracy. "
            "Install the generation stack with `uv sync --group generation`."
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def compute_nli_accuracies(
    groups: list[PromptSampleGroup],
    *,
    model_name: str = DEFAULT_NLI_MODEL_NAME,
    threshold: float = DEFAULT_NLI_THRESHOLD,
    batch_size: int = 64,
    max_length: int = 256,
    progress: bool = True,
    return_sample_max: bool = False,
):
    """NLI 양방향 entailment 양방향 max > threshold 매칭.

    return_sample_max=False (default): {pid: (accuracy, n_matches)}
    return_sample_max=True: ({pid: (accuracy, n_matches)}, {(pid, sample_idx): max_entail_prob})
    """
    torch, ModelCls, TokCls = _import_nli_deps()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = TokCls.from_pretrained(model_name)
    model = ModelCls.from_pretrained(model_name).to(device).eval()
    # entailment label index
    id2label = model.config.id2label
    entail_idx = next(
        i for i, lbl in id2label.items() if "entail" in str(lbl).lower()
    )

    pair_premises: list[str] = []
    pair_hypotheses: list[str] = []
    pair_keys: list[tuple[str, int]] = []  # (pid, sample_idx)
    for g in groups:
        for si, sample in enumerate(g.samples):
            for cand in g.candidates:
                if not sample or not cand:
                    continue
                pair_premises.append(cand)
                pair_hypotheses.append(sample)
                pair_keys.append((g.prompt_id, si))
                pair_premises.append(sample)
                pair_hypotheses.append(cand)
                pair_keys.append((g.prompt_id, si))

    if progress:
        print(f"  [NLI] device={device}, pairs={len(pair_premises):,}", flush=True)

    sample_max: dict[tuple[str, int], float] = defaultdict(float)
    with torch.no_grad():
        for i in range(0, len(pair_premises), batch_size):
            bp = pair_premises[i : i + batch_size]
            bh = pair_hypotheses[i : i + batch_size]
            toks = tokenizer(
                bp, bh, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            logits = model(**toks).logits
            probs = torch.softmax(logits, dim=-1)[:, int(entail_idx)].float().cpu().numpy()
            for k, p in zip(pair_keys[i : i + batch_size], probs, strict=True):
                if p > sample_max[k]:
                    sample_max[k] = float(p)

    out: dict[str, tuple[float, int]] = {}
    for g in groups:
        n = len(g.samples)
        if n == 0:
            continue
        matches = sum(
            1 for si in range(n) if sample_max.get((g.prompt_id, si), 0.0) >= threshold
        )
        out[g.prompt_id] = (matches / n, matches)
    if return_sample_max:
        return out, dict(sample_max)
    return out


# ============================================================
# Generation-level correctness (per-sample, Farquhar/Ma 호환 단위)
# ============================================================

def build_generation_correctness_frame(
    free_sample_rows: list[dict],
    *,
    use_nli: bool = True,
    nli_model_name: str = DEFAULT_NLI_MODEL_NAME,
    threshold: float = DEFAULT_NLI_THRESHOLD,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Per-sample correctness 라벨 (NLI 양방향 entailment 매칭).

    Output schema (row=(prompt_id, sample_index)):
      prompt_id, dataset, sample_index, response_text, n_candidates,
      nli_max_prob, is_correct (binary, 1=정답), match_method,
      token_overlap_match (sanity column).
    """
    groups = group_free_samples(free_sample_rows)

    # NLI sample_max via compute_nli_accuracies(return_sample_max=True)
    nli_sample_max: dict[tuple[str, int], float] = {}
    if use_nli:
        nli_groups = [g for g in groups if g.candidates]
        _, nli_sample_max = compute_nli_accuracies(  # type: ignore[misc]
            nli_groups,
            model_name=nli_model_name,
            threshold=threshold,
            batch_size=batch_size,
            return_sample_max=True,
        )

    # build per-(prompt, sample) rows
    rows = []
    for g in groups:
        if not g.candidates:
            continue
        for si, sample_text in enumerate(g.samples):
            tok_match = overlap_match(sample_text, list(g.candidates))
            if (g.prompt_id, si) in nli_sample_max:
                p = float(nli_sample_max[(g.prompt_id, si)])
                is_corr = int(p >= threshold)
                method = MATCH_METHOD_NLI
            else:
                p = float("nan")
                is_corr = int(tok_match)
                method = MATCH_METHOD_TOKEN
            rows.append({
                "prompt_id": g.prompt_id,
                "dataset": g.dataset,
                "sample_index": si,
                "response_text": sample_text,
                "n_candidates": len(g.candidates),
                "nli_max_prob": p,
                "is_correct": is_corr,
                "match_method": method,
                "token_overlap_match": int(bool(tok_match)),
            })
    return pd.DataFrame(rows)


def write_generation_correctness_artifacts(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    nli_model_name: str,
    threshold: float,
    use_nli: bool,
) -> tuple[Path, Path]:
    """Persist generation_correctness.parquet + .audit.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "generation_correctness.parquet"
    audit_path = out_dir / "generation_correctness.audit.json"
    df.to_parquet(table_path, index=False)

    by_ds = df.groupby("dataset")["is_correct"].agg(["count", "mean"]).to_dict()
    audit = {
        "match_method": MATCH_METHOD_NLI if use_nli else MATCH_METHOD_TOKEN,
        "nli_model": nli_model_name if use_nli else None,
        "nli_threshold": threshold if use_nli else None,
        "n_generations": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "per_dataset_correctness": {
            ds: {"n": int(by_ds["count"][ds]), "is_correct_rate": float(by_ds["mean"][ds])}
            for ds in by_ds["count"]
        },
        "overall_is_correct_rate": float(df["is_correct"].mean()),
        "label_change_vs_token": int(
            (df["is_correct"] != df["token_overlap_match"]).sum()
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return table_path, audit_path
