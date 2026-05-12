"""Compute QA Bridge corpus features per (prompt_id, candidate_role).

(question entity, answer entity) bridge co-occurrence — fact-level corpus
support 측정. 기존 entity_pair_cooccurrence 의 한계 (답변 안 entity-pair 만)
를 보완.

Usage:
  uv run python experiments/scripts/compute_qa_bridge_features.py \
      --candidates experiments/results/datasets/candidate_rows.jsonl \
      --out experiments/results/qa_bridge_features.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from experiments.adapters.corpus_counts import build_corpus_count_backend
from experiments.adapters.entity_extractor_spacy import SpacyEntityExtractor
from experiments.adapters.qa_bridge_features import compute_qa_bridge, record_to_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="Path to candidate_rows.jsonl")
    ap.add_argument("--out", required=True,
                    help="Path to qa_bridge_features.parquet")
    ap.add_argument("--ner-batch-size", type=int, default=64)
    ap.add_argument("--exclude-question-entities", action="store_true", default=True,
                    help="hallu candidate entity 에서 question entity 와 lowercase 일치 항목 제외 (default True).")
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading candidates {cand_path}", flush=True)
    rows = []
    with open(cand_path) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"  {len(rows)} candidates")

    print(f"[2/5] building backend + spaCy extractor", flush=True)
    backend = build_corpus_count_backend(cand_path)
    extractor = SpacyEntityExtractor(batch_size=args.ner_batch_size)

    # question 단위 NER 캐시 (prompt_id 당 한 번)
    print(f"[3/5] extracting question entities (per prompt)", flush=True)
    questions_by_pid: dict[str, str] = {}
    for r in rows:
        if r["prompt_id"] not in questions_by_pid:
            questions_by_pid[r["prompt_id"]] = r["question"]
    pids = list(questions_by_pid.keys())
    qtexts = [questions_by_pid[p] for p in pids]
    t0 = time.time()
    q_entities_list = extractor.extract_many(qtexts, ["declarative"] * len(qtexts))
    q_entities_by_pid = {pid: ents for pid, ents in zip(pids, q_entities_list)}
    print(f"  {len(q_entities_by_pid)} prompts NER done in {time.time()-t0:.1f}s")

    # candidate text NER (every row)
    print(f"[4/5] extracting candidate entities + computing bridge counts", flush=True)
    cand_texts = [r["candidate_text"] for r in rows]
    cand_roles = [r["candidate_role"] for r in rows]
    cand_role_lookup = {"right": "declarative", "hallucinated": "hallucinated"}
    cand_ner_roles = [cand_role_lookup.get(role, "declarative") for role in cand_roles]
    t0 = time.time()
    cand_entities_list = extractor.extract_many(cand_texts, cand_ner_roles)
    print(f"  {len(rows)} candidates NER done in {time.time()-t0:.1f}s")

    out_rows = []
    t0 = time.time()
    n_pair_total = 0
    last_log = t0
    for i, (r, c_E) in enumerate(zip(rows, cand_entities_list)):
        q_E = q_entities_by_pid[r["prompt_id"]]
        rec = compute_qa_bridge(
            prompt_id=r["prompt_id"],
            candidate_id=r["candidate_id"],
            candidate_role=r["candidate_role"],
            question_entities=list(q_E),
            candidate_entities=list(c_E),
            backend=backend,
            exclude_question_entities=args.exclude_question_entities,
        )
        n_pair_total += rec.qa_bridge_pair_count
        out_rows.append({
            "dataset": r["dataset"],
            "pair_id": r.get("pair_id"),
            **record_to_row(rec),
        })
        now = time.time()
        if now - last_log > 30:
            elapsed = now - t0
            eta = elapsed / max(1, i + 1) * (len(rows) - i - 1)
            print(f"    {i+1}/{len(rows)} ({100*(i+1)/len(rows):.1f}%) "
                  f"pairs={n_pair_total:,} elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
            last_log = now
    print(f"  total pairs queried: {n_pair_total:,}, elapsed {time.time()-t0:.1f}s")

    print(f"[5/5] writing {out_path}", flush=True)
    df = pd.DataFrame(out_rows)
    df.to_parquet(out_path, index=False)
    print(f"  saved {out_path}  rows={len(df)}")
    # quick stats
    by_role = df.groupby("candidate_role")[
        ["qa_bridge_pair_count", "qa_bridge_min", "qa_bridge_mean", "qa_bridge_axis", "qa_bridge_zero_flag"]
    ].agg(["mean", "median"])
    print("\n=== by candidate_role ===")
    print(by_role.to_string())
    by_ds = df.groupby(["dataset", "candidate_role"])[
        ["qa_bridge_pair_count", "qa_bridge_axis", "qa_bridge_zero_flag"]
    ].mean()
    print("\n=== by dataset × role ===")
    print(by_ds.to_string())


if __name__ == "__main__":
    main()
