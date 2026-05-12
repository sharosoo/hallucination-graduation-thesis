"""Compute n-gram corpus coverage features per candidate.

답변 token 의 n-gram (3-gram, 5-gram) 이 corpus 에 얼마나 자주 등장하는지
Infini-gram count() 로 산출. entity-level corpus signal 을 phrase-level 로
보완.

Usage:
  uv run python experiments/scripts/compute_ngram_coverage_features.py \
      --candidates $RUN/results/datasets/candidate_rows.jsonl \
      --out $RUN/qwen/results/ngram_coverage_features.parquet \
      --n 3 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from experiments.adapters.corpus_counts import build_corpus_count_backend
from experiments.adapters.ngram_coverage_features import (
    compute_ngram_coverage,
    record_to_row,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[3, 5])
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] loading candidates {cand_path}", flush=True)
    rows = []
    with open(cand_path) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"  {len(rows)} candidates")

    print(f"[2/3] building backend", flush=True)
    backend = build_corpus_count_backend(cand_path)

    print(f"[3/3] computing n-gram coverage (n={args.n})", flush=True)
    out_rows = []
    t0 = time.time()
    last_log = t0
    n_total_grams = 0
    for i, r in enumerate(rows):
        rec = compute_ngram_coverage(
            prompt_id=r["prompt_id"],
            candidate_id=r["candidate_id"],
            candidate_role=r["candidate_role"],
            candidate_text=r.get("candidate_text") or "",
            backend=backend,
            n_values=tuple(args.n),
        )
        n_total_grams += sum(s["count"] for s in rec.per_n_stats.values())
        out_rows.append({
            "dataset": r["dataset"],
            "pair_id": r.get("pair_id"),
            **record_to_row(rec),
        })
        now = time.time()
        if now - last_log > 30:
            elapsed = now - t0
            eta = elapsed / max(1, i + 1) * (len(rows) - i - 1)
            print(f"    {i+1}/{len(rows)} elapsed={elapsed:.0f}s ETA={eta:.0f}s grams={n_total_grams:,}",
                  flush=True)
            last_log = now

    df = pd.DataFrame(out_rows)
    df.to_parquet(out_path, index=False)
    print(f"  saved {out_path}  rows={len(df)}  total grams queried={n_total_grams:,}")
    # quick stats
    cols_show = [c for c in df.columns if c.startswith("ans_ngram_") and ("axis" in c or "zero" in c)]
    print("\n=== by dataset × role (axis + zero_count) ===")
    print(df.groupby(["dataset","candidate_role"])[cols_show].mean().to_string())


if __name__ == "__main__":
    main()
