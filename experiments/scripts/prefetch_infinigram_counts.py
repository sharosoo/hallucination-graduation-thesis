#!/usr/bin/env python3
"""Prefetch Infini-gram entity/pair counts into the corpus-axis on-disk cache.

Uses the same factory as `compute_corpus_features.py`, so env vars and the
``<candidates_path>.corpus_backend.json`` sidecar drive backend selection
(local engine over downloaded Dolma index, public REST API, or fixture).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters.corpus_counts import (
    INFINIGRAM_DEFAULT_ENDPOINT,
    INFINIGRAM_DEFAULT_INDEX,
    INFINIGRAM_DEFAULT_LOCAL_TOKENIZER,
    build_corpus_count_backend,
    infinigram_cache_path,
)
from experiments.adapters.corpus_features import combine_entities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidate_rows.jsonl path")
    parser.add_argument("--parallelism", type=int, default=8, help="Concurrent workers (8 is safe for local engine)")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for sampling (0 = all rows)")
    parser.add_argument("--cache", help="Override cache path (default: <candidates>.infinigram_cache.json)")
    return parser.parse_args()


def collect_terms(candidates_path: Path, *, limit: int) -> tuple[list[str], list[str]]:
    entities: set[str] = set()
    pairs: set[str] = set()
    with candidates_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if limit and line_index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            _, _, row_entities = combine_entities(row)
            unique = sorted(set(filter(None, row_entities)))
            entities.update(unique)
            for left, right in combinations(unique, 2):
                pairs.add(f"{left} AND {right}")
    return sorted(entities), sorted(pairs)


def main() -> int:
    args = parse_args()
    candidates_path = Path(args.candidates).resolve()
    if not candidates_path.exists():
        print(f"candidates path not found: {candidates_path}", file=sys.stderr)
        return 2

    backend = build_corpus_count_backend(candidates_path)
    description = backend.describe()
    backend_id = description.get("backend_id")
    if backend_id == "infini_gram_local_count" and description.get("status") == "local_index_absent":
        print(json.dumps({
            "phase": "abort",
            "reason": "no infini-gram backend configured",
            "candidates": str(candidates_path),
            "describe": description,
        }, ensure_ascii=False), file=sys.stderr)
        return 2

    started_at = time.monotonic()
    entities, pairs = collect_terms(candidates_path, limit=int(args.limit))
    print(json.dumps({
        "phase": "collected_terms",
        "candidates": str(candidates_path),
        "entities": len(entities),
        "pairs": len(pairs),
        "limit": int(args.limit),
        "backend": description,
        "parallelism": int(args.parallelism),
        "default_index": INFINIGRAM_DEFAULT_INDEX,
        "default_endpoint": INFINIGRAM_DEFAULT_ENDPOINT,
        "default_local_tokenizer": INFINIGRAM_DEFAULT_LOCAL_TOKENIZER,
    }, ensure_ascii=False), flush=True)

    warmup = getattr(backend, "warmup", None)
    if not callable(warmup):
        print(json.dumps({"phase": "abort", "reason": "backend does not implement warmup"}, ensure_ascii=False), file=sys.stderr)
        return 2
    summary = warmup(entities=entities, pairs=pairs, parallelism=int(args.parallelism))
    elapsed = time.monotonic() - started_at
    cache_path = Path(args.cache).resolve() if args.cache else infinigram_cache_path(candidates_path)
    print(json.dumps({
        "phase": "complete",
        "elapsed_seconds": round(elapsed, 2),
        "queue_summary": summary,
        "cache": str(cache_path),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
