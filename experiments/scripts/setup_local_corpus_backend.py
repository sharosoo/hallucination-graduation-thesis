#!/usr/bin/env python3
"""Verify a downloaded Infini-gram local index and pin it to candidate rows via a sidecar config.

After running this script, ``compute_corpus_features.py`` will pick up the
local backend without env vars: ``build_corpus_count_backend`` reads
``<candidates>.corpus_backend.json`` next to the candidate-rows JSONL and
constructs ``InfinigramLocalEngineBackend`` for it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters.corpus_counts import (
    INFINIGRAM_DEFAULT_LOCAL_TOKENIZER,
    InfinigramLocalEngineBackend,
    infinigram_cache_path,
)

REQUIRED_INDEX_FILES = ("metadata.0", "metaoff.0", "offset.0", "table.0", "tokenized.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidate_rows.jsonl path the corpus stage will read")
    parser.add_argument("--index-dir", required=True, help="Local directory containing the Infini-gram index files")
    parser.add_argument("--tokenizer", default=INFINIGRAM_DEFAULT_LOCAL_TOKENIZER, help="HuggingFace tokenizer name (default: allenai/OLMo-7B-hf)")
    parser.add_argument("--probe", default="George Washington", help="Entity probe for end-to-end smoke test")
    parser.add_argument("--probe-pair", default="George Washington AND Mount Vernon", help="Pair probe for CNF AND smoke test")
    return parser.parse_args()


def verify_index_dir(index_dir: Path) -> list[str]:
    problems: list[str] = []
    if not index_dir.exists():
        return [f"index dir does not exist: {index_dir}"]
    have = {p.name for p in index_dir.iterdir() if p.is_file()}
    for required in REQUIRED_INDEX_FILES:
        if required not in have:
            partial_matches = [name for name in have if name.startswith(required + ".")]
            if partial_matches:
                problems.append(f"file {required!r} only present as partial download(s): {partial_matches}")
            else:
                problems.append(f"missing required index file: {required}")
    return problems


def main() -> int:
    args = parse_args()
    candidates_path = Path(args.candidates).resolve()
    index_dir = Path(args.index_dir).resolve()

    if not candidates_path.exists():
        print(f"candidates path does not exist: {candidates_path}", file=sys.stderr)
        return 2

    problems = verify_index_dir(index_dir)
    if problems:
        print(json.dumps({"phase": "verify_index_failed", "index_dir": str(index_dir), "problems": problems}, ensure_ascii=False), file=sys.stderr)
        return 2

    cache_path = infinigram_cache_path(candidates_path)
    backend = InfinigramLocalEngineBackend(
        candidates_path=candidates_path,
        index_dir=index_dir,
        tokenizer_name=str(args.tokenizer),
        cache_path=cache_path,
    )

    started = time.monotonic()
    entity_result = backend.count_entity(str(args.probe))
    pair_result = backend.count_pair(*str(args.probe_pair).split(" AND ", 1)) if " AND " in str(args.probe_pair) else None
    elapsed = time.monotonic() - started

    if entity_result.raw_count is None:
        print(json.dumps({
            "phase": "probe_failed",
            "entity_probe": str(args.probe),
            "metadata": entity_result.provenance.metadata,
            "note": entity_result.provenance.note,
        }, ensure_ascii=False), file=sys.stderr)
        return 2

    sidecar_path = candidates_path.with_suffix(candidates_path.suffix + ".corpus_backend.json")
    sidecar_payload = {
        "schema_version": "corpus_backend_pin_v1",
        "backend": "infini_gram_local_count",
        "index_dir": str(index_dir),
        "tokenizer": str(args.tokenizer),
        "candidates_path": str(candidates_path),
        "cache_path": str(cache_path),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sidecar_path.write_text(json.dumps(sidecar_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "phase": "complete",
        "elapsed_seconds": round(elapsed, 3),
        "sidecar": str(sidecar_path),
        "entity_probe": {"query": str(args.probe), "count": entity_result.raw_count, "metadata": entity_result.provenance.metadata},
        "pair_probe": (
            {
                "query": pair_result.provenance.query,
                "count": pair_result.raw_count,
                "metadata": pair_result.provenance.metadata,
            } if pair_result is not None else None
        ),
        "describe": backend.describe(),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
