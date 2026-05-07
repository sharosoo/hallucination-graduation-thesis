#!/usr/bin/env python3
"""Cheap pre-flight check that S2 free-sample output is ready for S4 NLI.

Reads either the final ``free_sample_rows.json`` (if S2 has aggregated) or
walks the on-disk checkpoint shards (if S2 is still running) and reports:
  * total prompts seen, prompts with all 10 samples
  * sample-index coverage holes
  * required-field presence on a 50-shard sample
  * essential schema metadata (model_name, tokenizer_name, schema_version)

Exit code 0 means S4's strict gate (every prompt has exactly N=10 valid
samples with logits/logsumexp/selected_token_ids/response_text) would pass.
Exit code 1 means S4 would currently abort; prints the offending prompts.
Exit code 2 means the input path is unusable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_SAMPLE_COUNT = 10
REQUIRED_SAMPLE_FIELDS = (
    "response_text",
    "selected_token_logits",
    "logsumexp",
    "selected_token_ids",
    "generated_token_ids",
    "sample_index",
    "prompt_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to free_sample_rows.json (final) or its checkpoint dir")
    parser.add_argument("--sample-shards", type=int, default=50, help="Random shards to deep-check (checkpoint mode)")
    return parser.parse_args()


def check_final_json(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        print(f"FATAL: 'samples' missing or not a list in {path}", file=sys.stderr)
        return 2
    by_prompt: dict[str, set[int]] = defaultdict(set)
    field_problems: list[str] = []
    for sample in samples:
        pid = sample.get("prompt_id")
        sidx = sample.get("sample_index")
        if not isinstance(pid, str) or not isinstance(sidx, int):
            field_problems.append(f"sample missing prompt_id/sample_index: {sample.get('prompt_id')!r}/{sample.get('sample_index')!r}")
            continue
        by_prompt[pid].add(sidx)
        for f in REQUIRED_SAMPLE_FIELDS:
            if f not in sample:
                field_problems.append(f"prompt {pid[:60]!r} sample {sidx}: missing {f}")
                break
    return summarize(by_prompt, field_problems, source=str(path), schema_meta=payload)


def check_checkpoint_dir(path: Path, sample_n: int) -> int:
    if not path.is_dir():
        print(f"FATAL: not a directory: {path}", file=sys.stderr)
        return 2
    by_prompt: dict[str, set[int]] = defaultdict(set)
    field_problems: list[str] = []
    schema_meta: dict[str, object] = {}
    shard_dirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".tmp")]
    if not shard_dirs:
        print(f"FATAL: no shards in {path}", file=sys.stderr)
        return 2
    # Coverage scan: only need prompt_id + sample_index (cheap text scan).
    import re
    for d in shard_dirs:
        sj = d / "shard.json"
        if not sj.exists():
            continue
        try:
            head = sj.read_bytes()[:4096].decode("utf-8", errors="ignore")
            m = re.search(r'"prompt_id"\s*:\s*"([^"]+)"', head)
            si = re.search(r'"sample_index"\s*:\s*(\d+)', head)
            if m and si:
                by_prompt[m.group(1)].add(int(si.group(1)))
        except OSError:
            pass

    # Deep-validate a random 50-shard sample for required fields.
    random.seed(0)
    for d in random.sample(shard_dirs, min(sample_n, len(shard_dirs))):
        sj = d / "shard.json"
        if not sj.exists():
            field_problems.append(f"{d.name}: missing shard.json")
            continue
        try:
            payload = json.loads(sj.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            field_problems.append(f"{d.name}: invalid JSON ({exc})")
            continue
        sample = payload.get("sample") or {}
        for f in REQUIRED_SAMPLE_FIELDS:
            if f not in sample:
                field_problems.append(f"{d.name}: sample missing {f}")
                break
        # Capture schema meta from the first shard we inspect successfully.
        if not schema_meta:
            for k in ("model_name", "tokenizer_name", "schema_version", "logits_schema_version"):
                if k in payload:
                    schema_meta[k] = payload[k]
    return summarize(by_prompt, field_problems, source=str(path), schema_meta=schema_meta)


def summarize(
    by_prompt: dict[str, set[int]],
    field_problems: list[str],
    *,
    source: str,
    schema_meta: dict,
) -> int:
    n_prompts = len(by_prompt)
    fully_complete = sum(1 for v in by_prompt.values() if v == set(range(EXPECTED_SAMPLE_COUNT)))
    n_dist = Counter(len(v) for v in by_prompt.values())
    incomplete = [(pid, sorted(set(range(EXPECTED_SAMPLE_COUNT)) - v)) for pid, v in by_prompt.items() if v != set(range(EXPECTED_SAMPLE_COUNT))]

    print(f"source: {source}")
    print(f"schema_meta: {schema_meta}")
    print(f"prompts seen: {n_prompts}")
    print(f"fully complete (all 10 samples): {fully_complete}")
    print(f"samples-per-prompt distribution: {dict(sorted(n_dist.items()))}")
    print(f"field-shape problems on inspected shards: {len(field_problems)}")
    for fp in field_problems[:5]:
        print(f"  - {fp}")
    if incomplete:
        print(f"\nincomplete prompts: {len(incomplete)} (first 5)")
        for pid, missing in incomplete[:5]:
            print(f"  - {pid[:80]} missing samples {missing}")
        if len(incomplete) > 0:
            print("\nVERDICT: NOT READY for S4 (strict N=10 gate would fail)")
            return 1
    if field_problems:
        print("\nVERDICT: NOT READY for S4 (field-shape problems above)")
        return 1
    print("\nVERDICT: READY for S4")
    return 0


def main() -> int:
    args = parse_args()
    path = Path(args.path)
    if path.is_file() and path.suffix == ".json":
        return check_final_json(path)
    if path.is_dir():
        return check_checkpoint_dir(path, args.sample_shards)
    # Auto-resolve: maybe user pointed at the run results/generation/free_sample_rows.json
    # before S2 finalized — fall back to checkpoint dir.
    checkpoint = path.with_suffix(path.suffix + ".checkpoint") / "free_sample_rows"
    if checkpoint.is_dir():
        print(f"final JSON not present; falling back to checkpoint shards at {checkpoint}")
        return check_checkpoint_dir(checkpoint, args.sample_shards)
    print(f"FATAL: {path} is neither a final JSON nor a checkpoint dir", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
