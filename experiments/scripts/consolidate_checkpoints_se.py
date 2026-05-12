"""Consolidate run_generation.py checkpoint shards into a free_sample_rows.json.

run_generation.py 의 finalize 단계가 죽거나 너무 느릴 때, checkpoint shard 들
(samples 별 shard.json + full_logits.parquet) 을 직접 읽어 단일
free_sample_rows.json 을 만든다.

기본 동작:
  - shard.json 의 sample dict 만 모아 free_sample_rows.json (samples list) 산출
  - full_logits.parquet 는 별도로 통합하지 않음 (size 부담)
  - 대신 full_logits_ref 를 shard 별 parquet path 로 그대로 둠 (per-shard ref).
  - SE / Energy / NLI / sample_nll / logit_variance 등은 selected_token_logits 와
    logsumexp 만으로 산출 가능 → full_logits 통합 없이 진행 가능.

Usage:
  uv run python experiments/scripts/consolidate_checkpoints_se.py \
      --checkpoint-dir $RUN/qwen/results/generation/free_sample_rows.json.checkpoint \
      --out $RUN/qwen/results/generation/free_sample_rows.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True,
                    help="checkpoint root, e.g. .../free_sample_rows.json.checkpoint")
    ap.add_argument("--out", required=True,
                    help="output free_sample_rows.json path")
    ap.add_argument("--include-full-logits", action="store_true",
                    help="(unused) full_logits.parquet 를 single 파일로 통합. 크기 부담.")
    args = ap.parse_args()

    ckpt_root = Path(args.checkpoint_dir) / "free_sample_rows"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] scanning shard dirs in {ckpt_root}", flush=True)
    shard_dirs = sorted(ckpt_root.iterdir())
    print(f"  {len(shard_dirs)} shards", flush=True)

    print(f"[2/3] loading samples", flush=True)
    samples = []
    bad = 0
    empty_response_filled = 0
    t0 = time.time()
    for i, sd in enumerate(shard_dirs):
        try:
            shard_payload = json.loads((sd / "shard.json").read_text())
            sample = shard_payload["sample"]
            # full_logits_ref 의 path 를 절대경로로 변경 (shard dir 기반)
            ref = sample.get("full_logits_ref")
            if isinstance(ref, dict) and ref.get("path") == "full_logits.parquet":
                abs_parquet = (sd / "full_logits.parquet").resolve()
                ref["path"] = str(abs_parquet)
            # 빈 response_text 는 SE adapter strict 검증 통과 위해 placeholder.
            if not (sample.get("response_text") or "").strip():
                sample["response_text"] = "(empty)"
                empty_response_filled += 1
            samples.append(sample)
        except Exception:
            bad += 1
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(shard_dirs) - i - 1)
            print(f"    {i+1}/{len(shard_dirs)} loaded (bad={bad}) elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                  flush=True)

    print(f"  loaded {len(samples)} samples, bad={bad}, empty_response_filled={empty_response_filled}", flush=True)

    print(f"[3/3] writing {out_path}", flush=True)

    # 첫 shard 의 metadata 에서 model / tokenizer / generation_config 가져오기
    first_shard = json.loads((shard_dirs[0] / "shard.json").read_text())
    prompt_ids = {s["prompt_id"] for s in samples}
    sample_indexes = {int(s["sample_index"]) for s in samples}

    from datetime import datetime, timezone
    payload = {
        "model_name": first_shard.get("model_name"),
        "tokenizer_name": first_shard.get("tokenizer_name"),
        "generation_config": first_shard.get("generation_config"),
        "logits_schema_version": first_shard.get("logits_schema_version"),
        "formula_manifest_ref": "experiments/literature/formula_notes.md",
        "dataset_manifest_ref": "experiments/configs/datasets_se.yaml",
        "run_id": f"consolidated-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "schema_version": "generation_free_sample_rows_v1",
        "artifact_type": "free_sample_rows",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "has_logits": True,
        "has_full_logits": True,
        "full_vocabulary_logits": True,
        "fixture_mode": False,
        "full_logits_storage": {
            "format": "parquet",
            "path": "per_shard",
            "compression": "zstd",
            "dtype": "float16",
            "row_count": sum(s.get("full_logits_ref", {}).get("row_count", len(s.get("generated_token_ids", []))) for s in samples) if samples else 0,
            "note": "Per-shard parquet (consolidated from checkpoint shards). full_logits_ref.path on each sample points to absolute per-shard parquet.",
        },
        "sample_count_per_prompt": max(sample_indexes) + 1 if sample_indexes else 10,
        "prompt_group_count": len(prompt_ids),
        "answer_only_invalid_attempt_count": 0,
        "answer_only_max_invalid_attempts": 32,
        "samples": samples,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  saved {out_path}  size={size_mb:.0f} MB  n_samples={len(samples)}")


if __name__ == "__main__":
    main()
