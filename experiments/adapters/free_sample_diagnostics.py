"""Per-sample token-logit diagnostics for generation-level evaluation.

자유 생성된 free-sample 의 token-level logit 통계를 sample 단위 (prompt_id,
sample_index) 로 산출한다. generation-level fusion 의 sample-level 입력 신호로
사용된다.

산출 신호 (sample 단위):
    sample_nll                       = mean(logsumexp_t - selected_token_logit_t)
                                       = -(1/T) sum_t log p(x_t)
    sample_sequence_log_prob         = sum_t log p(x_t)
    sample_logit_variance            = var(selected_token_logit_t)
    sample_logsumexp_mean            = mean(logsumexp_t) — partition function flatness
    sample_confidence_margin_mean    = mean_t (top1_logit - top2_logit) — 정식 margin
    sample_confidence_margin_min     = min_t  (top1_logit - top2_logit) — token-level worst case
    sample_top1_logit_mean           = mean_t (top1_logit) — top-1 logit 평균

top1 / top2 는 free_sample_rows.json.full_logits.parquet (full vocabulary logits
half-precision cache) 에서 streaming 으로 추출한다. 73GB cache 를 row-group
단위로 iter 하며 token 당 np.partition 으로 top-2 만 추출하므로 메모리는
batch 단위로 제한된다.

Inputs:
  $RUN/results/generation/free_sample_rows.json
  $RUN/results/generation/free_sample_rows.json.full_logits.parquet  (full vocab logits)
Outputs:
  $RUN/results/free_sample_diagnostics.parquet
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def diagnostics_from_sample(selected_token_logits: list[float],
                            logsumexp: list[float]) -> dict[str, float]:
    sl = np.asarray(selected_token_logits, dtype=np.float64)
    lz = np.asarray(logsumexp, dtype=np.float64)
    n = int(len(sl))
    if n == 0 or len(lz) != n:
        return {
            "sample_nll": float("nan"),
            "sample_sequence_log_prob": float("nan"),
            "sample_logit_variance": float("nan"),
            "sample_logsumexp_mean": float("nan"),
            "n_tokens": n,
        }
    log_p = sl - lz  # log p(x_t) per token
    return {
        "sample_nll": float(-np.mean(log_p)),
        "sample_sequence_log_prob": float(np.sum(log_p)),
        "sample_logit_variance": float(np.var(sl)) if n > 1 else 0.0,
        "sample_logsumexp_mean": float(np.mean(lz)),
        "n_tokens": n,
    }


def build_diagnostics_frame(free_sample_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for s in free_sample_rows:
        diag = diagnostics_from_sample(
            s.get("selected_token_logits") or [],
            s.get("logsumexp") or [],
        )
        rows.append({
            "prompt_id": s["prompt_id"],
            "sample_index": int(s["sample_index"]),
            "dataset": s.get("dataset", ""),
            **diag,
        })
    return pd.DataFrame(rows)


def _process_batch_vectorized(args: tuple) -> dict[tuple[str, int], dict]:
    """Worker: full vocab logits batch → per-(pid,si) margin/min/top1 sums.

    Vectorized via np.partition on (batch, vocab) ndarray instead of per-row loop.
    """
    batch_dict, prompt_ids, sample_idxs = args
    flat = np.asarray(batch_dict["values"], dtype=np.float32)
    offsets = np.asarray(batch_dict["offsets"], dtype=np.int64)
    n = len(prompt_ids)
    if n == 0:
        return {}
    # Reshape: every row has same vocab length (V). Compute V from first row.
    v = int(offsets[1] - offsets[0])
    if any(int(offsets[i + 1] - offsets[i]) != v for i in range(n)):
        # ragged → fall back to per-row loop
        local: dict[tuple[str, int], dict] = defaultdict(
            lambda: {"margin_sum": 0.0, "margin_min": float("inf"),
                     "top1_sum": 0.0, "n": 0}
        )
        for i in range(n):
            arr = flat[offsets[i]:offsets[i + 1]]
            if arr.size < 2:
                continue
            part = np.partition(arr, -2)
            top1 = float(part[-1]); top2 = float(part[-2])
            if top1 < top2:
                top1, top2 = top2, top1
            margin = top1 - top2
            key = (prompt_ids[i], int(sample_idxs[i]))
            agg = local[key]
            agg["margin_sum"] += margin
            if margin < agg["margin_min"]:
                agg["margin_min"] = margin
            agg["top1_sum"] += top1
            agg["n"] += 1
        return dict(local)

    # Fast path: vectorized over batch
    mat = flat.reshape(n, v)  # (batch, vocab)
    part = np.partition(mat, -2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    margin = top1 - top2

    local2: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"margin_sum": 0.0, "margin_min": float("inf"),
                 "top1_sum": 0.0, "n": 0}
    )
    for i in range(n):
        key = (prompt_ids[i], int(sample_idxs[i]))
        agg = local2[key]
        m = float(margin[i])
        agg["margin_sum"] += m
        if m < agg["margin_min"]:
            agg["margin_min"] = m
        agg["top1_sum"] += float(top1[i])
        agg["n"] += 1
    return dict(local2)


def stream_top2_margins(
    full_logits_path: Path,
    *,
    batch_size: int = 1024,
    progress_every: int = 50_000,
    n_workers: int = 8,
) -> pd.DataFrame:
    """Streaming read + vectorized + multiprocessing pool 으로 top1-top2 margin 추출."""
    import multiprocessing as mp

    pf = pq.ParquetFile(str(full_logits_path))
    total_rows = pf.metadata.num_rows
    print(f"  streaming {full_logits_path.name}: {total_rows:,} token-rows  (batch={batch_size}, workers={n_workers})", flush=True)

    def gen_args():
        for batch in pf.iter_batches(
            batch_size=batch_size,
            columns=["prompt_id", "sample_index", "full_logits"],
        ):
            prompt_ids = batch.column("prompt_id").to_pylist()
            sample_idxs = batch.column("sample_index").to_pylist()
            logits = batch.column("full_logits")
            # Convert ListArray to (flat_values, offsets) for cheap pickle
            flat_arr = logits.values.to_numpy(zero_copy_only=False)
            offsets_arr = logits.offsets.to_numpy(zero_copy_only=False)
            yield ({"values": flat_arr, "offsets": offsets_arr},
                   prompt_ids, sample_idxs)

    # accumulator across all batches
    total: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"margin_sum": 0.0, "margin_min": float("inf"),
                 "top1_sum": 0.0, "n": 0}
    )
    seen = 0
    progress_anchor = 0
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        for partial in pool.imap_unordered(_process_batch_vectorized, gen_args(), chunksize=1):
            for k, agg in partial.items():
                ta = total[k]
                ta["margin_sum"] += agg["margin_sum"]
                if agg["margin_min"] < ta["margin_min"]:
                    ta["margin_min"] = agg["margin_min"]
                ta["top1_sum"] += agg["top1_sum"]
                ta["n"] += agg["n"]
            seen += sum(a["n"] for a in partial.values())
            if seen - progress_anchor >= progress_every:
                progress_anchor = seen
                print(f"    {seen:,} / {total_rows:,} ({100*seen/total_rows:.1f}%)", flush=True)

    rows = []
    for (pid, si), agg in total.items():
        n = agg["n"]
        if n == 0:
            continue
        rows.append({
            "prompt_id": pid,
            "sample_index": int(si),
            "sample_confidence_margin_mean": agg["margin_sum"] / n,
            "sample_confidence_margin_min": agg["margin_min"],
            "sample_top1_logit_mean": agg["top1_sum"] / n,
            "n_tokens_full_logits": n,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-vocab-margin", action="store_true",
                    help="full vocab logits parquet streaming 으로 top1-top2 margin 을 추출하지 않음 (빠른 경로).")
    ap.add_argument("--full-logits-batch-size", type=int, default=1024)
    ap.add_argument("--full-logits-workers", type=int, default=8)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    fs_path = run_dir / "results/generation/free_sample_rows.json"
    full_logits_path = run_dir / "results/generation/free_sample_rows.json.full_logits.parquet"
    out_path = run_dir / "results/free_sample_diagnostics.parquet"

    print(f"[1/3] loading {fs_path}", flush=True)
    samples = json.loads(fs_path.read_text())["samples"]
    print(f"  {len(samples)} samples")

    print("[2/3] computing per-sample selected-token diagnostics", flush=True)
    df = build_diagnostics_frame(samples)
    print(f"  selected-token frame: rows={len(df)}")

    if not args.no_vocab_margin:
        print(f"[3/3] streaming full vocab logits → top1-top2 margin", flush=True)
        margin_df = stream_top2_margins(
            full_logits_path,
            batch_size=args.full_logits_batch_size,
            n_workers=args.full_logits_workers,
        )
        print(f"  margin frame: rows={len(margin_df)}")
        df = df.merge(margin_df, on=["prompt_id", "sample_index"], how="left")
    else:
        print("[3/3] skipping full vocab margin (--no-vocab-margin)")

    df.to_parquet(out_path, index=False)
    print(f"saved {out_path}  rows={len(df)}  cols={list(df.columns)}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
