"""S2'-followup — detect prompts whose free-samples hit the max_new_tokens cap.

For each prompt, count how many of its N=10 free-samples have
``len(generated_token_ids) >= max_new_tokens`` (i.e. likely truncated).
A prompt is flagged when at least ``--min-truncated`` of its samples are
truncated. Outputs a filtered ``prompt_groups.jsonl`` + ``candidate_rows.jsonl``
suitable for re-running S2' with ``--config generation_se_qwen_long.yaml``
(``max_new_tokens=128``).

Thesis §3.1 reports ~12.6% (2,426 prompts) re-generated this way.

Usage:
  uv run python experiments/scripts/select_truncated_prompts.py \\
    --free-samples $RUN/qwen/results/generation/free_sample_rows.json \\
    --prompt-groups $RUN/results/datasets/prompt_groups.jsonl \\
    --candidates $RUN/results/datasets/candidate_rows.jsonl \\
    --max-new-tokens 64 \\
    --out-dir $RUN/qwen/results/datasets_truncated
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--free-samples",
        type=Path,
        required=True,
        help="free_sample_rows.json from S3' (consolidated).",
    )
    parser.add_argument(
        "--prompt-groups",
        type=Path,
        required=True,
        help="Original prompt_groups.jsonl (S1' output).",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Original candidate_rows.jsonl (S1' output).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        required=True,
        help="The max_new_tokens cap used in the S2' run (e.g. 64).",
    )
    parser.add_argument(
        "--min-truncated",
        type=int,
        default=1,
        help="Flag prompts with at least this many truncated samples (default: 1, "
             "matches thesis §3.1 'truncated answer 가 발생한 sample').",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for filtered prompt_groups.jsonl + candidate_rows.jsonl + truncated_report.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.free_samples.read_text())
    rows = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    print(f"loaded {len(rows)} free-sample rows", flush=True)

    # Count truncated samples per prompt_id
    truncated_count: dict[str, int] = defaultdict(int)
    sample_count: dict[str, int] = defaultdict(int)
    for r in rows:
        pid = r.get("prompt_id")
        if pid is None:
            continue
        sample_count[pid] += 1
        gids = r.get("generated_token_ids") or []
        if isinstance(gids, list) and len(gids) >= args.max_new_tokens:
            truncated_count[pid] += 1

    flagged = {pid for pid, n in truncated_count.items() if n >= args.min_truncated}
    print(
        f"prompts: {len(sample_count)} total, "
        f"{len(flagged)} flagged ({len(flagged) / max(len(sample_count), 1) * 100:.1f}%)",
        flush=True,
    )

    # Filter prompt_groups + candidate_rows by flagged prompt_id
    out_pg = args.out_dir / "prompt_groups.jsonl"
    out_cr = args.out_dir / "candidate_rows.jsonl"
    n_pg = n_cr = 0
    with open(out_pg, "w") as fp, open(args.prompt_groups) as fin:
        for line in fin:
            d = json.loads(line)
            if d.get("prompt_id") in flagged:
                fp.write(line)
                n_pg += 1
    with open(out_cr, "w") as fc, open(args.candidates) as fin:
        for line in fin:
            d = json.loads(line)
            if d.get("prompt_id") in flagged:
                fc.write(line)
                n_cr += 1

    report = {
        "max_new_tokens": args.max_new_tokens,
        "min_truncated_samples": args.min_truncated,
        "input_prompt_count": len(sample_count),
        "input_sample_count": len(rows),
        "flagged_prompt_count": len(flagged),
        "flagged_fraction": len(flagged) / max(len(sample_count), 1),
        "filtered_prompt_groups": n_pg,
        "filtered_candidate_rows": n_cr,
        "truncation_distribution": {
            f"{k}_truncated": sum(1 for v in truncated_count.values() if v == k)
            for k in range(0, 11)
        },
    }
    (args.out_dir / "truncated_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(
        f"wrote {n_pg} prompts → {out_pg}; {n_cr} candidates → {out_cr}; "
        f"report → {args.out_dir / 'truncated_report.json'}",
        flush=True,
    )
    print("Re-run S2' with --config experiments/configs/generation_se_qwen_long.yaml "
          f"using these filtered files, then merge the new free_sample_rows.json back "
          f"into the original (overwriting truncated samples).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
