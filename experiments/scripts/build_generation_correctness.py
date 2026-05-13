"""S7' wrapper — build generation_correctness.parquet from free_sample_rows.json.

NLI bidirectional entailment (microsoft/deberta-large-mnli, threshold 0.5)
between each free-sample answer and the dataset's gold candidate answers.
Output row=(prompt_id, sample_index, is_correct) consumed by S11'.

Usage:
  uv run python experiments/scripts/build_generation_correctness.py \\
    --free-samples $RUN/qwen/results/generation/free_sample_rows.json \\
    --out-dir $RUN/qwen/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.application.generation_correctness import (  # noqa: E402
    DEFAULT_NLI_MODEL_NAME,
    DEFAULT_NLI_THRESHOLD,
    build_generation_correctness_frame,
    write_generation_correctness_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--free-samples",
        type=Path,
        required=True,
        help="Path to free_sample_rows.json (S3' consolidated output).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory. Writes generation_correctness.{parquet,audit.json}.",
    )
    parser.add_argument(
        "--nli-model",
        default=DEFAULT_NLI_MODEL_NAME,
        help=f"HF NLI model id (default: {DEFAULT_NLI_MODEL_NAME}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_NLI_THRESHOLD,
        help=f"Bidirectional entailment threshold (default: {DEFAULT_NLI_THRESHOLD}).",
    )
    parser.add_argument(
        "--no-nli",
        action="store_true",
        help="Disable NLI matching, fall back to token-overlap (sanity only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="NLI inference batch size (default: 64).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.free_samples.exists():
        sys.exit(f"missing input: {args.free_samples}")

    payload = json.loads(args.free_samples.read_text())
    rows = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    print(f"loaded {len(rows)} free-sample rows from {args.free_samples}", flush=True)

    use_nli = not args.no_nli
    df = build_generation_correctness_frame(
        rows,
        use_nli=use_nli,
        nli_model_name=args.nli_model,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    table_path, audit_path = write_generation_correctness_artifacts(
        df,
        args.out_dir,
        nli_model_name=args.nli_model,
        threshold=args.threshold,
        use_nli=use_nli,
    )
    audit = json.loads(audit_path.read_text())
    print(
        f"wrote {len(df)} rows → {table_path} "
        f"(overall is_correct_rate={audit['overall_is_correct_rate']:.3f}, "
        f"per-dataset rates={audit['per_dataset_correctness']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
