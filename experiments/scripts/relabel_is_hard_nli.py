"""CLI wrapper: NLI 기반 prompt-level is_hard 재라벨링.

본 스크립트는 ``experiments.application.prompt_accuracy`` 의 thin wrapper 다.
실제 매칭 로직은 application 모듈에 정의되어 있고, run_prompt_level_analysis.py
와 동일한 코드를 호출한다.

Inputs:
  $RUN/results/generation/free_sample_rows.json
Outputs:
  $RUN/results/prompt_accuracy.parquet      (canonical 산출물)
  $RUN/results/prompt_accuracy.audit.json   (per-dataset is_hard rate, label change)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.application.prompt_accuracy import (
    DEFAULT_HARD_ACCURACY_CUTOFF,
    DEFAULT_NLI_MODEL_NAME,
    DEFAULT_NLI_THRESHOLD,
    build_prompt_accuracy_frame,
    write_prompt_accuracy_artifacts,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_NLI_MODEL_NAME)
    ap.add_argument("--threshold", type=float, default=DEFAULT_NLI_THRESHOLD,
                    help="NLI 양방향 max entailment 확률 임계값. 기본 0.5.")
    ap.add_argument("--hard-cutoff", type=float, default=DEFAULT_HARD_ACCURACY_CUTOFF,
                    help="accuracy < cutoff 면 is_hard=1. 기본 0.5.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-nli", action="store_true",
                    help="token-overlap 만 사용 (sanity check 모드).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fs_path = run_dir / "results/generation/free_sample_rows.json"
    out_dir = run_dir / "results"

    print(f"[1/2] loading {fs_path}", flush=True)
    free_samples = json.loads(fs_path.read_text())["samples"]

    use_nli = not args.no_nli
    print(f"[2/2] computing prompt accuracy (use_nli={use_nli}, threshold={args.threshold})", flush=True)
    df = build_prompt_accuracy_frame(
        free_samples,
        use_nli=use_nli,
        nli_model_name=args.model,
        threshold=args.threshold,
        hard_cutoff=args.hard_cutoff,
        batch_size=args.batch_size,
    )
    table_path, audit_path = write_prompt_accuracy_artifacts(
        df, out_dir,
        nli_model_name=args.model,
        threshold=args.threshold,
        hard_cutoff=args.hard_cutoff,
        use_nli=use_nli,
    )
    print(f"  saved {table_path}")
    print(f"  saved {audit_path}")
    print(json.dumps(json.loads(audit_path.read_text()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
