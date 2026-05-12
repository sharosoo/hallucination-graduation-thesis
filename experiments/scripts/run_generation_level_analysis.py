"""Generation-level fusion + robustness 분석 (Farquhar/Ma 평가 단위와 호환).

Inputs (read from --run-dir):
  results/generation/free_sample_rows.json  (read-only)
  results/free_sample_diagnostics.parquet   (sample 단위 token logit 통계, 없으면 자동 생성)
  results/generation_correctness.parquet    (NLI is_correct, 없으면 자동 생성)
  results/features.parquet                  (prompt-level SE / Energy / corpus axis broadcast)

Outputs (모두 신규 path, 기존 prompt_*.parquet / fusion.prompt_level / robustness.prompt_level
는 절대 건드리지 않음):
  results/generation_features.parquet
  results/fusion.generation_level/{summary.json, predictions.jsonl}
  results/robustness.generation_level/{summary.json, bootstrap_ci.json,
    corpus_bin_reliability.json, leave_one_dataset_out.json, threshold_calibration.json}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.application.generation_level_eval import (
    bootstrap_ci_per_decile,
    build_generation_features,
    calibration,
    corpus_bin_reliability,
    per_dataset_breakdown,
    run_generation_fusion,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-nli", action="store_true",
                    help="generation_correctness 산출 시 token-overlap 만 사용 (NLI skip).")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--nli-threshold", type=float, default=0.5)
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    print("[1/4] building generation features ...", flush=True)
    df = build_generation_features(
        run_dir,
        use_nli=not args.no_nli,
        nli_model_name=args.nli_model,
        nli_threshold=args.nli_threshold,
    )
    out_pf = run_dir / "results/generation_features.parquet"
    df.to_parquet(out_pf, index=False)
    print(f"  saved {out_pf} ({len(df)} samples, {df['prompt_id'].nunique()} prompts)")
    print(f"  is_correct rate overall: {df['is_correct'].mean():.3f}")
    print(f"  per-dataset: {df.groupby('dataset')['is_correct'].agg(['count','mean']).to_dict()}")

    print("[2/4] running generation-level fusion ...", flush=True)
    fusion_summary, preds = run_generation_fusion(df)
    fusion_dir = run_dir / "results/fusion.generation_level"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    (fusion_dir / "summary.json").write_text(json.dumps(fusion_summary, indent=2))
    preds.to_json(fusion_dir / "predictions.jsonl", orient="records", lines=True)
    print(f"  saved {fusion_dir}/")
    print("  AUROC ranking (with AURAC):")
    for name, m in sorted(fusion_summary["methods"].items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"    {name:<48} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} AURAC={m['aurac']:.3f}")

    print("[3/4] running generation-level robustness ...", flush=True)
    rob_dir = run_dir / "results/robustness.generation_level"
    rob_dir.mkdir(parents=True, exist_ok=True)

    methods = list(fusion_summary["methods"].keys())
    fusion_methods = [m for m in methods if m in preds["method"].unique()]

    # bootstrap (Fusion-Energy delta per decile, prompt-grouped)
    bs = bootstrap_ci_per_decile(
        df, preds,
        ref="Energy-only" if "Energy-only" in methods else fusion_methods[0],
        candidates=[m for m in fusion_methods if "fusion" in m.lower() or "boosting" in m.lower() or "forest" in m.lower() or "regression" in m.lower()],
        bin_field="corpus_axis_bin_10",
        n_boot=args.bootstrap_n,
    )
    (rob_dir / "bootstrap_ci.json").write_text(json.dumps(bs, indent=2))

    # per-decile reliability for all methods
    cb = {
        "corpus_axis_bin": corpus_bin_reliability(df, preds, fusion_methods, bin_field="corpus_axis_bin"),
        "corpus_axis_bin_5": corpus_bin_reliability(df, preds, fusion_methods, bin_field="corpus_axis_bin_5"),
        "corpus_axis_bin_10": corpus_bin_reliability(df, preds, fusion_methods, bin_field="corpus_axis_bin_10"),
    }
    (rob_dir / "corpus_bin_reliability.json").write_text(json.dumps(cb, indent=2))

    # per-dataset
    lo = per_dataset_breakdown(df, preds, fusion_methods)
    (rob_dir / "leave_one_dataset_out.json").write_text(json.dumps(lo, indent=2))

    # calibration
    cal = calibration(df, preds, fusion_methods)
    (rob_dir / "threshold_calibration.json").write_text(json.dumps(cal, indent=2))

    # robustness summary
    summary = {
        "row_count": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "is_correct_rate": float(df["is_correct"].mean()),
        "by_dataset": fusion_summary["by_dataset"],
        "n_methods": len(fusion_methods),
        "bootstrap_n": args.bootstrap_n,
    }
    (rob_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {rob_dir}/")

    print("[4/4] done.")


if __name__ == "__main__":
    main()
