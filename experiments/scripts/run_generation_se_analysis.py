"""SE 5-dataset 트랙 전용 generation-level fusion + robustness 분석.

기존 run_generation_level_analysis.py 는 paired track 의 features.parquet 단일
입력 가정. SE track 은 SE / Energy / corpus 분리이므로 mini script 가 직접 join.

Inputs ($RUN/qwen 또는 $RUN/gemma):
  results/generation_correctness.parquet
  results/free_sample_diagnostics.parquet
  results/semantic_entropy_features.parquet
  results/energy_features.parquet
  results/corpus_features.parquet

Outputs:
  results/generation_features.parquet
  results/fusion.generation_level/{summary.json, predictions.jsonl}
  results/robustness.generation_level/{summary.json, bootstrap_ci.json,
    corpus_bin_reliability.json, leave_one_dataset_out.json,
    threshold_calibration.json}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.application.generation_level_eval import (
    bootstrap_ci_per_decile,
    calibration,
    compute_aurac,
    corpus_bin_reliability,
    per_dataset_breakdown,
    run_generation_fusion,
)


def build_se_features(run_dir: Path) -> pd.DataFrame:
    print(f"[1/4] joining features", flush=True)
    gc = pd.read_parquet(run_dir / "results/generation_correctness.parquet")
    diag = pd.read_parquet(run_dir / "results/free_sample_diagnostics.parquet")
    se = pd.read_parquet(run_dir / "results/semantic_entropy_features.parquet")
    en = pd.read_parquet(run_dir / "results/energy_features.parquet")
    cf = pd.read_parquet(run_dir / "results/corpus_features.parquet")
    qb_path = run_dir / "results/qa_bridge_features.parquet"
    ng_path = run_dir / "results/ngram_coverage_features.parquet"
    qb = pd.read_parquet(qb_path) if qb_path.exists() else None
    ng = pd.read_parquet(ng_path) if ng_path.exists() else None

    print(f"  generation_correctness: {len(gc)}, diagnostics: {len(diag)}, SE: {len(se)}, Energy: {len(en)}, corpus: {len(cf)}, qa_bridge: {len(qb) if qb is not None else 0}, ngram: {len(ng) if ng is not None else 0}")

    # SE: prompt-level
    se_view = se[["prompt_id", "semantic_entropy_nli_likelihood",
                  "semantic_entropy_cluster_count",
                  "semantic_entropy_discrete_cluster_entropy"]].rename(
        columns={"semantic_entropy_nli_likelihood": "semantic_entropy"})
    # Energy: prompt-level
    en_view = en[["prompt_id", "semantic_energy_cluster_uncertainty",
                  "semantic_energy_sample_energy",
                  "semantic_energy_boltzmann"]]

    # Corpus: candidate-level (single right candidate per prompt_id)
    cf_inner = pd.json_normalize(cf["features"])
    cf_view = pd.concat([cf[["prompt_id"]], cf_inner], axis=1)
    cf_view = cf_view.drop_duplicates(subset=["prompt_id"])
    cf_keep = ["prompt_id", "entity_frequency_axis", "entity_pair_cooccurrence_axis",
               "entity_frequency_min", "entity_frequency_mean", "corpus_risk_only",
               "corpus_axis_bin", "corpus_axis_bin_5", "corpus_axis_bin_10"]
    cf_keep = [c for c in cf_keep if c in cf_view.columns]
    cf_view = cf_view[cf_keep]

    # Generation features: row=(prompt, sample_index)
    df = gc[["prompt_id", "dataset", "sample_index", "is_correct"]].copy()
    df = df.merge(diag.drop(columns=["dataset"], errors="ignore"),
                  on=["prompt_id", "sample_index"], how="left")
    df = df.merge(se_view, on="prompt_id", how="left")
    df = df.merge(en_view, on="prompt_id", how="left")
    df = df.merge(cf_view, on="prompt_id", how="left")

    # QA bridge (per-prompt, right candidate only)
    if qb is not None:
        qb_keep = ["prompt_id", "qa_bridge_pair_count", "qa_bridge_min", "qa_bridge_mean",
                   "qa_bridge_axis", "qa_bridge_zero_flag",
                   "n_question_entities", "n_candidate_entities"]
        qb_keep = [c for c in qb_keep if c in qb.columns]
        df = df.merge(qb[qb_keep].drop_duplicates("prompt_id"), on="prompt_id", how="left")

    # N-gram coverage (per-prompt, right candidate)
    if ng is not None:
        ng_keep = ["prompt_id"] + [c for c in ng.columns if c.startswith("ans_ngram_")]
        df = df.merge(ng[ng_keep].drop_duplicates("prompt_id"), on="prompt_id", how="left")

    # corpus axis bin (string) → ordinal
    for col in ["corpus_axis_bin", "corpus_axis_bin_5", "corpus_axis_bin_10"]:
        if col in df.columns and df[col].dtype == object:
            ordered = sorted(df[col].dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(ordered)}
            df[f"{col}_ord"] = df[col].map(mapping).astype("Int64")

    # 다양화 corpus axis: 각 corpus signal 을 quantile-decile bin 으로 → 영역별 분석 axis
    new_axes = {
        "qa_bridge_axis_bin_10": "qa_bridge_axis",
        "qa_bridge_min_bin_10": "qa_bridge_min",
        "ans_ngram_3_axis_bin_10": "ans_ngram_3_axis",
        "ans_ngram_5_axis_bin_10": "ans_ngram_5_axis",
        "ans_ngram_3_zero_count_bin_10": "ans_ngram_3_zero_count",
        "entity_pair_cooccurrence_axis_bin_10": "entity_pair_cooccurrence_axis",
        "entity_frequency_axis_bin_10": "entity_frequency_axis",
    }
    for new_col, src_col in new_axes.items():
        if src_col not in df.columns:
            continue
        # 같은 prompt 의 모든 sample 이 같은 값 (broadcast). prompt 단위 quantile.
        prompt_vals = df.drop_duplicates("prompt_id")[["prompt_id", src_col]].dropna()
        if len(prompt_vals) < 10:
            continue
        try:
            prompt_vals[new_col] = pd.qcut(
                prompt_vals[src_col].rank(method="first"),
                q=10, labels=[f"decile_{i:02d}_{i+1:02d}" for i in range(0, 100, 10)],
                duplicates="drop",
            )
        except Exception:
            # fallback to 5 bins on heavy ties
            try:
                prompt_vals[new_col] = pd.qcut(prompt_vals[src_col].rank(method="first"), q=5, duplicates="drop")
                prompt_vals[new_col] = prompt_vals[new_col].astype(str)
            except Exception:
                continue
        df = df.merge(prompt_vals[["prompt_id", new_col]].drop_duplicates("prompt_id"),
                      on="prompt_id", how="left")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Per-model run dir (e.g. .../qwen or .../gemma)")
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    df = build_se_features(run_dir)
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

    bs = bootstrap_ci_per_decile(
        df, preds,
        ref="Energy-only" if "Energy-only" in methods else fusion_methods[0],
        candidates=[m for m in fusion_methods if any(k in m for k in
                    ("regression", "forest", "boosting"))],
        bin_field="corpus_axis_bin_10",
        n_boot=args.bootstrap_n,
    )
    (rob_dir / "bootstrap_ci.json").write_text(json.dumps(bs, indent=2))

    cb = {}
    bin_fields = [
        "corpus_axis_bin", "corpus_axis_bin_5", "corpus_axis_bin_10",
        "qa_bridge_axis_bin_10", "qa_bridge_min_bin_10",
        "ans_ngram_3_axis_bin_10", "ans_ngram_5_axis_bin_10",
        "ans_ngram_3_zero_count_bin_10",
        "entity_pair_cooccurrence_axis_bin_10",
        "entity_frequency_axis_bin_10",
    ]
    # thesis Tab 4.5 의 per-decile range Δ 를 자동 산출하려면 fusion 뿐 아니라
    # 단일 환각 탐지 신호 (SE-only / Energy-only / logit-diagnostic-only =
    # sample_nll) 도 corpus_bin_reliability 에 포함해야 한다. 특히
    # HeadlineQaBridgeNllDelta 는 (qa_bridge_axis_bin_10, sample_nll) 조합.
    decomp_methods = [m for m in (
        fusion_methods
        + ["SE-only", "Energy-only", "logit-diagnostic-only"]
    ) if m in preds["method"].unique()]
    for bf in bin_fields:
        if bf in df.columns:
            cb[bf] = corpus_bin_reliability(df, preds, decomp_methods, bin_field=bf)
    (rob_dir / "corpus_bin_reliability.json").write_text(json.dumps(cb, indent=2))

    lo = per_dataset_breakdown(df, preds, fusion_methods)
    (rob_dir / "leave_one_dataset_out.json").write_text(json.dumps(lo, indent=2))

    cal = calibration(df, preds, fusion_methods)
    (rob_dir / "threshold_calibration.json").write_text(json.dumps(cal, indent=2))

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
