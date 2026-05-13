"""Review-driven ablations (efficient): bootstrap CI for Δ + Fusion lift CI
+ SVAMP sensitivity + per-dataset Δ + Spearman ρ per (axis, signal).

Vectorized prompt-grouped bootstrap using numpy index arrays (no pd.concat per iter).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# 7 corpus axes × 3 detection signals = 21 Spearman ρ cells (Tab 4.5).
CORPUS_AXES = [
    "entity_frequency_axis_bin_10",
    "corpus_axis_bin_10",  # baseline = (entity_freq + entity_pair) / 2
    "qa_bridge_axis_bin_10",
    "ans_ngram_3_axis_bin_10",
    "ans_ngram_5_axis_bin_10",
    "ans_ngram_3_zero_count_bin_10",
    "entity_pair_cooccurrence_axis_bin_10",
]
DETECTION_SIGNALS = [
    ("semantic_entropy", True),
    ("semantic_energy_cluster_uncertainty", True),
    ("sample_nll", True),
]


def per_decile_auroc(y: np.ndarray, score: np.ndarray, bins: np.ndarray) -> dict:
    """y: is_correct (0/1), score: signal (already flipped if needed), bins: decile labels."""
    out = {}
    for v in np.unique(bins):
        mask = bins == v
        ys = y[mask]
        if len(np.unique(ys)) < 2:
            continue
        try:
            out[v] = roc_auc_score(ys, score[mask])
        except Exception:
            continue
    return out


def delta(d: dict) -> float:
    if not d:
        return float("nan")
    vs = list(d.values())
    return max(vs) - min(vs)


def spearman_per_axis(df: pd.DataFrame, axis_col: str, signal_col: str, flip: bool) -> dict:
    """Spearman ρ between bin rank (0..9) and per-bin AUROC.

    Returns {"rho": float, "p": float, "n_bins": int, "per_bin_auroc": {bin: auroc}}.
    None if fewer than 3 valid bins.
    """
    sub = df.dropna(subset=[axis_col, signal_col, "is_correct"])
    if len(sub) == 0:
        return {"rho": None, "p": None, "n_bins": 0, "per_bin_auroc": {}}
    score = -sub[signal_col].values if flip else sub[signal_col].values
    bins = sub[axis_col].astype(int).values  # decile id 0..9
    per_bin = per_decile_auroc(sub["is_correct"].values, score, bins)
    if len(per_bin) < 3:
        return {"rho": None, "p": None, "n_bins": len(per_bin), "per_bin_auroc": per_bin}
    bin_ids = sorted(per_bin.keys())
    aurocs = [per_bin[b] for b in bin_ids]
    res = spearmanr(bin_ids, aurocs)
    rho = float(res[0])  # type: ignore[index]
    p = float(res[1])  # type: ignore[index]
    return {
        "rho": rho if rho == rho else None,  # NaN check
        "p": p if p == p else None,
        "n_bins": len(per_bin),
        "per_bin_auroc": {int(k): float(v) for k, v in per_bin.items()},
    }


def compute_decile_spearman_grid(df: pd.DataFrame) -> dict:
    """7 axes × 3 detection signals = 21 ρ cells for Tab 4.5."""
    grid = {}
    for axis in CORPUS_AXES:
        if axis not in df.columns:
            continue
        grid[axis] = {}
        for sig, flip in DETECTION_SIGNALS:
            if sig not in df.columns:
                continue
            grid[axis][sig] = spearman_per_axis(df, axis, sig, flip)
    return grid


def bootstrap_delta_diff_fast(df: pd.DataFrame, axis_a: str, axis_b: str,
                              signal: str, flip: bool, n_boot: int = 1000, seed: int = 0):
    """Prompt-grouped bootstrap. Vectorized.

    For each iteration: sample prompts with replacement, gather row indices once,
    run AUROC per decile.
    """
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=[axis_a, axis_b]).reset_index(drop=True)
    prompts = df["prompt_id"].unique()
    # Pre-build prompt -> row indices
    prompt_to_idx = {p: np.where(df["prompt_id"].values == p)[0]
                     for p in prompts}
    score_full = -df[signal].values if flip else df[signal].values
    y_full = df["is_correct"].values
    bin_a_full = df[axis_a].astype(str).values
    bin_b_full = df[axis_b].astype(str).values

    deltas_a, deltas_b = [], []
    for it in range(n_boot):
        sample = rng.choice(prompts, size=len(prompts), replace=True)
        idx = np.concatenate([prompt_to_idx[p] for p in sample])
        d_a = delta(per_decile_auroc(y_full[idx], score_full[idx], bin_a_full[idx]))
        d_b = delta(per_decile_auroc(y_full[idx], score_full[idx], bin_b_full[idx]))
        deltas_a.append(d_a)
        deltas_b.append(d_b)
        if (it + 1) % 100 == 0:
            print(f"  boot {it+1}/{n_boot}", file=sys.stderr)
    deltas_a = np.array(deltas_a)
    deltas_b = np.array(deltas_b)
    diffs = deltas_a - deltas_b

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "delta_a_mean": float(np.nanmean(deltas_a)),
        "delta_a_ci": ci(deltas_a),
        "delta_b_mean": float(np.nanmean(deltas_b)),
        "delta_b_ci": ci(deltas_b),
        "diff_mean": float(np.nanmean(diffs)),
        "diff_ci": ci(diffs),
        "diff_positive_frac": float((diffs > 0).mean()),
        "n_boot": n_boot,
    }


def bootstrap_fusion_lift_fast(preds_no: pd.DataFrame, preds_with: pd.DataFrame,
                               n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = preds_no.merge(
        preds_with,
        on=["prompt_id", "sample_index", "is_correct"],
        suffixes=("_nc", "_wc"),
    )
    prompts = base["prompt_id"].unique()
    prompt_to_idx = {p: np.where(base["prompt_id"].values == p)[0] for p in prompts}
    y = base["is_correct"].values
    s_nc = base["pred_nc"].values
    s_wc = base["pred_wc"].values

    lifts = []
    for it in range(n_boot):
        sample = rng.choice(prompts, size=len(prompts), replace=True)
        idx = np.concatenate([prompt_to_idx[p] for p in sample])
        if len(np.unique(y[idx])) < 2:
            continue
        a_nc = roc_auc_score(y[idx], s_nc[idx])
        a_wc = roc_auc_score(y[idx], s_wc[idx])
        lifts.append(a_wc - a_nc)
        if (it + 1) % 100 == 0:
            print(f"  fusion boot {it+1}/{n_boot}", file=sys.stderr)
    lifts = np.array(lifts)
    return {
        "lift_mean": float(np.nanmean(lifts)),
        "lift_ci": (float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))),
        "lift_positive_frac": float((lifts > 0).mean()),
        "n_boot": n_boot,
    }


def per_dataset_delta(df: pd.DataFrame, axis_col: str, signal_col: str, flip: bool):
    out = {}
    for ds, sub in df.groupby("dataset"):
        score = -sub[signal_col].values if flip else sub[signal_col].values
        d = delta(per_decile_auroc(sub["is_correct"].values, score,
                                   sub[axis_col].astype(str).values))
        out[ds] = d
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Per-model run dir containing results/generation_features.parquet "
             "and results/fusion.generation_level/predictions.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Default: <run-dir>/results/review_ablations.json",
    )
    parser.add_argument("--n-boot", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    run = args.run_dir
    if (run / "results/generation_features.parquet").exists():
        results_dir = run / "results"
    elif (run / "generation_features.parquet").exists():
        results_dir = run
    else:
        sys.exit(f"generation_features.parquet not found under {run} (or {run}/results)")
    out_path = args.out or (results_dir / "review_ablations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading generation_features ...", file=sys.stderr)
    df = pd.read_parquet(results_dir / "generation_features.parquet")
    df = df.dropna(subset=["entity_pair_cooccurrence_axis_bin_10",
                           "entity_frequency_axis_bin_10",
                           "is_correct"]).reset_index(drop=True)
    print(f"  rows={len(df)}, prompts={df['prompt_id'].nunique()}", file=sys.stderr)

    out = {}

    print("[2/6] decile Spearman ρ grid (7 axes × 3 signals) ...", file=sys.stderr)
    out["decile_spearman"] = compute_decile_spearman_grid(df)

    print(f"[3/6] bootstrap Δ (SE) entity_pair vs entity_freq, B={args.n_boot} ...",
          file=sys.stderr)
    out["bootstrap_se"] = bootstrap_delta_diff_fast(
        df, "entity_pair_cooccurrence_axis_bin_10",
        "entity_frequency_axis_bin_10",
        "semantic_entropy", flip=True, n_boot=args.n_boot,
    )
    print(f"[4/6] bootstrap Δ (Energy), B={args.n_boot} ...", file=sys.stderr)
    out["bootstrap_energy"] = bootstrap_delta_diff_fast(
        df, "entity_pair_cooccurrence_axis_bin_10",
        "entity_frequency_axis_bin_10",
        "semantic_energy_cluster_uncertainty", flip=True, n_boot=args.n_boot,
    )

    print("[5/6] SVAMP-excluded sensitivity ...", file=sys.stderr)
    df_ns = df[df["dataset"] != "SVAMP"]
    se_pair = delta(per_decile_auroc(
        df_ns["is_correct"].values, -df_ns["semantic_entropy"].values,
        df_ns["entity_pair_cooccurrence_axis_bin_10"].astype(str).values))
    se_freq = delta(per_decile_auroc(
        df_ns["is_correct"].values, -df_ns["semantic_entropy"].values,
        df_ns["entity_frequency_axis_bin_10"].astype(str).values))
    en_pair = delta(per_decile_auroc(
        df_ns["is_correct"].values, -df_ns["semantic_energy_cluster_uncertainty"].values,
        df_ns["entity_pair_cooccurrence_axis_bin_10"].astype(str).values))
    en_freq = delta(per_decile_auroc(
        df_ns["is_correct"].values, -df_ns["semantic_energy_cluster_uncertainty"].values,
        df_ns["entity_frequency_axis_bin_10"].astype(str).values))
    out["svamp_excluded"] = {
        "n_prompts": int(df_ns["prompt_id"].nunique()),
        "se_pair_delta": se_pair, "se_freq_delta": se_freq,
        "se_ratio": se_pair / se_freq if se_freq > 0 else None,
        "energy_pair_delta": en_pair, "energy_freq_delta": en_freq,
        "energy_ratio": en_pair / en_freq if en_freq > 0 else None,
    }

    print("[6/6] per-dataset Δ ...", file=sys.stderr)
    out["per_dataset_pair_se"] = per_dataset_delta(
        df, "entity_pair_cooccurrence_axis_bin_10", "semantic_entropy", flip=True)
    out["per_dataset_freq_se"] = per_dataset_delta(
        df, "entity_frequency_axis_bin_10", "semantic_entropy", flip=True)
    out["per_dataset_pair_energy"] = per_dataset_delta(
        df, "entity_pair_cooccurrence_axis_bin_10",
        "semantic_energy_cluster_uncertainty", flip=True)

    # Persist partial result before fusion lift in case of crash
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  partial result → {out_path}", file=sys.stderr)

    # Fusion lift CI from predictions.jsonl + features.is_correct
    preds_path = results_dir / "fusion.generation_level/predictions.jsonl"
    if preds_path.exists():
        print("[+] fusion lift CI ...", file=sys.stderr)
        preds = pd.read_json(preds_path, lines=True)
        print(f"  preds methods: {preds['method'].unique().tolist()}", file=sys.stderr)
        labels = df[["prompt_id", "sample_index", "is_correct"]].drop_duplicates(
            ["prompt_id", "sample_index"])
        nc = preds[preds["method"] == "gradient boosting (no corpus)"][
            ["prompt_id", "sample_index", "score"]].rename(columns={"score": "pred"})
        wc = preds[preds["method"] == "gradient boosting (with corpus)"][
            ["prompt_id", "sample_index", "score"]].rename(columns={"score": "pred"})
        nc = nc.merge(labels, on=["prompt_id", "sample_index"])
        wc = wc.merge(labels, on=["prompt_id", "sample_index"])
        out["fusion_lift_gbm"] = bootstrap_fusion_lift_fast(nc, wc, n_boot=args.n_boot)
    else:
        print(f"  fusion predictions not found at {preds_path}; skipping fusion lift",
              file=sys.stderr)

    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"final result → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
