"""Review-driven ablations (efficient): bootstrap CI for Δ + Fusion lift CI
+ SVAMP sensitivity + per-dataset Δ.

Vectorized prompt-grouped bootstrap using numpy index arrays (no pd.concat per iter).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RUN = Path("/mnt/data/hallucination-graduation-thesis-runs/se-pipeline-20260511T034406Z/qwen/results")


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


def main():
    print("[1/5] loading generation_features ...", file=sys.stderr)
    df = pd.read_parquet(RUN / "generation_features.parquet")
    df = df.dropna(subset=["entity_pair_cooccurrence_axis_bin_10",
                           "entity_frequency_axis_bin_10",
                           "is_correct"]).reset_index(drop=True)
    print(f"  rows={len(df)}, prompts={df['prompt_id'].nunique()}", file=sys.stderr)

    out = {}

    print("[2/5] bootstrap Δ (SE) entity_pair vs entity_freq, B=500 ...", file=sys.stderr)
    out["bootstrap_se"] = bootstrap_delta_diff_fast(
        df, "entity_pair_cooccurrence_axis_bin_10",
        "entity_frequency_axis_bin_10",
        "semantic_entropy", flip=True, n_boot=500,
    )
    print("[3/5] bootstrap Δ (Energy), B=500 ...", file=sys.stderr)
    out["bootstrap_energy"] = bootstrap_delta_diff_fast(
        df, "entity_pair_cooccurrence_axis_bin_10",
        "entity_frequency_axis_bin_10",
        "semantic_energy_cluster_uncertainty", flip=True, n_boot=500,
    )

    print("[4/5] SVAMP-excluded sensitivity ...", file=sys.stderr)
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

    print("[5/5] per-dataset Δ ...", file=sys.stderr)
    out["per_dataset_pair_se"] = per_dataset_delta(
        df, "entity_pair_cooccurrence_axis_bin_10", "semantic_entropy", flip=True)
    out["per_dataset_freq_se"] = per_dataset_delta(
        df, "entity_frequency_axis_bin_10", "semantic_entropy", flip=True)
    out["per_dataset_pair_energy"] = per_dataset_delta(
        df, "entity_pair_cooccurrence_axis_bin_10",
        "semantic_energy_cluster_uncertainty", flip=True)

    # Persist partial result before fusion lift in case of crash
    Path("/tmp/review_ablation/bootstrap_partial.json").write_text(
        json.dumps(out, indent=2, default=str))

    # Fusion lift CI from predictions.jsonl + features.is_correct
    print("[+] fusion lift CI ...", file=sys.stderr)
    preds = pd.read_json(RUN / "fusion.generation_level/predictions.jsonl",
                         lines=True)
    print(f"  preds methods: {preds['method'].unique().tolist()}", file=sys.stderr)
    labels = df[["prompt_id", "sample_index", "is_correct"]].drop_duplicates(
        ["prompt_id", "sample_index"])
    nc = preds[preds["method"] == "gradient boosting (no corpus)"][
        ["prompt_id", "sample_index", "score"]].rename(columns={"score": "pred"})
    wc = preds[preds["method"] == "gradient boosting (with corpus)"][
        ["prompt_id", "sample_index", "score"]].rename(columns={"score": "pred"})
    nc = nc.merge(labels, on=["prompt_id", "sample_index"])
    wc = wc.merge(labels, on=["prompt_id", "sample_index"])
    out["fusion_lift_gbm"] = bootstrap_fusion_lift_fast(nc, wc, n_boot=500)

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
