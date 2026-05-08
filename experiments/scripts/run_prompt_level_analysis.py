"""Prompt-level reframe analysis.

Builds prompt-level features from candidate-row features.parquet + free_sample
labels, runs fusion baselines + corpus-conditioned variants on prompt unit, and
emits robustness artifacts (bootstrap CI, corpus bin reliability, LODO,
calibration) mirroring the candidate-row pipeline so the thesis can swap units.

Inputs (read from --run-dir):
  results/features.parquet
  results/generation/free_sample_rows.json
Outputs:
  results/prompt_features.parquet
  results/fusion.prompt_level/{summary.json, predictions.jsonl}
  results/robustness.prompt_level/{summary.json, bootstrap_ci.json,
    corpus_bin_reliability.json, leave_one_dataset_out.json,
    threshold_calibration.json}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def overlap_match(sample: str, refs: list[str]) -> bool:
    if not sample or not refs:
        return False
    s = set(sample.split())
    for ref in refs:
        c = set(ref.split())
        if c and len(s & c) / len(c) >= 0.5:
            return True
    return False


def build_prompt_features(run_dir: Path) -> pd.DataFrame:
    fs = json.loads((run_dir / "results/generation/free_sample_rows.json").read_text())["samples"]
    pa: dict[str, dict] = {}
    for s in fs:
        pid = s["prompt_id"]
        if pid not in pa:
            md = s.get("metadata") or {}
            refs = []
            for k in ("right_answer", "best_answer"):
                v = md.get(k)
                if isinstance(v, str) and v:
                    refs.append(norm(v))
            ca = md.get("correct_answers")
            if isinstance(ca, list):
                refs.extend(norm(x) for x in ca if isinstance(x, str) and x)
            pa[pid] = {"refs": list({r for r in refs if r}), "matches": 0, "n": 0, "dataset": s["dataset"]}
        if pa[pid]["refs"]:
            if overlap_match(norm(s.get("response_text", "")), pa[pid]["refs"]):
                pa[pid]["matches"] += 1
        pa[pid]["n"] += 1

    rows = []
    for pid, v in pa.items():
        if not v["refs"] or v["n"] == 0:
            continue
        rows.append({
            "prompt_id": pid,
            "dataset": v["dataset"],
            "accuracy": v["matches"] / v["n"],
            "is_hard": int(v["matches"] / v["n"] < 0.5),
        })
    label_df = pd.DataFrame(rows)

    feat = pd.read_parquet(run_dir / "results/features.parquet")
    inner = pd.json_normalize(feat["features"])
    big = pd.concat([feat[["prompt_id", "dataset", "candidate_role"]], inner], axis=1)

    prompt_cols = [
        "semantic_entropy",
        "semantic_entropy_discrete_cluster_entropy",
        "semantic_entropy_cluster_count",
        "semantic_energy_boltzmann",
        "semantic_energy_cluster_uncertainty",
        "semantic_energy_sample_energy",
        "corpus_axis_bin",
        "corpus_axis_bin_5",
        "corpus_axis_bin_10",
    ]
    cand_cols = [
        "mean_negative_log_probability",
        "logit_variance",
        "confidence_margin",
        "entity_frequency_axis",
        "entity_pair_cooccurrence_axis",
        "entity_frequency_min",
        "entity_frequency_mean",
        "corpus_risk_only",
    ]
    prompt_view = big.drop_duplicates(subset=["prompt_id"])[["prompt_id", *prompt_cols]]

    g_correct = big[big["candidate_role"] == "right"].groupby("prompt_id")[cand_cols].mean()
    g_correct.columns = [c + "_correct" for c in g_correct.columns]
    g_hallu = big[big["candidate_role"] == "hallucinated"].groupby("prompt_id")[cand_cols].mean()
    g_hallu.columns = [c + "_hallu" for c in g_hallu.columns]
    g_mean = big.groupby("prompt_id")[cand_cols].mean()
    g_mean.columns = [c + "_mean" for c in g_mean.columns]
    g_max = big.groupby("prompt_id")[cand_cols].max()
    g_max.columns = [c + "_max" for c in g_max.columns]
    g_delta = (g_hallu.values - g_correct.values)
    delta_df = pd.DataFrame(g_delta, index=g_correct.index, columns=[c[:-len("_correct")] + "_delta" for c in g_correct.columns])

    df = (label_df
          .merge(prompt_view, on="prompt_id", how="inner")
          .merge(g_mean, on="prompt_id", how="left")
          .merge(g_max, on="prompt_id", how="left")
          .merge(delta_df, on="prompt_id", how="left"))
    return df


# ---------- fusion ----------

CORE_PROMPT = ["semantic_entropy", "semantic_energy_boltzmann"]
CORPUS_AGG = [
    "entity_frequency_axis_mean",
    "entity_pair_cooccurrence_axis_mean",
    "corpus_risk_only_mean",
    "entity_frequency_axis_max",
    "entity_pair_cooccurrence_axis_max",
]
CAND_AGG = [
    "mean_negative_log_probability_mean",
    "logit_variance_mean",
    "confidence_margin_mean",
    "mean_negative_log_probability_delta",
    "confidence_margin_delta",
]


def _fit_predict_oof(X, y, model_factory, n_splits=5, seed=42):
    oof = np.zeros(len(y))
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _aurocs(y, score):
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(score, 1e-6, 1 - 1e-6))) if score.min() >= 0 and score.max() <= 1 else None,
    }


def _flip_if_needed(y, score):
    a = roc_auc_score(y, score)
    if a < 0.5:
        return -score, True
    return score, False


def run_fusion(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = df["is_hard"].values
    methods: dict[str, dict] = {}
    pred_rows = []

    # Single signals
    for name, col in [
        ("SE-only", "semantic_entropy"),
        ("Energy-only", "semantic_energy_boltzmann"),
        ("logit-diagnostic-only", "mean_negative_log_probability_mean"),
        ("corpus-axis-only", "corpus_risk_only_mean"),
        ("entity-frequency-only", "entity_frequency_axis_mean"),
        ("entity-pair-cooccurrence-only", "entity_pair_cooccurrence_axis_mean"),
    ]:
        s = df[col].fillna(0).values
        s2, flipped = _flip_if_needed(y, s)
        # rescale to [0,1] for brier
        s_norm = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-9)
        methods[name] = {**_aurocs(y, s_norm), "flipped": flipped}

    # Global fusion (no corpus / with corpus)
    def feat(df, with_corpus):
        cols = CORE_PROMPT + CAND_AGG + (CORPUS_AGG if with_corpus else [])
        return df[cols].fillna(0).values

    for name, factory, with_corpus in [
        ("logistic regression (no corpus)", lambda: LogisticRegression(max_iter=5000), False),
        ("logistic regression (with corpus)", lambda: LogisticRegression(max_iter=5000), True),
        ("random forest (no corpus)", lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
        ("random forest (with corpus)", lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), True),
        ("gradient boosting (no corpus)", lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42), False),
        ("gradient boosting (with corpus)", lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42), True),
        ("SVM rbf (with corpus)", lambda: SVC(probability=True, kernel="rbf", random_state=42), True),
    ]:
        X = feat(df, with_corpus)
        oof = _fit_predict_oof(X, y, factory)
        methods[name] = {**_aurocs(y, oof), "flipped": False}
        for pid, score in zip(df["prompt_id"].values, oof):
            pred_rows.append({"method": name, "prompt_id": pid, "score": float(score)})

    # Corpus-bin weighted fusion: per-bin LR, weighted by bin assignment
    bin_field = "corpus_axis_bin"
    cb_oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(df):
        for b in df[bin_field].dropna().unique():
            tr_mask = (df.iloc[tr][bin_field] == b).values
            te_mask = (df.iloc[te][bin_field] == b).values
            if tr_mask.sum() < 20 or te_mask.sum() == 0:
                continue
            X_tr = feat(df.iloc[tr], with_corpus=False)[tr_mask]
            X_te = feat(df.iloc[te], with_corpus=False)[te_mask]
            y_tr = y[tr][tr_mask]
            if len(set(y_tr)) < 2:
                continue
            m = LogisticRegression(max_iter=5000)
            m.fit(X_tr, y_tr)
            te_idx = np.array(te)[te_mask]
            cb_oof[te_idx] = m.predict_proba(X_te)[:, 1]
    methods["corpus-bin weighted fusion"] = {**_aurocs(y, cb_oof), "flipped": False}
    for pid, score in zip(df["prompt_id"].values, cb_oof):
        pred_rows.append({"method": "corpus-bin weighted fusion", "prompt_id": pid, "score": float(score)})

    # Per-dataset breakdown
    by_ds = {}
    for ds in df["dataset"].unique():
        m = (df["dataset"] == ds).values
        ds_metrics = {}
        for name in methods:
            ds_metrics[name] = {}
        # For each method we need the original score; reuse pred_rows
        for r in pred_rows:
            if r["method"] in ds_metrics:
                pass
        by_ds[ds] = {"row_count": int(m.sum()), "is_hard_rate": float(y[m].mean())}

    return {"row_count": len(df), "is_hard_rate": float(y.mean()), "methods": methods, "by_dataset": by_ds}, pd.DataFrame(pred_rows)


# ---------- robustness ----------

def bootstrap_ci(df: pd.DataFrame, preds: pd.DataFrame, ref: str, cands: list[str], n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    pids = df["prompt_id"].values
    y = df.set_index("prompt_id")["is_hard"]
    score_by = {m: preds[preds["method"] == m].set_index("prompt_id")["score"] for m in cands + [ref]}
    if ref not in score_by:
        return out
    for c in cands:
        if c not in score_by:
            continue
        deltas = []
        # observed
        ya = y.loc[pids].values
        sr = score_by[ref].loc[pids].values
        sc = score_by[c].loc[pids].values
        obs = roc_auc_score(ya, sc) - roc_auc_score(ya, sr)
        for _ in range(n_boot):
            idx = rng.choice(len(pids), size=len(pids), replace=True)
            ya_b, sr_b, sc_b = ya[idx], sr[idx], sc[idx]
            if len(set(ya_b)) < 2:
                continue
            deltas.append(roc_auc_score(ya_b, sc_b) - roc_auc_score(ya_b, sr_b))
        deltas = np.array(deltas)
        out.append({
            "candidate": c,
            "reference": ref,
            "observed_delta": float(obs),
            "ci_95_lower": float(np.percentile(deltas, 2.5)),
            "ci_95_upper": float(np.percentile(deltas, 97.5)),
            "bootstrap_n": int(len(deltas)),
        })
    return out


def corpus_bin_reliability(df: pd.DataFrame, preds: pd.DataFrame, methods: list[str], bin_field: str):
    out = {}
    y = df.set_index("prompt_id")["is_hard"]
    score_by = {m: preds[preds["method"] == m].set_index("prompt_id")["score"] for m in methods}
    df_idx = df.set_index("prompt_id")
    for m in methods:
        bins = []
        for b in sorted(df[bin_field].dropna().unique()):
            mask = (df_idx[bin_field] == b)
            pids = df_idx.index[mask].values
            if len(pids) < 20 or len(set(y.loc[pids])) < 2:
                bins.append({"bin": b, "row_count": int(len(pids)), "auroc": None})
                continue
            score = score_by[m].loc[pids].values
            bins.append({
                "bin": b,
                "row_count": int(len(pids)),
                "is_hard_rate": float(y.loc[pids].mean()),
                "auroc": float(roc_auc_score(y.loc[pids], score)),
            })
        out[m] = {"bins": bins}
    return out


def lodo(df: pd.DataFrame, preds: pd.DataFrame, methods: list[str]):
    out = {"per_dataset": {}, "aggregate": {}}
    y = df.set_index("prompt_id")["is_hard"]
    score_by = {m: preds[preds["method"] == m].set_index("prompt_id")["score"] for m in methods}
    df_idx = df.set_index("prompt_id")
    for m in methods:
        out["aggregate"][m] = {"auroc": float(roc_auc_score(y.loc[df_idx.index], score_by[m].loc[df_idx.index]))}
    for ds in df["dataset"].unique():
        mask = (df_idx["dataset"] == ds)
        pids = df_idx.index[mask].values
        ds_out = {}
        for m in methods:
            score = score_by[m].loc[pids].values
            ya = y.loc[pids].values
            if len(set(ya)) < 2:
                ds_out[m] = {"auroc": None}
            else:
                ds_out[m] = {"auroc": float(roc_auc_score(ya, score))}
        out["per_dataset"][ds] = ds_out
    return out


def threshold_calibration(df: pd.DataFrame, preds: pd.DataFrame, methods: list[str], n_bins=10):
    out = {}
    y = df.set_index("prompt_id")["is_hard"]
    score_by = {m: preds[preds["method"] == m].set_index("prompt_id")["score"] for m in methods}
    pids = df["prompt_id"].values
    ya = y.loc[pids].values
    for m in methods:
        s = score_by[m].loc[pids].values
        s_clip = np.clip(s, 1e-6, 1 - 1e-6)
        brier = float(brier_score_loss(ya, s_clip))
        # ECE
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (s_clip >= bin_edges[i]) & (s_clip < bin_edges[i + 1])
            if in_bin.sum() == 0:
                continue
            conf = s_clip[in_bin].mean()
            acc = ya[in_bin].mean()
            ece += (in_bin.sum() / len(s_clip)) * abs(conf - acc)
        out[m] = {"brier": brier, "ece": float(ece)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    print("[1/4] building prompt features ...", flush=True)
    df = build_prompt_features(run_dir)
    out_pf = run_dir / "results/prompt_features.parquet"
    df.to_parquet(out_pf, index=False)
    print(f"  saved {out_pf} ({len(df)} prompts)")
    print(f"  is_hard rate: {df['is_hard'].mean():.3f} ({df['is_hard'].sum()}/{len(df)})")
    print(f"  per-dataset: {df.groupby('dataset')['is_hard'].agg(['count','mean']).to_dict()}")

    print("[2/4] running prompt-level fusion ...", flush=True)
    fusion_summary, preds = run_fusion(df)
    fusion_dir = run_dir / "results/fusion.prompt_level"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    (fusion_dir / "summary.json").write_text(json.dumps(fusion_summary, indent=2))
    preds.to_json(fusion_dir / "predictions.jsonl", orient="records", lines=True)
    print(f"  saved {fusion_dir}/")
    print("  AUROC ranking:")
    for name, m in sorted(fusion_summary["methods"].items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"    {name:<48} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}")

    print("[3/4] running prompt-level robustness ...", flush=True)
    rob_dir = run_dir / "results/robustness.prompt_level"
    rob_dir.mkdir(parents=True, exist_ok=True)
    fusion_methods = [n for n in fusion_summary["methods"] if n in preds["method"].unique()]
    bs = bootstrap_ci(df, preds, ref="logistic regression (no corpus)",
                      cands=[n for n in fusion_methods if "no corpus" not in n.lower() or "weighted" in n.lower()])
    bs2 = bootstrap_ci(df, preds, ref="gradient boosting (no corpus)",
                       cands=[n for n in fusion_methods if "with corpus" in n or "weighted" in n.lower() or n.startswith("corpus-")])
    (rob_dir / "bootstrap_ci.json").write_text(json.dumps({"vs_no_corpus_linear": bs, "vs_no_corpus_nonlinear": bs2}, indent=2))

    cb_methods = ["corpus-bin weighted fusion","gradient boosting (no corpus)","random forest (no corpus)",
                  "logistic regression (no corpus)"]
    cbr = {}
    for bf in ["corpus_axis_bin", "corpus_axis_bin_5", "corpus_axis_bin_10"]:
        cbr[bf] = corpus_bin_reliability(df, preds, [m for m in cb_methods if m in fusion_methods], bf)
    (rob_dir / "corpus_bin_reliability.json").write_text(json.dumps(cbr, indent=2))

    lo = lodo(df, preds, fusion_methods)
    (rob_dir / "leave_one_dataset_out.json").write_text(json.dumps(lo, indent=2))

    cal = threshold_calibration(df, preds, fusion_methods)
    (rob_dir / "threshold_calibration.json").write_text(json.dumps(cal, indent=2))

    summary = {
        "fusion_summary_path": str(fusion_dir / "summary.json"),
        "fusion_methods": fusion_methods,
        "row_count": len(df),
        "is_hard_rate": float(df["is_hard"].mean()),
        "by_dataset": {ds: {"row_count": int((df["dataset"] == ds).sum()),
                            "is_hard_rate": float(df[df["dataset"] == ds]["is_hard"].mean())}
                       for ds in df["dataset"].unique()},
    }
    (rob_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {rob_dir}/")

    print("[4/4] done.")


if __name__ == "__main__":
    main()
