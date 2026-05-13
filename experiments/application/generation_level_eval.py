"""Generation-level fusion + robustness 분석 (Farquhar/Ma 평가 단위와 호환).

row 단위 = (prompt_id, sample_index). label = is_correct (NLI 양방향 entailment
매칭 결과 binary). prompt-level 신호 (SE / Semantic Energy / corpus axis) 는
같은 prompt 의 모든 sample 에 broadcast. sample-level 신호 (sample_nll /
sample_logit_variance / sample_logsumexp_mean / sample_sequence_log_prob) 는
free_sample_diagnostics 어댑터의 row 단위 값을 그대로 사용.

KFold split 은 GroupKFold(prompt_id) — 같은 prompt 의 sample 이 train/test 에
나뉘는 누수 방지.

Metrics: AUROC, AUPRC, Brier, ECE, AURAC (Farquhar Nature 2024 main metric).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

# Feature column names (semantic_entropy_features.parquet / energy / corpus 통합 후)
PROMPT_BROADCAST: list[str] = [
    "semantic_entropy",
    "semantic_entropy_discrete_cluster_entropy",
    "semantic_entropy_cluster_count",
    "semantic_energy_boltzmann",
    "semantic_energy_cluster_uncertainty",
    "semantic_energy_sample_energy",
]
SAMPLE_LEVEL: list[str] = [
    "sample_nll",
    "sample_sequence_log_prob",
    "sample_logit_variance",
    "sample_logsumexp_mean",
]
CORPUS_BROADCAST: list[str] = [
    "corpus_axis_bin",
    "corpus_axis_bin_5",
    "corpus_axis_bin_10",
]

# Fusion 입력 set
CORE_INPUTS = ["semantic_entropy", "semantic_energy_cluster_uncertainty",
               "sample_nll", "sample_logit_variance",
               "sample_logsumexp_mean", "sample_sequence_log_prob"]
# corpus 다양화 — with_corpus=True 시 fusion 에 추가될 모든 corpus signal.
# 데이터에 컬럼이 있는 것만 사용 (없으면 skip).
CORPUS_INPUTS = [
    "corpus_axis_bin_10_ord",          # entity-level, ordinal bin
    "entity_frequency_axis",           # entity-level, raw axis
    "entity_pair_cooccurrence_axis",   # entity-pair, raw axis
    "entity_frequency_min",            # raw min
    "qa_bridge_axis",                  # question-answer entity bridge
    "qa_bridge_min",                   # bridge min
    "qa_bridge_zero_flag",             # CHOKE proxy
    "ans_ngram_3_axis",                # 3-gram coverage
    "ans_ngram_5_axis",                # 5-gram coverage
    "ans_ngram_3_zero_count",          # novel phrase count
    "ans_ngram_5_zero_count",
]
# 단일 신호 corpus AUROC 평가용 (with_corpus 와 별개)
CORPUS_INPUT = "corpus_axis_bin_10_ord"


# ---------------- features ----------------

def build_generation_features(
    run_dir: Path,
    *,
    use_nli: bool = True,
    nli_model_name: str = "microsoft/deberta-large-mnli",
    nli_threshold: float = 0.5,
) -> pd.DataFrame:
    """row=(prompt_id, sample_index), label=is_correct, with all signals joined."""
    # local imports to keep module load cheap when only utilities are needed
    from experiments.adapters.free_sample_diagnostics import (
        build_diagnostics_frame,
    )
    from experiments.application.generation_correctness import (
        build_generation_correctness_frame,
        write_generation_correctness_artifacts,
    )

    fs_path = run_dir / "results/generation/free_sample_rows.json"
    gc_path = run_dir / "results/generation_correctness.parquet"
    diag_path = run_dir / "results/free_sample_diagnostics.parquet"

    # 1) generation correctness
    if gc_path.exists():
        gc = pd.read_parquet(gc_path)
        print(f"  reusing {gc_path.name} ({len(gc)} samples)", flush=True)
    else:
        print(f"  building {gc_path.name} via NLI matching ...", flush=True)
        fs_rows = json.loads(fs_path.read_text())["samples"]
        gc = build_generation_correctness_frame(
            fs_rows,
            use_nli=use_nli,
            nli_model_name=nli_model_name,
            threshold=nli_threshold,
        )
        write_generation_correctness_artifacts(
            gc, run_dir / "results",
            nli_model_name=nli_model_name,
            threshold=nli_threshold,
            use_nli=use_nli,
        )

    # 2) free-sample diagnostics
    if diag_path.exists():
        diag = pd.read_parquet(diag_path)
        print(f"  reusing {diag_path.name} ({len(diag)} samples)", flush=True)
    else:
        print(f"  building {diag_path.name} ...", flush=True)
        fs_rows = json.loads(fs_path.read_text())["samples"]
        diag = build_diagnostics_frame(fs_rows)
        diag.to_parquet(diag_path, index=False)

    # 3) prompt-level broadcast features
    feat = pd.read_parquet(run_dir / "results/features.parquet")
    inner = pd.json_normalize(feat["features"])
    big = pd.concat([feat[["prompt_id", "dataset", "candidate_role"]], inner], axis=1)
    keep_cols = ["prompt_id", *(c for c in PROMPT_BROADCAST if c in big.columns),
                 *(c for c in CORPUS_BROADCAST if c in big.columns)]
    prompt_view = big.drop_duplicates(subset=["prompt_id"])[keep_cols]

    # 4) join
    df = gc[["prompt_id", "dataset", "sample_index", "is_correct"]].copy()
    df = df.merge(
        diag[["prompt_id", "sample_index", *SAMPLE_LEVEL]],
        on=["prompt_id", "sample_index"], how="left",
    )
    df = df.merge(prompt_view, on="prompt_id", how="left")

    # corpus_axis_bin_* string labels (e.g. "decile_60_70") → ordinal int
    # for downstream fusion / single-signal scoring.
    for col in CORPUS_BROADCAST:
        if col in df.columns and df[col].dtype == object:
            ordered = sorted(df[col].dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(ordered)}
            df[f"{col}_ord"] = df[col].map(mapping).astype("Int64")
    return df


# ---------------- metrics ----------------

def _aurocs(y, score) -> dict:
    s_clip = np.clip(score, 1e-6, 1 - 1e-6) if score.min() >= 0 and score.max() <= 1 else score
    out = {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }
    if score.min() >= 0 and score.max() <= 1:
        out["brier"] = float(brier_score_loss(y, s_clip))
    out["aurac"] = float(compute_aurac(y, score))
    return out


def _flip_if_needed(y, score) -> tuple[np.ndarray, bool]:
    if roc_auc_score(y, score) < 0.5:
        return -score, True
    return score, False


def compute_aurac(y_correct: np.ndarray, score: np.ndarray) -> float:
    """Area Under Rejection-Accuracy Curve (Farquhar Nature 2024 main metric).

    score 는 높을수록 correctness 확률이 높음을 의미. cumulative top-k 의
    accuracy 평균.
    """
    n = int(len(y_correct))
    if n == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y_correct[order]
    cum_correct = np.cumsum(y_sorted)
    cum_acc = cum_correct / np.arange(1, n + 1)
    return float(np.mean(cum_acc))


def _ece(y, score, n_bins: int = 10) -> float:
    s = np.clip(score, 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (s >= edges[i]) & (s < edges[i + 1])
        if in_bin.sum() == 0:
            continue
        conf = s[in_bin].mean()
        acc = y[in_bin].mean()
        ece += (in_bin.sum() / len(s)) * abs(conf - acc)
    return float(ece)


# ---------------- fusion ----------------

def _fit_predict_oof_grouped(X, y, groups, model_factory, n_splits=5):
    oof = np.zeros(len(y))
    splitter = GroupKFold(n_splits=n_splits)
    for tr, te in splitter.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def run_generation_fusion(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """5-fold GroupKFold(prompt_id) fusion + per-method 단일 신호 baseline."""
    y = df["is_correct"].values.astype(int)
    groups = df["prompt_id"].values
    methods: dict[str, dict] = {}
    pred_rows: list[dict] = []

    # --- single signal baselines (no learning, score-based AUROC) ---
    # thesis Tab 4.1 / Tab 4.3 의 단일 신호 행들과 1:1 대응. 환각 탐지
    # 신호 (SE / Energy / logit 통계) + 7 corpus signals 모두 포함.
    single_signal_cols = [
        # detection signals
        ("SE-only", "semantic_entropy"),
        ("Energy-only", "semantic_energy_cluster_uncertainty"),
        ("logit-diagnostic-only", "sample_nll"),
        ("logit-variance-only", "sample_logit_variance"),
        ("sequence-log-prob-only", "sample_sequence_log_prob"),
        # corpus signals (Tab 4.3)
        ("corpus-axis-only", "corpus_axis_bin_10_ord"),
        ("entity-freq-only", "entity_frequency_axis"),
        ("entity-pair-cooc-only", "entity_pair_cooccurrence_axis"),
        ("qa-bridge-mean-only", "qa_bridge_axis"),
        ("qa-bridge-zero-only", "qa_bridge_zero_flag"),
        ("ngram3-mean-only", "ans_ngram_3_axis"),
        ("ngram3-zero-only", "ans_ngram_3_zero_count"),
        ("ngram5-mean-only", "ans_ngram_5_axis"),
    ]
    for name, col in single_signal_cols:
        if col not in df.columns:
            continue
        s = df[col].fillna(0).astype(float).values
        s2, flipped = _flip_if_needed(y, s)
        s_norm = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-9)
        m = _aurocs(y, s_norm)
        m["flipped"] = flipped
        methods[name] = m
        # single-signal score 도 predictions 에 write (bootstrap CI 비교 용)
        for pid, si, score in zip(df["prompt_id"].values, df["sample_index"].values, s_norm):
            pred_rows.append({
                "method": name,
                "prompt_id": str(pid),
                "sample_index": int(si),
                "score": float(score),
            })

    # --- fusion ---
    def feat(use_corpus: bool) -> np.ndarray:
        cols = list(CORE_INPUTS)
        if use_corpus:
            for col in CORPUS_INPUTS:
                if col in df.columns:
                    cols.append(col)
        return df[cols].fillna(0).astype(float).values

    fusion_specs = [
        ("logistic regression (no corpus)", lambda: LogisticRegression(max_iter=5000), False),
        ("logistic regression (with corpus)", lambda: LogisticRegression(max_iter=5000), True),
        ("random forest (no corpus)",
         lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
        ("random forest (with corpus)",
         lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), True),
        ("gradient boosting (no corpus)",
         lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42), False),
        ("gradient boosting (with corpus)",
         lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42), True),
    ]
    for name, factory, with_corpus in fusion_specs:
        X = feat(with_corpus)
        oof = _fit_predict_oof_grouped(X, y, groups, factory)
        m = _aurocs(y, oof)
        m["flipped"] = False
        methods[name] = m
        for pid, si, score in zip(df["prompt_id"].values, df["sample_index"].values, oof):
            pred_rows.append({
                "method": name,
                "prompt_id": str(pid),
                "sample_index": int(si),
                "score": float(score),
            })

    # per-dataset row counts
    by_ds = {ds: {
        "row_count": int((df["dataset"] == ds).sum()),
        "is_correct_rate": float(df.loc[df["dataset"] == ds, "is_correct"].mean()),
    } for ds in df["dataset"].unique()}

    summary = {
        "row_count": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "is_correct_rate": float(y.mean()),
        "methods": methods,
        "by_dataset": by_ds,
    }
    return summary, pd.DataFrame(pred_rows)


# ---------------- robustness ----------------

def _scores_by_method(preds: pd.DataFrame, methods: list[str]) -> dict[str, pd.Series]:
    """Index by (prompt_id, sample_index) for fast lookup."""
    out = {}
    for m in methods:
        sub = preds[preds["method"] == m]
        out[m] = sub.set_index(["prompt_id", "sample_index"])["score"]
    return out


def bootstrap_ci_per_decile(
    df: pd.DataFrame, preds: pd.DataFrame, ref: str, candidates: list[str],
    *, bin_field: str = "corpus_axis_bin_10", n_boot: int = 1000, seed: int = 42,
) -> dict:
    """prompt-grouped bootstrap (같은 prompt 의 모든 sample 함께 resample)."""
    rng = np.random.default_rng(seed)
    score_by = _scores_by_method(preds, candidates + [ref])
    out = {}
    df_idx = df.set_index(["prompt_id", "sample_index"])
    for b in sorted(df[bin_field].dropna().unique()):
        sub = df[df[bin_field] == b]
        prompts = sub["prompt_id"].unique()
        keys = list(zip(sub["prompt_id"].values, sub["sample_index"].values))
        ya = sub["is_correct"].values.astype(int)
        if len(set(ya)) < 2 or len(keys) < 30:
            out[b] = None
            continue
        sr = score_by[ref].loc[keys].values
        per_cand = {}
        for c in candidates:
            sc = score_by[c].loc[keys].values
            obs = roc_auc_score(ya, sc) - roc_auc_score(ya, sr)
            # prompt-grouped resample
            prompt_to_indices = defaultdict(list)
            for i, pid in enumerate(sub["prompt_id"].values):
                prompt_to_indices[pid].append(i)
            deltas = []
            for _ in range(n_boot):
                sampled_prompts = rng.choice(prompts, size=len(prompts), replace=True)
                idx = []
                for pid in sampled_prompts:
                    idx.extend(prompt_to_indices[pid])
                idx_arr = np.array(idx)
                yb = ya[idx_arr]
                if len(set(yb)) < 2:
                    continue
                deltas.append(roc_auc_score(yb, sc[idx_arr]) - roc_auc_score(yb, sr[idx_arr]))
            deltas_arr = np.array(deltas)
            per_cand[c] = {
                "observed_delta": float(obs),
                "ci_95_lower": float(np.percentile(deltas_arr, 2.5)),
                "ci_95_upper": float(np.percentile(deltas_arr, 97.5)),
                "bootstrap_n": int(len(deltas_arr)),
            }
        out[b] = per_cand
    return out


def corpus_bin_reliability(
    df: pd.DataFrame, preds: pd.DataFrame, methods: list[str],
    *, bin_field: str,
) -> dict:
    out = {}
    score_by = _scores_by_method(preds, methods)
    for m in methods:
        bins = []
        for b in sorted(df[bin_field].dropna().unique()):
            sub = df[df[bin_field] == b]
            keys = list(zip(sub["prompt_id"].values, sub["sample_index"].values))
            y = sub["is_correct"].values.astype(int)
            if len(keys) < 30 or len(set(y)) < 2:
                bins.append({"bin": b, "row_count": int(len(keys)), "auroc": None})
                continue
            score = score_by[m].loc[keys].values
            bins.append({
                "bin": b,
                "row_count": int(len(keys)),
                "is_correct_rate": float(y.mean()),
                "auroc": float(roc_auc_score(y, score)),
                "aurac": float(compute_aurac(y, score)),
            })
        out[m] = {"bins": bins}
    return out


def per_dataset_breakdown(df: pd.DataFrame, preds: pd.DataFrame, methods: list[str]) -> dict:
    out = {"per_dataset": {}, "aggregate": {}}
    score_by = _scores_by_method(preds, methods)
    all_keys = list(zip(df["prompt_id"].values, df["sample_index"].values))
    y_all = df["is_correct"].values.astype(int)
    for m in methods:
        s = score_by[m].loc[all_keys].values
        out["aggregate"][m] = {
            "auroc": float(roc_auc_score(y_all, s)),
            "aurac": float(compute_aurac(y_all, s)),
        }
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        keys = list(zip(sub["prompt_id"].values, sub["sample_index"].values))
        ya = sub["is_correct"].values.astype(int)
        ds_out = {}
        for m in methods:
            sc = score_by[m].loc[keys].values
            if len(set(ya)) < 2:
                ds_out[m] = {"auroc": None, "aurac": None}
            else:
                ds_out[m] = {
                    "auroc": float(roc_auc_score(ya, sc)),
                    "aurac": float(compute_aurac(ya, sc)),
                }
        out["per_dataset"][ds] = ds_out
    return out


def calibration(df: pd.DataFrame, preds: pd.DataFrame, methods: list[str]) -> dict:
    score_by = _scores_by_method(preds, methods)
    keys = list(zip(df["prompt_id"].values, df["sample_index"].values))
    y = df["is_correct"].values.astype(int)
    out = {}
    for m in methods:
        s = score_by[m].loc[keys].values
        s_clip = np.clip(s, 1e-6, 1 - 1e-6)
        out[m] = {
            "brier": float(brier_score_loss(y, s_clip)),
            "ece": _ece(y, s),
        }
    return out
