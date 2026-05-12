"""Question-entity-only axis: build entity_pair / entity_freq axes from
QUESTION text only (no gold answer entity, no free-sample entity).
Robustness check for review item 한계 8.

Pipeline:
  1. Read candidate_rows.jsonl -> question text per prompt
  2. spaCy NER on question text only (12 NER labels we keep)
  3. Infini-gram count for single entity freq + entity-pair co-occurrence
  4. Bin into rank-quantile deciles
  5. Compute Δ on SE / Energy / sample_nll
  6. Compare with reference Δ (question + gold answer)

Output: JSON to stdout
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import roc_auc_score


RUN = Path("/mnt/data/hallucination-graduation-thesis-runs/se-pipeline-20260511T034406Z")
KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT",
               "WORK_OF_ART", "FAC", "NORP", "PRODUCT", "LANGUAGE", "LAW"}


def normalize(t: str) -> str:
    return " ".join(t.lower().strip().split())


def extract_entities_spacy(nlp, text: str, max_entities: int = 8) -> list[str]:
    doc = nlp(text)
    ents = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ not in KEEP_LABELS:
            continue
        n = normalize(ent.text)
        if not n or n in seen:
            continue
        seen.add(n)
        ents.append(n)
        if len(ents) >= max_entities:
            break
    if not ents:
        # noun chunk fallback
        for chunk in doc.noun_chunks:
            n = normalize(chunk.text)
            if not n or n in seen:
                continue
            seen.add(n)
            ents.append(n)
            if len(ents) >= max_entities:
                break
    return ents


def main():
    sys.path.insert(0, "/home/global/workspaces/sharosoo/hallucination-graduation-thesis")
    from experiments.adapters.corpus_counts import build_corpus_count_backend

    print("[1/6] loading candidate_rows.jsonl ...", file=sys.stderr)
    cand_path = RUN / "results/datasets/candidate_rows.jsonl"
    rows = []
    with open(cand_path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "prompt_id": d["prompt_id"],
                "question": d["question"],
            })
    df_q = pd.DataFrame(rows).drop_duplicates("prompt_id").reset_index(drop=True)
    print(f"  {len(df_q)} unique prompts", file=sys.stderr)

    print("[2/6] spaCy NER on question text ...", file=sys.stderr)
    nlp = spacy.load("en_core_web_lg")
    q_entities = []
    t0 = time.time()
    for i, q in enumerate(df_q["question"]):
        ents = extract_entities_spacy(nlp, q)
        q_entities.append(ents)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(df_q)} elapsed={time.time()-t0:.0f}s", file=sys.stderr)
    df_q["q_entities"] = q_entities
    df_q["n_q_entities"] = df_q["q_entities"].map(len)
    print(f"  done. avg ents/Q = {df_q['n_q_entities'].mean():.2f}", file=sys.stderr)

    print("[3/6] building corpus backend ...", file=sys.stderr)
    backend = build_corpus_count_backend(cand_path)

    print("[4/6] computing entity_freq_min and entity_pair_cooc_mean (question-only) ...",
          file=sys.stderr)
    freq_min, pair_mean = [], []
    t0 = time.time()
    for i, ents in enumerate(df_q["q_entities"]):
        if not ents:
            freq_min.append(np.nan)
            pair_mean.append(np.nan)
            continue
        # single entity freq
        try:
            counts = []
            for e in ents:
                r = backend.count_entity(e)
                if r.raw_count is not None:
                    counts.append(r.raw_count)
            freq_min.append(float(min(counts)) if counts else np.nan)
        except Exception:
            freq_min.append(np.nan)
        # entity-pair co-occurrence
        if len(ents) < 2:
            pair_mean.append(np.nan)
        else:
            pair_counts = []
            for a in range(len(ents)):
                for b in range(a + 1, len(ents)):
                    try:
                        r = backend.count_pair(ents[a], ents[b])
                        if r.raw_count is not None:
                            pair_counts.append(float(r.raw_count))
                    except Exception:
                        pass
            pair_mean.append(float(np.mean(pair_counts)) if pair_counts else np.nan)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df_q)} elapsed={time.time()-t0:.0f}s", file=sys.stderr)
    df_q["q_entity_freq_min"] = freq_min
    df_q["q_entity_pair_mean"] = pair_mean

    # axis (rank-quantile decile) — only over rows with non-NaN
    print("[5/6] binning to deciles ...", file=sys.stderr)
    for src, dst in [("q_entity_freq_min", "q_entity_freq_axis_bin_10"),
                     ("q_entity_pair_mean", "q_entity_pair_axis_bin_10")]:
        valid = df_q[src].notna()
        df_q[dst] = pd.NA
        if valid.sum() < 20:
            print(f"  WARN: too few valid for {src} ({valid.sum()})", file=sys.stderr)
            continue
        ranks = df_q.loc[valid, src].rank(method="first")
        df_q.loc[valid, dst] = pd.qcut(
            ranks, q=10,
            labels=[f"decile_{i:02d}_{i+1:02d}" for i in range(0, 100, 10)],
            duplicates="drop",
        ).astype(str)
    print(f"  q_entity_freq valid: {df_q['q_entity_freq_axis_bin_10'].notna().sum()}",
          file=sys.stderr)
    print(f"  q_entity_pair valid: {df_q['q_entity_pair_axis_bin_10'].notna().sum()}",
          file=sys.stderr)

    print("[6/6] joining with generation features and computing Δ ...", file=sys.stderr)
    gf = pd.read_parquet(RUN / "qwen/results/generation_features.parquet")
    needed = ["prompt_id", "is_correct", "semantic_entropy",
              "semantic_energy_cluster_uncertainty", "sample_nll"]
    gf2 = gf[needed].merge(df_q[["prompt_id", "q_entity_freq_axis_bin_10",
                                  "q_entity_pair_axis_bin_10"]], on="prompt_id")

    def per_decile(df, axis, signal, flip=True):
        out = {}
        for v, sub in df.dropna(subset=[axis]).groupby(axis):
            if sub["is_correct"].nunique() < 2:
                continue
            score = -sub[signal].values if flip else sub[signal].values
            try:
                out[v] = roc_auc_score(sub["is_correct"], score)
            except Exception:
                pass
        return out

    def delta(d):
        if not d:
            return float("nan")
        return max(d.values()) - min(d.values())

    results = {}
    for signal, flip in [("semantic_entropy", True),
                         ("semantic_energy_cluster_uncertainty", True),
                         ("sample_nll", True)]:
        for axis, label in [("q_entity_freq_axis_bin_10", "q_entity_freq"),
                            ("q_entity_pair_axis_bin_10", "q_entity_pair")]:
            d = per_decile(gf2, axis, signal, flip=flip)
            sig_short = signal.split("_")[0] if signal != "sample_nll" else "sample_nll"
            sig_short = {"semantic": "SE",
                         "sample_nll": "sample_nll"}.get(sig_short, signal)
            if "energy" in signal:
                sig_short = "Energy"
            key = f"{sig_short}_{label}"
            results[key] = {
                "delta": delta(d),
                "min": min(d.values()) if d else None,
                "max": max(d.values()) if d else None,
                "n_deciles": len(d),
            }

    print(json.dumps({
        "n_prompts_total": int(len(df_q)),
        "n_q_freq_valid": int(df_q["q_entity_freq_axis_bin_10"].notna().sum()),
        "n_q_pair_valid": int(df_q["q_entity_pair_axis_bin_10"].notna().sum()),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
