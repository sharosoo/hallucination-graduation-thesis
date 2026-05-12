"""Compute paper-faithful Semantic Energy from free-samples + SE only.

기존 compute_energy_features.py 가 candidate_scores 필수 (candidate-level
diagnostic 산출 위해) 인데, SE 5-dataset 트랙은 single-candidate 라
candidate_scores 가 없다. 본 논문 generation-level fusion 입력에는 paper-
faithful Energy 만 필요하므로 mini script.

Inputs:
  --free-samples : free_sample_rows.json (consolidated)
  --semantic-entropy : semantic_entropy_features.parquet
Outputs:
  --out : energy_features.parquet (per-prompt paper-faithful Semantic Energy)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-samples", required=True)
    ap.add_argument("--semantic-entropy", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[1/3] loading free samples", flush=True)
    fs_data = json.loads(Path(args.free_samples).read_text())
    samples = fs_data["samples"]
    print(f"  {len(samples)} samples")

    print(f"[2/3] loading semantic_entropy clusters", flush=True)
    se_df = pd.read_parquet(args.semantic_entropy)
    print(f"  {len(se_df)} prompt rows; cols={[c for c in se_df.columns if 'cluster' in c.lower() or 'sample' in c.lower()][:8]}")

    # SE row 의 cluster_likelihoods → cluster_id, member_sample_indexes, cluster_log_probability
    # 또한 sample_log_likelihoods → sample_index → sample_log_likelihood
    # sample_index 가 어느 cluster 에 속하는지 inverse map 필요
    se_clusters_by_pid = {}
    se_sample_loglik_by_pid = {}
    for _, row in se_df.iterrows():
        pid = row["prompt_id"]
        clusters = row.get("cluster_log_likelihoods")
        if clusters is None:
            clusters = row.get("semantic_clusters")
        if hasattr(clusters, "tolist"): clusters = clusters.tolist()
        clusters = list(clusters) if clusters is not None else []
        sample_logliks = row.get("sample_log_likelihoods")
        if hasattr(sample_logliks, "tolist"): sample_logliks = sample_logliks.tolist()
        sample_logliks = list(sample_logliks) if sample_logliks is not None else []
        se_clusters_by_pid[pid] = clusters
        se_sample_loglik_by_pid[pid] = sample_logliks

    # samples 를 prompt_id 별 그룹화
    samples_by_pid = defaultdict(list)
    for s in samples:
        samples_by_pid[s["prompt_id"]].append(s)
    for pid in samples_by_pid:
        samples_by_pid[pid].sort(key=lambda s: int(s["sample_index"]))

    print(f"[3/3] computing paper-faithful Semantic Energy per prompt", flush=True)
    rows = []
    for pid, prompt_samples in samples_by_pid.items():
        # per-sample energy = mean(-selected_token_logits)
        sample_energies = {}
        for s in prompt_samples:
            sl = np.asarray(s.get("selected_token_logits") or [], dtype=np.float64)
            if sl.size == 0:
                sample_energies[int(s["sample_index"])] = 0.0
                continue
            sample_energies[int(s["sample_index"])] = float(np.mean(-sl))

        # cluster info
        clusters = se_clusters_by_pid.get(pid, [])
        if not clusters:
            # fallback: 단일 cluster 가정
            mean_e = float(np.mean(list(sample_energies.values()))) if sample_energies else 0.0
            rows.append({
                "prompt_id": pid,
                "dataset": prompt_samples[0]["dataset"],
                "semantic_energy_cluster_uncertainty": mean_e,
                "semantic_energy_sample_energy": mean_e,
                "semantic_energy_boltzmann": mean_e,
                "n_clusters": 1,
                "n_samples": len(sample_energies),
            })
            continue

        # cluster_uncertainty = sum_k p(C_k) * E_Bolt(C_k)
        # E_Bolt(C_k) = sum over members of sample_energy
        # p(C_k) = exp(cluster_log_probability)
        total_uncertainty = 0.0
        boltzmann_total = 0.0
        for c in clusters:
            members = c.get("member_sample_indexes")
            if hasattr(members, "tolist"): members = members.tolist()
            members = list(members) if members is not None else []
            if len(members) == 0: continue
            cluster_energies = [sample_energies.get(int(m), 0.0) for m in members]
            e_bolt = float(np.sum(cluster_energies))
            p_c = float(np.exp(c.get("cluster_log_probability", -1e9)))
            total_uncertainty += p_c * e_bolt
            boltzmann_total += e_bolt
        mean_e = float(np.mean(list(sample_energies.values()))) if sample_energies else 0.0
        rows.append({
            "prompt_id": pid,
            "dataset": prompt_samples[0]["dataset"],
            "semantic_energy_cluster_uncertainty": total_uncertainty,
            "semantic_energy_sample_energy": mean_e,
            "semantic_energy_boltzmann": boltzmann_total,
            "n_clusters": len(clusters),
            "n_samples": len(sample_energies),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"  saved {args.out}  rows={len(out_df)}")
    print(out_df.describe().to_string())


if __name__ == "__main__":
    main()
