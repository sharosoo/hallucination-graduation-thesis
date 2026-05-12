"""Prepare prompt_groups + candidate_rows for the SE 5-dataset (single-candidate) track.

Reads experiments/configs/datasets_se.yaml (JSON) and emits
  $RUN/results/datasets/prompt_groups.jsonl
  $RUN/results/datasets/candidate_rows.jsonl
  $RUN/results/datasets/dataset_manifest.json

Single-candidate (right answer only). No paired hallucinated candidate.
Generation-level evaluation by post-hoc NLI matching of free-sample answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.adapters.hf_datasets_single_candidate import (
    materialize_se_dataset,
    record_to_candidate_row,
    record_to_prompt_group,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/datasets_se.yaml")
    ap.add_argument("--out-dir", required=True, help="Run dir results/datasets path")
    args = ap.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config_path.read_text())
    if cfg.get("registry_name") != "single_candidate_short_answer_experiment_datasets":
        raise SystemExit(f"unexpected registry_name in {config_path}")
    datasets = cfg["datasets"]

    pg_path = out_dir / "prompt_groups.jsonl"
    cr_path = out_dir / "candidate_rows.jsonl"
    manifest_path = out_dir / "dataset_manifest.json"

    print(f"[1/3] materializing {len(datasets)} datasets", flush=True)
    all_records = []
    per_dataset_counts = {}
    for ds_cfg in datasets:
        name = ds_cfg["name"]
        print(f"  - {name}: hf_id={ds_cfg['hf_id']}, target={ds_cfg['target_sample_count']}", flush=True)
        recs = materialize_se_dataset(
            dataset_name=name,
            hf_id=ds_cfg["hf_id"],
            config=ds_cfg.get("config"),
            split=ds_cfg["split"],
            split_id=ds_cfg["split_id"],
            target_sample_count=int(ds_cfg["target_sample_count"]),
            seed=int(ds_cfg["seed"]),
        )
        per_dataset_counts[name] = len(recs)
        all_records.extend(recs)
        print(f"    materialized {len(recs)} prompts", flush=True)

    print(f"[2/3] writing prompt_groups + candidate_rows", flush=True)
    with open(pg_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(record_to_prompt_group(rec), ensure_ascii=False) + "\n")
    with open(cr_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(record_to_candidate_row(rec), ensure_ascii=False) + "\n")
    print(f"  saved {pg_path} ({len(all_records)} prompts)")
    print(f"  saved {cr_path} ({len(all_records)} candidates)")

    print(f"[3/3] writing manifest", flush=True)
    manifest = {
        "registry_name": cfg["registry_name"],
        "dataset_contract": cfg["experiment_dataset_policy"]["dataset_contract"],
        "candidate_rows_per_prompt": cfg["experiment_dataset_policy"]["candidate_rows_per_prompt"],
        "n_total_prompts": len(all_records),
        "n_total_candidates": len(all_records),
        "per_dataset_prompt_count": per_dataset_counts,
        "datasets_config_path": str(config_path),
        "prompt_template": "se_sentence_length_v1",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  saved {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
