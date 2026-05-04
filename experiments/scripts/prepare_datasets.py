#!/usr/bin/env python3
"""Prepare or dry-run the dataset registry without downloading full datasets by default."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.validate_datasets import load_json, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the dataset registry config")
    parser.add_argument("--out", required=True, help="Output directory for dataset preparation metadata")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report selected datasets without downloading large artifacts",
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def selected_datasets(config: dict) -> tuple[list[dict], list[dict]]:
    datasets = config.get("datasets", [])
    core = [dataset for dataset in datasets if dataset.get("role") == "core"]
    promoted = [
        dataset
        for dataset in datasets
        if isinstance(dataset.get("promotion"), dict) and dataset["promotion"].get("promoted_to_core") is True
    ]
    return core, promoted


def build_report(config_path: Path, config: dict, out_dir: Path, *, dry_run: bool) -> dict:
    core, promoted = selected_datasets(config)
    selected = [*core, *promoted]
    label_policy = config["label_policy"]

    return {
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "mode": "dry-run" if dry_run else "prepare-plan",
        "downloads_performed": False,
        "download_policy": config["dataset_expansion_policy"]["download_policy"],
        "mandatory_core_datasets": config["mandatory_core_datasets"],
        "selected_core_datasets": [dataset["name"] for dataset in selected],
        "promoted_optional_datasets": [dataset["name"] for dataset in promoted],
        "row_level_output_policy": label_policy["row_level_output"],
        "fixed_operational_thresholds": label_policy["fixed_operational_thresholds"],
        "analysis_se_bins": {
            "scheme_name": label_policy["analysis_se_bins"]["scheme_name"],
            "analysis_only": label_policy["analysis_se_bins"]["analysis_only"],
            "special_attention_thresholds": label_policy["analysis_se_bins"]["special_attention_thresholds"],
            "bin_count": len(label_policy["analysis_se_bins"]["bins"]),
        },
        "datasets": [
            {
                "name": dataset["name"],
                "role": dataset["role"],
                "split": dataset["split"],
                "split_id": dataset["split_id"],
                "target_sample_count": dataset["target_sample_count"],
                "seed": dataset["seed"],
                "label_source_type": dataset["label_source"]["type"],
                "label_policy": dataset["label_policy"],
                "notes": dataset["notes"],
                "promotion": dataset.get("promotion", {}),
            }
            for dataset in selected
        ],
    }


def print_report(report: dict) -> None:
    print("Dataset preparation report")
    print(f"- Mode: {report['mode']}")
    print(f"- Downloads performed: {report['downloads_performed']}")
    print(f"- Selected core datasets: {', '.join(report['selected_core_datasets'])}")
    if report["promoted_optional_datasets"]:
        print(f"- Promoted optional datasets: {', '.join(report['promoted_optional_datasets'])}")
    else:
        print("- Promoted optional datasets: none")
    print(f"- Row-level label field: {report['row_level_output_policy']['required_label_field']}")
    print(
        "- SE analysis bins: "
        f"{report['analysis_se_bins']['scheme_name']} "
        f"({report['analysis_se_bins']['bin_count']} bins; thresholds {report['analysis_se_bins']['special_attention_thresholds']})"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(config_path)
    problems = validate_config(config)
    if problems:
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit("Dataset registry validation failed before preparation.")

    report = build_report(config_path, config, out_dir, dry_run=args.dry_run)
    write_json(out_dir / "dataset_preparation_report.json", report)
    print_report(report)

    if not args.dry_run:
        print(
            "Preparation remains metadata-only in this task. "
            "Actual dataset downloading is intentionally deferred to a later implementation task."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
