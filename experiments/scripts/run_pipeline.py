#!/usr/bin/env python3
"""Create or execute a reproducible experiment pipeline run manifest."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[2]
Mode = Literal["smoke", "dev", "full-core", "full-extended"]


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    name: str
    command: tuple[str, ...]
    expected_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    thesis_gate: bool
    implemented: bool = True
    skip_reason: str | None = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def command(*parts: str) -> tuple[str, ...]:
    return parts


def build_stages(mode: Mode) -> tuple[PipelineStage, ...]:
    prompt_rows = f"experiments/results/datasets/prompt_rows.{mode}.jsonl"
    generation_out = f"experiments/results/generation/full_logits.{mode}.json"
    dataset_mode_supported = False
    direct_corpus_supported = False
    return (
        PipelineStage(
            stage_id="S0",
            name="contract validation",
            command=command(
                "uv", "run", "python", "experiments/scripts/validate_pipeline_contract.py", "experiments/PIPELINE.md"
            ),
            expected_inputs=("experiments/PIPELINE.md",),
            expected_outputs=(),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S0b",
            name="paper-feature alignment validation",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/validate_paper_feature_alignment.py",
                "--formulas",
                "experiments/configs/formulas.yaml",
                "--notes",
                "experiments/literature/formula_notes.md",
                "--pipeline",
                "experiments/PIPELINE.md",
            ),
            expected_inputs=("experiments/configs/formulas.yaml", "experiments/literature/formula_notes.md", "experiments/PIPELINE.md"),
            expected_outputs=(),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S1",
            name="dataset materialization",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/prepare_datasets.py",
                "--config",
                "experiments/configs/datasets.yaml",
                "--out",
                "experiments/results/datasets",
                "--mode",
                mode,
            ),
            expected_inputs=("experiments/configs/datasets.yaml",),
            expected_outputs=("experiments/results/datasets/dataset_preparation_report.json", prompt_rows),
            thesis_gate=True,
            implemented=dataset_mode_supported,
            skip_reason="prepare_datasets.py is currently metadata-only and has no --mode prompt materialization implementation",
        ),
        PipelineStage(
            stage_id="S2",
            name="full-logits generation",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/run_generation.py",
                "--config",
                "experiments/configs/generation.yaml",
                "--prompts",
                prompt_rows,
                "--out",
                generation_out,
            ),
            expected_inputs=("experiments/configs/generation.yaml", prompt_rows),
            expected_outputs=(generation_out,),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S2b",
            name="full-logits validation",
            command=command("uv", "run", "python", "experiments/scripts/validate_generation_logits.py", generation_out),
            expected_inputs=(generation_out,),
            expected_outputs=(),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S3",
            name="direct corpus features",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/compute_corpus_features.py",
                "--manifests",
                "experiments/manifests",
                "--out",
                "experiments/results/corpus_features.parquet",
                "--mode",
                "direct-corpus",
            ),
            expected_inputs=("experiments/manifests",),
            expected_outputs=("experiments/results/corpus_features.parquet",),
            thesis_gate=True,
            implemented=direct_corpus_supported,
            skip_reason="direct-corpus mode is required by PIPELINE.md but not implemented yet",
        ),
        PipelineStage(
            stage_id="S4",
            name="true Semantic Energy",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/compute_energy_features.py",
                "--manifests",
                "experiments/manifests",
                "--out",
                "experiments/results/energy_features.parquet",
                "--require-true-boltzmann",
            ),
            expected_inputs=("experiments/manifests", generation_out),
            expected_outputs=("experiments/results/energy_features.parquet",),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S5",
            name="feature table",
            command=command(
                "uv", "run", "python", "experiments/scripts/build_feature_table.py", "--inputs", "experiments/results", "--out", "experiments/results/features.parquet"
            ),
            expected_inputs=("experiments/results/corpus_features.parquet", "experiments/results/energy_features.parquet"),
            expected_outputs=("experiments/results/features.parquet",),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S6",
            name="type analysis",
            command=command(
                "uv", "run", "python", "experiments/scripts/run_type_analysis.py", "--features", "experiments/results/features.parquet", "--out", "experiments/results/type_analysis"
            ),
            expected_inputs=("experiments/results/features.parquet",),
            expected_outputs=("experiments/results/type_analysis/summary.json",),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S7",
            name="logistic fusion",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/run_fusion.py",
                "--features",
                "experiments/results/features.parquet",
                "--config",
                "experiments/configs/fusion.yaml",
                "--out",
                "experiments/results/fusion",
            ),
            expected_inputs=("experiments/results/features.parquet", "experiments/configs/fusion.yaml"),
            expected_outputs=("experiments/results/fusion/summary.json", "experiments/results/fusion/predictions.jsonl"),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S8",
            name="robustness",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/run_robustness.py",
                "--features",
                "experiments/results/features.parquet",
                "--fusion",
                "experiments/results/fusion",
                "--out",
                "experiments/results/robustness",
            ),
            expected_inputs=("experiments/results/features.parquet", "experiments/results/fusion/summary.json"),
            expected_outputs=("experiments/results/robustness/summary.json",),
            thesis_gate=True,
        ),
        PipelineStage(
            stage_id="S9",
            name="thesis evidence export",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/export_thesis_evidence.py",
                "--results",
                "experiments/results",
                "--out",
                "experiments/results/thesis_evidence_table.tex",
            ),
            expected_inputs=("experiments/results/type_analysis/summary.json", "experiments/results/fusion/summary.json", "experiments/results/robustness/summary.json"),
            expected_outputs=("experiments/results/thesis_evidence_table.tex", "experiments/results/thesis_evidence_summary.json"),
            thesis_gate=True,
        ),
    )


def execute_stage(stage: PipelineStage, logs_dir: Path) -> dict[str, object]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{stage.stage_id}.stdout.log"
    stderr_path = logs_dir / f"{stage.stage_id}.stderr.log"
    if not stage.implemented:
        return {
            "stage_id": stage.stage_id,
            "status": "skipped_not_implemented",
            "skip_reason": stage.skip_reason,
            "command": list(stage.command),
        }
    completed = subprocess.run(stage.command, cwd=ROOT, capture_output=True, text=True, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "stage_id": stage.stage_id,
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": list(stage.command),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def build_manifest(mode: Mode, out_root: Path, *, execute: bool) -> dict[str, object]:
    run_id = f"pipeline-{mode}-{utc_stamp()}"
    run_dir = out_root / run_id
    logs_dir = run_dir / "logs"
    stages = build_stages(mode)
    stage_records: list[dict[str, object]] = []
    for stage in stages:
        record = asdict(stage)
        record["command"] = list(stage.command)
        if execute:
            execution = execute_stage(stage, logs_dir)
            record["execution"] = execution
            if execution.get("status") == "failed":
                stage_records.append(record)
                break
        stage_records.append(record)
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "git_commit": git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "execute": execute,
        "thesis_valid_candidate": execute and all(stage.implemented for stage in stages),
        "thesis_validity_note": "All thesis gates must pass; skipped/not-implemented stages make this non-thesis-valid.",
        "stages": stage_records,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "dev", "full-core", "full-extended"), required=True)
    parser.add_argument("--out", default="experiments/results/runs", help="Directory for run manifests and logs")
    parser.add_argument("--execute", action="store_true", help="Execute implemented stages and capture stdout/stderr logs")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest without executing commands")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execute and args.dry_run:
        print("Choose only one of --execute or --dry-run.", file=sys.stderr)
        return 2
    manifest = build_manifest(args.mode, Path(args.out), execute=bool(args.execute))
    print(json.dumps({"run_id": manifest["run_id"], "manifest_path": manifest["manifest_path"], "execute": manifest["execute"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
