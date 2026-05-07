#!/usr/bin/env python3
"""Create or execute a reproducible paired experiment pipeline run manifest."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.stage_control import (
    CORPUS_AXIS_SCHEMA_VERSION,
    DATASET_PREPARATION_SCHEMA_VERSION,
    FEATURE_TABLE_SCHEMA_VERSION,
    GENERATION_CANDIDATE_SCORE_SCHEMA_VERSION,
    GENERATION_FREE_SAMPLE_SCHEMA_VERSION,
    SEMANTIC_ENERGY_SCHEMA_VERSION,
    SEMANTIC_ENTROPY_SCHEMA_VERSION,
    write_json_atomic,
    write_text_atomic,
)


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    name: str
    command: tuple[str, ...]
    expected_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    thesis_gate: bool
    artifact_keys: tuple[str, ...] = ()
    validation_commands: tuple[tuple[str, ...], ...] = ()
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


def path_string(path: Path) -> str:
    return str(path)


def command_string(command_parts: tuple[str, ...] | list[str]) -> str:
    return shlex.join(list(command_parts))


def planned_log_paths(stage: PipelineStage, logs_dir: Path) -> dict[str, object]:
    return {
        "primary": {
            "stdout_log": str(logs_dir / f"{stage.stage_id}.stdout.log"),
            "stderr_log": str(logs_dir / f"{stage.stage_id}.stderr.log"),
        },
        "validations": [
            {
                "stdout_log": str(logs_dir / f"{stage.stage_id}.validation{index}.stdout.log"),
                "stderr_log": str(logs_dir / f"{stage.stage_id}.validation{index}.stderr.log"),
            }
            for index, _ in enumerate(stage.validation_commands, start=1)
        ],
    }


def artifact_paths(run_dir: Path) -> dict[str, str]:
    results = run_dir / "results"
    datasets = results / "datasets"
    generation = results / "generation"
    correctness = results / "correctness"
    feature_table = results / "features.parquet"
    fusion = results / "fusion"
    robustness = results / "robustness"

    return {
        "results_dir": path_string(results),
        "datasets_dir": path_string(datasets),
        "prompt_groups": path_string(datasets / "prompt_groups.jsonl"),
        "candidate_rows": path_string(datasets / "candidate_rows.jsonl"),
        "dataset_manifest": path_string(datasets / "dataset_manifest.json"),
        "dataset_preparation_report": path_string(datasets / "dataset_preparation_report.json"),
        "free_samples": path_string(generation / "free_sample_rows.json"),
        "candidate_scores": path_string(generation / "candidate_scores.json"),
        "candidate_label_rows": path_string(correctness / "data" / "correctness_judgments.jsonl"),
        "correctness_manifest": path_string(correctness / "dataset_manifest.json"),
        "correctness_readme": path_string(correctness / "README.md"),
        "semantic_entropy_rows": path_string(results / "semantic_entropy_features.parquet"),
        "corpus_rows": path_string(results / "corpus_features.parquet"),
        "energy_rows": path_string(results / "energy_features.parquet"),
        "feature_table": path_string(feature_table),
        "feature_table_report": path_string(feature_table.with_suffix(feature_table.suffix + ".report.json")),
        "type_label_validation_report": path_string(results / "type_label_validation_report.json"),
        "fusion_summary": path_string(fusion / "summary.json"),
        "fusion_baseline_metrics": path_string(fusion / "baseline_metrics.csv"),
        "fusion_feature_importance_csv": path_string(fusion / "feature_importance.csv"),
        "fusion_feature_importance_json": path_string(fusion / "feature_importance.json"),
        "fusion_learned_comparison": path_string(fusion / "learned_fusion_comparison.json"),
        "fusion_predictions": path_string(fusion / "predictions.jsonl"),
        "fusion_report": path_string(fusion / "report.md"),
        "robustness_summary": path_string(robustness / "summary.json"),
        "robustness_bootstrap_json": path_string(robustness / "bootstrap_ci.json"),
        "robustness_bootstrap_csv": path_string(robustness / "bootstrap_ci.csv"),
        "robustness_lodo": path_string(robustness / "leave_one_dataset_out.json"),
        "robustness_threshold_sensitivity": path_string(robustness / "threshold_sensitivity.json"),
        "robustness_within_dataset": path_string(robustness / "within_dataset_checks.json"),
        "robustness_selective_risk": path_string(robustness / "selective_risk.json"),
        "robustness_report": path_string(robustness / "report.md"),
    }


def artifact_schema_versions() -> dict[str, str]:
    return {
        "prompt_groups": DATASET_PREPARATION_SCHEMA_VERSION,
        "candidate_rows": DATASET_PREPARATION_SCHEMA_VERSION,
        "dataset_manifest": DATASET_PREPARATION_SCHEMA_VERSION,
        "dataset_preparation_report": DATASET_PREPARATION_SCHEMA_VERSION,
        "free_samples": GENERATION_FREE_SAMPLE_SCHEMA_VERSION,
        "candidate_scores": GENERATION_CANDIDATE_SCORE_SCHEMA_VERSION,
        "semantic_entropy_rows": SEMANTIC_ENTROPY_SCHEMA_VERSION,
        "corpus_rows": CORPUS_AXIS_SCHEMA_VERSION,
        "energy_rows": SEMANTIC_ENERGY_SCHEMA_VERSION,
        "feature_table": FEATURE_TABLE_SCHEMA_VERSION,
        "feature_table_report": FEATURE_TABLE_SCHEMA_VERSION,
        "type_label_validation_report": FEATURE_TABLE_SCHEMA_VERSION,
    }


def build_stages(run_dir: Path) -> tuple[PipelineStage, ...]:
    artifacts = artifact_paths(run_dir)
    results = artifacts["results_dir"]
    datasets = artifacts["datasets_dir"]
    correctness = str(Path(artifacts["candidate_label_rows"]).parents[1])
    fusion = str(Path(artifacts["fusion_summary"]).parent)
    robustness = str(Path(artifacts["robustness_summary"]).parent)
    logs = run_dir / "logs"

    return (
        PipelineStage(
            stage_id="S0",
            name="contract validation",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/validate_pipeline_contract.py",
                "experiments/PIPELINE.md",
            ),
            expected_inputs=(
                "experiments/PIPELINE.md",
                "experiments/configs/formulas.yaml",
                "experiments/literature/formula_notes.md",
            ),
            expected_outputs=(),
            thesis_gate=True,
            validation_commands=(
                command(
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
            ),
        ),
        PipelineStage(
            stage_id="S1",
            name="dataset preparation",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/prepare_datasets.py",
                "--config",
                "experiments/configs/datasets.yaml",
                "--out",
                datasets,
                "--progress",
                path_string(logs / "S1.dataset_preparation.progress.json"),
            ),
            expected_inputs=("experiments/configs/datasets.yaml",),
            expected_outputs=(
                artifacts["prompt_groups"],
                artifacts["candidate_rows"],
                artifacts["dataset_manifest"],
                artifacts["dataset_preparation_report"],
            ),
            thesis_gate=True,
            artifact_keys=("prompt_groups", "candidate_rows", "dataset_manifest", "dataset_preparation_report"),
        ),
        PipelineStage(
            stage_id="S2",
            name="generation",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/run_generation.py",
                "--config",
                "experiments/configs/generation.yaml",
                "--prompt-groups",
                artifacts["prompt_groups"],
                "--candidates",
                artifacts["candidate_rows"],
                "--out-free-samples",
                artifacts["free_samples"],
                "--out-candidate-scores",
                artifacts["candidate_scores"],
                "--progress",
                path_string(logs / "S2.generation.progress.json"),
                "--resume",
            ),
            expected_inputs=("experiments/configs/generation.yaml", artifacts["prompt_groups"], artifacts["candidate_rows"]),
            expected_outputs=(artifacts["free_samples"], artifacts["candidate_scores"]),
            thesis_gate=True,
            artifact_keys=("free_samples", "candidate_scores"),
            validation_commands=(
                command("uv", "run", "python", "experiments/scripts/validate_generation_logits.py", artifacts["free_samples"]),
                command("uv", "run", "python", "experiments/scripts/validate_generation_logits.py", artifacts["candidate_scores"]),
            ),
        ),
        PipelineStage(
            stage_id="S3",
            name="correctness labels",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/build_correctness_dataset.py",
                "--candidates",
                artifacts["candidate_rows"],
                "--out",
                correctness,
                "--progress",
                path_string(logs / "S3.correctness.progress.json"),
            ),
            expected_inputs=(artifacts["candidate_rows"],),
            expected_outputs=(
                artifacts["candidate_label_rows"],
                artifacts["correctness_manifest"],
                artifacts["correctness_readme"],
            ),
            thesis_gate=True,
            artifact_keys=("candidate_label_rows", "correctness_manifest", "correctness_readme"),
        ),
        PipelineStage(
            stage_id="S4",
            name="Semantic Entropy",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/compute_semantic_entropy.py",
                "--free-samples",
                artifacts["free_samples"],
                "--out",
                artifacts["semantic_entropy_rows"],
                "--progress",
                path_string(logs / "S4.semantic_entropy.progress.json"),
                "--resume",
            ),
            expected_inputs=(artifacts["free_samples"],),
            expected_outputs=(artifacts["semantic_entropy_rows"],),
            thesis_gate=True,
            artifact_keys=("semantic_entropy_rows",),
        ),
        PipelineStage(
            stage_id="S5",
            name="corpus axis",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/compute_corpus_features.py",
                "--candidates",
                artifacts["candidate_rows"],
                "--out",
                artifacts["corpus_rows"],
                "--progress",
                path_string(logs / "S5.corpus.progress.json"),
                "--resume",
            ),
            expected_inputs=(artifacts["candidate_rows"],),
            expected_outputs=(artifacts["corpus_rows"],),
            thesis_gate=True,
            artifact_keys=("corpus_rows",),
            validation_commands=(
                command("uv", "run", "python", "experiments/scripts/validate_feature_provenance.py", artifacts["corpus_rows"]),
            ),
        ),
        PipelineStage(
            stage_id="S6",
            name="Semantic Energy",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/compute_energy_features.py",
                "--candidate-scores",
                artifacts["candidate_scores"],
                "--free-samples",
                artifacts["free_samples"],
                "--semantic-entropy",
                artifacts["semantic_entropy_rows"],
                "--out",
                artifacts["energy_rows"],
                "--progress",
                path_string(logs / "S6.energy.progress.json"),
                "--resume",
            ),
            expected_inputs=(artifacts["candidate_scores"], artifacts["free_samples"], artifacts["semantic_entropy_rows"]),
            expected_outputs=(artifacts["energy_rows"],),
            thesis_gate=True,
            artifact_keys=("energy_rows",),
            validation_commands=(
                command("uv", "run", "python", "experiments/scripts/validate_energy_features.py", artifacts["energy_rows"]),
            ),
        ),
        PipelineStage(
            stage_id="S7",
            name="feature table",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/build_feature_table.py",
                "--inputs",
                results,
                "--out",
                artifacts["feature_table"],
                "--progress",
                path_string(logs / "S7.feature_table.progress.json"),
                "--resume",
            ),
            expected_inputs=(
                artifacts["candidate_label_rows"],
                artifacts["semantic_entropy_rows"],
                artifacts["corpus_rows"],
                artifacts["energy_rows"],
            ),
            expected_outputs=(
                artifacts["feature_table"],
                artifacts["feature_table_report"],
                artifacts["type_label_validation_report"],
            ),
            thesis_gate=True,
            artifact_keys=("feature_table", "feature_table_report", "type_label_validation_report"),
            validation_commands=(
                command("uv", "run", "python", "experiments/scripts/validate_type_labels.py", artifacts["feature_table"]),
            ),
        ),
        PipelineStage(
            stage_id="S8",
            name="fusion",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/run_fusion.py",
                "--features",
                artifacts["feature_table"],
                "--config",
                "experiments/configs/fusion.yaml",
                "--out",
                fusion,
                "--progress",
                path_string(logs / "S8.fusion.progress.json"),
            ),
            expected_inputs=(artifacts["feature_table"], "experiments/configs/fusion.yaml"),
            expected_outputs=(
                artifacts["fusion_summary"],
                artifacts["fusion_baseline_metrics"],
                artifacts["fusion_feature_importance_csv"],
                artifacts["fusion_feature_importance_json"],
                artifacts["fusion_learned_comparison"],
                artifacts["fusion_predictions"],
                artifacts["fusion_report"],
            ),
            thesis_gate=True,
            artifact_keys=(
                "fusion_summary",
                "fusion_baseline_metrics",
                "fusion_feature_importance_csv",
                "fusion_feature_importance_json",
                "fusion_learned_comparison",
                "fusion_predictions",
                "fusion_report",
            ),
        ),
        PipelineStage(
            stage_id="S9",
            name="robustness",
            command=command(
                "uv",
                "run",
                "python",
                "experiments/scripts/run_robustness.py",
                "--features",
                artifacts["feature_table"],
                "--fusion",
                fusion,
                "--out",
                robustness,
                "--progress",
                path_string(logs / "S9.robustness.progress.json"),
            ),
            expected_inputs=(
                artifacts["feature_table"],
                artifacts["fusion_summary"],
                artifacts["fusion_baseline_metrics"],
                artifacts["fusion_predictions"],
            ),
            expected_outputs=(
                artifacts["robustness_summary"],
                artifacts["robustness_bootstrap_json"],
                artifacts["robustness_bootstrap_csv"],
                artifacts["robustness_lodo"],
                artifacts["robustness_threshold_sensitivity"],
                artifacts["robustness_within_dataset"],
                artifacts["robustness_selective_risk"],
                artifacts["robustness_report"],
            ),
            thesis_gate=True,
            artifact_keys=(
                "robustness_summary",
                "robustness_bootstrap_json",
                "robustness_bootstrap_csv",
                "robustness_lodo",
                "robustness_threshold_sensitivity",
                "robustness_within_dataset",
                "robustness_selective_risk",
                "robustness_report",
            ),
            validation_commands=(
                command("uv", "run", "python", "experiments/scripts/validate_report_claims.py", artifacts["robustness_summary"]),
            ),
        ),
    )


def execute_command(command_parts: tuple[str, ...], stdout_path: Path, stderr_path: Path) -> dict[str, object]:
    completed = subprocess.run(command_parts, cwd=ROOT, capture_output=True, text=True, check=False)
    write_text_atomic(stdout_path, completed.stdout)
    write_text_atomic(stderr_path, completed.stderr)
    return {
        "returncode": completed.returncode,
        "command": list(command_parts),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def execute_stage(stage: PipelineStage, logs_dir: Path) -> dict[str, object]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not stage.implemented:
        return {
            "stage_id": stage.stage_id,
            "status": "skipped_not_implemented",
            "skip_reason": stage.skip_reason,
            "command": list(stage.command),
        }

    primary = execute_command(stage.command, logs_dir / f"{stage.stage_id}.stdout.log", logs_dir / f"{stage.stage_id}.stderr.log")
    if primary["returncode"] != 0:
        return {"stage_id": stage.stage_id, "status": "failed", "primary": primary}

    validations: list[dict[str, object]] = []
    for index, validation_command in enumerate(stage.validation_commands, start=1):
        validation = execute_command(
            validation_command,
            logs_dir / f"{stage.stage_id}.validation{index}.stdout.log",
            logs_dir / f"{stage.stage_id}.validation{index}.stderr.log",
        )
        validations.append(validation)
        if validation["returncode"] != 0:
            return {"stage_id": stage.stage_id, "status": "failed", "primary": primary, "validations": validations}

    return {"stage_id": stage.stage_id, "status": "passed", "primary": primary, "validations": validations}


def build_script_execution_log(stage_records: list[dict[str, object]]) -> list[dict[str, object]]:
    log_rows: list[dict[str, object]] = []
    for record in stage_records:
        command_value = record.get("command")
        command = command_value if isinstance(command_value, list) else []
        command_string_value = record.get("command_string")
        command_string_text = command_string_value if isinstance(command_string_value, str) else command_string(command)
        validation_specs_value = record.get("validation_command_specs")
        validation_specs = validation_specs_value if isinstance(validation_specs_value, list) else []
        planned_logs_value = record.get("planned_logs")
        planned_logs = planned_logs_value if isinstance(planned_logs_value, dict) else {}
        primary_planned_value = planned_logs.get("primary")
        primary_planned = primary_planned_value if isinstance(primary_planned_value, dict) else {}
        row: dict[str, object] = {
            "stage_id": record.get("stage_id"),
            "name": record.get("name"),
            "primary_command": command,
            "primary_command_string": command_string_text,
            "validation_commands": [
                spec.get("command") for spec in validation_specs if isinstance(spec, dict)
            ],
            "artifact_keys": record.get("artifact_keys"),
            "expected_inputs": record.get("expected_inputs"),
            "expected_outputs": record.get("expected_outputs"),
            "validations": validation_specs,
            "validators": validation_specs,
            "status": "not_executed",
            "stdout_log": primary_planned.get("stdout_log"),
            "stderr_log": primary_planned.get("stderr_log"),
            "returncode": None,
        }
        execution = record.get("execution")
        if isinstance(execution, dict):
            row["status"] = execution.get("status")
            primary = execution.get("primary")
            if isinstance(primary, dict):
                row["returncode"] = primary.get("returncode")
                row["stdout_log"] = primary.get("stdout_log")
                row["stderr_log"] = primary.get("stderr_log")
            executed_validations = execution.get("validations")
            if isinstance(executed_validations, list):
                merged_validations: list[dict[str, object]] = []
                for spec, executed_validation in zip(validation_specs, executed_validations):
                    if not isinstance(spec, dict):
                        continue
                    merged = dict(spec)
                    if isinstance(executed_validation, dict):
                        merged["returncode"] = executed_validation.get("returncode")
                        merged["stdout_log"] = executed_validation.get("stdout_log")
                        merged["stderr_log"] = executed_validation.get("stderr_log")
                    merged_validations.append(merged)
                row["validations"] = merged_validations
        log_rows.append(row)
    return log_rows


def build_manifest(out_root: Path, *, execute: bool) -> dict[str, object]:
    run_id = f"pipeline-{utc_stamp()}"
    run_dir = out_root / run_id
    logs_dir = run_dir / "logs"
    stages = build_stages(run_dir)
    artifacts = artifact_paths(run_dir)
    schemas = artifact_schema_versions()
    stage_records: list[dict[str, object]] = []
    failed_stage_id: str | None = None

    for stage in stages:
        record = asdict(stage)
        record["command"] = list(stage.command)
        record["command_string"] = command_string(stage.command)
        record["validation_commands"] = [list(validation) for validation in stage.validation_commands]
        record["validators"] = record["validation_commands"]
        planned_logs = planned_log_paths(stage, logs_dir)
        primary_logs = cast(dict[str, str], planned_logs["primary"])
        validation_logs = cast(list[dict[str, str]], planned_logs["validations"])
        record["planned_logs"] = planned_logs
        record["planned_stdout_log"] = primary_logs["stdout_log"]
        record["planned_stderr_log"] = primary_logs["stderr_log"]
        record["validation_command_specs"] = [
            {
                "index": index,
                "command": list(validation),
                "command_string": command_string(validation),
                "stdout_log": validation_log["stdout_log"],
                "stderr_log": validation_log["stderr_log"],
                "returncode": None,
            }
            for index, (validation, validation_log) in enumerate(
                zip(stage.validation_commands, validation_logs),
                start=1,
            )
        ]
        record["validator_specs"] = record["validation_command_specs"]
        record["thesis_facing_artifacts"] = [
            {
                "artifact_key": artifact_key,
                "path": artifacts[artifact_key],
                "schema_version": schemas.get(artifact_key),
                "expected_inputs": list(stage.expected_inputs),
                "expected_outputs": list(stage.expected_outputs),
                "primary_command": list(stage.command),
                "primary_command_string": record["command_string"],
                "validators": record["validation_command_specs"],
                "stdout_log": primary_logs["stdout_log"],
                "stderr_log": primary_logs["stderr_log"],
            }
            for artifact_key in stage.artifact_keys
            if artifact_key in artifacts
        ]
        if execute:
            execution = execute_stage(stage, logs_dir)
            record["execution"] = execution
            if execution.get("status") == "failed":
                failed_stage_id = stage.stage_id
                stage_records.append(record)
                break
        stage_records.append(record)

    status = "dry_run"
    if execute:
        status = "failed" if failed_stage_id else "passed"

    manifest = {
        "run_id": run_id,
        "dataset": "paired_discriminative_experiment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "git_commit": git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "execute": execute,
        "status": status,
        "failed_stage_id": failed_stage_id,
        "thesis_valid_candidate": execute and failed_stage_id is None and all(stage.implemented for stage in stages),
        "thesis_validity_note": "Every stage and validation command must pass before this run can support thesis claims.",
        "artifact_schema_versions": schemas,
        "expected_artifacts": artifacts,
        "stages": stage_records,
        "script_execution_log": build_script_execution_log(stage_records),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    write_json_atomic(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/results/runs", help="Directory for run manifests and logs")
    parser.add_argument("--execute", action="store_true", help="Execute real pipeline commands and capture stdout/stderr logs")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest without executing commands")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execute and args.dry_run:
        print("Choose only one of --execute or --dry-run.", file=sys.stderr)
        return 2
    manifest = build_manifest(Path(args.out), execute=bool(args.execute))
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "manifest_path": manifest["manifest_path"],
                "execute": manifest["execute"],
                "status": manifest["status"],
                "failed_stage_id": manifest["failed_stage_id"],
            },
            indent=2,
        )
    )
    return 1 if manifest["execute"] and manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
