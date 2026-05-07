#!/usr/bin/env python3
"""Thin CLI for split free-sampling and teacher-forced generation artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters.model_generation import (
    GenerationConfigError,
    GenerationDependencyError,
    LocalModelGenerationAdapter,
    ModelLoadError,
    checkpoint_root_for_artifact,
    validate_generation_artifact,
)
from experiments.scripts.stage_control import progress_snapshot, write_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiments/configs/generation.yaml")
    parser.add_argument(
        "--prompt-groups",
        default=None,
        help="Prompt-group input file (.json or .jsonl) for prompt-level N=10 free sampling.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Candidate-row input file (.json or .jsonl) for teacher-forced candidate scoring.",
    )
    parser.add_argument(
        "--out-free-samples",
        default=None,
        help="Output JSON path for prompt-level free-sample rows.",
    )
    parser.add_argument(
        "--out-candidate-scores",
        default=None,
        help="Output JSON path for teacher-forced candidate-score rows.",
    )
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="Write deterministic fixture artifacts instead of calling a live model.",
    )
    parser.add_argument(
        "--fixture-variant",
        choices=("full_logits", "missing_full_logits"),
        default="full_logits",
        help="Fixture schema variant to write when --write-fixture is set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a phase when its final JSON artifact and full-logits sidecar already validate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing phase outputs before writing. Required to overwrite invalid partial artifacts.",
    )
    parser.add_argument("--progress", help="Optional progress JSON path updated atomically.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.prompt_groups and not args.candidates:
        raise GenerationConfigError("provide at least one of --prompt-groups or --candidates")
    if args.prompt_groups and not args.out_free_samples:
        raise GenerationConfigError("--prompt-groups requires --out-free-samples")
    if args.out_free_samples and not args.prompt_groups:
        raise GenerationConfigError("--out-free-samples requires --prompt-groups")
    if args.candidates and not args.out_candidate_scores:
        raise GenerationConfigError("--candidates requires --out-candidate-scores")
    if args.out_candidate_scores and not args.candidates:
        raise GenerationConfigError("--out-candidate-scores requires --candidates")
    if args.resume and args.force:
        raise GenerationConfigError("--resume and --force are mutually exclusive")


def _sidecar_path_for_json(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".full_logits.parquet")


def _delete_phase_outputs(path: Path) -> None:
    for candidate in (path, _sidecar_path_for_json(path)):
        if candidate.exists():
            candidate.unlink()
    checkpoint_root = checkpoint_root_for_artifact(path)
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)


def _validated_existing_phase(path: Path, *, artifact_type: str) -> dict[str, object] | None:
    if not path.exists():
        if _sidecar_path_for_json(path).exists():
            if checkpoint_root_for_artifact(path).exists():
                _sidecar_path_for_json(path).unlink()
                return None
            raise GenerationConfigError(
                f"found partial sidecar without final JSON artifact for {path}; use --force to delete and regenerate"
            )
        return None
    payload = validate_generation_artifact(path)
    if payload.get("artifact_type") != artifact_type:
        raise GenerationConfigError(
            f"existing artifact {path} has artifact_type={payload.get('artifact_type')!r}; expected {artifact_type!r}"
        )
    if artifact_type == "free_sample_rows":
        raw_samples = payload.get("samples")
        samples: list[object] = raw_samples if isinstance(raw_samples, list) else []
        return {
            "artifact_path": str(path),
            "sample_count": len(samples),
            "artifact_type": artifact_type,
            "resumed": True,
        }
    raw_candidate_rows = payload.get("candidate_score_rows")
    raw_token_rows = payload.get("token_score_rows")
    candidate_rows: list[object] = raw_candidate_rows if isinstance(raw_candidate_rows, list) else []
    token_rows: list[object] = raw_token_rows if isinstance(raw_token_rows, list) else []
    return {
        "artifact_path": str(path),
        "candidate_count": len(candidate_rows),
        "token_score_count": len(token_rows),
        "artifact_type": artifact_type,
        "resumed": True,
    }


def _emit(
    progress_path: Path | None,
    *,
    phase: str,
    completed: int,
    total: int,
    message: str,
    output_path: Path,
) -> None:
    write_progress(
        progress_path,
        progress_snapshot(phase=phase, completed=completed, total=total, message=message, output_path=output_path),
    )


def main() -> int:
    progress_path: Path | None = None
    progress_output_path = Path("generation-artifacts.json")
    try:
        args = parse_args()
        _validate_args(args)
        progress_path = Path(args.progress) if args.progress else None
        first_output = args.out_free_samples or args.out_candidate_scores or "generation-artifacts.json"
        progress_output_path = Path(first_output)
        total_phases = int(bool(args.prompt_groups)) + int(bool(args.candidates))
        completed_phases = 0
        _emit(
            progress_path,
            phase="start",
            completed=0,
            total=total_phases,
            message="starting generation artifact preparation",
            output_path=progress_output_path,
        )
        adapter = LocalModelGenerationAdapter(Path(args.config))
        result: dict[str, object] = {}
        if args.prompt_groups:
            free_sample_path = Path(args.out_free_samples)
            if args.force:
                _delete_phase_outputs(free_sample_path)
            elif args.resume:
                existing = _validated_existing_phase(free_sample_path, artifact_type="free_sample_rows")
                if existing is not None:
                    result["free_samples"] = existing
                else:
                    result["free_samples"] = None
            elif free_sample_path.exists() or _sidecar_path_for_json(free_sample_path).exists():
                raise GenerationConfigError(
                    f"refusing to overwrite existing free-sample outputs at {free_sample_path}; use --resume to validate/skip or --force to regenerate"
                )
            elif checkpoint_root_for_artifact(free_sample_path).exists():
                raise GenerationConfigError(
                    f"refusing to overwrite existing free-sample checkpoints at {checkpoint_root_for_artifact(free_sample_path)}; use --resume or --force"
                )
            if args.resume and result.get("free_samples") is not None:
                pass
            elif args.write_fixture:
                result["free_samples"] = adapter.write_free_sample_fixture(
                    out_path=args.out_free_samples,
                    prompt_groups_path=args.prompt_groups,
                    variant=args.fixture_variant,
                )
            else:
                result["free_samples"] = adapter.build_free_sample_artifact(
                    out_path=args.out_free_samples,
                    prompt_groups_path=args.prompt_groups,
                    resume=args.resume,
                )
            completed_phases += 1
            _emit(
                progress_path,
                phase="free_samples_complete",
                completed=completed_phases,
                total=total_phases,
                message="free-sample artifact ready",
                output_path=free_sample_path,
            )
        if args.candidates:
            candidate_scores_path = Path(args.out_candidate_scores)
            if args.force:
                _delete_phase_outputs(candidate_scores_path)
            elif args.resume:
                existing = _validated_existing_phase(candidate_scores_path, artifact_type="teacher_forced_candidate_scores")
                if existing is not None:
                    result["candidate_scores"] = existing
                else:
                    result["candidate_scores"] = None
            elif candidate_scores_path.exists() or _sidecar_path_for_json(candidate_scores_path).exists():
                raise GenerationConfigError(
                    f"refusing to overwrite existing candidate-score outputs at {candidate_scores_path}; use --resume to validate/skip or --force to regenerate"
                )
            elif checkpoint_root_for_artifact(candidate_scores_path).exists():
                raise GenerationConfigError(
                    f"refusing to overwrite existing candidate-score checkpoints at {checkpoint_root_for_artifact(candidate_scores_path)}; use --resume or --force"
                )
            if args.resume and result.get("candidate_scores") is not None:
                pass
            elif args.write_fixture:
                result["candidate_scores"] = adapter.write_candidate_score_fixture(
                    out_path=args.out_candidate_scores,
                    candidates_path=args.candidates,
                    variant=args.fixture_variant,
                )
            else:
                result["candidate_scores"] = adapter.build_candidate_score_artifact(
                    out_path=args.out_candidate_scores,
                    candidates_path=args.candidates,
                    resume=args.resume,
                )
            completed_phases += 1
            _emit(
                progress_path,
                phase="candidate_scores_complete",
                completed=completed_phases,
                total=total_phases,
                message="candidate-score artifact ready",
                output_path=candidate_scores_path,
            )
    except (GenerationConfigError, GenerationDependencyError, ModelLoadError, RuntimeError) as exc:
        _emit(
            progress_path,
            phase="failed",
            completed=0,
            total=0,
            message=str(exc),
            output_path=progress_output_path,
        )
        print(str(exc), file=sys.stderr)
        return 2
    _emit(
        progress_path,
        phase="complete",
        completed=completed_phases,
        total=total_phases,
        message="generation artifacts complete",
        output_path=progress_output_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
