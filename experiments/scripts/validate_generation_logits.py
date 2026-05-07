#!/usr/bin/env python3
"""Validate that a generation artifact preserves row-level full logits and logsumexp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters.model_generation import (
    GenerationValidationError,
    expected_free_sample_indexes,
    free_sample_index_coverage,
    validate_generation_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Path to a generated logits artifact JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = validate_generation_artifact(Path(args.artifact))
    except GenerationValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    artifact_type = payload.get("artifact_type")
    if artifact_type == "free_sample_rows":
        raw_samples = payload.get("samples")
        samples: list[object] = raw_samples if isinstance(raw_samples, list) else []
        coverage = free_sample_index_coverage(payload)
        expected_coverage = list(expected_free_sample_indexes())
        print(
            f"Validated free-sample logits artifact {args.artifact} with {len(samples)} sample(s); "
            f"schema_version={payload.get('schema_version')} logits_schema={payload.get('logits_schema_version')} "
            f"full_logits={payload.get('has_full_logits')} "
            f"expected_sample_indexes={expected_coverage} observed_prompt_coverages={coverage}"
        )
        return 0
    if artifact_type == "teacher_forced_candidate_scores":
        raw_candidate_rows = payload.get("candidate_score_rows")
        raw_token_rows = payload.get("token_score_rows")
        candidate_rows: list[object] = raw_candidate_rows if isinstance(raw_candidate_rows, list) else []
        token_rows: list[object] = raw_token_rows if isinstance(raw_token_rows, list) else []
        print(
            f"Validated teacher-forced candidate-score artifact {args.artifact} with {len(candidate_rows)} candidate row(s) "
            f"and {len(token_rows)} token row(s); schema_version={payload.get('schema_version')} "
            f"logits_schema={payload.get('logits_schema_version')} "
            f"full_logits={payload.get('has_full_logits')}"
        )
        return 0
    print(f"Unsupported validated generation artifact_type {artifact_type!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
