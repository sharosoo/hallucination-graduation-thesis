#!/usr/bin/env python3
"""Validate that a generation artifact preserves row-level full logits and logsumexp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters.model_generation import GenerationValidationError, validate_generation_artifact


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
    raw_samples = payload.get("samples")
    samples: list[object] = raw_samples if isinstance(raw_samples, list) else []
    print(
        f"Validated generation logits artifact {args.artifact} with {len(samples)} sample(s); "
        f"schema={payload.get('logits_schema_version')} full_logits={payload.get('has_full_logits')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
