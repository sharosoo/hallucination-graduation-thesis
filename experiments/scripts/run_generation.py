#!/usr/bin/env python3
"""Thin CLI for repo-owned prompt generation with preserved row-level logits."""

from __future__ import annotations

import argparse
import json
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiments/configs/generation.yaml")
    parser.add_argument("--out", required=True, help="Output artifact path (.json)")
    parser.add_argument("--prompts", default=None, help="Optional prompt rows file (.json or .jsonl) overriding config prompt_rows")
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="Write a deterministic fixture artifact instead of calling a live model.",
    )
    parser.add_argument(
        "--fixture-variant",
        choices=("full_logits", "missing_full_logits"),
        default="full_logits",
        help="Fixture schema variant to write when --write-fixture is set.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        adapter = LocalModelGenerationAdapter(Path(args.config))
        if args.write_fixture:
            result = adapter.write_fixture(out_path=args.out, variant=args.fixture_variant, prompt_rows_path=args.prompts)
        else:
            result = adapter.build_artifact(out_path=args.out, prompt_rows_path=args.prompts)
    except (GenerationConfigError, GenerationDependencyError, ModelLoadError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
