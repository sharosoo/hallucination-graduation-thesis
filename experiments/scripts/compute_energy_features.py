#!/usr/bin/env python3
"""Compute semantic energy features, blocking by default when full logits are unavailable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import EnergyFeatureAdapter, write_feature_artifact
from experiments.adapters.energy_features import EnergyFeatureUnavailableError, write_rerun_required_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True, help="Path to experiments/manifests")
    parser.add_argument("--out", required=True, help="Output path (.parquet or .jsonl)")
    parser.add_argument(
        "--require-true-boltzmann",
        action="store_true",
        help="Fail unless selected upstream artifacts expose row-level full logits for true Boltzmann energy.",
    )
    parser.add_argument(
        "--allow-proxy-diagnostic",
        action="store_true",
        help="Allow diagnostic-only proxy export when full logits are absent. Output is marked not_for_thesis_claims and not_for_validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = EnergyFeatureAdapter(manifest_dir=Path(args.manifests))
    try:
        rows, report = adapter.build_feature_rows(
            require_true_boltzmann=args.require_true_boltzmann,
            allow_proxy_diagnostic=args.allow_proxy_diagnostic,
        )
    except EnergyFeatureUnavailableError as exc:
        storage = write_rerun_required_report(Path(args.out), exc.report)
        print(json.dumps({"report": exc.report, "storage": storage}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    storage = write_feature_artifact(Path(args.out), rows, report)
    print(json.dumps({"report": report, "storage": storage}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
