#!/usr/bin/env python3
"""Compute direct corpus feature rows from cached upstream artifacts or an explicit proxy branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters import CachedOrProxyCorpusAdapter, write_feature_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True, help="Path to experiments/manifests")
    parser.add_argument("--out", required=True, help="Output path (.parquet or .jsonl)")
    parser.add_argument(
        "--mode",
        default="cache-or-proxy",
        choices=["cache-or-proxy", "cache-only", "service-unavailable"],
        help="Corpus feature acquisition mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_dir = Path(args.manifests)
    out_path = Path(args.out)
    dataset_config_path = ROOT / "experiments" / "configs" / "datasets.yaml"
    adapter = CachedOrProxyCorpusAdapter(manifest_dir=manifest_dir, dataset_config_path=dataset_config_path)
    mode = args.mode
    if mode == "service-unavailable":
        adapter.proxy_index = adapter.proxy_index.empty()
    rows, report = adapter.build_feature_rows(mode=mode)
    storage = write_feature_artifact(out_path, rows, report)
    print(json.dumps({"report": report, "storage": storage}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
