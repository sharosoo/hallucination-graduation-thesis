#!/usr/bin/env python3
"""Summarize derived upstream manifests into an artifact inventory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True, help="Manifest directory produced by build_manifests.py")
    parser.add_argument("--out", required=True, help="Output JSON path for the artifact inventory")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse manifest {path}: {exc}") from exc


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    dataset = artifact.get("dataset", {}) if isinstance(artifact.get("dataset"), dict) else {}
    availability = artifact.get("availability", {}) if isinstance(artifact.get("availability"), dict) else {}
    metadata = artifact.get("prompt_model_metadata", {}) if isinstance(artifact.get("prompt_model_metadata"), dict) else {}
    return {
        "artifact_id": artifact.get("artifact_id"),
        "branch": artifact.get("branch"),
        "artifact_kind": artifact.get("artifact_kind"),
        "relative_path": artifact.get("relative_path"),
        "sha256": artifact.get("sha256"),
        "sample_count": artifact.get("sample_count"),
        "dataset_id": dataset.get("dataset_id"),
        "dataset_name": dataset.get("dataset_name"),
        "registry_role": dataset.get("registry_role"),
        "top_level_fields": artifact.get("top_level_fields", []),
        "sample_fields": artifact.get("sample_fields", []),
        "metric_fields": artifact.get("metric_fields", []),
        "statistics_fields": artifact.get("statistics_fields", []),
        "has_logits": bool(availability.get("has_logits")),
        "has_full_logits": bool(availability.get("has_full_logits")),
        "has_corpus_stats": bool(availability.get("has_corpus_stats")),
        "has_labels": bool(availability.get("has_labels")),
        "prompt_model_metadata": metadata,
    }


def dataset_summary(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        dataset = artifact.get("dataset", {}) if isinstance(artifact.get("dataset"), dict) else {}
        key = (str(dataset.get("dataset_id") or "unknown"), str(dataset.get("dataset_name") or "unknown"))
        grouped[key].append(artifact)

    summary = []
    for (dataset_id, dataset_name), records in sorted(grouped.items()):
        sample_counts: list[int] = []
        for record in records:
            sample_count = record.get("sample_count")
            if isinstance(sample_count, int):
                sample_counts.append(sample_count)
        availability_counts = Counter()
        branches = sorted({str(record.get("branch")) for record in records})
        for record in records:
            availability = record.get("availability", {}) if isinstance(record.get("availability"), dict) else {}
            for key in ("has_logits", "has_full_logits", "has_corpus_stats", "has_labels"):
                if availability.get(key):
                    availability_counts[key] += 1
        summary.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "artifact_count": len(records),
                "branches": branches,
                "min_sample_count": min(sample_counts) if sample_counts else None,
                "max_sample_count": max(sample_counts) if sample_counts else None,
                "artifacts_with_logits": availability_counts["has_logits"],
                "artifacts_with_full_logits": availability_counts["has_full_logits"],
                "artifacts_with_corpus_stats": availability_counts["has_corpus_stats"],
                "artifacts_with_labels": availability_counts["has_labels"],
            }
        )
    return summary


def capability_summary(artifacts: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter({"has_logits": 0, "has_full_logits": 0, "has_corpus_stats": 0, "has_labels": 0})
    for artifact in artifacts:
        availability = artifact.get("availability", {}) if isinstance(artifact.get("availability"), dict) else {}
        for key in ("has_logits", "has_full_logits", "has_corpus_stats", "has_labels"):
            if availability.get(key):
                totals[key] += 1
    totals["artifact_count"] = len(artifacts)
    return dict(totals)


def branch_gap_summary(branch_status: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for record in branch_status:
        missing_paths = record.get("missing_paths", []) if isinstance(record.get("missing_paths"), list) else []
        summary.append(
            {
                "branch": record.get("branch"),
                "required": bool(record.get("required")),
                "category": record.get("category"),
                "status": record.get("status"),
                "present_count": len(record.get("present_paths", [])) if isinstance(record.get("present_paths"), list) else 0,
                "missing_count": len(missing_paths),
                "missing_paths": missing_paths,
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    manifest_dir = Path(args.manifests)
    out_path = Path(args.out)

    if not manifest_dir.exists() or not manifest_dir.is_dir():
        raise SystemExit("manifest path not found")

    artifact_manifest_path = manifest_dir / "upstream_artifacts_manifest.json"
    branch_manifest_path = manifest_dir / "upstream_branch_status.json"
    if not artifact_manifest_path.exists():
        raise SystemExit("artifact manifest not found")
    if not branch_manifest_path.exists():
        raise SystemExit("branch status manifest not found")

    artifact_manifest = load_json(artifact_manifest_path)
    branch_manifest = load_json(branch_manifest_path)

    artifacts = artifact_manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise SystemExit("artifact manifest malformed")

    branch_status = branch_manifest.get("branch_status", [])
    if not isinstance(branch_status, list):
        raise SystemExit("branch status manifest malformed")

    inventory = {
        "inventory_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_dir": str(manifest_dir.resolve()),
        "source_root": artifact_manifest.get("source_root"),
        "dataset_registry_name": artifact_manifest.get("dataset_registry_name"),
        "artifact_count": len(artifacts),
        "capability_summary": capability_summary(artifacts),
        "branch_status": branch_gap_summary(branch_status),
        "dataset_summary": dataset_summary(artifacts),
        "artifacts": [summarize_artifact(artifact) for artifact in artifacts],
    }

    write_json(out_path, inventory)
    print(json.dumps(inventory, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
