#!/usr/bin/env python3
"""Fetch literature artifacts and write a manifest with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


USER_AGENT = "Mozilla/5.0 (compatible; literature-fetch/1.0)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to literature config")
    parser.add_argument("--out", required=True, help="Output literature directory")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse config {path}: {exc}") from exc


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fetch_source(source: dict, out_dir: Path) -> dict:
    source_urls = source.get("source_urls") or []
    if not source_urls:
        raise SystemExit(f"Source {source['id']} is missing source_urls in the config.")

    artifact_dir = out_dir / source["id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    filename = source.get("filename") or f"{source['id']}.bin"
    artifact_path = artifact_dir / filename

    errors: list[str] = []
    payload: bytes | None = None
    resolved_url: str | None = None
    for url in source_urls:
        try:
            payload = fetch_bytes(url)
            resolved_url = url
            break
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            errors.append(f"{url}: {exc}")

    if payload is None or resolved_url is None:
        joined = "\n".join(errors) if errors else "no URL attempts made"
        raise SystemExit(f"Failed to fetch source {source['id']}. Attempts:\n{joined}")

    artifact_path.write_bytes(payload)
    entry = {
        "id": source["id"],
        "title": source["title"],
        "artifact_type": source.get("artifact_type", "pdf"),
        "canonical_url": source.get("canonical_url"),
        "source_url": resolved_url,
        "official_urls": source.get("official_urls", []),
        "status": source["status"],
        "citation_caveat": source["citation_caveat"],
        "local_path": str(artifact_path.relative_to(out_dir.parent)),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }
    return entry


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SystemExit("literature config must define a non-empty 'sources' list")

    manifest_entries = [fetch_source(source, out_dir) for source in sources]
    manifest = {
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "sources": manifest_entries,
    }
    write_json(out_dir / "literature_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
