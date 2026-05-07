"""Annotation-driven correctness label manifest builder."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.domain import CandidateLabelRow


FORBIDDEN_LABEL_FIELDS = frozenset({"judge_name", "judge_mode", "raw_judge_response", "rationale"})
CORRECT_ROLES = frozenset({"right", "correct"})
INCORRECT_ROLES = frozenset({"hallucinated", "incorrect"})


class CorrectnessDatasetError(ValueError):
    """Raised when candidate rows cannot be converted to annotation labels."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise CorrectnessDatasetError(f"candidate row {line_number} must be a JSON object")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _required_text(row: dict[str, Any], field_name: str, *, line_number: int) -> str:
    value = row.get(field_name)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise CorrectnessDatasetError(f"candidate row {line_number} missing non-empty {field_name!r}")
    return text


def _required_bool(row: dict[str, Any], field_name: str, *, line_number: int) -> bool:
    value = row.get(field_name)
    if not isinstance(value, bool):
        raise CorrectnessDatasetError(f"candidate row {line_number} must contain boolean {field_name!r}")
    return value


def candidate_label_from_row(row: dict[str, Any], *, line_number: int) -> CandidateLabelRow:
    candidate_role = _required_text(row, "candidate_role", line_number=line_number)
    normalized_role = candidate_role.strip().lower()
    is_correct = _required_bool(row, "is_correct", line_number=line_number)
    if normalized_role in CORRECT_ROLES and not is_correct:
        raise CorrectnessDatasetError(f"candidate row {line_number} has correct role {candidate_role!r} with is_correct=false")
    if normalized_role in INCORRECT_ROLES and is_correct:
        raise CorrectnessDatasetError(f"candidate row {line_number} has incorrect role {candidate_role!r} with is_correct=true")
    if normalized_role not in CORRECT_ROLES | INCORRECT_ROLES:
        raise CorrectnessDatasetError(
            f"candidate row {line_number} has unsupported candidate_role {candidate_role!r}; "
            "expected right/correct or hallucinated/incorrect"
        )

    return CandidateLabelRow(
        prompt_id=_required_text(row, "prompt_id", line_number=line_number),
        candidate_id=_required_text(row, "candidate_id", line_number=line_number),
        pair_id=_required_text(row, "pair_id", line_number=line_number),
        candidate_role=candidate_role,
        candidate_text=_required_text(row, "candidate_text", line_number=line_number),
        is_correct=is_correct,
        label_source=_required_text(row, "label_source", line_number=line_number),
        source_row_id=_required_text(row, "source_row_id", line_number=line_number),
        dataset=_required_text(row, "dataset", line_number=line_number),
        split_id=_required_text(row, "split_id", line_number=line_number),
    )


def candidate_label_row_to_json(label_row: CandidateLabelRow) -> dict[str, Any]:
    row = asdict(label_row)
    row["sample_id"] = label_row.candidate_id
    row["is_hallucination"] = not label_row.is_correct
    row["dataset_id"] = label_row.split_id
    return row


def build_candidate_label_rows(candidate_rows_path: Path) -> tuple[CandidateLabelRow, ...]:
    raw_rows = load_jsonl(candidate_rows_path)
    return tuple(candidate_label_from_row(row, line_number=index) for index, row in enumerate(raw_rows, start=1))


def forbidden_field_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted({field for row in rows for field in row if field in FORBIDDEN_LABEL_FIELDS})
    return {
        "forbidden_fields": sorted(FORBIDDEN_LABEL_FIELDS),
        "present_forbidden_fields": present,
        "passed": not present,
    }


def _counts_by_field(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field_name, "")) for row in rows).items()))


def _correctness_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter("correct" if bool(row["is_correct"]) else "incorrect" for row in rows)
    return {key: counts.get(key, 0) for key in ("correct", "incorrect")}


def _sample_rows_by_dataset(rows: list[dict[str, Any]], *, per_dataset: int = 2) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = str(row["dataset"])
        if len(samples[dataset]) < per_dataset:
            samples[dataset].append(row)
    return dict(sorted(samples.items()))


def build_label_manifest_evidence(candidate_rows_path: Path, data_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scan = forbidden_field_scan(rows)
    return {
        "artifact_type": "annotation_driven_label_manifest_evidence",
        "schema_version": "candidate_label_manifest_evidence_v1",
        "candidate_rows_path": str(candidate_rows_path),
        "label_manifest_path": str(data_path),
        "row_count": len(rows),
        "heuristic_matching_used": False,
        "llm_as_judge_used": False,
        "forbidden_field_scan": scan,
        "counts_by_dataset": _counts_by_field(rows, "dataset"),
        "counts_by_candidate_role": _counts_by_field(rows, "candidate_role"),
        "counts_by_label_source": _counts_by_field(rows, "label_source"),
        "counts_by_correctness": _correctness_counts(rows),
        "sample_rows_by_dataset": _sample_rows_by_dataset(rows),
    }


def build_halueval_direct_evidence(candidate_rows_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    halueval_rows = [row for row in rows if row.get("dataset") == "HaluEval-QA"]
    right_rows = [row for row in halueval_rows if str(row.get("candidate_role", "")).lower() in CORRECT_ROLES]
    incorrect_rows = [row for row in halueval_rows if str(row.get("candidate_role", "")).lower() in INCORRECT_ROLES]
    return {
        "artifact_type": "halueval_direct_annotation_evidence",
        "schema_version": "halueval_direct_annotation_evidence_v1",
        "candidate_rows_path": str(candidate_rows_path),
        "dataset": "HaluEval-QA",
        "row_count": len(halueval_rows),
        "right_row_count": len(right_rows),
        "hallucinated_or_incorrect_row_count": len(incorrect_rows),
        "right_rows_all_correct": all(bool(row["is_correct"]) for row in right_rows),
        "hallucinated_or_incorrect_rows_all_incorrect": all(not bool(row["is_correct"]) for row in incorrect_rows),
        "label_sources": _counts_by_field(halueval_rows, "label_source"),
        "heuristic_matching_used": False,
        "llm_as_judge_used": False,
        "sample_rows": {
            "right": right_rows[:2],
            "hallucinated_or_incorrect": incorrect_rows[:2],
        },
    }


def dataset_card_text(manifest: dict[str, Any]) -> str:
    return f"""---
dataset_info:
  features:
    - name: prompt_id
      dtype: string
    - name: candidate_id
      dtype: string
    - name: pair_id
      dtype: string
    - name: dataset
      dtype: string
    - name: candidate_role
      dtype: string
    - name: is_correct
      dtype: bool
    - name: label_source
      dtype: string
---

# Annotation-driven correctness labels

This correctness label manifest is derived only from `candidate_rows.jsonl` dataset annotations.
It does not call an LLM judge, does not read generated answers, and does not use heuristic text matching.

## Construction

- Source candidate rows: `{manifest.get('candidate_rows_path')}`
- Label rows: `{manifest.get('files', {}).get('data')}`
- Row count: {manifest.get('row_count', 0)}
- Heuristic matching used: `{manifest.get('heuristic_matching_used')}`
- LLM-as-judge used: `{manifest.get('llm_as_judge_used')}`
"""


def build_annotation_correctness_dataset(
    candidate_rows_path: Path,
    out_dir: Path,
    *,
    evidence_path: Path | None = None,
    halueval_evidence_path: Path | None = None,
) -> dict[str, Any]:
    label_rows = build_candidate_label_rows(candidate_rows_path)
    rows = [candidate_label_row_to_json(label_row) for label_row in label_rows]
    scan = forbidden_field_scan(rows)
    if not scan["passed"]:
        raise CorrectnessDatasetError(f"label manifest contains forbidden fields: {scan['present_forbidden_fields']}")

    data_path = out_dir / "data" / "correctness_judgments.jsonl"
    manifest_path = out_dir / "dataset_manifest.json"
    card_path = out_dir / "README.md"
    write_jsonl(data_path, rows)

    evidence = build_label_manifest_evidence(candidate_rows_path, data_path, rows)
    halueval_evidence = build_halueval_direct_evidence(candidate_rows_path, rows)
    manifest = {
        "artifact_type": "annotation_driven_correctness_label_manifest",
        "schema_version": "candidate_label_manifest_v1",
        "candidate_rows_path": str(candidate_rows_path),
        "row_count": len(rows),
        "heuristic_matching_used": False,
        "llm_as_judge_used": False,
        "forbidden_field_scan": scan,
        "counts_by_dataset": evidence["counts_by_dataset"],
        "counts_by_candidate_role": evidence["counts_by_candidate_role"],
        "counts_by_label_source": evidence["counts_by_label_source"],
        "counts_by_correctness": evidence["counts_by_correctness"],
        "files": {
            "data": "data/correctness_judgments.jsonl",
            "manifest": "dataset_manifest.json",
            "card": "README.md",
        },
    }
    write_json(manifest_path, manifest)
    card_path.write_text(dataset_card_text(manifest), encoding="utf-8")
    if evidence_path is not None:
        write_json(evidence_path, evidence)
    if halueval_evidence_path is not None:
        write_json(halueval_evidence_path, halueval_evidence)

    return {
        "dataset_dir": str(out_dir),
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
        "card_path": str(card_path),
        "row_count": len(rows),
        "heuristic_matching_used": False,
        "llm_as_judge_used": False,
        "forbidden_field_scan": scan,
        "counts_by_dataset": evidence["counts_by_dataset"],
        "counts_by_correctness": evidence["counts_by_correctness"],
    }
