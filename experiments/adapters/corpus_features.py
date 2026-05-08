"""Corpus-axis feature computation using required direct corpus-count backends."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.adapters.corpus_counts import ALLOWED_BACKEND_IDS, build_corpus_count_backend, serialize_count_result
from experiments.domain import AnalysisBin, FeatureProvenance, FeatureRole

LOW_FREQUENCY_THRESHOLD = 1000
MAX_ENTITY_COUNT = 8
ENTITY_FREQUENCY_SCALE = 1_000_000
PAIR_COOCCURRENCE_SCALE = 100_000
THREE_BIN_RULES = (
    ("low_support", 0.333333),
    ("medium_support", 0.666667),
    ("high_support", None),
)
FIVE_BIN_RULES = (
    ("very_low_support", 0.2),
    ("low_support", 0.4),
    ("mid_support", 0.6),
    ("high_support", 0.8),
    ("very_high_support", None),
)
TEN_BIN_RULES = (
    ("decile_00_10", 0.1),
    ("decile_10_20", 0.2),
    ("decile_20_30", 0.3),
    ("decile_30_40", 0.4),
    ("decile_40_50", 0.5),
    ("decile_50_60", 0.6),
    ("decile_60_70", 0.7),
    ("decile_70_80", 0.8),
    ("decile_80_90", 0.9),
    ("decile_90_100", None),
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}
REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "prompt_id",
    "pair_id",
    "dataset",
    "split_id",
    "candidate_text",
    "candidate_role",
)
SOURCE_TEXT_FIELD = "candidate_text"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{uuid4().hex}")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_output_path(path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def normalize_term(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip().lower())
    return re.sub(r"[^a-z0-9'\- ]+", "", collapsed).strip()


def tokenize_text(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", value)]


def phrase_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(match.group(1) for match in re.finditer(r'"([^\"]{3,})"', text))
    candidates.extend(match.group(0) for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9'\-]+(?:\s+[A-Z][A-Za-z0-9'\-]+){0,3})\b", text))
    tokens = tokenize_text(text)
    filtered = [token for token in tokens if len(token) >= 5 and token not in STOPWORDS]
    candidates.extend(filtered[: MAX_ENTITY_COUNT * 2])
    seen: set[str] = set()
    normalized: list[str] = []
    for candidate in candidates:
        term = normalize_term(candidate)
        if not term or term in STOPWORDS or term in seen:
            continue
        seen.add(term)
        normalized.append(term)
        if len(normalized) >= MAX_ENTITY_COUNT:
            break
    return normalized


_DEFAULT_ENTITY_EXTRACTOR: Any = None  # set lazily in combine_entities


def _get_default_entity_extractor() -> Any:
    """Resolve the default entity extractor (regex) once."""
    global _DEFAULT_ENTITY_EXTRACTOR
    if _DEFAULT_ENTITY_EXTRACTOR is None:
        # Local import avoids a circular dep when the adapter is imported
        # before the ports package is on sys.path.
        from experiments.adapters.entity_extractor_regex import RegexEntityExtractor

        _DEFAULT_ENTITY_EXTRACTOR = RegexEntityExtractor()
    return _DEFAULT_ENTITY_EXTRACTOR


def combine_entities(
    row: dict[str, Any],
    *,
    extractor: Any = None,
) -> tuple[list[str], list[str], list[str]]:
    """Extract question / answer / merged entity lists for a candidate row.

    ``extractor`` accepts any object satisfying ``EntityExtractorPort``:
    ``RegexEntityExtractor`` (default, legacy regex heuristic) or
    ``QucoEntityExtractor`` (QuCo-extractor-0.5B, recommended for new runs).
    """
    extractor = extractor or _get_default_entity_extractor()
    question = str(row.get("question", ""))
    response = str(row.get(SOURCE_TEXT_FIELD, ""))
    question_entities = extractor.extract(question, role="question")
    answer_entities = extractor.extract(response, role="declarative")
    ordered: list[str] = []
    for entity in question_entities + answer_entities:
        if entity and entity not in ordered:
            ordered.append(entity)
    return question_entities, answer_entities, ordered[:MAX_ENTITY_COUNT]


def load_analysis_bins(config_path: Path) -> tuple[AnalysisBin, ...]:
    config = load_json(config_path)
    bins = config["label_policy"]["analysis_se_bins"]["bins"]
    return tuple(
        AnalysisBin(
            scheme_name=config["label_policy"]["analysis_se_bins"]["scheme_name"],
            bin_id=entry["bin_id"],
            lower_bound=entry.get("lower_bound"),
            upper_bound=entry.get("upper_bound"),
            includes_upper_bound=bool(entry.get("includes_upper_bound", True)),
            note=entry.get("note", "Analysis-only bin."),
        )
        for entry in bins
    )


def read_analysis_bin_config(config_path: Path) -> tuple[tuple[AnalysisBin, ...], list[dict[str, Any]]]:
    config = load_json(config_path)
    raw_bins = list(config["label_policy"]["analysis_se_bins"]["bins"])
    return load_analysis_bins(config_path), raw_bins


def select_analysis_bin(value: float, bins: tuple[AnalysisBin, ...], raw_config: list[dict[str, Any]]) -> AnalysisBin | None:
    for bin_spec, entry in zip(bins, raw_config, strict=True):
        lower = entry.get("lower_bound")
        upper = entry.get("upper_bound")
        lower_inclusive = bool(entry.get("lower_inclusive", True))
        upper_inclusive = bool(entry.get("upper_inclusive", True))
        lower_ok = True if lower is None else value >= lower if lower_inclusive else value > lower
        upper_ok = True if upper is None else value <= upper if upper_inclusive else value < upper
        if lower_ok and upper_ok:
            return bin_spec
    return None


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def coerce_optional_bool(value: Any, *, path: Path, row_index: int) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"{path}: candidate row {row_index} field 'is_correct' must be boolean when present")


def normalize_candidate_rows(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        normalized: dict[str, Any] = {}
        for field_name in REQUIRED_CANDIDATE_FIELDS:
            value = row.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: candidate row {index} must include non-empty string field '{field_name}'")
            normalized[field_name] = value.strip()
        for optional_text in ("source_row_id", "dataset_id", "question", "prompt", "label_source"):
            value = row.get(optional_text)
            if value is not None:
                normalized[optional_text] = str(value)
        is_correct = coerce_optional_bool(row.get("is_correct"), path=path, row_index=index)
        if is_correct is not None:
            normalized["is_correct"] = is_correct
        normalized_rows.append(normalized)
    return normalized_rows


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        rows, _storage = read_feature_rows(path)
        return normalize_candidate_rows(rows, path)
    if path.suffix == ".jsonl":
        return normalize_candidate_rows(load_jsonl_rows(path), path)
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("candidate_rows"), list):
        return normalize_candidate_rows([row for row in payload["candidate_rows"] if isinstance(row, dict)], path)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return normalize_candidate_rows([row for row in payload["rows"] if isinstance(row, dict)], path)
    if isinstance(payload, list):
        return normalize_candidate_rows([row for row in payload if isinstance(row, dict)], path)
    raise ValueError(f"Unsupported candidate corpus source artifact: {path}")


def log_normalize(value: float, scale: int) -> float:
    if value <= 0 or scale <= 0:
        return 0.0
    return min(round(math.log1p(value) / math.log1p(scale), 6), 1.0)


def assign_axis_bin(value: float | None, rules: tuple[tuple[str, float | None], ...]) -> str | None:
    if value is None:
        return None
    for label, upper_bound in rules:
        if upper_bound is None or value <= upper_bound:
            return label
    return rules[-1][0]


def pair_key(entity1: str, entity2: str) -> tuple[str, str]:
    left, right = sorted((entity1, entity2))
    return left, right


def serialize_provenance(entry: FeatureProvenance) -> dict[str, Any]:
    payload = asdict(entry)
    payload["role"] = entry.role.value
    return payload


def build_candidate_corpus_provenance_entries(
    *,
    source_artifact_path: str,
    corpus_source: str,
    corpus_status: str,
) -> tuple[FeatureProvenance, ...]:
    source = f"{corpus_source}:{corpus_status}"
    return (
        FeatureProvenance(
            feature_name="entity_frequency_mean",
            role=FeatureRole.TRAINABLE,
            source=source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="Raw direct corpus mean entity count from a required corpus backend.",
        ),
        FeatureProvenance(
            feature_name="entity_frequency_min",
            role=FeatureRole.TRAINABLE,
            source=source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="Raw direct corpus minimum entity count from a required corpus backend.",
        ),
        FeatureProvenance(
            feature_name="entity_frequency",
            role=FeatureRole.TRAINABLE,
            source=source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="Continuous entity-frequency axis alias retained for downstream compatibility.",
        ),
        FeatureProvenance(
            feature_name="entity_frequency_axis",
            role=FeatureRole.TRAINABLE,
            source=source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="entity_pair_cooccurrence",
            role=FeatureRole.TRAINABLE,
            source=source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="Continuous entity-pair co-occurrence axis derived from direct pair counts.",
        ),
        FeatureProvenance(
            feature_name="entity_pair_cooccurrence_axis",
            role=FeatureRole.TRAINABLE,
            source=source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="low_frequency_entity_flag",
            role=FeatureRole.TRAINABLE,
            source=f"threshold({LOW_FREQUENCY_THRESHOLD}) on raw direct corpus entity counts",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="zero_cooccurrence_flag",
            role=FeatureRole.TRAINABLE,
            source=source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="coverage_score",
            role=FeatureRole.TRAINABLE,
            source=source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="Combined continuous corpus-support axis used for reliability conditioning.",
        ),
        FeatureProvenance(
            feature_name="corpus_risk_only",
            role=FeatureRole.ANALYSIS_ONLY,
            source="derived from transformed corpus axes and threshold flags",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=False,
            note="Reliability conditioning metadata; not a correctness label.",
        ),
        FeatureProvenance(
            feature_name="corpus_source",
            role=FeatureRole.EXTERNAL_CORPUS,
            source=corpus_source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=False,
            note="Direct count backend identifier. Candidate-text proxy counting is forbidden.",
        ),
    )


def compute_corpus_risk_only(
    *,
    entity_frequency_axis: float,
    entity_pair_cooccurrence_axis: float,
    low_frequency_entity_flag: bool,
    zero_cooccurrence_flag: bool,
    coverage_score: float,
) -> float:
    freq_risk = 1.0 - entity_frequency_axis
    pair_risk = 1.0 - entity_pair_cooccurrence_axis
    low_flag_risk = 1.0 if low_frequency_entity_flag else 0.0
    zero_flag_risk = 1.0 if zero_cooccurrence_flag else 0.0
    coverage_risk = 1.0 - coverage_score
    return round((freq_risk + pair_risk + low_flag_risk + zero_flag_risk + coverage_risk) / 5.0, 6)


class CorpusFeatureAdapter:
    """Computes corpus-axis rows from candidate artifacts using a required count backend port."""

    def __init__(
        self,
        candidates_path: Path,
        dataset_config_path: Path,
        *,
        entity_extractor: Any | None = None,
    ) -> None:
        self.candidates_path = candidates_path
        self.rows = load_candidate_rows(candidates_path)
        self.backend = build_corpus_count_backend(candidates_path)
        self.analysis_bins, self.analysis_bin_specs = read_analysis_bin_config(dataset_config_path)
        self.entity_extractor = entity_extractor or _get_default_entity_extractor()
        self._maybe_warmup_backend()

    def _maybe_warmup_backend(self) -> None:
        warmup_fn = getattr(self.backend, "warmup", None)
        if not callable(warmup_fn):
            return
        all_entities: set[str] = set()
        all_pairs: set[str] = set()
        for source_row in self.rows:
            _, _, entities = combine_entities(source_row, extractor=self.entity_extractor)
            unique = sorted(set(entities))
            for entity in unique:
                if entity:
                    all_entities.add(entity)
            for left, right in combinations(unique, 2):
                all_pairs.add(f"{left} AND {right}")
        warmup_fn(entities=sorted(all_entities), pairs=sorted(all_pairs))

    def build_feature_rows(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        run_id = f"corpus-features-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        rows: list[dict[str, Any]] = []
        provenance_records: list[dict[str, Any]] = []
        for source_row in self.rows:
            slim_row, heavy_provenance = self._build_row(run_id, source_row)
            rows.append(slim_row)
            provenance_records.append(heavy_provenance)
        self._provenance_records = provenance_records  # consumed by sidecar writer
        rows_by_source = Counter(str(row["features"].get("corpus_source")) for row in rows)
        rows_by_status = Counter(str(row["features"].get("corpus_status")) for row in rows)
        report = {
            "run_id": run_id,
            "row_count": len(rows),
            "candidate_rows_path": str(self.candidates_path),
            "source_artifact_path": str(self.candidates_path),
            "source_text_field": SOURCE_TEXT_FIELD,
            "candidate_identity_fields": ["dataset", "split_id", "prompt_id", "pair_id", "candidate_id"],
            "corpus_backend": self.backend.describe(),
            "entity_extractor": self.entity_extractor.describe(),
            "corpus_source_counts": dict(rows_by_source),
            "corpus_status_counts": dict(rows_by_status),
            "allowed_backend_ids": sorted(ALLOWED_BACKEND_IDS),
            "count_backend_contract": "direct corpus counts only; candidate_text proxy counting and BM25 retrieval scores are forbidden",
        }
        return rows, report

    def _build_row(self, run_id: str, source_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        question_entities, answer_entities, entities = combine_entities(source_row, extractor=self.entity_extractor)
        pair_terms = [pair_key(left, right) for left, right in combinations(sorted(set(entities)), 2)]
        entity_results = {entity: self.backend.count_entity(entity) for entity in entities}
        pair_results = {pair: self.backend.count_pair(pair[0], pair[1]) for pair in pair_terms}

        entity_payloads = {entity: serialize_count_result(result) for entity, result in entity_results.items()}
        pair_payloads = {f"{left} AND {right}": serialize_count_result(result) for (left, right), result in pair_results.items()}

        combined_payloads = list(entity_payloads.values()) + list(pair_payloads.values())
        disallowed_backend = any(payload["backend_id"] not in ALLOWED_BACKEND_IDS for payload in combined_payloads)
        missing_entity_counts = [entity for entity, payload in entity_payloads.items() if payload["raw_count"] is None]
        missing_pair_counts = [pair for pair, payload in pair_payloads.items() if payload["raw_count"] is None]
        approximate_entities = [entity for entity, payload in entity_payloads.items() if bool(payload.get("approximate", False))]
        approximate_pairs = [pair for pair, payload in pair_payloads.items() if bool(payload.get("approximate", False))]
        error_statuses = [
            str(payload.get("status", ""))
            for payload in combined_payloads
            if str(payload.get("status", "")).startswith(("api_", "timeout", "rate_limit", "missing_", "local_index_absent", "cache_miss"))
        ]

        counts_complete = bool(entities) and not missing_entity_counts and not missing_pair_counts and not approximate_entities and not approximate_pairs and not disallowed_backend
        if not entities:
            row_status = "excluded_no_entities"
        elif disallowed_backend:
            row_status = "excluded_disallowed_backend"
        elif approximate_entities or approximate_pairs:
            row_status = "excluded_approximate_counts"
        elif missing_entity_counts or missing_pair_counts or error_statuses:
            row_status = "excluded_missing_counts"
        else:
            fallback_present = any(payload.get("status") == "fallback_resolved" for payload in combined_payloads)
            row_status = "fallback_resolved" if fallback_present else "resolved"

        raw_entity_counts = {
            entity: int(payload["raw_count"])
            for entity, payload in entity_payloads.items()
            if payload["raw_count"] is not None
        }
        raw_pair_counts = {
            pair: int(payload["raw_count"])
            for pair, payload in pair_payloads.items()
            if payload["raw_count"] is not None
        }

        entity_frequency_mean = sum(raw_entity_counts.values()) / len(raw_entity_counts) if counts_complete else None
        entity_frequency_min = min(raw_entity_counts.values()) if counts_complete else None
        pair_count_mean = sum(raw_pair_counts.values()) / len(raw_pair_counts) if counts_complete and raw_pair_counts else 0.0 if counts_complete else None
        entity_frequency_axis = log_normalize(entity_frequency_min if entity_frequency_min is not None else 0.0, ENTITY_FREQUENCY_SCALE) if counts_complete and entity_frequency_min is not None else None
        entity_pair_cooccurrence_axis = log_normalize(pair_count_mean if pair_count_mean is not None else 0.0, PAIR_COOCCURRENCE_SCALE) if counts_complete and pair_count_mean is not None else None
        low_frequency_flag = bool(counts_complete and entity_frequency_min is not None and entity_frequency_min < LOW_FREQUENCY_THRESHOLD)
        zero_cooccurrence_flag = bool(counts_complete and pair_terms and pair_count_mean == 0.0)
        coverage_score = (
            round(((entity_frequency_axis or 0.0) + (entity_pair_cooccurrence_axis or 0.0)) / 2.0, 6)
            if counts_complete and entity_frequency_axis is not None and entity_pair_cooccurrence_axis is not None
            else None
        )
        corpus_axis_bin = assign_axis_bin(coverage_score, THREE_BIN_RULES)
        corpus_axis_bin_5 = assign_axis_bin(coverage_score, FIVE_BIN_RULES)
        corpus_axis_bin_10 = assign_axis_bin(coverage_score, TEN_BIN_RULES)
        corpus_risk_only = (
            compute_corpus_risk_only(
                entity_frequency_axis=entity_frequency_axis,
                entity_pair_cooccurrence_axis=entity_pair_cooccurrence_axis,
                low_frequency_entity_flag=low_frequency_flag,
                zero_cooccurrence_flag=zero_cooccurrence_flag,
                coverage_score=coverage_score,
            )
            if counts_complete and entity_frequency_axis is not None and entity_pair_cooccurrence_axis is not None and coverage_score is not None
            else None
        )

        analysis_bin = select_analysis_bin(0.0, self.analysis_bins, self.analysis_bin_specs)
        backend_summary = self.backend.describe()
        corpus_source = str(backend_summary.get("backend_id") or "unknown_backend")
        # Slim corpus_axis: scalar fields only. Per-row entity/pair lists,
        # raw_*_counts dicts, and missing/approximate lists are heavy nested
        # provenance that bloats parquet row-groups by 100x+ at read time.
        # Global count provenance is preserved in the infini-gram cache file;
        # per-row entity/pair detail can be reconstructed deterministically by
        # rerunning combine_entities + cache lookup if ever needed.
        missing_entity_count_total = len(missing_entity_counts)
        missing_pair_count_total = len(missing_pair_counts)
        approximate_entity_total = len(approximate_entities)
        approximate_pair_total = len(approximate_pairs)
        corpus_axis = {
            "backend_id": corpus_source,
            "index_ref": backend_summary.get("index_ref"),
            "cache_ref": backend_summary.get("cache_ref"),
            "row_status": row_status,
            "counts_complete": counts_complete,
            "entity_count": len(entities),
            "pair_count": len(pair_terms),
            "entity_frequency_mean_raw": entity_frequency_mean,
            "entity_frequency_min_raw": entity_frequency_min,
            "entity_pair_cooccurrence_raw_mean": pair_count_mean,
            "entity_frequency_axis": entity_frequency_axis,
            "entity_pair_cooccurrence_axis": entity_pair_cooccurrence_axis,
            "corpus_axis_score": coverage_score,
            "corpus_axis_bin": corpus_axis_bin,
            "corpus_axis_bin_5": corpus_axis_bin_5,
            "corpus_axis_bin_10": corpus_axis_bin_10,
            "excluded_reason": None if counts_complete else row_status,
            "missing_entity_count_total": missing_entity_count_total,
            "missing_pair_count_total": missing_pair_count_total,
            "approximate_entity_total": approximate_entity_total,
            "approximate_pair_total": approximate_pair_total,
            "source_text_field": SOURCE_TEXT_FIELD,
        }
        features = {
            "semantic_entropy": None,
            "cluster_count": None,
            "semantic_energy_proxy": None,
            "logit_variance": None,
            "confidence_margin": None,
            "entity_frequency": entity_frequency_axis,
            "entity_frequency_mean": entity_frequency_mean,
            "entity_frequency_min": entity_frequency_min,
            "entity_pair_cooccurrence": entity_pair_cooccurrence_axis,
            "entity_frequency_axis": entity_frequency_axis,
            "entity_pair_cooccurrence_axis": entity_pair_cooccurrence_axis,
            "low_frequency_entity_flag": low_frequency_flag,
            "zero_cooccurrence_flag": zero_cooccurrence_flag,
            "coverage_score": coverage_score,
            "corpus_source": corpus_source,
            "corpus_risk_only": corpus_risk_only,
            "corpus_status": row_status,
            "corpus_axis_bin": corpus_axis_bin,
            "corpus_axis_bin_5": corpus_axis_bin_5,
            "corpus_axis_bin_10": corpus_axis_bin_10,
            "se_bin": None
            if analysis_bin is None
            else {
                "scheme_name": analysis_bin.scheme_name,
                "bin_id": analysis_bin.bin_id,
                "lower_bound": analysis_bin.lower_bound,
                "upper_bound": analysis_bin.upper_bound,
                "includes_upper_bound": analysis_bin.includes_upper_bound,
                "note": analysis_bin.note,
            },
        }
        provenance = [
            serialize_provenance(entry)
            for entry in build_candidate_corpus_provenance_entries(
                source_artifact_path=str(self.candidates_path),
                corpus_source=corpus_source,
                corpus_status=row_status,
            )
        ]
        slim_row = {
            "run_id": run_id,
            "dataset": str(source_row["dataset"]),
            "dataset_id": str(source_row.get("dataset_id") or source_row["split_id"]),
            "split_id": str(source_row["split_id"]),
            "candidate_id": str(source_row["candidate_id"]),
            "prompt_id": str(source_row["prompt_id"]),
            "pair_id": str(source_row["pair_id"]),
            "sample_id": str(source_row["candidate_id"]),
            "candidate_role": str(source_row["candidate_role"]),
            "is_correct": source_row.get("is_correct"),
            "is_hallucination": None if source_row.get("is_correct") is None else not bool(source_row["is_correct"]),
            "label": None,
            "label_status": "not_assigned_by_corpus_features",
            "source_text_field": SOURCE_TEXT_FIELD,
            "source_artifact_path": str(self.candidates_path),
            "artifact_id": self.candidates_path.name,
            "features": features,
            "corpus_axis": corpus_axis,
            "feature_provenance": provenance,
            "formula_manifest_ref": "experiments/README.md#4-feature-contract",
            "dataset_manifest_ref": str(self.candidates_path),
        }
        # Heavy per-row provenance is preserved verbatim in the sidecar JSONL.
        # Downstream stages do NOT read this — they consume the slim parquet —
        # but it keeps the per-row entity/pair attribution recoverable for
        # thesis paper-trail spot checks without rerunning combine_entities.
        heavy_provenance = {
            "run_id": run_id,
            "candidate_id": str(source_row["candidate_id"]),
            "prompt_id": str(source_row["prompt_id"]),
            "pair_id": str(source_row["pair_id"]),
            "candidate_role": str(source_row["candidate_role"]),
            "row_status": row_status,
            "counts_complete": counts_complete,
            "question_entities": question_entities,
            "answer_entities": answer_entities,
            "entities": [{"term": entity, **payload} for entity, payload in entity_payloads.items()],
            "pairs": [{"pair": pair, **payload} for pair, payload in pair_payloads.items()],
            "raw_entity_counts": raw_entity_counts,
            "raw_pair_counts": raw_pair_counts,
            "missing_entity_counts": missing_entity_counts,
            "missing_pair_counts": missing_pair_counts,
            "approximate_entities": approximate_entities,
            "approximate_pairs": approximate_pairs,
            "query_rules": {
                "entity_frequency": "direct count(entity)",
                "entity_pair_cooccurrence": "direct count(entity_a AND entity_b)",
            },
        }
        return slim_row, heavy_provenance


def write_feature_artifact(
    out_path: Path,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    schema_version: str | None = None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_summary = {
        "requested_out_path": str(out_path),
        "row_count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": report,
        "storage_kind": "jsonl",
    }
    if schema_version is not None:
        payload_summary["schema_version"] = schema_version
    if out_path.suffix == ".parquet":
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pylist(rows)
            temp_path = _temporary_output_path(out_path)
            try:
                pq.write_table(table, temp_path)
                temp_path.replace(out_path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise
            payload_summary["storage_kind"] = "parquet"
            payload_summary["materialized_path"] = str(out_path)
            return payload_summary
        except Exception as exc:  # pragma: no cover
            fallback_path = out_path.with_suffix(out_path.suffix + ".jsonl")
            _write_jsonl_atomic(fallback_path, rows)
            payload_summary["storage_kind"] = "jsonl_fallback_for_requested_parquet"
            payload_summary["materialized_path"] = str(fallback_path)
            payload_summary["parquet_unavailable_reason"] = repr(exc)
            write_json(out_path.with_suffix(out_path.suffix + ".storage.json"), payload_summary)
            return payload_summary

    _write_jsonl_atomic(out_path, rows)
    payload_summary["materialized_path"] = str(out_path)
    return payload_summary


def read_feature_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    storage_report = None
    actual_path = path
    if path.suffix == ".parquet":
        storage_report_path = path.with_suffix(path.suffix + ".storage.json")
        if storage_report_path.exists():
            storage_report = load_json(storage_report_path)
            actual_path = Path(storage_report["materialized_path"])
            if actual_path.suffix == ".parquet":
                try:
                    import pyarrow.parquet as pq  # type: ignore

                    table = pq.read_table(actual_path)
                    return table.to_pylist(), storage_report
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(f"Unable to read parquet artifact {actual_path}: {exc}") from exc
        elif path.exists():
            try:
                import pyarrow.parquet as pq  # type: ignore

                table = pq.read_table(path)
                return table.to_pylist(), None
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Unable to read parquet artifact {path}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    with actual_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows, storage_report
