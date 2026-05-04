"""Manifest-backed corpus feature adapters and serialization helpers."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from experiments.domain import AnalysisBin, FeatureProvenance, FeatureRole, FeatureVector, TypeLabel

LOW_FREQUENCY_THRESHOLD = 1000
MAX_ENTITY_COUNT = 8
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
FEATURE_ORDER = (
    "semantic_entropy",
    "cluster_count",
    "semantic_energy_proxy",
    "logit_variance",
    "confidence_margin",
    "entity_frequency",
    "entity_frequency_mean",
    "entity_frequency_min",
    "entity_pair_cooccurrence",
    "low_frequency_entity_flag",
    "zero_cooccurrence_flag",
    "coverage_score",
    "corpus_source",
    "corpus_risk_only",
)


@dataclass(frozen=True)
class ManifestArtifact:
    """Normalized row-level artifact descriptor."""

    artifact_id: str
    artifact_kind: str
    dataset_id: str
    dataset_name: str
    split_id: str
    registry_role: str | None
    absolute_path: str
    has_corpus_stats: bool
    sample_count: int
    timestamp: str | None


@dataclass
class CorpusProxyIndex:
    """Locally cached term and pair support derived from upstream corpus artifacts."""

    term_counts: dict[str, int]
    pair_counts: dict[tuple[str, str], int]
    source_artifact_paths: tuple[str, ...]

    @classmethod
    def empty(cls) -> "CorpusProxyIndex":
        return cls(term_counts={}, pair_counts={}, source_artifact_paths=())

    @property
    def is_available(self) -> bool:
        return bool(self.term_counts or self.pair_counts)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def combine_entities(sample: dict[str, Any]) -> list[str]:
    corpus_stats = sample.get("corpus_stats")
    if isinstance(corpus_stats, dict):
        sources: list[str] = []
        for key in ("entities_q", "entities_a"):
            values = corpus_stats.get(key)
            if isinstance(values, list):
                sources.extend(str(value) for value in values if isinstance(value, str))
        entity_freqs = corpus_stats.get("entity_frequencies")
        if isinstance(entity_freqs, dict):
            sources.extend(str(key) for key in entity_freqs)
        normalized = [normalize_term(value) for value in sources]
        return [value for index, value in enumerate(normalized) if value and value not in normalized[:index]][:MAX_ENTITY_COUNT]

    question = str(sample.get("question", ""))
    response = first_response(sample)
    combined = phrase_candidates(question) + phrase_candidates(response)
    deduped: list[str] = []
    for value in combined:
        if value and value not in deduped:
            deduped.append(value)
    return deduped[:MAX_ENTITY_COUNT]


def first_response(sample: dict[str, Any]) -> str:
    responses = sample.get("responses")
    if isinstance(responses, list) and responses:
        return str(responses[0])
    return ""


def infer_label(is_hallucination: Any, semantic_entropy: float) -> TypeLabel:
    hallucination = bool(is_hallucination)
    if not hallucination:
        return TypeLabel.NORMAL
    if semantic_entropy <= 0.1:
        return TypeLabel.LOW_DIVERSITY
    if semantic_entropy <= 0.5:
        return TypeLabel.AMBIGUOUS_INCORRECT
    return TypeLabel.HIGH_DIVERSITY


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


def read_analysis_bin_config(config_path: Path) -> tuple[tuple[AnalysisBin, ...], list[dict[str, Any]]]:
    config = load_json(config_path)
    raw_bins = list(config["label_policy"]["analysis_se_bins"]["bins"])
    return load_analysis_bins(config_path), raw_bins


class UpstreamManifestCatalog:
    """Loads row-level upstream artifacts and chooses preferred inputs."""

    def __init__(self, manifest_dir: Path) -> None:
        self.manifest_dir = manifest_dir
        self.manifest_path = manifest_dir / "upstream_artifacts_manifest.json"
        payload = load_json(self.manifest_path)
        self.source_root = payload.get("source_root", "")
        self.artifacts = tuple(self._normalize_artifact(entry) for entry in payload.get("artifacts", []))

    def _normalize_artifact(self, entry: dict[str, Any]) -> ManifestArtifact:
        dataset = entry.get("dataset") or {}
        metadata = entry.get("metadata") or {}
        prompt_model_metadata = entry.get("prompt_model_metadata") or {}
        raw_metadata_config = metadata.get("config")
        metadata_config = raw_metadata_config if isinstance(raw_metadata_config, dict) else {}
        split_id = (
            dataset.get("registry_split_id")
            or prompt_model_metadata.get("dataset_split")
            or metadata_config.get("dataset_split")
            or metadata_config.get("split")
            or "unknown_split"
        )
        availability = entry.get("availability") or {}
        return ManifestArtifact(
            artifact_id=str(entry.get("artifact_id", "unknown_artifact")),
            artifact_kind=str(entry.get("artifact_kind", "unknown_kind")),
            dataset_id=str(dataset.get("dataset_id", "unknown")),
            dataset_name=str(dataset.get("dataset_name", dataset.get("source_dataset_value", "unknown"))),
            split_id=str(split_id),
            registry_role=dataset.get("registry_role"),
            absolute_path=str(entry.get("absolute_path", "")),
            has_corpus_stats=bool(availability.get("has_corpus_stats", False)),
            sample_count=int(entry.get("sample_count", 0) or 0),
            timestamp=(metadata.get("timestamp") or prompt_model_metadata.get("timestamp")),
        )

    def preferred_row_artifacts(self) -> tuple[ManifestArtifact, ...]:
        grouped: dict[str, list[ManifestArtifact]] = defaultdict(list)
        for artifact in self.artifacts:
            if artifact.artifact_kind not in {"row_results", "row_results_with_corpus"}:
                continue
            if not artifact.absolute_path:
                continue
            if artifact.dataset_id == "unknown":
                continue
            grouped[artifact.dataset_id].append(artifact)

        selected: list[ManifestArtifact] = []
        for dataset_id, candidates in grouped.items():
            candidates.sort(
                key=lambda artifact: (
                    0 if artifact.has_corpus_stats else 1,
                    -artifact.sample_count,
                    artifact.timestamp or "",
                    artifact.artifact_id,
                )
            )
            selected.append(candidates[0])
        selected.sort(key=lambda artifact: (artifact.dataset_name, artifact.split_id, artifact.artifact_id))
        return tuple(selected)

    def corpus_cache_artifacts(self) -> tuple[ManifestArtifact, ...]:
        return tuple(artifact for artifact in self.preferred_row_artifacts() if artifact.has_corpus_stats)


def build_proxy_index(artifacts: tuple[ManifestArtifact, ...]) -> CorpusProxyIndex:
    term_counts: dict[str, int] = {}
    pair_counts: Counter[tuple[str, str]] = Counter()
    source_paths: list[str] = []

    for artifact in artifacts:
        payload = load_json(Path(artifact.absolute_path))
        samples = payload.get("samples")
        if not isinstance(samples, list):
            continue
        source_paths.append(artifact.absolute_path)
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            corpus_stats = sample.get("corpus_stats")
            if not isinstance(corpus_stats, dict):
                continue
            entity_frequencies = corpus_stats.get("entity_frequencies")
            normalized_terms: list[str] = []
            if isinstance(entity_frequencies, dict):
                for raw_term, raw_count in entity_frequencies.items():
                    term = normalize_term(str(raw_term))
                    if not term:
                        continue
                    try:
                        count = int(raw_count)
                    except (TypeError, ValueError):
                        count = 0
                    term_counts[term] = max(term_counts.get(term, 0), count)
                    normalized_terms.append(term)
            if not normalized_terms:
                normalized_terms = combine_entities(sample)
            unique_terms = sorted(set(normalized_terms))
            for entity1, entity2 in combinations(unique_terms, 2):
                pair_counts[(entity1, entity2)] += 1

    return CorpusProxyIndex(
        term_counts=term_counts,
        pair_counts=dict(pair_counts),
        source_artifact_paths=tuple(sorted(set(source_paths))),
    )


def max_term_count(index: CorpusProxyIndex) -> int:
    return max(index.term_counts.values(), default=0)


def log_normalize_frequency(value: float, max_count: int) -> float:
    if value <= 0 or max_count <= 0:
        return 0.0
    return math.log1p(value) / math.log1p(max_count)


def pair_key(entity1: str, entity2: str) -> tuple[str, str]:
    left, right = sorted((entity1, entity2))
    return left, right


class CachedOrProxyCorpusAdapter:
    """Computes direct corpus feature rows from cache, proxy, or explicit unavailable branches."""

    def __init__(self, manifest_dir: Path, dataset_config_path: Path) -> None:
        self.catalog = UpstreamManifestCatalog(manifest_dir)
        self.dataset_config_path = dataset_config_path
        self.analysis_bins, self.analysis_bin_specs = read_analysis_bin_config(dataset_config_path)
        self.proxy_index = build_proxy_index(self.catalog.corpus_cache_artifacts())

    def build_feature_rows(self, mode: str = "cache-or-proxy") -> tuple[list[dict[str, Any]], dict[str, Any]]:
        run_id = f"corpus-features-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        rows: list[dict[str, Any]] = []
        source_counts: Counter[str] = Counter()
        skipped: list[dict[str, Any]] = []
        for artifact in self.catalog.preferred_row_artifacts():
            path = Path(artifact.absolute_path)
            if not path.exists():
                skipped.append({"artifact_id": artifact.artifact_id, "reason": "missing_artifact_path", "path": artifact.absolute_path})
                continue
            payload = load_json(path)
            samples = payload.get("samples")
            if not isinstance(samples, list):
                skipped.append({"artifact_id": artifact.artifact_id, "reason": "missing_samples_array", "path": artifact.absolute_path})
                continue
            for index, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    continue
                row = self._build_row(run_id, artifact, payload, sample, index, mode)
                source_counts[row["features"]["corpus_source"]] += 1
                rows.append(row)

        report = {
            "run_id": run_id,
            "mode": mode,
            "row_count": len(rows),
            "selected_artifact_count": len(self.catalog.preferred_row_artifacts()),
            "proxy_index": {
                "available": self.proxy_index.is_available,
                "term_count": len(self.proxy_index.term_counts),
                "pair_count": len(self.proxy_index.pair_counts),
                "source_artifact_paths": list(self.proxy_index.source_artifact_paths),
            },
            "corpus_source_counts": dict(source_counts),
            "skipped_artifacts": skipped,
            "supported_methods": {
                "learned_fusion_with_corpus": {
                    "uses_direct_corpus_features": True,
                    "proposed_method_not_scalar_coverage_only": True,
                },
                "corpus_risk_only": {
                    "column": "corpus_risk_only",
                    "uses_only_corpus_features_or_proxies": True,
                    "note": "Baseline support only. This is not the proposed learned fusion method.",
                },
            },
        }
        return rows, report

    def _build_row(
        self,
        run_id: str,
        artifact: ManifestArtifact,
        payload: dict[str, Any],
        sample: dict[str, Any],
        sample_index: int,
        mode: str,
    ) -> dict[str, Any]:
        semantic_entropy = float(sample.get("semantic_entropy", 0.0) or 0.0)
        label = infer_label(sample.get("is_hallucination", 0), semantic_entropy)
        analysis_bin = select_analysis_bin(semantic_entropy, self.analysis_bins, self.analysis_bin_specs)
        feature_vector, corpus_details = self._feature_vector_for_sample(
            run_id=run_id,
            artifact=artifact,
            payload=payload,
            sample=sample,
            sample_index=sample_index,
            label=label,
            analysis_bin=analysis_bin,
            mode=mode,
        )
        features = serialize_feature_vector(feature_vector)
        features["corpus_details"] = corpus_details
        return {
            "run_id": run_id,
            "dataset": artifact.dataset_name,
            "dataset_id": artifact.dataset_id,
            "split_id": artifact.split_id,
            "sample_id": feature_vector.sample_id,
            "label": label.value,
            "source_artifact_path": artifact.absolute_path,
            "artifact_id": artifact.artifact_id,
            "features": features,
            "feature_provenance": [serialize_provenance(entry) for entry in feature_vector.provenance],
            "formula_manifest_ref": "experiments/README.md#4-feature-contract",
            "dataset_manifest_ref": str(self.catalog.manifest_path),
        }

    def _feature_vector_for_sample(
        self,
        *,
        run_id: str,
        artifact: ManifestArtifact,
        payload: dict[str, Any],
        sample: dict[str, Any],
        sample_index: int,
        label: TypeLabel,
        analysis_bin: AnalysisBin | None,
        mode: str,
    ) -> tuple[FeatureVector, dict[str, Any]]:
        dataset_meta = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        sample_id = infer_sample_id(artifact.dataset_id, sample, sample_index)
        cluster_count = int(sample.get("num_clusters", 0) or 0)
        semantic_energy = float(sample.get("semantic_energy", 0.0) or 0.0)
        corpus_stats = sample.get("corpus_stats") if isinstance(sample.get("corpus_stats"), dict) else None
        if corpus_stats:
            feature_payload, source_label = self._from_cached_corpus_stats(sample, corpus_stats)
        elif mode == "cache-or-proxy" and self.proxy_index.is_available:
            feature_payload, source_label = self._from_proxy(sample)
        else:
            feature_payload, source_label = self._service_unavailable(sample)

        feature_vector = FeatureVector(
            run_id=run_id,
            dataset=artifact.dataset_name,
            split_id=artifact.split_id,
            sample_id=sample_id,
            label=label,
            semantic_entropy=float(sample.get("semantic_entropy", 0.0) or 0.0),
            cluster_count=cluster_count,
            semantic_energy=semantic_energy,
            energy_kind=None,
            logit_variance=None,
            confidence_margin=None,
            entity_frequency=feature_payload["entity_frequency_mean"],
            entity_frequency_mean=feature_payload["entity_frequency_mean"],
            entity_frequency_min=feature_payload["entity_frequency_min"],
            entity_pair_cooccurrence=feature_payload["entity_pair_cooccurrence"],
            low_frequency_entity_flag=feature_payload["low_frequency_entity_flag"],
            zero_cooccurrence_flag=feature_payload["zero_cooccurrence_flag"],
            coverage_score=feature_payload["coverage_score"],
            corpus_source=feature_payload["corpus_source"],
            corpus_risk_only=feature_payload["corpus_risk_only"],
            corpus_status=feature_payload["corpus_status"],
            se_bin=analysis_bin,
            provenance=build_provenance_entries(
                source_artifact_path=artifact.absolute_path,
                corpus_source=feature_payload["corpus_source"],
                corpus_status=feature_payload["corpus_status"],
            ),
        )
        details = {
            "corpus_source_label": source_label,
            "dataset_config": dataset_meta,
            "entity_terms": feature_payload["entity_terms"],
            "entity_frequencies": feature_payload["entity_frequencies"],
            "pair_support_pairs": feature_payload["pair_support_pairs"],
            "coverage_formula": feature_payload["coverage_formula"],
            "corpus_risk_formula": feature_payload["corpus_risk_formula"],
            "corpus_status": feature_payload["corpus_status"],
        }
        return feature_vector, details

    def _from_cached_corpus_stats(self, sample: dict[str, Any], corpus_stats: dict[str, Any]) -> tuple[dict[str, Any], str]:
        entity_frequencies = corpus_stats.get("entity_frequencies")
        frequencies: list[int] = []
        frequency_map: dict[str, int] = {}
        if isinstance(entity_frequencies, dict):
            for raw_term, raw_value in entity_frequencies.items():
                term = normalize_term(str(raw_term))
                if not term:
                    continue
                try:
                    count = int(raw_value)
                except (TypeError, ValueError):
                    count = 0
                frequency_map[term] = count
                frequencies.append(count)
        entities = combine_entities(sample)
        num_pairs = int(corpus_stats.get("num_cooc_pairs", 0) or 0)
        cached_cooc_score = float(corpus_stats.get("cooc_score", 0.0) or 0.0)
        zero_cooccurrence_flag = num_pairs > 0 and cached_cooc_score == 0.0
        entity_frequency_mean = sum(frequencies) / len(frequencies) if frequencies else 0.0
        entity_frequency_min = min(frequencies) if frequencies else 0.0
        low_frequency_flag = bool(frequencies) and entity_frequency_min < LOW_FREQUENCY_THRESHOLD
        coverage_score = float(corpus_stats.get("coverage", corpus_stats.get("coverage_score", 0.0)) or 0.0)
        corpus_risk_only = compute_corpus_risk_only(
            entity_frequency_mean=entity_frequency_mean,
            entity_frequency_min=entity_frequency_min,
            entity_pair_cooccurrence=cached_cooc_score,
            low_frequency_entity_flag=low_frequency_flag,
            zero_cooccurrence_flag=zero_cooccurrence_flag,
            coverage_score=coverage_score,
            max_count=max_term_count(self.proxy_index),
        )
        return {
            "entity_terms": entities,
            "entity_frequencies": frequency_map,
            "pair_support_pairs": [],
            "entity_frequency_mean": entity_frequency_mean,
            "entity_frequency_min": entity_frequency_min,
            "low_frequency_entity_flag": low_frequency_flag,
            "entity_pair_cooccurrence": cached_cooc_score,
            "zero_cooccurrence_flag": zero_cooccurrence_flag,
            "coverage_score": coverage_score,
            "corpus_source": "cache_upstream_corpus_stats",
            "corpus_status": "cache-derived_direct_counts_and_cached_cooccurrence_ratio",
            "corpus_risk_only": corpus_risk_only,
            "coverage_formula": "upstream cached coverage field",
            "corpus_risk_formula": "mean(freq_risk, pair_risk, low_flag_risk, coverage_risk)",
        }, "cache"

    def _from_proxy(self, sample: dict[str, Any]) -> tuple[dict[str, Any], str]:
        entities = combine_entities(sample)
        frequency_map = {entity: self.proxy_index.term_counts.get(entity, 0) for entity in entities}
        frequencies = list(frequency_map.values())
        pairs = [pair_key(entity1, entity2) for entity1, entity2 in combinations(sorted(set(entities)), 2)]
        supported_pairs = [pair for pair in pairs if self.proxy_index.pair_counts.get(pair, 0) > 0]
        if pairs:
            pair_score = sum(self.proxy_index.pair_counts.get(pair, 0) for pair in pairs) / len(pairs)
        else:
            pair_score = 0.0
        max_pair_count = max(self.proxy_index.pair_counts.values(), default=0)
        normalized_pair_score = pair_score / max_pair_count if max_pair_count > 0 else 0.0
        entity_frequency_mean = sum(frequencies) / len(frequencies) if frequencies else 0.0
        entity_frequency_min = min(frequencies) if frequencies else 0.0
        low_frequency_flag = bool(frequencies) and entity_frequency_min < LOW_FREQUENCY_THRESHOLD
        zero_cooccurrence_flag = bool(pairs) and not supported_pairs
        max_count = max_term_count(self.proxy_index)
        freq_presence = sum(1 for value in frequencies if value > 0) / len(frequencies) if frequencies else 0.0
        pair_support_ratio = len(supported_pairs) / len(pairs) if pairs else 0.0
        coverage_score = round((freq_presence + pair_support_ratio) / 2.0, 6)
        corpus_risk_only = compute_corpus_risk_only(
            entity_frequency_mean=entity_frequency_mean,
            entity_frequency_min=entity_frequency_min,
            entity_pair_cooccurrence=normalized_pair_score,
            low_frequency_entity_flag=low_frequency_flag,
            zero_cooccurrence_flag=zero_cooccurrence_flag,
            coverage_score=coverage_score,
            max_count=max_count,
        )
        return {
            "entity_terms": entities,
            "entity_frequencies": frequency_map,
            "pair_support_pairs": [[left, right] for left, right in supported_pairs],
            "entity_frequency_mean": entity_frequency_mean,
            "entity_frequency_min": entity_frequency_min,
            "low_frequency_entity_flag": low_frequency_flag,
            "entity_pair_cooccurrence": normalized_pair_score,
            "zero_cooccurrence_flag": zero_cooccurrence_flag,
            "coverage_score": coverage_score,
            "corpus_source": "proxy_cached_artifact_index",
            "corpus_status": "proxy_from_cached_entity_and_pair_support_index",
            "corpus_risk_only": corpus_risk_only,
            "coverage_formula": "mean(entity_presence_ratio, supported_pair_ratio)",
            "corpus_risk_formula": "mean(freq_risk, pair_risk, low_flag_risk, coverage_risk)",
        }, "proxy"

    def _service_unavailable(self, sample: dict[str, Any]) -> tuple[dict[str, Any], str]:
        entities = combine_entities(sample)
        return {
            "entity_terms": entities,
            "entity_frequencies": {entity: 0 for entity in entities},
            "pair_support_pairs": [],
            "entity_frequency_mean": 0.0,
            "entity_frequency_min": 0.0,
            "low_frequency_entity_flag": bool(entities),
            "entity_pair_cooccurrence": 0.0,
            "zero_cooccurrence_flag": len(entities) >= 2,
            "coverage_score": 0.0,
            "corpus_source": "service_unavailable_no_cache",
            "corpus_status": "service_unavailable_explicit_zero_fallback",
            "corpus_risk_only": 1.0 if entities else 0.0,
            "coverage_formula": "explicit zero fallback due to unavailable corpus service/cache",
            "corpus_risk_formula": "hard fallback: 1.0 when entities exist else 0.0",
        }, "service_unavailable"


def infer_sample_id(dataset_id: str, sample: dict[str, Any], sample_index: int) -> str:
    if "sample_id" in sample and str(sample["sample_id"]).strip():
        return str(sample["sample_id"])
    if "idx" in sample:
        return f"{dataset_id}-{sample['idx']}"
    question = str(sample.get("question", ""))
    snippet = re.sub(r"\W+", "-", question.lower()).strip("-")[:48] or "sample"
    return f"{dataset_id}-{sample_index:05d}-{snippet}"


def build_provenance_entries(
    *,
    source_artifact_path: str,
    corpus_source: str,
    corpus_status: str,
) -> tuple[FeatureProvenance, ...]:
    return (
        FeatureProvenance(
            feature_name="label",
            role=FeatureRole.LABEL_ONLY,
            source="sample.is_hallucination + semantic_entropy thresholds",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=True,
            trainable=False,
            note="Operational label only. Never use as a trainable feature.",
        ),
        FeatureProvenance(
            feature_name="semantic_entropy",
            role=FeatureRole.TRAINABLE,
            source="sample.semantic_entropy",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="cluster_count",
            role=FeatureRole.TRAINABLE,
            source="sample.num_clusters",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="semantic_energy_proxy",
            role=FeatureRole.TRAINABLE,
            source="sample.semantic_energy",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="Upstream artifacts expose semantic energy only; no full-logit Boltzmann recomputation is claimed here.",
        ),
        FeatureProvenance(
            feature_name="entity_frequency_mean",
            role=FeatureRole.TRAINABLE,
            source=f"{corpus_source}:{corpus_status}",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="entity_frequency_min",
            role=FeatureRole.TRAINABLE,
            source=f"{corpus_source}:{corpus_status}",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="entity_frequency",
            role=FeatureRole.TRAINABLE,
            source=f"{corpus_source}:{corpus_status}",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="Alias of entity_frequency_mean retained for README contract compatibility.",
        ),
        FeatureProvenance(
            feature_name="entity_pair_cooccurrence",
            role=FeatureRole.TRAINABLE,
            source=f"{corpus_source}:{corpus_status}",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="May be cached co-occurrence ratio or proxy support score; never a correctness-derived label feature.",
        ),
        FeatureProvenance(
            feature_name="low_frequency_entity_flag",
            role=FeatureRole.TRAINABLE,
            source=f"threshold({LOW_FREQUENCY_THRESHOLD}) on corpus entity frequencies",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="zero_cooccurrence_flag",
            role=FeatureRole.TRAINABLE,
            source=f"{corpus_source}:{corpus_status}",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
        ),
        FeatureProvenance(
            feature_name="coverage_score",
            role=FeatureRole.TRAINABLE,
            source=f"{corpus_source}:{corpus_status}",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=True,
            note="Direct corpus feature retained for learned fusion; not the proposed scalar-only method.",
        ),
        FeatureProvenance(
            feature_name="corpus_risk_only",
            role=FeatureRole.ANALYSIS_ONLY,
            source="derived from corpus-only features for baseline support",
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=False,
            note="Baseline metadata for corpus-risk-only scoring support.",
        ),
        FeatureProvenance(
            feature_name="corpus_source",
            role=FeatureRole.EXTERNAL_CORPUS,
            source=corpus_source,
            source_artifact_path=source_artifact_path,
            depends_on_correctness=False,
            trainable=False,
            note="Explicit provenance marker so proxy and unavailable branches are never silent.",
        ),
    )


def compute_corpus_risk_only(
    *,
    entity_frequency_mean: float,
    entity_frequency_min: float,
    entity_pair_cooccurrence: float,
    low_frequency_entity_flag: bool,
    zero_cooccurrence_flag: bool,
    coverage_score: float,
    max_count: int,
) -> float:
    freq_anchor = entity_frequency_min if entity_frequency_min > 0 else entity_frequency_mean
    freq_risk = 1.0 - log_normalize_frequency(freq_anchor, max_count)
    pair_risk = 1.0 if zero_cooccurrence_flag else max(0.0, 1.0 - entity_pair_cooccurrence)
    low_flag_risk = 1.0 if low_frequency_entity_flag else 0.0
    coverage_risk = max(0.0, 1.0 - coverage_score)
    return round((freq_risk + pair_risk + low_flag_risk + coverage_risk) / 4.0, 6)


def serialize_provenance(entry: FeatureProvenance) -> dict[str, Any]:
    payload = asdict(entry)
    payload["role"] = entry.role.value
    return payload


def serialize_feature_vector(vector: FeatureVector) -> dict[str, Any]:
    features = {
        "semantic_entropy": vector.semantic_entropy,
        "cluster_count": vector.cluster_count,
        "semantic_energy_proxy": vector.semantic_energy,
        "logit_variance": vector.logit_variance,
        "confidence_margin": vector.confidence_margin,
        "entity_frequency": vector.entity_frequency,
        "entity_frequency_mean": vector.entity_frequency_mean,
        "entity_frequency_min": vector.entity_frequency_min,
        "entity_pair_cooccurrence": vector.entity_pair_cooccurrence,
        "low_frequency_entity_flag": vector.low_frequency_entity_flag,
        "zero_cooccurrence_flag": vector.zero_cooccurrence_flag,
        "coverage_score": vector.coverage_score,
        "corpus_source": vector.corpus_source,
        "corpus_risk_only": vector.corpus_risk_only,
        "corpus_status": vector.corpus_status,
        "se_bin": None
        if vector.se_bin is None
        else {
            "scheme_name": vector.se_bin.scheme_name,
            "bin_id": vector.se_bin.bin_id,
            "lower_bound": vector.se_bin.lower_bound,
            "upper_bound": vector.se_bin.upper_bound,
            "includes_upper_bound": vector.se_bin.includes_upper_bound,
            "note": vector.se_bin.note,
        },
    }
    return features


def write_feature_artifact(out_path: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_summary = {
        "requested_out_path": str(out_path),
        "row_count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": report,
        "storage_kind": "jsonl",
    }
    if out_path.suffix == ".parquet":
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pylist(rows)
            pq.write_table(table, out_path)
            payload_summary["storage_kind"] = "parquet"
            payload_summary["materialized_path"] = str(out_path)
            return payload_summary
        except Exception as exc:  # pragma: no cover - optional dependency branch
            fallback_path = out_path.with_suffix(out_path.suffix + ".jsonl")
            with fallback_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            payload_summary["storage_kind"] = "jsonl_fallback_for_requested_parquet"
            payload_summary["materialized_path"] = str(fallback_path)
            payload_summary["parquet_unavailable_reason"] = repr(exc)
            write_json(out_path.with_suffix(out_path.suffix + ".storage.json"), payload_summary)
            return payload_summary

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
        elif path.exists():
            try:
                import pyarrow.parquet as pq  # type: ignore

                table = pq.read_table(path)
                return table.to_pylist(), None
            except Exception as exc:  # pragma: no cover - optional dependency branch
                raise RuntimeError(f"Unable to read parquet artifact {path}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    with actual_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows, storage_report
