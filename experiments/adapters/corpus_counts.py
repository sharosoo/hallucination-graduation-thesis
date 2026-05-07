"""Concrete corpus count backends for thesis-valid corpus-axis computation."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.ports import CorpusCountBackendPort, CorpusCountProvenance, CorpusCountResult

ALLOWED_BACKEND_IDS = frozenset({"infini_gram_api_count", "infini_gram_local_count", "quco_cache_infini_gram"})
FIXTURE_SCHEMA_VERSION = "corpus_count_fixture_v1"

INFINIGRAM_BACKEND_ENV = "THESIS_CORPUS_BACKEND"
INFINIGRAM_INDEX_ENV = "INFINIGRAM_INDEX"
INFINIGRAM_ENDPOINT_ENV = "INFINIGRAM_ENDPOINT"
INFINIGRAM_LOCAL_INDEX_DIR_ENV = "INFINIGRAM_LOCAL_INDEX_DIR"
INFINIGRAM_LOCAL_TOKENIZER_ENV = "INFINIGRAM_LOCAL_TOKENIZER"
INFINIGRAM_DEFAULT_INDEX = "v4_dolma-v1_7_llama"
INFINIGRAM_DEFAULT_ENDPOINT = "https://api.infini-gram.io/"
INFINIGRAM_DEFAULT_LOCAL_TOKENIZER = "allenai/OLMo-7B-hf"
INFINIGRAM_CACHE_SCHEMA_VERSION = "infinigram_api_cache_v1"
INFINIGRAM_DEFAULT_TIMEOUT_SECONDS = 30.0
INFINIGRAM_DEFAULT_MAX_RETRIES = 3
INFINIGRAM_DEFAULT_PARALLELISM = 8


def normalize_term(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip().lower())
    return re.sub(r"[^a-z0-9'\- ]+", "", collapsed).strip()


def pair_query(left: str, right: str) -> str:
    ordered = sorted((normalize_term(left), normalize_term(right)))
    return f"{ordered[0]} AND {ordered[1]}"


def pair_storage_key(left: str, right: str) -> str:
    ordered = sorted((normalize_term(left), normalize_term(right)))
    return f"{ordered[0]} && {ordered[1]}"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FixtureCountRecord:
    count: int | None
    status: str
    approximate: bool = False
    note: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: object, *, kind: str, key: str) -> "FixtureCountRecord":
        if isinstance(payload, int):
            return cls(count=payload, status="resolved")
        if not isinstance(payload, dict):
            raise ValueError(f"{kind} count fixture for {key!r} must be an int or object")
        raw_count = payload.get("count")
        count = None if raw_count is None else int(raw_count)
        status = str(payload.get("status") or "missing_status")
        approximate = bool(payload.get("approximate", False))
        note = payload.get("note")
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"{kind} count fixture metadata for {key!r} must be an object when present")
        return cls(count=count, status=status, approximate=approximate, note=str(note) if note is not None else None, metadata=metadata)


class FixtureCorpusCountBackend(CorpusCountBackendPort):
    """Offline deterministic corpus-count backend backed by a checked fixture/cache snapshot."""

    def __init__(self, fixture_path: Path, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
            raise ValueError(
                f"{fixture_path}: schema_version must be {FIXTURE_SCHEMA_VERSION!r}; got {payload.get('schema_version')!r}"
            )
        backend = payload.get("backend")
        if not isinstance(backend, dict):
            raise ValueError(f"{fixture_path}: backend must be an object")
        backend_id = str(backend.get("backend_id") or "")
        if backend_id not in ALLOWED_BACKEND_IDS:
            raise ValueError(f"{fixture_path}: backend.backend_id must be one of {sorted(ALLOWED_BACKEND_IDS)}")

        entities = payload.get("entities")
        pairs = payload.get("pairs")
        if not isinstance(entities, dict) or not isinstance(pairs, dict):
            raise ValueError(f"{fixture_path}: entities and pairs must both be objects")

        self.fixture_path = fixture_path
        self.backend_id = backend_id
        self.index_ref = str(backend.get("index_ref") or "") or None
        self.cache_ref = str(backend.get("cache_ref") or fixture_path.name)
        self.max_diff_tokens = int(backend.get("max_diff_tokens") or 0) or None
        self.max_clause_freq = int(backend.get("max_clause_freq") or 0) or None
        self.note = str(backend.get("note") or "") or None
        self._entities = {
            normalize_term(str(key)): FixtureCountRecord.from_payload(value, kind="entity", key=str(key))
            for key, value in entities.items()
        }
        self._pairs = {
            str(key): FixtureCountRecord.from_payload(value, kind="pair", key=str(key))
            for key, value in pairs.items()
        }

    @classmethod
    def from_path(cls, fixture_path: Path) -> "FixtureCorpusCountBackend":
        payload = load_json(fixture_path)
        if not isinstance(payload, dict):
            raise ValueError(f"{fixture_path}: fixture payload must be an object")
        return cls(fixture_path, payload)

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "index_ref": self.index_ref,
            "cache_ref": self.cache_ref,
            "max_diff_tokens": self.max_diff_tokens,
            "max_clause_freq": self.max_clause_freq,
            "fixture_path": str(self.fixture_path),
            "entity_fixture_count": len(self._entities),
            "pair_fixture_count": len(self._pairs),
            "note": self.note,
        }

    def count_entity(self, entity: str) -> CorpusCountResult:
        normalized = normalize_term(entity)
        record = self._entities.get(normalized)
        status = record.status if record is not None else "cache_miss"
        count = record.count if record is not None else None
        approximate = record.approximate if record is not None else False
        note = record.note if record is not None else "fixture cache miss"
        metadata = dict(record.metadata or {}) if record is not None else {}
        return CorpusCountResult(
            raw_count=count,
            provenance=CorpusCountProvenance(
                backend_id=self.backend_id,
                query=normalized,
                query_kind="entity_frequency",
                status=status,
                index_ref=self.index_ref,
                cache_ref=self.cache_ref,
                approximate=approximate,
                max_diff_tokens=self.max_diff_tokens,
                max_clause_freq=self.max_clause_freq,
                note=note,
                metadata=metadata,
            ),
        )

    def count_pair(self, left: str, right: str) -> CorpusCountResult:
        query = pair_query(left, right)
        storage_key = pair_storage_key(left, right)
        record = self._pairs.get(storage_key)
        status = record.status if record is not None else "cache_miss"
        count = record.count if record is not None else None
        approximate = record.approximate if record is not None else False
        note = record.note if record is not None else "fixture cache miss"
        metadata = dict(record.metadata or {}) if record is not None else {}
        return CorpusCountResult(
            raw_count=count,
            provenance=CorpusCountProvenance(
                backend_id=self.backend_id,
                query=query,
                query_kind="entity_pair_cooccurrence",
                status=status,
                index_ref=self.index_ref,
                cache_ref=self.cache_ref,
                approximate=approximate,
                max_diff_tokens=self.max_diff_tokens,
                max_clause_freq=self.max_clause_freq,
                note=note,
                metadata=metadata,
            ),
        )


class MissingCorpusCountBackend(CorpusCountBackendPort):
    """Explicit fail-closed backend used when no direct corpus-count source is configured."""

    def __init__(self, *, candidates_path: Path) -> None:
        self.candidates_path = candidates_path

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": "infini_gram_local_count",
            "index_ref": f"local://{self.candidates_path.stem}",
            "cache_ref": None,
            "status": "local_index_absent",
            "candidate_path": str(self.candidates_path),
            "note": "No direct corpus-count sidecar was discovered. Candidate-text proxy counting is forbidden.",
        }

    def count_entity(self, entity: str) -> CorpusCountResult:
        normalized = normalize_term(entity)
        return CorpusCountResult(
            raw_count=None,
            provenance=CorpusCountProvenance(
                backend_id="infini_gram_local_count",
                query=normalized,
                query_kind="entity_frequency",
                status="local_index_absent",
                index_ref=f"local://{self.candidates_path.stem}",
                cache_ref=None,
                approximate=False,
                note="No local Infini-gram-compatible index or cache snapshot is configured.",
            ),
        )

    def count_pair(self, left: str, right: str) -> CorpusCountResult:
        return CorpusCountResult(
            raw_count=None,
            provenance=CorpusCountProvenance(
                backend_id="infini_gram_local_count",
                query=pair_query(left, right),
                query_kind="entity_pair_cooccurrence",
                status="local_index_absent",
                index_ref=f"local://{self.candidates_path.stem}",
                cache_ref=None,
                approximate=False,
                note="No local Infini-gram-compatible index or cache snapshot is configured.",
            ),
        )


def fixture_sidecar_candidates(candidates_path: Path) -> tuple[Path, ...]:
    return (
        candidates_path.with_suffix(candidates_path.suffix + ".corpus_counts.json"),
        candidates_path.with_name(f"{candidates_path.stem}.corpus_counts.json"),
    )


def infinigram_cache_path(candidates_path: Path) -> Path:
    return candidates_path.with_suffix(candidates_path.suffix + ".infinigram_cache.json")


class InfinigramApiBackend(CorpusCountBackendPort):
    """Live Infini-gram REST count backend with on-disk JSON cache.

    Entity counts come from infini-gram exact count queries. Pair counts use
    AND queries, which infini-gram returns as upper-bound (approx=true) values.
    For the corpus-support binning axis we treat the upper bound as the count
    and record the raw infini-gram approx flag in metadata so the provenance
    chain stays auditable.
    """

    BACKEND_ID = "infini_gram_api_count"

    def __init__(
        self,
        *,
        candidates_path: Path,
        index: str,
        endpoint: str,
        cache_path: Path,
        timeout: float = INFINIGRAM_DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = INFINIGRAM_DEFAULT_MAX_RETRIES,
    ) -> None:
        self.candidates_path = candidates_path
        self.index = index
        self.endpoint = endpoint
        self.cache_path = cache_path
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._cache_lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_dirty = False
        self._load_cache()

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.BACKEND_ID,
            "index_ref": f"infini_gram://{self.index}",
            "cache_ref": str(self.cache_path),
            "endpoint": self.endpoint,
            "cache_entries": len(self._cache),
            "cache_schema_version": INFINIGRAM_CACHE_SCHEMA_VERSION,
            "approximate_pair_semantics": "infini_gram AND queries return an upper-bound count; treated as count for axis binning, raw approx flag stored in metadata",
            "note": "Live Infini-gram REST backend with persistent JSON cache.",
        }

    def count_entity(self, entity: str) -> CorpusCountResult:
        normalized = normalize_term(entity)
        if not normalized:
            return self._empty_result(query=normalized, kind="entity_frequency", reason="empty_normalized_query")
        record = self._fetch_or_query(query=normalized, kind="entity_frequency")
        return self._materialize(query=normalized, kind="entity_frequency", record=record)

    def count_pair(self, left: str, right: str) -> CorpusCountResult:
        query = pair_query(left, right)
        record = self._fetch_or_query(query=query, kind="entity_pair_cooccurrence")
        return self._materialize(query=query, kind="entity_pair_cooccurrence", record=record)

    # --- prefetch ---
    def warmup(self, *, entities: list[str], pairs: list[str], parallelism: int = INFINIGRAM_DEFAULT_PARALLELISM) -> dict[str, int]:
        normalized_entities = sorted({normalize_term(entity) for entity in entities if normalize_term(entity)})
        normalized_pairs = sorted(set(pairs))
        todo: list[tuple[str, str]] = []
        with self._cache_lock:
            for query in normalized_entities:
                if self._cache_key("entity_frequency", query) not in self._cache:
                    todo.append(("entity_frequency", query))
            for query in normalized_pairs:
                if self._cache_key("entity_pair_cooccurrence", query) not in self._cache:
                    todo.append(("entity_pair_cooccurrence", query))
        if not todo:
            return {"queued": 0, "fetched": 0, "errors": 0, "cached_total": len(self._cache)}
        fetched = 0
        errors = 0
        flush_every = max(50, len(todo) // 20)
        with ThreadPoolExecutor(max_workers=max(1, int(parallelism))) as executor:
            futures = [executor.submit(self._fetch_and_cache, kind, query) for kind, query in todo]
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    future.result()
                    fetched += 1
                except Exception:  # pragma: no cover - network failures bookkeeping
                    errors += 1
                if index % flush_every == 0:
                    self._flush_cache()
        self._flush_cache()
        return {"queued": len(todo), "fetched": fetched, "errors": errors, "cached_total": len(self._cache)}

    # --- internals ---
    @staticmethod
    def _cache_key(kind: str, query: str) -> str:
        return f"{kind}::{query}"

    def _empty_result(self, *, query: str, kind: str, reason: str) -> CorpusCountResult:
        return CorpusCountResult(
            raw_count=None,
            provenance=CorpusCountProvenance(
                backend_id=self.BACKEND_ID,
                query=query,
                query_kind=kind,
                status=f"excluded_{reason}",
                index_ref=f"infini_gram://{self.index}",
                cache_ref=str(self.cache_path),
                approximate=False,
                note=f"infinigram backend skipped query: {reason}",
            ),
        )

    def _materialize(self, *, query: str, kind: str, record: dict[str, Any]) -> CorpusCountResult:
        raw_count = record.get("count")
        infini_approx = bool(record.get("infinigram_approx", False))
        # We faithfully record the infini-gram approx flag in metadata but normalize the
        # `approximate` field to False so the upper-bound pair count is usable for the
        # corpus-support binning axis (entity counts remain exact).
        metadata = {
            "infinigram_approx": infini_approx,
            "infinigram_index": self.index,
            "infinigram_endpoint": self.endpoint,
            "fetched_at": record.get("fetched_at"),
            "latency_ms": record.get("latency_ms"),
        }
        if record.get("error"):
            metadata["error"] = record["error"]
        if raw_count is None:
            status = "excluded_missing_counts"
        elif kind == "entity_pair_cooccurrence" and infini_approx:
            status = "resolved"
        else:
            status = "resolved"
        note_parts = []
        if kind == "entity_pair_cooccurrence" and infini_approx:
            note_parts.append("infini_gram AND query returned upper-bound count")
        if record.get("error"):
            note_parts.append(f"last_error={record['error']}")
        return CorpusCountResult(
            raw_count=int(raw_count) if isinstance(raw_count, int) else None,
            provenance=CorpusCountProvenance(
                backend_id=self.BACKEND_ID,
                query=query,
                query_kind=kind,
                status=status,
                index_ref=f"infini_gram://{self.index}",
                cache_ref=str(self.cache_path),
                approximate=False,
                note="; ".join(note_parts) if note_parts else None,
                metadata=metadata,
            ),
        )

    def _fetch_or_query(self, *, query: str, kind: str) -> dict[str, Any]:
        key = self._cache_key(kind, query)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        record = self._call_api(query=query, kind=kind)
        with self._cache_lock:
            self._cache[key] = record
            self._cache_dirty = True
        return record

    def _fetch_and_cache(self, kind: str, query: str) -> None:
        key = self._cache_key(kind, query)
        with self._cache_lock:
            if key in self._cache:
                return
        record = self._call_api(query=query, kind=kind)
        with self._cache_lock:
            self._cache[key] = record
            self._cache_dirty = True

    def _call_api(self, *, query: str, kind: str) -> dict[str, Any]:
        payload = json.dumps({"index": self.index, "query_type": "count", "query": query}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                request = urllib.request.Request(self.endpoint, data=payload, headers=headers, method="POST")
                start = time.monotonic()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - controlled endpoint
                    body = response.read()
                latency_ms = (time.monotonic() - start) * 1000.0
                parsed = json.loads(body.decode("utf-8"))
                if not isinstance(parsed, dict) or "count" not in parsed:
                    raise ValueError(f"infini-gram response missing 'count': {parsed}")
                return {
                    "count": int(parsed["count"]),
                    "infinigram_approx": bool(parsed.get("approx", False)),
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "latency_ms": round(float(latency_ms), 2),
                    "kind": kind,
                    "query": query,
                }
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
                continue
        return {
            "count": None,
            "infinigram_approx": False,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": kind,
            "query": query,
            "error": last_error or "unknown_error",
        }

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or payload.get("schema_version") != INFINIGRAM_CACHE_SCHEMA_VERSION:
            return
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return
        if payload.get("index") and payload.get("index") != self.index:
            return
        with self._cache_lock:
            for key, value in entries.items():
                if isinstance(value, dict):
                    self._cache[str(key)] = value

    def _flush_cache(self) -> None:
        with self._cache_lock:
            if not self._cache_dirty:
                return
            entries_snapshot = dict(self._cache)
            self._cache_dirty = False
        payload = {
            "schema_version": INFINIGRAM_CACHE_SCHEMA_VERSION,
            "index": self.index,
            "endpoint": self.endpoint,
            "entries": entries_snapshot,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_name(f".{self.cache_path.name}.tmp-{uuid4().hex}")
        try:
            temp_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            temp_path.replace(self.cache_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise


class InfinigramLocalEngineBackend(CorpusCountBackendPort):
    """Local Infini-gram count backend using ``infini_gram.engine.InfiniGramEngine``.

    Tokenizes queries with a configured HuggingFace tokenizer (OLMo by default
    for ``v4_dolmasample_olmo``), then issues exact ``count`` for single-term
    queries and CNF ``count_cnf`` (AND) for entity-pair co-occurrence. Pair
    queries return upper-bound counts; the engine's approx flag is captured in
    metadata while the ``approximate`` field is normalized to False so the
    upper bound can serve the corpus-support binning axis.
    """

    BACKEND_ID = "infini_gram_local_count"

    def __init__(
        self,
        *,
        candidates_path: Path,
        index_dir: Path,
        tokenizer_name: str,
        cache_path: Path,
        max_clause_freq: int = 50000,
        max_diff_tokens: int = 100,
    ) -> None:
        self.candidates_path = candidates_path
        self.index_dir = index_dir
        self.tokenizer_name = tokenizer_name
        self.cache_path = cache_path
        self.max_clause_freq = int(max_clause_freq)
        self.max_diff_tokens = int(max_diff_tokens)
        self._cache_lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_dirty = False
        self._engine_lock = threading.Lock()
        self._engine = None  # lazy
        self._tokenizer = None  # lazy
        self._eos_token_id: int | None = None
        self._vocab_size: int | None = None
        self._load_cache()

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        with self._engine_lock:
            if self._engine is not None:
                return
            from infini_gram.engine import InfiniGramEngine  # type: ignore
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                add_bos_token=False,
                add_eos_token=False,
            )
            vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
            eos_token_id = int(getattr(tokenizer, "eos_token_id", 0) or 0)
            engine = InfiniGramEngine(
                index_dir=str(self.index_dir),
                eos_token_id=eos_token_id,
                vocab_size=vocab_size,
            )
            self._tokenizer = tokenizer
            self._engine = engine
            self._eos_token_id = eos_token_id
            self._vocab_size = vocab_size

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.BACKEND_ID,
            "index_ref": f"infini_gram_local://{self.index_dir.name}",
            "cache_ref": str(self.cache_path),
            "index_dir": str(self.index_dir),
            "tokenizer_name": self.tokenizer_name,
            "max_clause_freq": self.max_clause_freq,
            "max_diff_tokens": self.max_diff_tokens,
            "cache_entries": len(self._cache),
            "cache_schema_version": INFINIGRAM_CACHE_SCHEMA_VERSION,
            "approximate_pair_semantics": "infini_gram CNF AND queries return an upper-bound count (max_clause_freq capped); treated as count for axis binning",
            "note": "Local Infini-gram engine backed by an on-disk index; persistent JSON cache mirrors API backend schema.",
        }

    @staticmethod
    def _cache_key(kind: str, query: str) -> str:
        return f"{kind}::{query}"

    def _empty_result(self, *, query: str, kind: str, reason: str) -> CorpusCountResult:
        return CorpusCountResult(
            raw_count=None,
            provenance=CorpusCountProvenance(
                backend_id=self.BACKEND_ID,
                query=query,
                query_kind=kind,
                status=f"excluded_{reason}",
                index_ref=f"infini_gram_local://{self.index_dir.name}",
                cache_ref=str(self.cache_path),
                approximate=False,
                note=f"infinigram local backend skipped query: {reason}",
            ),
        )

    def _materialize(self, *, query: str, kind: str, record: dict[str, Any]) -> CorpusCountResult:
        raw_count = record.get("count")
        infini_approx = bool(record.get("infinigram_approx", False))
        metadata = {
            "infinigram_approx": infini_approx,
            "infinigram_index": self.index_dir.name,
            "tokenizer_name": self.tokenizer_name,
            "input_token_count": record.get("input_token_count"),
            "max_clause_freq": self.max_clause_freq,
            "max_diff_tokens": self.max_diff_tokens,
        }
        if record.get("error"):
            metadata["error"] = record["error"]
        if raw_count is None:
            status = "excluded_missing_counts"
        else:
            status = "resolved"
        notes = []
        if kind == "entity_pair_cooccurrence" and infini_approx:
            notes.append("infini_gram CNF AND query returned upper-bound count")
        if record.get("error"):
            notes.append(f"last_error={record['error']}")
        return CorpusCountResult(
            raw_count=int(raw_count) if isinstance(raw_count, int) else None,
            provenance=CorpusCountProvenance(
                backend_id=self.BACKEND_ID,
                query=query,
                query_kind=kind,
                status=status,
                index_ref=f"infini_gram_local://{self.index_dir.name}",
                cache_ref=str(self.cache_path),
                approximate=False,
                max_clause_freq=self.max_clause_freq,
                max_diff_tokens=self.max_diff_tokens,
                note="; ".join(notes) if notes else None,
                metadata=metadata,
            ),
        )

    def _tokenize_term(self, term: str) -> list[int]:
        assert self._tokenizer is not None  # set by _ensure_engine
        # Encode with a leading space to match how multi-word entities appear mid-sentence
        # in pretraining text. Strip BOS/EOS by tokenizer config (add_bos/eos=False).
        ids = self._tokenizer.encode(" " + term)
        return [int(token_id) for token_id in ids if isinstance(token_id, int)]

    def _execute_count(self, *, query: str, kind: str) -> dict[str, Any]:
        self._ensure_engine()
        engine = self._engine
        assert engine is not None  # populated by _ensure_engine
        try:
            if kind == "entity_frequency":
                token_ids = self._tokenize_term(query)
                if not token_ids:
                    return {"count": None, "infinigram_approx": False, "input_token_count": 0, "error": "empty_token_ids"}
                response = engine.count(input_ids=token_ids)
                count = getattr(response, "count", None)
                if count is None and isinstance(response, dict):
                    count = response.get("count")
                approx = bool(getattr(response, "approx", False))
                return {
                    "count": int(count) if isinstance(count, int) else None,
                    "infinigram_approx": approx,
                    "input_token_count": len(token_ids),
                    "kind": kind,
                    "query": query,
                }
            if kind == "entity_pair_cooccurrence":
                # query format: "<left> AND <right>"
                if " AND " not in query:
                    return {"count": None, "infinigram_approx": False, "error": "malformed_pair_query"}
                left, right = query.split(" AND ", 1)
                left_ids = self._tokenize_term(left)
                right_ids = self._tokenize_term(right)
                if not left_ids or not right_ids:
                    return {"count": None, "infinigram_approx": False, "input_token_count": len(left_ids) + len(right_ids), "error": "empty_token_ids"}
                cnf = [[left_ids], [right_ids]]
                response = engine.count_cnf(
                    cnf=cnf,
                    max_clause_freq=self.max_clause_freq,
                    max_diff_tokens=self.max_diff_tokens,
                )
                count = getattr(response, "count", None)
                if count is None and isinstance(response, dict):
                    count = response.get("count")
                approx = bool(getattr(response, "approx", False))
                return {
                    "count": int(count) if isinstance(count, int) else None,
                    "infinigram_approx": approx,
                    "input_token_count": len(left_ids) + len(right_ids),
                    "kind": kind,
                    "query": query,
                }
            return {"count": None, "infinigram_approx": False, "error": f"unknown_kind:{kind}"}
        except Exception as exc:  # pragma: no cover - engine error reporting
            return {"count": None, "infinigram_approx": False, "error": f"{type(exc).__name__}: {exc}"}

    def _fetch_or_query(self, *, query: str, kind: str) -> dict[str, Any]:
        key = self._cache_key(kind, query)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        record = self._execute_count(query=query, kind=kind)
        with self._cache_lock:
            self._cache[key] = record
            self._cache_dirty = True
        return record

    def count_entity(self, entity: str) -> CorpusCountResult:
        normalized = normalize_term(entity)
        if not normalized:
            return self._empty_result(query=normalized, kind="entity_frequency", reason="empty_normalized_query")
        record = self._fetch_or_query(query=normalized, kind="entity_frequency")
        return self._materialize(query=normalized, kind="entity_frequency", record=record)

    def count_pair(self, left: str, right: str) -> CorpusCountResult:
        query = pair_query(left, right)
        record = self._fetch_or_query(query=query, kind="entity_pair_cooccurrence")
        return self._materialize(query=query, kind="entity_pair_cooccurrence", record=record)

    def warmup(self, *, entities: list[str], pairs: list[str], parallelism: int = INFINIGRAM_DEFAULT_PARALLELISM) -> dict[str, int]:
        self._ensure_engine()
        normalized_entities = sorted({normalize_term(entity) for entity in entities if normalize_term(entity)})
        normalized_pairs = sorted(set(pairs))
        todo: list[tuple[str, str]] = []
        with self._cache_lock:
            for query in normalized_entities:
                if self._cache_key("entity_frequency", query) not in self._cache:
                    todo.append(("entity_frequency", query))
            for query in normalized_pairs:
                if self._cache_key("entity_pair_cooccurrence", query) not in self._cache:
                    todo.append(("entity_pair_cooccurrence", query))
        if not todo:
            return {"queued": 0, "fetched": 0, "errors": 0, "cached_total": len(self._cache)}
        fetched = 0
        errors = 0
        flush_every = max(200, len(todo) // 20)

        def worker(item: tuple[str, str]) -> None:
            kind, query = item
            record = self._execute_count(query=query, kind=kind)
            with self._cache_lock:
                self._cache[self._cache_key(kind, query)] = record
                self._cache_dirty = True

        with ThreadPoolExecutor(max_workers=max(1, int(parallelism))) as executor:
            futures = [executor.submit(worker, item) for item in todo]
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    future.result()
                    fetched += 1
                except Exception:
                    errors += 1
                if index % flush_every == 0:
                    self._flush_cache()
        self._flush_cache()
        return {"queued": len(todo), "fetched": fetched, "errors": errors, "cached_total": len(self._cache)}

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or payload.get("schema_version") != INFINIGRAM_CACHE_SCHEMA_VERSION:
            return
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return
        if payload.get("index") and payload.get("index") != self.index_dir.name:
            return
        with self._cache_lock:
            for key, value in entries.items():
                if isinstance(value, dict):
                    self._cache[str(key)] = value

    def _flush_cache(self) -> None:
        with self._cache_lock:
            if not self._cache_dirty:
                return
            entries_snapshot = dict(self._cache)
            self._cache_dirty = False
        payload = {
            "schema_version": INFINIGRAM_CACHE_SCHEMA_VERSION,
            "index": self.index_dir.name,
            "tokenizer_name": self.tokenizer_name,
            "entries": entries_snapshot,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_name(f".{self.cache_path.name}.tmp-{uuid4().hex}")
        try:
            temp_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            temp_path.replace(self.cache_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise


def _read_backend_config_sidecar(candidates_path: Path) -> dict[str, Any] | None:
    sidecar = candidates_path.with_suffix(candidates_path.suffix + ".corpus_backend.json")
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def build_corpus_count_backend(candidates_path: Path) -> CorpusCountBackendPort:
    for sidecar_path in fixture_sidecar_candidates(candidates_path):
        if sidecar_path.exists():
            return FixtureCorpusCountBackend.from_path(sidecar_path)
    backend_choice = (os.environ.get(INFINIGRAM_BACKEND_ENV) or "").strip().lower()
    cache_path = infinigram_cache_path(candidates_path)

    sidecar_config = _read_backend_config_sidecar(candidates_path) or {}
    sidecar_choice = str(sidecar_config.get("backend") or "").strip().lower()
    sidecar_index_dir = sidecar_config.get("index_dir")
    sidecar_tokenizer = sidecar_config.get("tokenizer")

    local_index_dir_str = (os.environ.get(INFINIGRAM_LOCAL_INDEX_DIR_ENV) or "").strip()
    if not local_index_dir_str and sidecar_choice in {"infini_gram_local", "infini_gram_local_count", "local"} and isinstance(sidecar_index_dir, str):
        local_index_dir_str = sidecar_index_dir.strip()
    local_index_dir = Path(local_index_dir_str) if local_index_dir_str else None
    use_local_backend = (
        backend_choice in {"infini_gram_local", "infini_gram_local_count", "local"}
        or sidecar_choice in {"infini_gram_local", "infini_gram_local_count", "local"}
        or (local_index_dir is not None and local_index_dir.exists())
    )
    if use_local_backend and local_index_dir is not None and local_index_dir.exists():
        tokenizer_name = (os.environ.get(INFINIGRAM_LOCAL_TOKENIZER_ENV) or "").strip()
        if not tokenizer_name and isinstance(sidecar_tokenizer, str) and sidecar_tokenizer.strip():
            tokenizer_name = sidecar_tokenizer.strip()
        if not tokenizer_name:
            tokenizer_name = INFINIGRAM_DEFAULT_LOCAL_TOKENIZER
        return InfinigramLocalEngineBackend(
            candidates_path=candidates_path,
            index_dir=local_index_dir,
            tokenizer_name=tokenizer_name,
            cache_path=cache_path,
        )
    use_api_backend = backend_choice in {"infini_gram_api", "infini_gram_api_count", "infinigram", "api"} or cache_path.exists()
    if use_api_backend:
        index = (os.environ.get(INFINIGRAM_INDEX_ENV) or INFINIGRAM_DEFAULT_INDEX).strip() or INFINIGRAM_DEFAULT_INDEX
        endpoint = (os.environ.get(INFINIGRAM_ENDPOINT_ENV) or INFINIGRAM_DEFAULT_ENDPOINT).strip() or INFINIGRAM_DEFAULT_ENDPOINT
        return InfinigramApiBackend(
            candidates_path=candidates_path,
            index=index,
            endpoint=endpoint,
            cache_path=cache_path,
        )
    return MissingCorpusCountBackend(candidates_path=candidates_path)


def serialize_count_result(result: CorpusCountResult) -> dict[str, Any]:
    payload = asdict(result.provenance)
    payload["raw_count"] = result.raw_count
    payload["excluded"] = result.raw_count is None or result.provenance.approximate or result.provenance.status.startswith("excluded")
    return payload
