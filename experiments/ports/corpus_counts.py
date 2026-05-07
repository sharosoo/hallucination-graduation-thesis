"""Typed corpus count backend port for thesis-valid corpus axes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CorpusCountProvenance:
    """Per-query provenance for direct corpus count evidence."""

    backend_id: str
    query: str
    query_kind: str
    status: str
    index_ref: str | None = None
    cache_ref: str | None = None
    approximate: bool = False
    fallback_backend_id: str | None = None
    max_diff_tokens: int | None = None
    max_clause_freq: int | None = None
    note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusCountResult:
    """Result of a direct corpus count query."""

    raw_count: int | None
    provenance: CorpusCountProvenance


class CorpusCountBackendPort(ABC):
    """Abstract interface for direct corpus-backed entity and pair counts."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return backend metadata for artifact-level reporting."""

    @abstractmethod
    def count_entity(self, entity: str) -> CorpusCountResult:
        """Return a direct corpus count for a single normalized entity."""

    @abstractmethod
    def count_pair(self, left: str, right: str) -> CorpusCountResult:
        """Return a direct corpus co-occurrence count for a normalized entity pair."""
