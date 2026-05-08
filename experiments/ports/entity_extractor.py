"""Entity extractor port.

Returns a list of entity strings for a single text. Implementations may use
regex heuristics (legacy) or LLM-based extraction (QuCo-extractor-0.5B).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

EntityRole = Literal["question", "declarative"]


class EntityExtractorPort(ABC):
    """Extract a set of entity strings from a free-form text."""

    @abstractmethod
    def extract(self, text: str, *, role: EntityRole = "declarative") -> list[str]:
        """Return entities ordered by appearance, deduplicated, lower-cased.

        ``role`` lets implementations apply a different prompt or rule for
        question-style vs. declarative sentences (e.g., QuCo emits
        ``[[entity, relation]]`` for questions and ``[[head, relation,
        tail]]`` for declaratives).
        """

    @abstractmethod
    def describe(self) -> dict[str, str]:
        """Return provenance metadata to record on every emitted artifact."""
