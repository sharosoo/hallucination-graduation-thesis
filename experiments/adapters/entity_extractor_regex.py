"""Regex-based entity extractor (legacy, kept as fallback)."""

from __future__ import annotations

from experiments.adapters import corpus_features as _legacy
from experiments.ports.entity_extractor import EntityExtractorPort, EntityRole


class RegexEntityExtractor(EntityExtractorPort):
    """Wraps the regex-based ``phrase_candidates`` heuristic.

    Behaviour matches the pre-pivot extractor exactly so old artifacts can be
    reproduced. New runs should prefer ``QucoEntityExtractor``.
    """

    version = "regex_v1"

    def extract(self, text: str, *, role: EntityRole = "declarative") -> list[str]:
        # Role is ignored — the regex heuristic does not differentiate.
        return _legacy.phrase_candidates(text)

    def describe(self) -> dict[str, str]:
        return {
            "entity_extractor_version": self.version,
            "entity_extractor_kind": "regex_phrase_candidates",
            "entity_extractor_model_ref": "(none)",
        }
