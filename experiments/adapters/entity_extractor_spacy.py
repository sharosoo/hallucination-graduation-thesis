"""spaCy NER-based entity extractor.

Uses ``en_core_web_lg`` (or any spaCy pipeline with a NER component) to
identify named entities in the candidate / question text. Entity types are
filtered to the set most relevant for factoid-QA hallucination detection
(PERSON, ORG, GPE, LOC, DATE, EVENT, WORK_OF_ART, FAC, NORP, PRODUCT,
LANGUAGE, LAW). All other types (e.g. CARDINAL, ORDINAL, MONEY, PERCENT,
QUANTITY, TIME) are dropped because they are usually not the target entity
of a factoid answer.

For very short candidate texts that contain no recognized named entity
(``"Apples"``, ``"Bach"``), spaCy may return nothing. In that case we fall
back to the noun chunks (``en_core_web_lg`` always tags noun chunks via the
parser). If still empty, the cleaned text itself is added as a single entity
— the candidate IS the answer in HaluEval-QA / TruthfulQA, so it should
always have at least one entity to query against the corpus.

Implementation notes.

* spaCy is small (~600MB for ``_lg``) and runs CPU-only, so we do not need
  GPU. It also has no chat / prompt overhead — pure pipeline call.
* The spaCy ``Language`` object is loaded once and reused across calls.
* Batched inference uses ``nlp.pipe()`` which streams documents through the
  pipeline in parallel under the hood.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from experiments.ports.entity_extractor import EntityExtractorPort, EntityRole


DEFAULT_MODEL_NAME = "en_core_web_lg"
DEFAULT_BATCH_SIZE = 64

# spaCy entity labels we keep. See https://spacy.io/models/en#en_core_web_lg.
DEFAULT_KEEP_LABELS: tuple[str, ...] = (
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "DATE",
    "EVENT",
    "WORK_OF_ART",
    "FAC",
    "NORP",
    "PRODUCT",
    "LANGUAGE",
    "LAW",
)

_DISCOURSE_PREFIXES = (
    "yes, ",
    "no, ",
    "actually, ",
    "actually,",
    "so the answer is ",
    "the answer is ",
)


def _strip_discourse_prefix(text: str) -> str:
    lower = text.lower()
    for prefix in _DISCOURSE_PREFIXES:
        if lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text


_LEADING_ARTICLES = ("a ", "an ", "the ")


def _normalize_unique(entities: list[str]) -> list[str]:
    """Lower-case, strip punctuation noise, drop leading articles, dedupe."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in entities:
        if not raw:
            continue
        norm = re.sub(r"\s+", " ", raw.strip().lower())
        norm = re.sub(r"[^a-z0-9'\- ]+", "", norm).strip()
        for article in _LEADING_ARTICLES:
            if norm.startswith(article):
                norm = norm[len(article):]
                break
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


@dataclass
class SpacyEntityExtractor(EntityExtractorPort):
    """spaCy-based named entity extractor.

    Parameters
    ----------
    model_name:
        spaCy pipeline to load (``"en_core_web_lg"`` recommended).
    keep_labels:
        Entity types to keep. All others dropped.
    batch_size:
        Documents per ``nlp.pipe`` batch.
    add_noun_chunks_fallback:
        When the NER component returns nothing for a short text, fall back to
        noun chunks (covers cases like ``"Apples"`` where there is no NER
        match but the chunk parser identifies a head noun).
    add_text_fallback:
        Final fallback: if both NER and noun-chunks return nothing, add the
        cleaned text itself as a single entity (only for short candidates,
        controlled by ``short_word_limit``).
    short_word_limit:
        Word-count threshold below which the text-itself fallback applies.
    """

    model_name: str = DEFAULT_MODEL_NAME
    keep_labels: tuple[str, ...] = DEFAULT_KEEP_LABELS
    batch_size: int = DEFAULT_BATCH_SIZE
    add_noun_chunks_fallback: bool = True
    add_text_fallback: bool = True
    short_word_limit: int = 6
    _nlp: Any = field(default=None, repr=False, init=False)

    version: str = "spacy_en_core_web_lg_v1"

    # ------------------------------------------------------------------ #
    # Port surface
    # ------------------------------------------------------------------ #

    def extract(self, text: str, *, role: EntityRole = "declarative") -> list[str]:
        return self.extract_many([text], [role])[0]

    def extract_many(
        self,
        texts: list[str],
        roles: list[EntityRole],
    ) -> list[list[str]]:
        if len(texts) != len(roles):
            raise ValueError("texts and roles must have the same length")
        self._ensure_loaded()
        # Pre-clean discourse prefixes on declarative side to give spaCy a
        # cleaner sentence to parse.
        cleaned: list[str] = []
        for raw_text, role in zip(texts, roles):
            text = (raw_text or "").strip()
            if role == "declarative" and text:
                text = _strip_discourse_prefix(text)
            cleaned.append(text)

        # Empty inputs short-circuit. Track which indices need processing.
        active_indices = [i for i, t in enumerate(cleaned) if t]
        active_texts = [cleaned[i] for i in active_indices]
        results: list[list[str]] = [[] for _ in cleaned]
        if not active_texts:
            return results

        # Batched pipeline call.
        for idx, doc in zip(
            active_indices,
            self._nlp.pipe(active_texts, batch_size=self.batch_size),
        ):
            text = cleaned[idx]
            entities = self._entities_from_doc(doc)
            if not entities and self.add_text_fallback:
                if len(text.split()) <= self.short_word_limit:
                    entities = [text]
            results[idx] = _normalize_unique(entities)
        return results

    def describe(self) -> dict[str, str]:
        return {
            "entity_extractor_version": self.version,
            "entity_extractor_kind": "spacy_ner",
            "entity_extractor_model_ref": self.model_name,
            "entity_extractor_keep_labels": ",".join(self.keep_labels),
            "entity_extractor_short_word_limit": str(self.short_word_limit),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "SpacyEntityExtractor requires the 'spacy' package and the "
                "model package to be installed. Run "
                "`uv pip install spacy` and `python -m spacy download "
                f"{self.model_name}`."
            ) from exc
        self._nlp = spacy.load(self.model_name)

    def _entities_from_doc(self, doc: Any) -> list[str]:
        keep = set(self.keep_labels)
        ents: list[str] = []
        for ent in doc.ents:
            if ent.label_ in keep:
                ents.append(ent.text)
        if ents:
            return ents
        if self.add_noun_chunks_fallback:
            chunks: list[str] = []
            for chunk in getattr(doc, "noun_chunks", []):
                # Drop pronoun-only and very short chunks.
                token_text = chunk.text.strip()
                if not token_text:
                    continue
                if len(token_text) <= 1:
                    continue
                chunks.append(token_text)
            if chunks:
                return chunks
        return []
