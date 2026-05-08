"""QuCo-extractor-0.5B based entity extractor.

Uses ``ZhishanQ/QuCo-extractor-0.5B`` (Qwen2.5-0.5B-Instruct distilled from
GPT-4o-mini, released by the QuCo-RAG authors) to produce knowledge triplets
``(head, relation, tail)`` per sentence. We keep the ``head`` and ``tail`` as
entities of interest and discard ``relation`` text — same choice as QuCo-RAG
(see ``experiments/literature/evidence_notes/quco_extractor_adoption.md``).

The class is lazy: the model is loaded on first ``extract`` call, so unit
tests that never call extract() do not require the model on disk.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from experiments.ports.entity_extractor import EntityExtractorPort, EntityRole

# --------------------------------------------------------------------------- #
# Prompt templates (ported verbatim from QuCo-RAG src/prompt_templates.json).
# --------------------------------------------------------------------------- #

_DECLARATIVE_PROMPT = """\
Extract the knowledge triples that convey core semantic information from the given sentence, strictly following the output format used in the examples.
The sentence may be a question or a declarative sentence.
* **If the sentence is a question**, return [[entity, relation]], because you don't know the answer.
  * This question could be a multi-hop question. In that case, **only consider the first relation**.
  * You don't need to include any query about entities **not explicitly mentioned** in the question.
* **If the sentence contains one knowledge triple**, return a list in the format:
  [[head_entity_1, relation_1, tail_entity_1]]
* **If the sentence contains no factual semantic information involving entities**, return an empty list: `[]`
* **If the sentence contains more than one knowledge triple**, return a list in the format:
  [[head_entity_1, relation_1, tail_entity_1], [head_entity_2, relation_2, tail_entity_2], ...]
---
**Example 1:**
Sentence:
Which film came out first, Kumbasaram or Mystery Of The 13Th Guest?
entities:
[["Kumbasaram", "came out"], ["Mystery of the 13th Guest", "came out"]]

**Example 2:**
Sentence:
Kumbasaram was released in 2017.
entities:
[["Kumbasaram", "released in", "2017"]]

**Example 3:**
Sentence:
Thus, Kumbasaram came out first.
entities:
[]

**Example 4:**
Sentence:
Diane Meyer Simon's husband is George F. Simon.
entities:
[["Diane Meyer Simon", "husband", "George F. Simon"]]

**Example 5:**
Sentence:
Rajiv Rai was born in Mumbai, Maharashtra, India.
entities:
[["Rajiv Rai", "place of birth", "Mumbai, Maharashtra, India"]]

Your Task:
Sentence:{}
entities:"""

_QUESTION_PROMPT = """\
Extract the key entities that convey core semantic information from the given question, strictly following the output format used in the examples.
---
**Example 1:**
Sentence:
Which film came out first, Kumbasaram or Mystery Of The 13Th Guest?
entities:
[["Kumbasaram"], ["Mystery of the 13th Guest"]]

**Example 2:**
Sentence:
Who is the mother of the director of film Polish-Russian War (Film)?
entities:
[["Polish-Russian War (Film)"]]

**Example 3:**
Sentence:
When did John V, Prince Of Anhalt-Zerbst's father die?
entities:
[["John V, Prince of Anhalt-Zerbst"]]

**Example 4:**
Sentence:
Are more people today related to Genghis Khan than Julius Caesar?
entities:
[["Genghis Khan"], ["Julius Caesar"]]

Your Task:
Sentence:{}
entities:"""


# --------------------------------------------------------------------------- #
# Implementation.
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_REF = "ZhishanQ/QuCo-extractor-0.5B"
DEFAULT_PROMPT_TEMPLATE_VERSION = "quco_entity_extraction_v1"
MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 32


@dataclass
class QucoEntityExtractor(EntityExtractorPort):
    """Knowledge-triplet extractor using QuCo-extractor-0.5B.

    Parameters
    ----------
    model_ref:
        HF model id or local path. Default ``ZhishanQ/QuCo-extractor-0.5B``.
    device:
        ``"cuda:0"`` / ``"cuda"`` / ``"cpu"``. ``None`` uses ``cuda`` if
        available else ``cpu``.
    max_new_tokens:
        Per-sentence generation cap.
    batch_size:
        Number of prompts per generation call. Larger = faster on GPU.
    cache_path:
        Optional JSONL path. If set, results are cached by ``(text, role)``
        and persisted across runs — kill-restart resumes from cached rows.
    """

    model_ref: str = DEFAULT_MODEL_REF
    device: str | None = None
    max_new_tokens: int = MAX_NEW_TOKENS
    batch_size: int = DEFAULT_BATCH_SIZE
    cache_path: str | None = None
    _model: Any = field(default=None, repr=False, init=False)
    _tokenizer: Any = field(default=None, repr=False, init=False)
    _cache: dict[tuple[str, str], list[str]] = field(default_factory=dict, repr=False, init=False)
    _cache_loaded: bool = field(default=False, repr=False, init=False)

    version: str = "quco_extractor_0_5b_v1"
    prompt_template_version: str = DEFAULT_PROMPT_TEMPLATE_VERSION

    # ------------------------------------------------------------------ #
    # Port surface
    # ------------------------------------------------------------------ #

    def extract(self, text: str, *, role: EntityRole = "declarative") -> list[str]:
        results = self.extract_many([text], [role])
        return results[0]

    def extract_many(
        self,
        texts: list[str],
        roles: list[EntityRole],
    ) -> list[list[str]]:
        """Batched extraction with on-disk cache and fallback chain.

        Fallback chain (for declarative role only — question role is single-pass):
        1. Try the declarative prompt (extracts knowledge triplets).
        2. If empty AND text is short (<=8 words), try the question prompt
           (extracts standalone entities — better for short noun-phrase answers
           like "Delhi", "1941", "The Mock Turtle").
        3. If still empty AND text is short, treat the normalized text itself
           as a single entity (final fallback for atomic answers).
        """
        if len(texts) != len(roles):
            raise ValueError("texts and roles must have the same length")
        self._load_cache()

        # Pass 1: try the requested role for every (text, role) pair.
        primary = self._lookup_or_run(list(zip(texts, roles)))

        # Pass 2: declarative-empty + short → re-try with question prompt.
        retry_idx: list[int] = []
        retry_keys: list[tuple[str, str]] = []
        for i, (raw_text, role) in enumerate(zip(texts, roles)):
            if role != "declarative":
                continue
            if primary[i]:
                continue
            text = (raw_text or "").strip()
            if not text:
                continue
            if len(text.split()) > 8:
                continue
            retry_idx.append(i)
            retry_keys.append((text, "question"))
        if retry_keys:
            secondary = self._lookup_or_run(retry_keys)
            for j, i in enumerate(retry_idx):
                if secondary[j]:
                    primary[i] = secondary[j]

        # Pass 3: still empty + short → treat normalized text as single entity.
        for i, (raw_text, role) in enumerate(zip(texts, roles)):
            if primary[i]:
                continue
            text = (raw_text or "").strip()
            if not text or len(text.split()) > 8:
                continue
            fallback = _normalize_unique([text])
            if fallback:
                primary[i] = fallback

        return primary

    def _lookup_or_run(
        self,
        items: list[tuple[str, EntityRole]],
    ) -> list[list[str]]:
        """Cache-aware batched run for a list of (text, role) pairs."""
        results: list[list[str] | None] = [None] * len(items)
        unique_prompts: list[tuple[str, EntityRole]] = []
        unique_idx: dict[tuple[str, str], int] = {}
        for i, (raw_text, role) in enumerate(items):
            text = (raw_text or "").strip()
            if not text:
                results[i] = []
                continue
            key = (text, role)
            if key in self._cache:
                results[i] = self._cache[key]
                continue
            if key not in unique_idx:
                unique_idx[key] = len(unique_prompts)
                unique_prompts.append(key)
        if unique_prompts:
            self._ensure_loaded()
            new_records: list[tuple[tuple[str, str], list[str]]] = []
            for start in range(0, len(unique_prompts), self.batch_size):
                batch = unique_prompts[start:start + self.batch_size]
                completions = self._generate_batch(batch)
                for (text, role), completion in zip(batch, completions):
                    triplets = self._parse_triplets(completion)
                    entities_raw = self._triplets_to_entities(triplets, role=role)
                    entities = _normalize_unique(entities_raw)
                    self._cache[(text, role)] = entities
                    new_records.append(((text, role), entities))
            if new_records:
                self._append_cache(new_records)
        for i, (raw_text, role) in enumerate(items):
            if results[i] is not None:
                continue
            text = (raw_text or "").strip()
            results[i] = self._cache.get((text, role), [])
        return [r if r is not None else [] for r in results]

    def describe(self) -> dict[str, str]:
        return {
            "entity_extractor_version": self.version,
            "entity_extractor_kind": "quco_extractor_0_5b",
            "entity_extractor_model_ref": self.model_ref,
            "entity_extractor_prompt_template": self.prompt_template_version,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _format_prompt(self, text: str, role: EntityRole) -> str:
        template = _QUESTION_PROMPT if role == "question" else _DECLARATIVE_PROMPT
        return template.format(text)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "QucoEntityExtractor requires the 'transformers' and 'torch' "
                "packages. Install via `uv pip install transformers torch`."
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_ref, trust_remote_code=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_ref,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            trust_remote_code=False,
        ).to(device)
        self._model.eval()
        self.device = device

    def _generate_batch(self, batch: list[tuple[str, EntityRole]]) -> list[str]:
        """Batched chat-template inference. Returns raw text per input."""
        self._ensure_loaded()
        import torch

        chat_prompts = []
        for text, role in batch:
            prompt = self._format_prompt(text, role)
            chat = [{"role": "user", "content": prompt}]
            chat_prompts.append(
                self._tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        # Left-pad for decoder-only batched generation so that the new tokens
        # for each row are aligned at the end.
        prev_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(
                chat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
        finally:
            self._tokenizer.padding_side = prev_padding_side
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = output[:, prompt_len:]
        return [
            self._tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in gen_tokens
        ]

    # ------------------------------------------------------------------ #
    # Cache (JSONL on disk, keyed by (text, role)).
    # ------------------------------------------------------------------ #

    def _load_cache(self) -> None:
        if self._cache_loaded:
            return
        self._cache_loaded = True
        if not self.cache_path:
            return
        from pathlib import Path
        path = Path(self.cache_path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record["text"]
                    role = record["role"]
                    entities = record["entities"]
                    if isinstance(entities, list):
                        self._cache[(text, role)] = [str(e) for e in entities]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def _append_cache(self, records: list[tuple[tuple[str, str], list[str]]]) -> None:
        if not self.cache_path:
            return
        from pathlib import Path
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for (text, role), entities in records:
                fh.write(
                    json.dumps(
                        {"text": text, "role": role, "entities": entities},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @staticmethod
    def _parse_triplets(raw: str) -> list[list[str]]:
        """Parse model output into a list of triplets / pairs / singletons."""
        # Strip leading/trailing whitespace and any 'entities:' prefix.
        body = raw.strip()
        body = re.sub(r"^entities\s*:\s*", "", body, flags=re.IGNORECASE)
        # Try direct JSON parse first, then ast.literal_eval fallback.
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(body)
                if isinstance(value, list):
                    triplets: list[list[str]] = []
                    for item in value:
                        if isinstance(item, (list, tuple)):
                            triplets.append([str(v).strip() for v in item if v is not None])
                        elif isinstance(item, str):
                            triplets.append([item.strip()])
                    return triplets
            except (ValueError, SyntaxError):
                continue
        # Last-resort: regex extract first list-of-lists.
        match = re.search(r"\[\s*\[.*?\]\s*\]", body, flags=re.DOTALL)
        if match:
            return QucoEntityExtractor._parse_triplets(match.group(0))
        return []

    @staticmethod
    def _triplets_to_entities(
        triplets: list[list[str]],
        *,
        role: EntityRole,
    ) -> list[str]:
        entities: list[str] = []
        for item in triplets:
            if not item:
                continue
            if role == "question":
                # Question prompt emits [[entity]] or [[entity, relation]].
                # Keep only the first slot (entity).
                entities.append(item[0])
            else:
                # Declarative prompt emits [[head, relation, tail]] or
                # [[entity, relation]] for embedded questions.
                if len(item) >= 3:
                    entities.append(item[0])
                    entities.append(item[2])
                else:
                    entities.append(item[0])
        return entities


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


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
        # Strip leading articles (Infini-gram queries are sensitive to surface form).
        for article in _LEADING_ARTICLES:
            if norm.startswith(article):
                norm = norm[len(article):]
                break
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out
