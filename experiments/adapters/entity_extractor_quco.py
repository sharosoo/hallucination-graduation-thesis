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
    """

    model_ref: str = DEFAULT_MODEL_REF
    device: str | None = None
    max_new_tokens: int = MAX_NEW_TOKENS
    _model: Any = field(default=None, repr=False, init=False)
    _tokenizer: Any = field(default=None, repr=False, init=False)

    version: str = "quco_extractor_0_5b_v1"
    prompt_template_version: str = DEFAULT_PROMPT_TEMPLATE_VERSION

    # ------------------------------------------------------------------ #
    # Port surface
    # ------------------------------------------------------------------ #

    def extract(self, text: str, *, role: EntityRole = "declarative") -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        prompt = self._format_prompt(text, role)
        completion = self._generate(prompt)
        triplets = self._parse_triplets(completion)
        entities = self._triplets_to_entities(triplets, role=role)
        return _normalize_unique(entities)

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

    def _generate(self, prompt: str) -> str:
        self._ensure_loaded()
        import torch  # local — guaranteed by _ensure_loaded

        chat = [{"role": "user", "content": prompt}]
        prompt_text = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

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


def _normalize_unique(entities: list[str]) -> list[str]:
    """Lower-case, strip punctuation noise, drop duplicates while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in entities:
        if not raw:
            continue
        norm = re.sub(r"\s+", " ", raw.strip().lower())
        norm = re.sub(r"[^a-z0-9'\- ]+", "", norm).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out
