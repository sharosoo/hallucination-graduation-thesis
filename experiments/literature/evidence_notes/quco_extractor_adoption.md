# QuCo-extractor-0.5B adoption — evidence + decision

## Source

- HF model: `ZhishanQ/QuCo-extractor-0.5B`
  https://huggingface.co/ZhishanQ/QuCo-extractor-0.5B
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Distillation source: `gpt-4o-mini` (40K annotated examples, in-context learning
  + full-parameter SFT)
- Paper: QuCo-RAG (Min et al. 2025, arXiv:2512.19134, ACL Findings 2026)
- Repo: https://github.com/ZhishanQ/QuCo-RAG
- License: see HF model card

## What this note records

Decision to **replace** our regex-based `phrase_candidates` extractor in
`experiments/adapters/corpus_features.py` with the publicly released
QuCo-extractor-0.5B. This affects S5 (corpus features), S7 (feature merge),
S8 (fusion), S9 (robustness). See sisyphus notepad
`.sisyphus/notepads/structured-experiments-upgrade/ner_pivot_plan.md` for the
full pivot plan and re-run scope.

## Why this extractor

QuCo-RAG repo (README, Important Notes section) explicitly recommends the
released distilled model:

> **Entity extraction options**: By default, we use QuCo-extractor-0.5B for
> entity extraction, which is distilled from gpt-4o-mini and handles most
> domains and datasets well. For reproducibility, this default model is
> sufficient.

Our research uses corpus statistics (entity frequency + entity-pair
co-occurrence) for the same purpose QuCo-RAG does (uncertainty proxy from
pretraining-corpus exposure). Using the same extractor lets us:

1. Compare effect sizes across the two papers on equivalent entity inputs.
2. Avoid debate about whether weak corpus-level signal in our experiment
   (paired win-rate 0.551) reflects extractor noise vs. genuine effect size.
3. Inherit QuCo-RAG's reproducibility (no in-house NER tuning).

## Triplet output, head/tail use

QuCo-extractor-0.5B emits `(head, relation, tail)` knowledge triplets per
sentence. In our use we keep `head` and `tail` as the entities of interest
and discard `relation` text — same choice as QuCo-RAG (paper section 3.3):

> We compute cooc(h, t) rather than cooc(h, r, t) because relational
> predicates exhibit high lexical variability (e.g., "employed by" vs.
> "worked at"), while named entities are more lexically stable (Galárraga et
> al., 2014).

This is consistent with our entity-pair co-occurrence axis definition.

## Caveats

- **Triplet model trained on QA datasets** (2WikiMultihopQA, HotpotQA). Our
  data is TruthfulQA + HaluEval-QA. Domain shift expected to be small (both
  are factoid QA), but we should manually inspect 100 sample entities after
  first run.
- **Same external corpus snapshot caveat as before**: Infini-gram 16B index
  ≠ Qwen2.5-3B pretraining corpus. Therefore corpus support remains a
  *proxy* for model exposure, not a direct measure. This note does not change
  that limitation.
- **Distilled model**, not base model. QuCo-RAG also offers `gpt-4o-mini`
  API as alternative; we don't use it (cost, reproducibility).

## Provenance fields to add

After S5 re-run, every row in `features.parquet` and
`corpus_features.parquet` will carry:

- `entity_extractor_version`: `"quco_extractor_0_5b_v1"` (vs prior
  `"regex_v1"`)
- `entity_extractor_model_ref`: `"ZhishanQ/QuCo-extractor-0.5B"`
- `entity_extractor_prompt_version`: `"quco_new_prompt_v3"`

This lets future audits distinguish pre-pivot and post-pivot artifacts.

## Thesis-safe interpretation

- Use QuCo-extractor-0.5B as "entity recognition tool" only. Do not claim
  any property of QuCo-RAG's *retrieval* method or its uncertainty
  framework as inherited.
- Our research framing (corpus support as conditioning axis, NOT as direct
  hallucination detector or retrieval trigger) remains independent of
  QuCo-RAG's framing.
- In the thesis methodology section, cite QuCo-RAG (`\cite{qucorag}`) for
  the extractor and Liu et al. 2024 (`\cite{infinigram2024}`) for the
  Infini-gram backend separately.

## Decision date and authority

- Date: 2026-05-08
- Author: thesis author (with reviewer feedback round 4)
- Driver: reviewer concern that regex-based entity extraction may have
  weakened corpus-level CHOKE evidence (paired win-rate 0.551 vs.
  candidate-level 0.36/0.34).
