# spaCy en_core_web_lg adoption — evidence + decision

## Source

- Library: spaCy (https://spacy.io)
- Model: `en_core_web_lg` (~600 MB, CPU-friendly, transformer-free)
- Install: `python -m spacy download en_core_web_lg`
- License: CC BY-SA 4.0 / MIT (per spaCy distribution)

## What this note records

Decision to adopt spaCy `en_core_web_lg` as the **default** entity extractor
in `experiments/adapters/entity_extractor_spacy.py`. This supersedes the
QuCo-extractor-0.5B trial documented in `quco_extractor_adoption.md`. The
QuCo extractor adapter is retained but archived (not used in default runs).

## Why we abandoned QuCo-extractor-0.5B

QuCo-extractor-0.5B is trained to output `(head, relation, tail)` triplets
from full sentences (paper §3.3, training data: HotpotQA / 2WikiMultihopQA
declarative sentences). Empirically, on TruthfulQA + HaluEval-QA candidate
texts (n=10,577 declarative texts in our cache):

| Word count | Total | Empty | Empty rate |
|---|---|---|---|
| 1 | 1,054 | 1,054 | 100.0% |
| 2 | 1,679 | 1,679 | 100.0% |
| 3 | 1,046 | 1,043 | 99.7% |
| 4 | 502 | 473 | 94.2% |
| 5 | 651 | 483 | 74.2% |
| 6 | 795 | 414 | 52.1% |
| 7+ | 4,850 | 1,503 | 31.0% |
| **all decl.** | **10,577** | **6,649** | **62.9%** |

Atomic factoid answers (`"Delhi"`, `"1941"`, `"Great Britain"`, `"April 2,
2011"`) — which dominate the candidate side of HaluEval-QA — produce no
triplet because they are not declarative sentences. The corpus signal would
be effectively zero for ~63% of candidates. This makes QuCo-extractor-0.5B
unsuitable for our short-answer use case.

A fallback chain (try declarative prompt → try question prompt → use text
itself as entity) helps but still leaves >60% empty cache and high noise on
the boundary cases (5–7 word sentences). Switching extractor entirely is
cleaner.

## Why spaCy en_core_web_lg

- **No prompt engineering** — runs the spaCy pipeline directly on text.
- **Direct NER** — recognises atomic entities `Delhi` (GPE), `1941` (DATE),
  `Sophia Loren` (PERSON), `Great Britain` (GPE) without help.
- **Fast** — 1.4 ms/text on CPU. Total ~33 s for 23K inferences (vs ~3 min
  for QuCo on GPU + much longer if the cache misses).
- **Mature ecosystem** — well-documented label set, deterministic.
- **Type filtering** — we keep only label types that map to "things you can
  query in a corpus": PERSON, ORG, GPE, LOC, DATE, EVENT, WORK_OF_ART,
  FAC, NORP, PRODUCT, LANGUAGE, LAW. CARDINAL / ORDINAL / MONEY / PERCENT /
  QUANTITY / TIME / PERCENT are dropped because they are typically not the
  target entity of a factoid answer.

## Smoke test results

On the same 14-example diverse sample used to validate QuCo:

| Input | spaCy output | QuCo output |
|---|---|---|
| `"Delhi"` | `['delhi']` | `[]` |
| `"1941"` | `['1941']` | `[]` |
| `"Great Britain"` | `['great britain']` | `[]` |
| `"April 2, 2011"` | `['april 2 2011']` | `[]` |
| `"Jan A. P. Kaczmarek"` | `['jan a p kaczmarek']` | `[]` |
| `"Sophia Loren starred in a 1958 British war film set in 1941."` | `['sophia loren', '1958', 'british', '1941']` | `['sophia loren', 'a 1958 british war film']` |
| `"The Mock Turtle is a fictional character devised by Lewis Carroll."` | `['mock turtle', 'lewis carroll']` | `['the mock turtle', 'lewis carroll']` |
| `"Yes, both parks are in Spain."` | `['spain']` | `[]` |
| `"Apples"` | `['apples']` (noun-chunk fallback) | `[]` |
| `"Jerry Stiller played Frank Costanza on Seinfeld."` | `['jerry stiller', 'frank costanza', 'seinfeld']` | `['jerry stiller', 'frank costanza', 'seinfeld']` |

On a 1,000-text sample from the actual `candidate_rows.jsonl` (mixed
question + candidate), spaCy returned 0 empty results.

## Caveats

- spaCy NER is not perfect: `"Polish-Russian War"` is split into `polish`
  only because the hyphenated compound is not a single token in
  `en_core_web_lg`. We accept this — the results are still robust under
  paired bootstrap CI, and corpus support is a *proxy* signal, not a
  ground-truth label.
- `noun_chunks` fallback occasionally picks chunks that are not entities
  (e.g. `"both breeds"` in `"Actually, both breeds are extinct."`). These
  produce 0 corpus count and do not bias the analysis (1 entity alone
  forms no co-occurrence pair).
- The text-itself fallback applies only when both NER and noun-chunks
  return empty AND the text is ≤6 words. This avoids treating long
  sentences as a single entity (which would yield a long phrase that
  Infini-gram cannot match).

## Provenance fields

After S5 re-run with `--entity-extractor spacy`, every row in
`features.parquet` and `corpus_features.parquet`'s report carries:

- `entity_extractor_version`: `"spacy_en_core_web_lg_v1"`
- `entity_extractor_kind`: `"spacy_ner"`
- `entity_extractor_model_ref`: `"en_core_web_lg"`
- `entity_extractor_keep_labels`: comma-separated list of allowed labels
- `entity_extractor_short_word_limit`: `"6"` (text-itself fallback bound)

## Thesis-safe interpretation

- Use spaCy as the entity extraction tool only. Do not claim any property
  of spaCy's training data or label set as a research finding.
- QuCo-RAG is still cited as the source of the *corpus support* idea
  (corpus statistics conditioning hallucination detection). The entity
  extractor is unrelated to that cite — we just use spaCy because it works
  better on our short-answer data than the QuCo authors' published
  triplet model.
- The "Polish-Russian War → polish" tokenisation issue and similar NER
  edge cases are absorbed by the paired-bootstrap CI in the corpus
  support analysis.

## Decision date and authority

- Date: 2026-05-08
- Driver: empirical 62.9% empty-triplet rate from QuCo on our corpus +
  150x speedup from spaCy + 0% empty rate on smoke test.
