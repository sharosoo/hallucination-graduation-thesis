# Reproducible Experiment Pipeline

This file is the binding source of truth for the thesis experiment pipeline. A result is thesis-valid only when it is produced by this pipeline, recorded in a run manifest, and passes the validation gates below.

## 1. Non-negotiable principles

- **Smoke results are never thesis evidence.** A 100-200 row run is for checking code paths only.
- **Full logits must be repo-owned.** Semantic Energy claims require row-level generation artifacts that preserve full-vocabulary logits or a documented lossless equivalent sufficient for the paper formula.
- **Corpus statistics must be repo-owned.** Cached upstream corpus stats and proxy indexes are allowed only as smoke/dev fallbacks and must be labeled as such.
- **QuCo-RAG semantics are split:** QuCo-RAG uses an Infini-gram-compatible count backend for entity frequency and entity-pair `AND` counts, while Elasticsearch/BM25 is used for retrieval over the `wiki` passage index.
- **Fusion is logistic fusion.** The current implementation is a custom stdlib L2 logistic regression, not `sklearn.linear_model.LogisticRegression`. The exact command, config, coefficients, predictions, and logs must be preserved.
- **The thesis is downstream.** LaTeX tables and prose may cite only validated artifacts from `experiments/results/` plus their manifest checksums.

## 2. Evaluation modes

| Mode | Intended rows | Purpose | Thesis-valid? |
| --- | ---: | --- | --- |
| `smoke` | 25-50 rows per core dataset | Validate data loading, prompting, full-logit shape, corpus-stat query wiring, and metric scripts | No |
| `dev` | 200-500 rows per core dataset | Debug feature schemas, thresholds, calibration, and runtime/storage cost | No |
| `full-core` | TruthfulQA 817 + TriviaQA 2000 + HaluEval-QA 5000 = 7817 rows | Main thesis evaluation over mandatory core datasets | Yes, after gates |
| `full-extended` | full-core + NQ 2000 + HotpotQA 2000 + FEVER 2000 + BioASQ 1000 = 14817 rows | Optional domain/generalization extension | Yes, after explicit promotion and gates |

Current checked-in results are a 1000-row upstream/cache/proxy run. They are useful for diagnosing the code path, but not final thesis evidence for full-logit Semantic Energy or QuCo-RAG-style direct corpus execution.

## 3. Stage graph

Each stage writes a machine-readable artifact and records its command in the run manifest.

### S0. Environment and contract validation

Command:

```bash
uv sync --group generation
uv run python experiments/scripts/validate_readme_contract.py experiments/README.md
uv run python experiments/scripts/validate_formula_specs.py experiments/configs/formulas.yaml experiments/literature/formula_notes.md
uv run python experiments/scripts/validate_evidence_notes.py experiments/literature/evidence_notes
uv run python experiments/scripts/validate_pipeline_contract.py experiments/PIPELINE.md
uv run python experiments/scripts/validate_paper_feature_alignment.py --formulas experiments/configs/formulas.yaml --notes experiments/literature/formula_notes.md --pipeline experiments/PIPELINE.md
```

Inputs: `pyproject.toml`, `uv.lock`, `experiments/README.md`, `experiments/PIPELINE.md`, formula/evidence notes.

Outputs: validator stdout and `experiments/results/runs/<run_id>/manifest.json` when orchestrated.

Gate: fail on missing paper references, missing pipeline sections, stale QuCo wording, or smoke/proxy overclaims.

### S1. Dataset materialization and prompt rows

Current status: `experiments/scripts/prepare_datasets.py` is metadata-only. A thesis-valid run requires a concrete Hugging Face dataset adapter and prompt-row materializer.

Required command shape:

```bash
uv run python experiments/scripts/prepare_datasets.py --config experiments/configs/datasets.yaml --out experiments/results/datasets --mode <smoke|dev|full-core|full-extended>
```

Required outputs:

- `experiments/results/datasets/dataset_preparation_report.json`
- `experiments/results/datasets/prompt_rows.<mode>.jsonl`
- `experiments/results/datasets/dataset_manifest.<mode>.json`

Required row fields: `dataset`, `dataset_id`, `split_id`, `sample_id`, `question`, `context`, `prompt`, `gold_answers` or dataset label source metadata, prompt hash, dataset revision.

Gate: row counts must match the selected mode; each dataset must have both positive and negative label sources or a documented reason before metric claims.

### S2. Repo-owned generation with full logits

Command:

```bash
uv run python experiments/scripts/run_generation.py --config experiments/configs/generation.yaml --prompts experiments/results/datasets/prompt_rows.<mode>.jsonl --out experiments/results/generation/full_logits.<mode>.json
uv run python experiments/scripts/validate_generation_logits.py experiments/results/generation/full_logits.<mode>.json
```

Outputs must include `full_logits`, `logsumexp`, generated token ids, selected token ids, selected token logits, full vocabulary flag, model/tokenizer names and revisions, generation config, schema version, and created timestamp.

Gate: randomly inspected rows must have shape `[generated_steps, tokenizer_vocab_size]`. Top-k-only logits do not satisfy this gate unless the metric is renamed and the thesis explicitly states the deviation.

Storage note: full-vocabulary logits are large. Estimate `rows × generated_tokens × vocab_size × bytes_per_value` from smoke artifacts before full-core execution.

### S3. QuCo-RAG-style corpus statistics

Command shape:

```bash
uv run python experiments/scripts/compute_corpus_features.py --manifests experiments/manifests --out experiments/results/corpus_features.parquet --mode direct-corpus
```

Current status: only `cache-or-proxy`, `cache-only`, and `service-unavailable` are implemented. `direct-corpus` is required for thesis-valid QuCo-style corpus features.

Required implementation contract:

- Entity extraction must record the extractor prompt/version or deterministic parser version.
- Entity frequency must call an Infini-gram-compatible count backend, not Elasticsearch, unless the deviation is renamed.
- Entity-pair co-occurrence must query pair counts using the QuCo-style `head AND tail` count semantics.
- Elasticsearch/BM25 may be used for retrieval evidence, with index name, UUID/snapshot, analyzer/settings hash, document count, text fields, and query logs recorded.

Required outputs:

- `entity_frequency_mean`, `entity_frequency_min`
- `entity_pair_cooccurrence`
- `low_frequency_entity_flag`
- `zero_cooccurrence_flag`
- `coverage_score` only if its formula/source is documented
- corpus backend metadata and query audit records

Gate: thesis-valid rows must not report `cache_upstream_corpus_stats` or `proxy_cached_artifact_index` as their primary corpus source.

### S4. Semantic Energy from logits

Command:

```bash
uv run python experiments/scripts/compute_energy_features.py --manifests experiments/manifests --out experiments/results/energy_features.parquet --require-true-boltzmann
uv run python experiments/scripts/validate_energy_features.py experiments/results/energy_features.parquet
```

Required correction: the energy stage must compute `semantic_energy_boltzmann`, `logit_variance`, and `confidence_margin` from generated `full_logits` / `logsumexp` / selected token ids. It must not simply trust an upstream scalar named `semantic_energy` because an artifact claims `has_full_logits=true`.

Gate: a deterministic fixture with known logits must match the paper-derived formula before real rows are accepted.

### S5. Feature table and labels

Command:

```bash
uv run python experiments/scripts/build_feature_table.py --inputs experiments/results --out experiments/results/features.parquet
uv run python experiments/scripts/validate_type_labels.py experiments/results/features.parquet
```

Required behavior:

- Labels are operational: `NORMAL`, `HIGH_DIVERSITY`, `LOW_DIVERSITY`, `AMBIGUOUS_INCORRECT`.
- Correctness/gold-answer data is label-only and forbidden as trainable features.
- True Energy values must be row-joined into `features.semantic_energy_boltzmann`; missing full logits must fail thesis validation rather than silently falling back to proxy energy.

Gate: report per-dataset and per-label counts; do not run claims where a slice has a single class or too few rows without caveat.

### S6. Type analysis

Command:

```bash
uv run python experiments/scripts/run_type_analysis.py --features experiments/results/features.parquet --out experiments/results/type_analysis
```

Outputs: `summary.json`, `metrics.csv`, `report.md`.

Gate: Energy metrics must be numeric only when S4 passed true-Boltzmann validation.

### S7. Logistic fusion and baselines

Command:

```bash
uv run python experiments/scripts/run_fusion.py --features experiments/results/features.parquet --config experiments/configs/fusion.yaml --out experiments/results/fusion
```

Current implementation: custom stdlib L2 logistic regression with standardization, gradient descent, leave-one-dataset-out folds, and train-max-F1 thresholds.

Required outputs:

- `summary.json`
- `baseline_metrics.csv`
- `predictions.jsonl`
- `feature_importance.json`
- `feature_importance.csv`
- `learned_fusion_comparison.json`
- `report.md`
- `run_command.json` and stdout/stderr logs when run through `run_pipeline.py`

Gate: all baselines listed in `experiments/README.md` must either produce metrics or a documented unavailable reason. Energy-only, fixed linear SE/Energy, and hard cascade are invalid until true Energy exists.

### S8. Robustness and selective-risk checks

Command:

```bash
uv run python experiments/scripts/run_robustness.py --features experiments/results/features.parquet --fusion experiments/results/fusion --out experiments/results/robustness
uv run python experiments/scripts/validate_report_claims.py experiments/results/robustness/summary.json
```

Outputs: bootstrap CIs, leave-one-dataset-out summaries, threshold sensitivity, within-dataset checks, selective-risk report.

Gate: if confidence intervals cross zero, prose must not claim a stable improvement.

### S9. Thesis evidence export and LaTeX build

Command:

```bash
uv run python experiments/scripts/export_thesis_evidence.py --results experiments/results --out experiments/results/thesis_evidence_table.tex
uv run python experiments/scripts/validate_thesis_evidence_links.py thesis/main.tex experiments/literature/evidence_notes experiments/results/thesis_evidence_summary.json
cd thesis && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

Gate: LaTeX must build cleanly, and every numeric claim must trace to `thesis_evidence_summary.json` or an evidence note with page/table/section/equation references.

## 4. Paper-derived feature alignment

| Feature family | Source | Required implementation check |
| --- | --- | --- |
| `semantic_entropy`, `cluster_count` | Farquhar et al. Semantic Entropy | Formula notes must cite page/section/equation; implementation must preserve cluster probability/SE semantics or mark deviation. |
| `semantic_energy_boltzmann` | Ma Semantic Energy | Must compute from logits/logsumexp according to the saved formula notes; proxy selected-logit scalars are not thesis-valid. |
| `logit_variance`, `confidence_margin` | Downstream logits diagnostics | Must be labeled as derived diagnostics, not named Ma formulas. |
| `entity_frequency_mean`, `entity_frequency_min`, `low_frequency_entity_flag` | QuCo-RAG | Must use Infini-gram-compatible entity count semantics and threshold metadata. |
| `entity_pair_cooccurrence`, `zero_cooccurrence_flag` | QuCo-RAG | Must use `head AND tail` pair-count semantics or explicitly rename the deviation. |
| `learned fusion with corpus` | This thesis + selective prediction framing | Must use non-probe, non-label trainable features only and export logistic coefficients. |

Run:

```bash
uv run python experiments/scripts/validate_paper_feature_alignment.py --formulas experiments/configs/formulas.yaml --notes experiments/literature/formula_notes.md --pipeline experiments/PIPELINE.md
```

## 5. Reproducible orchestration

Use `run_pipeline.py` to create a manifest even for dry runs:

```bash
uv run python experiments/scripts/run_pipeline.py --mode smoke --dry-run --out experiments/results/runs
uv run python experiments/scripts/run_pipeline.py --mode smoke --execute --out experiments/results/runs
```

The orchestrator records:

- run id, mode, git commit if available, Python version, platform, UTC timestamp
- exact commands and working directory
- stdout/stderr log paths for executed stages
- expected inputs and outputs
- skipped stages and skip reasons

Do not write thesis claims from a run unless the manifest says all thesis gates passed.
