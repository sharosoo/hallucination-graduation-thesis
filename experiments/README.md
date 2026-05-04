# Experiments Contract

This directory is the binding experiment contract for the `Corpus-Grounded Selective Fusion Detector`. Write code under `experiments/` only after this contract validates.

The executable pipeline is fixed in [`experiments/PIPELINE.md`](PIPELINE.md). If this README and `PIPELINE.md` disagree, treat `PIPELINE.md` as the operational source of truth for run order, validation gates, and thesis-valid evidence requirements.

## 1. Method name and thesis claims

The method name is exactly `Corpus-Grounded Selective Fusion Detector`.

The experiment tests three thesis claims:

1. High-diversity hallucinations are targeted by Semantic Entropy.
2. Low-diversity hallucinations are targeted by Semantic Energy.
3. QuCo-RAG-style corpus statistics support selective fusion by adding corpus-grounded entity signals to the fusion model.

The proposed method uses corpus features as direct learned-fusion features, not a scalar coverage-only proposed method. Prior scalar corpus coverage weighting failed as a selection rule, so coverage-only adaptive weighting is not the proposed method.

## 2. Label contract

Every sample receives one of these exact labels:

| Label | Rule | Role |
| --- | --- | --- |
| `NORMAL` | correct -> NORMAL | Non-hallucinated sample. |
| `HIGH_DIVERSITY` | incorrect and SE > 0.5 -> HIGH_DIVERSITY | Type-specific hallucination target for Semantic Entropy. |
| `LOW_DIVERSITY` | incorrect and SE <= 0.1 -> LOW_DIVERSITY | Type-specific hallucination target for Semantic Energy. |
| `AMBIGUOUS_INCORRECT` | incorrect and 0.1 < SE <= 0.5 -> AMBIGUOUS_INCORRECT | Incorrect gray zone. Exclude from high-vs-low type-specific claims, but include in overall hallucination detection. |

Ambiguous incorrect samples are never treated as normal.

The four operational labels above keep their fixed thresholds. Fine-grained SE bins are analysis bins only and must not replace `NORMAL`, `HIGH_DIVERSITY`, `LOW_DIVERSITY`, or `AMBIGUOUS_INCORRECT` as class labels.

Every row-level output must include an explicit operational type label marker for every sample, using the `label` field and one of the four labels above.

## 3. Dataset contract

Core datasets:

- TruthfulQA
- TriviaQA
- HaluEval-QA

Stretch or future datasets, future only:

- Natural Questions
- HotpotQA
- FEVER
- BioASQ

Dataset-level and cross-dataset metrics are required. Pooled random splits are secondary checks only.

Dataset expansion is allowed when clearly configured and reported, but TruthfulQA, TriviaQA, and HaluEval-QA remain mandatory core datasets. Natural Questions, HotpotQA, FEVER, and BioASQ remain future or stretch datasets unless a later checked-in config explicitly promotes them.

## 4. Feature contract

Trainable features must not use correctness labels, gold answers, or hidden-state probe features. Label-only values may be used to assign the four labels above, but they must not leak into trainable features unless explicitly listed here as trainable.

Required feature families:

- `semantic_entropy`
- `cluster_count`
- `semantic_energy_boltzmann` or `semantic_energy_proxy`
- `logit_variance`
- `confidence_margin`
- `entity_frequency`
- `entity_pair_cooccurrence`
- `low_frequency_entity_flag`

Corpus feature meanings follow the QuCo-RAG-style reference implementation pattern: entity frequency comes from direct corpus count queries, entity-pair co-occurrence comes from corpus windowed pair counts, and low-frequency flags come from thresholded entity frequency. These corpus features are direct learned-fusion features, not a scalar coverage-only proposed method.

## 5. Baseline contract

All baseline methods must be implemented and reported:

- SE-only
- Energy-only
- corpus-risk-only
- fixed linear 0.1/0.9
- fixed linear 0.5/0.5
- fixed linear 0.9/0.1
- hard cascade
- learned fusion without corpus
- learned fusion with corpus

The main method comparison is learned fusion with corpus against learned fusion without corpus and against the single-signal and fixed-rule baselines.

## 6. PC Probe guardrail

PC Probe is reference-only and not implemented. Hidden-state/probe features are excluded.

Phillips/PC Probe may be cited only for conceptual framing around learned combination and confidently wrong errors. It must not appear as an implemented method, baseline, feature source, or training input.

## 7. Metrics contract

Report these metrics at overall, per-dataset, and per-label levels where labels apply:

- AUROC for hallucination detection.
- AUPRC for hallucination detection under class imbalance.
- Accuracy, precision, recall, and F1 at the selected operating point.
- Calibration or threshold diagnostics for fixed linear, hard cascade, and learned fusion methods.
- Type-specific AUROC or paired comparison for `HIGH_DIVERSITY` vs `LOW_DIVERSITY` subsets.
- Corpus feature coefficient or importance summaries for learned fusion with corpus.
- Fine-grained SE bin analysis for crossover and threshold sensitivity, using deciles or config-defined bins spanning `[0, +inf)` with special attention around 0.1 and 0.5.

`AMBIGUOUS_INCORRECT` is included in overall hallucination detection metrics and excluded from high-vs-low type-specific thesis claims.

Fine-grained SE bins are for analysis, crossover checks, and threshold sensitivity only. They do not change the fixed operational label thresholds at SE <= 0.1 and SE > 0.5.

## 8. Output schema contract

Every result artifact must be reproducible from a manifest and include enough metadata to trace formulas, features, datasets, and splits.

Required output elements:

- `run_id`: stable run identifier.
- `method_name`: one of the baseline names or `Corpus-Grounded Selective Fusion Detector`.
- `dataset`: source dataset name.
- `split_id`: split or cross-dataset evaluation identifier.
- `sample_id`: stable sample identifier for row-level outputs.
- `label`: explicit operational type label marker for every sample, one of `NORMAL`, `HIGH_DIVERSITY`, `LOW_DIVERSITY`, `AMBIGUOUS_INCORRECT`.
- `features`: object containing the required feature families used by the method.
- `prediction_score`: hallucination score.
- `prediction_label`: predicted hallucination decision at the selected threshold.
- `metrics`: aggregate object with AUROC, AUPRC, accuracy, precision, recall, and F1 when applicable.
- `feature_importance`: coefficient or importance object for learned fusion methods.
- `formula_manifest_ref`: pointer to paper-derived formula specs.
- `dataset_manifest_ref`: pointer to dataset and corpus snapshot manifests.

Future self-contained generation runs under this repo must no longer depend on `../hallucination_lfe` for prompt generation or logits export. New row-level generation artifacts must preserve machine-readable per-token full logits (for example `full_logits`), per-step `logsumexp`, selected/generated token ids, and provenance fields such as `model_name`, `tokenizer_name`, `generation_config`, `logits_schema_version`, `formula_manifest_ref`, `dataset_manifest_ref`, and `created_at`.

## 9. Hexagonal architecture convention

The experiment implementation must use backend-style typed hexagonal architecture, not notebook-style scripts.

- `domain`: typed dataclasses, enums, value objects, labels, feature records, predictions, metrics, and provenance objects.
- `ports`: abstract interfaces for datasets, formula specs, corpus statistics, model logits, storage, and evaluation output.
- `adapters`: concrete readers and clients for upstream artifacts, Hugging Face datasets, corpus-stat services, model/logit sources, local storage, and serialized manifests.
- `application`: use-case services that orchestrate labeling, feature extraction, fusion training, evaluation, and export without depending on concrete adapters.
- `scripts`: thin CLI entrypoints that parse arguments and call application services.
- `configs`: dataset, literature, formula, feature, fusion, and evaluation configuration files.
- `manifests`: derived dataset, corpus, feature, formula, and run manifests with hashes or stable references.
- `results`: generated metrics, row-level predictions, tables, figures, and final reports.
- `literature`: saved papers, checksums, extracted formula notes, and evidence notes used by thesis prose.

## 10. Reference-status caveats

- QuCo-RAG is used as corpus frequency and entity-pair co-occurrence inspiration. Unless an official archival status is verified in a later literature task, cite it with a preprint or status caveat.
- Ma Semantic Energy is used as the Semantic Energy source or motivation. Until later PDF-grounded validation, cite it with a preprint or submission caveat.
- Phillips/PC Probe is reference-only. It supplies conceptual framing for learned combination and confidently wrong cases, but PC Probe is not implemented and hidden-state/probe features are excluded.

## 11. Validation command

Run the contract validator after editing this file:

```bash
uv run python experiments/scripts/validate_readme_contract.py experiments/README.md
```

## 12. Environment and execution contract

- Create or refresh the repo-local virtual environment with `uv sync`.
- Install heavyweight generation dependencies only when needed with `uv sync --group generation`.
- Run experiment CLIs through `uv run`, for example `uv run python -m experiments.scripts.validate_architecture` or `uv run python experiments/scripts/run_generation.py --config experiments/configs/generation.yaml --out experiments/results/generation/full_logits_fixture.json --write-fixture --fixture-variant full_logits`.
- The checked-in `experiments/configs/generation.yaml` fixture path is safe to validate without downloading model weights. A live local generation run still requires the optional `generation` dependency group and locally available model files unless `runtime.local_files_only` is disabled on purpose.
- Use `uv run python experiments/scripts/run_pipeline.py --mode smoke --dry-run --out experiments/results/runs` to materialize the reproducible command manifest before executing any run.
- Smoke/dev runs are not thesis evidence. Thesis claims require `full-core` or explicitly promoted `full-extended` mode plus all gates in `experiments/PIPELINE.md`.
