# Experiments Contract

This directory is the binding experiment contract for the `Corpus-Conditioned Hallucination Metric Reliability Study`. Write code under `experiments/` only after this contract validates.

The executable pipeline is fixed in [`experiments/PIPELINE.md`](PIPELINE.md). If this README and `PIPELINE.md` disagree, treat `PIPELINE.md` as the operational source of truth for run order, validation gates, and thesis-valid evidence requirements.

## 1. Method name and thesis claims

The method name is exactly `Corpus-Conditioned Hallucination Metric Reliability Study`.

The experiment tests three thesis claims:

1. Corpus entity frequency and entity-pair co-occurrence define continuous conditioning axes for hallucination metric reliability.
2. Paper-faithful Semantic Entropy and Semantic Energy must be evaluated as metric families whose reliability can change across corpus-axis bins, not as globally fixed hallucination solutions.
3. Condition-aware fusion is evaluated by comparing global fusion against corpus-bin-aware or axis-interaction fusion; any improvement is an empirical result, not an assumed claim.

The proposed contribution is not a RAG system, not answer-generation improvement, and not a production hallucination detector. QuCo-RAG-style corpus statistics are used as corpus-support axes that may condition metric reliability. Low entity frequency or zero entity-pair co-occurrence is not itself a hallucination label.

## 2. Label contract

Every candidate row keeps an annotation-backed binary correctness label:

| Field | Rule | Role |
| --- | --- | --- |
| `is_correct` | dataset annotation says candidate is correct | Non-hallucinated candidate. |
| `is_hallucination` | `not is_correct` | Hallucinated candidate. |
| `candidate_label` | `right` or `hallucinated` | Paired candidate role from the dataset. |

`NORMAL`, `HIGH_DIVERSITY`, `LOW_DIVERSITY`, and `AMBIGUOUS_INCORRECT` may appear only as archived operational labels from the earlier diagnostic run. They are not the thesis-defining class ontology for the redesigned experiment. New thesis-valid analyses use corpus-axis bins, optional SE bins, and annotation-backed hallucination labels.

Correctness labels, gold answers, and dataset annotations are label-only metadata. They must not leak into trainable features.

## 3. Dataset contract

The active thesis dataset is exactly one paired discriminative experiment dataset built from:

- TruthfulQA
- HaluEval-QA

Each prompt contributes exactly two candidate rows: one annotation-backed correct answer and one annotation-backed hallucinated answer. The model scores those fixed candidates with teacher forcing. It does not generate an answer for later correctness labeling.

Excluded datasets:

- TriviaQA
- Natural Questions
- HotpotQA
- FEVER
- BioASQ

These datasets are excluded from the active thesis dataset because they do not provide clean dataset-level `(right_answer, hallucinated_answer)` candidate pairs. Adding synthetic hallucinated candidates would reintroduce heuristic matching or judge-based label construction, which this experiment forbids.

Dataset-level and cross-dataset metrics are required across TruthfulQA and HaluEval-QA. Pooled random splits are secondary checks only.

Dataset expansion is not part of this contract. A later task may add new paired datasets only if the checked-in dataset itself provides clean candidate pairs and the contract is updated before implementation.

Reference-only non-paired datasets:

- TriviaQA may appear in literature notes or archived results, but it is not an active or core thesis dataset.

## 4. Feature contract

Trainable features must not use correctness labels, gold answers, or hidden-state probe features. Required feature families are grouped by research role:

### Paper-faithful uncertainty features

- `semantic_entropy_nli_likelihood`
- `semantic_entropy_cluster_count`
- `semantic_entropy_discrete_cluster_entropy`
- `semantic_energy_cluster_uncertainty`
- `semantic_energy_sample_energy`

Semantic Entropy must use N=10 answer-only samples, NLI-based semantic clustering, and likelihood-based cluster probabilities. Semantic Energy must use multiple generated answers, the semantic clusters shared with SE, selected-token logit-derived energy, and cluster-level aggregation before it is called paper-faithful Semantic Energy.

### Candidate-level diagnostic features

- `mean_negative_log_probability`
- `logit_variance`
- `confidence_margin`
- `semantic_energy_boltzmann_diagnostic`

These diagnostics may be useful baselines, but they must not be described as the full Semantic Energy paper implementation unless the paper-faithful multi-generation cluster path is present.

### Corpus-axis features

- `entity_frequency`
- `entity_frequency_axis`
- `entity_pair_cooccurrence`
- `entity_pair_cooccurrence_axis`
- `low_frequency_entity_flag`
- `zero_cooccurrence_flag`
- `corpus_axis_bin`

Corpus feature meanings follow the QuCo-RAG-style reference pattern: entity frequency comes from direct corpus count queries, entity-pair co-occurrence comes from corpus windowed pair counts, and low/zero flags are thresholded derivatives. These values are corpus-support conditions for reliability analysis, not direct correctness labels.

## 5. Baseline contract

All baseline methods must be implemented and reported:

- SE-only
- Energy-only
- logit-diagnostic-only
- corpus-axis-only
- global learned fusion without corpus axis
- global learned fusion with corpus axis
- corpus-bin feature selection
- corpus-bin weighted fusion
- axis-interaction logistic fusion

The main comparison is global fusion versus condition-aware fusion across corpus-axis bins. Single-signal baselines remain necessary to show which feature is reliable in which bin.

## 6. PC Probe guardrail

PC Probe is reference-only and not implemented. Hidden-state/probe features are excluded.

Phillips/PC Probe may be cited only for conceptual framing around entropy-only failure and confidently wrong errors. It must not appear as an implemented method, baseline, feature source, or training input.

## 7. Metrics contract

Report these metrics overall, per dataset, and per corpus-axis bin:

- AUROC for hallucination detection.
- AUPRC for hallucination detection under class imbalance.
- Accuracy, precision, recall, and F1 at the selected operating point.
- Calibration or threshold diagnostics for global fusion and condition-aware fusion.
- Per-bin paired win rate for candidate-level features, with hallucinated-minus-normal paired deltas.
- Prompt-grouped bootstrap confidence intervals.
- Corpus-axis bin analysis using quantile or predeclared fixed bins for entity frequency and entity-pair co-occurrence.
- Axis sensitivity analysis comparing 3-bin versus 5-bin settings where sample size allows.
- Feature alignment table marking each feature as paper-faithful, adapted, or diagnostic.

SE bins may be used as secondary analysis bins, but the main thesis axis is corpus support. Fine-grained bins are analysis strata only; they do not create new correctness labels.

## 8. Output schema contract

Every result artifact must be reproducible from a manifest and include enough metadata to trace formulas, features, datasets, corpus snapshots, and splits.

Required output elements:

- `run_id`: stable run identifier.
- `method_name`: one of the baseline names or `Corpus-Conditioned Hallucination Metric Reliability Study`.
- `dataset`: source dataset name.
- `split_id`: split or cross-dataset evaluation identifier.
- `prompt_id`, `pair_id`, `candidate_id`: stable paired identities.
- `candidate_label`: `right` or `hallucinated`.
- `is_correct` and `is_hallucination`: annotation-backed labels.
- `features`: object containing the required feature families used by the method.
- `corpus_axis`: object containing raw counts, transformed continuous scores, bins, and corpus provenance.
- `prediction_score`: hallucination score.
- `prediction_label`: predicted hallucination decision at the selected threshold.
- `metrics`: aggregate object with AUROC, AUPRC, accuracy, precision, recall, F1, paired win rate, and confidence intervals when applicable.
- `feature_alignment`: paper-faithful/adapted/diagnostic status.
- `feature_importance`: coefficient or importance object for learned fusion methods.
- `formula_manifest_ref`: pointer to paper-derived formula specs.
- `dataset_manifest_ref`: pointer to dataset and corpus snapshot manifests.

Future self-contained generation runs under this repo must no longer depend on `../hallucination_lfe` for prompt generation or logits export. New row-level generation artifacts must preserve machine-readable per-token full logits either inline for tiny fixtures or through `full_logits_ref` pointing to a same-stem `.full_logits.parquet` sidecar for live Qwen-scale runs. They must also preserve per-step `logsumexp`, selected/generated token ids, and provenance fields such as `model_name`, `tokenizer_name`, `generation_config`, `logits_schema_version`, `formula_manifest_ref`, `dataset_manifest_ref`, and `created_at`.

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
- Phillips/PC Probe is reference-only. It supplies conceptual framing for entropy-only failure and confidently wrong cases, but PC Probe is not implemented and hidden-state/probe features are excluded.

## 11. Validation command

Run the contract validator after editing this file:

```bash
uv run python experiments/scripts/validate_readme_contract.py experiments/README.md
```

## 12. Environment and execution contract

- Create or refresh the repo-local virtual environment with `uv sync`.
- Install heavyweight generation dependencies only when needed with `uv sync --group generation`.
- Run experiment CLIs through `uv run`.
- The checked-in `experiments/configs/generation.yaml` live path uses the Qwen2.5 causal LM and tokenizer specified by `model.model_name` / `model.tokenizer_name` on CUDA by default.
- Semantic Entropy free sampling uses an answer-only answer-span protocol. The redesigned thesis-valid SE artifact requires prompt-level N=10 valid samples, NLI cluster metadata, and likelihood-based cluster probability fields.
- Live generation uses `runtime.free_sample_batch_size` and `runtime.candidate_score_batch_size` to batch Qwen forward passes for throughput while preserving checkpoint-backed resume semantics.
- `run_generation.py --resume` is the default S2 policy in the pipeline manifest.
- Use `uv run python experiments/scripts/run_pipeline.py --dry-run --out experiments/results/runs` to materialize the reproducible command manifest before executing the experiment.
- Thesis claims require the single experiment dataset pipeline plus all gates in `experiments/PIPELINE.md`.
