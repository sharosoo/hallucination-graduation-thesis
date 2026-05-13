# Experiments Contract

This directory is the binding experiment contract for the `Corpus-Conditioned Hallucination Metric Reliability Study`. Write code under `experiments/` only after this contract validates.

The executable pipeline is fixed in [`experiments/PIPELINE.md`](PIPELINE.md). The active main evaluation is the **트랙 B (SE 5-dataset, generation-level)** track. If this README and `PIPELINE.md` disagree, treat `PIPELINE.md` as the operational source of truth for run order, validation gates, and thesis-valid evidence requirements.

## 1. Method name and thesis claims

The method name is exactly `Corpus-Conditioned Hallucination Metric Reliability Study`.

The active thesis (Phase 3) is the SE 5-dataset, generation-level evaluation. The unit of evaluation is one free-sample generation. Total 35,000 generations are produced from 3,500 prompts × 10 free samples per prompt with Qwen2.5-3B (base, fp16).

The experiment tests three thesis claims:

1. Per-generation correctness on the SE 5-dataset (TriviaQA, SQuAD-1.1, BioASQ, NQ-Open, SVAMP) is the appropriate evaluation unit for Semantic Entropy and Semantic Energy because it matches the unit used by Farquhar (Nature 2024) and Ma (2025).
2. Paper-faithful Semantic Entropy and Semantic Energy must be evaluated as metric families whose AUROC range can change across corpus-axis bins, not as globally fixed hallucination solutions. The decomposition power depends on which corpus signal unit (entity frequency, entity-pair co-occurrence, question-answer bridge, n-gram coverage) defines the bin axis.
3. Fusion of paper-faithful uncertainty signals with corpus signals is evaluated under prompt-grouped 5-fold GroupKFold to prevent same-sample leakage across train and test.

The proposed contribution is not a RAG system, not answer-generation improvement, and not a production hallucination detector. QuCo-RAG-style corpus statistics are used as corpus-support axes that may condition metric reliability. Low entity frequency or zero entity-pair co-occurrence is not itself a hallucination label.

## 2. Label contract

Every free-sample row keeps an NLI-derived binary correctness label.

| Field | Rule | Role |
| --- | --- | --- |
| `is_correct` | $M(s_i) = \max_{c \in C} \max(p(c \to s_i), p(s_i \to c)) \geq 0.5$ | Generation-level correctness label. |
| `nli_max_prob` | $M(s_i)$ | Raw NLI bidirectional entailment score. |
| `match_method` | `nli_bidirectional_max_entail` | Default; falls back to `token_overlap` only when transformers / torch are missing. |
| `token_overlap_match` | jaccard ≥ 0.5 or substring | Sanity column kept alongside the NLI label. |

The candidate set $C$ is the union of dataset-provided ground-truth answer, equivalent expressions, and aliases (lower-cased, deduplicated). The NLI model is `microsoft/deberta-large-mnli`, identical to the Semantic Entropy clustering model. Row unit is `(prompt_id, sample_index)`.

이 라벨은 Farquhar 등 (2024) 와 Ma 등 (2025) 의 generation-level correctness 와 동일 평가 단위이며, candidate-level `is_hallucination` 라벨은 본 트랙에서 산출하지 않는다 (paired hallucinated candidate 가 데이터셋에 존재하지 않음). 학습 target 은 `is_correct` (이진) 이며, main 분석 축은 corpus 신호 단위별 10분위 구간이다.

Correctness labels, gold answers, and dataset annotations are label-only metadata. They must not leak into trainable features.

## 3. Dataset contract

The active thesis dataset is the SE 5-dataset single-candidate experiment dataset built from the same 5 datasets used by Farquhar (Nature 2024) Semantic Entropy and Ma (2025) Semantic Energy.

| Dataset | HF id | Config | Split | Sample count | Seed |
| --- | --- | --- | --- | ---: | ---: |
| TriviaQA | `trivia_qa` | `rc.nocontext` | `validation` | 800 | 13 |
| SQuAD-1.1 | `rajpurkar/squad` | (none) | `validation` | 800 | 23 |
| BioASQ | `kroshan/BioASQ` | (none) | `train` | 800 | 31 |
| NQ-Open | `nq_open` | (none) | `validation` | 800 | 41 |
| SVAMP | `ChilleD/SVAMP` | (none) | `test` | 300 | 53 |

총 3,500 sample. 각 sample 마다 데이터셋이 제공한 정답 표현 (단일 ground-truth 또는 alias 목록) 한 개를 right candidate 로 보존하며, paired hallucinated candidate 는 만들지 않는다. 정답성 판정은 모델이 자유 생성한 답변 10개에 대해 §2 의 NLI 양방향 함의 매칭으로 사후 산출한다.

Prompt template (Farquhar 2024 의 sentence-length 시나리오):

```
Answer the following question in a single brief but complete sentence.
Question: {question}
Answer:
```

Context passage 는 모든 데이터셋에서 의도적으로 제외한다 (Farquhar §Methods — confabulation 유도). 정확한 hf id, split, seed, candidate policy 는 `experiments/configs/datasets_se.yaml` 에 고정되어 있다.

## 4. Feature contract

Trainable features must not use correctness labels, gold answers, or hidden-state probe features. Required feature families are grouped by research role.

### Paper-faithful uncertainty features (sample 단위)

- `semantic_entropy_nli_likelihood`
- `semantic_entropy_cluster_count`
- `semantic_entropy_discrete_cluster_entropy`
- `semantic_energy_cluster_uncertainty`
- `semantic_energy_sample_energy`

Semantic Entropy must use N=10 free samples, NLI-based semantic clustering, and likelihood-based cluster probabilities. Semantic Energy must use the same semantic clusters, selected-token logit-derived energy, and cluster-level aggregation (sum) before it is called paper-faithful Semantic Energy. 두 신호는 sample 단위로 산출되어, 같은 sample 의 모든 답변에 동일한 값이 broadcast 된다.

### Free-sample diagnostic features (답변 단위)

- `sample_nll` (평균 음의 로그우도)
- `sample_sequence_log_prob` (시퀀스 로그우도)
- `sample_logit_variance` (logit 분산)
- `sample_logsumexp_mean` (평균 로그 분배함수)
- `sample_confidence_margin_mean`, `sample_confidence_margin_min`, `sample_top1_logit_mean` (top1-top2 streaming, 선택)

이 신호들은 답변마다 산출된 변동 정보를 담아 fusion 모델의 입력으로 사용된다. paper-faithful Semantic Energy 와 같은 column 으로 합치지 않는다.

### Corpus-axis features (sample 단위)

- `entity_frequency`, `entity_frequency_axis`, `entity_frequency_min`
- `entity_pair_cooccurrence`, `entity_pair_cooccurrence_axis`
- `low_frequency_entity_flag`, `zero_cooccurrence_flag`
- `corpus_axis_bin`, `corpus_axis_bin_5`, `corpus_axis_bin_10`
- `qa_bridge_axis`, `qa_bridge_min`, `qa_bridge_zero_flag`
- `ans_ngram_3_axis`, `ans_ngram_5_axis`, `ans_ngram_3_zero_count`, `ans_ngram_5_zero_count`

Corpus feature meanings follow QuCo-RAG-style direct count semantics on a fixed Infini-gram local backend (`v4_dolmasample_olmo`, 16B Dolma sample tokens). 답변에서 추출한 entity 의 단일 등장 빈도와 entity 쌍 동시 등장이 entity 수준 신호이고, 질문 entity 와 답변 entity (질문과 겹치지 않는 부분) 의 동시 등장이 question-answer bridge 신호이며, 답변 토큰 시퀀스의 3-gram / 5-gram 등장 빈도가 n-gram 수준 신호이다. 이 값들은 신뢰도 conditioning axis 이며 직접적인 정답성 라벨이 아니다.

### Entity extractor backends

Entity extraction is pluggable via `EntityExtractorPort` (see `experiments/ports/entity_extractor.py`).

- `spacy` (**default, recommended**): spaCy `en_core_web_lg` NER, filtered to PERSON / ORG / GPE / LOC / DATE / EVENT / WORK_OF_ART / FAC / NORP / PRODUCT / LANGUAGE / LAW. Falls back to noun chunks for short texts that the NER pipeline misses, and finally to the cleaned text itself for atomic candidates (e.g. `"Delhi"`). CPU-only, ~1.4 ms/text.
- `regex` (legacy, archived): quote-wrapped phrases, 1–4 word capitalized n-grams, non-stopword tokens of length ≥5. Misses short atomic entities, includes noisy verbs / common-nouns. Kept for reproducing pre-2026-05 artifacts.
- `quco` (experimental, archived): `ZhishanQ/QuCo-extractor-0.5B` knowledge triplet model. On short factoid answers it returns 100% empty triplets. Adapter retained for reference.

Selection is via the `--entity-extractor {spacy,regex,quco}` flag on `compute_corpus_features.py`. Switching extractor only requires re-running S8' → S11'; S2'/S4'/S5'/S6' artifacts are entity-independent and can be reused.

## 5. Baseline contract

All baseline methods must be implemented and reported.

- SE-only (no training; rank-based AUROC)
- Energy-only (rank-based)
- 답변 단위 logit 통계 단독 (sample_nll, sample_logit_variance 등)
- corpus-signal-only (단위별 — entity 빈도, entity 쌍, qa-bridge, n-gram)
- Logistic Regression CORE (fusion: SE + Energy + 답변 단위 통계 4종)
- Logistic Regression CORE + CORPUS
- Random Forest CORE / CORE + CORPUS
- Gradient Boosting CORE / CORE + CORPUS

The main comparison is fusion AUROC under the CORE vs CORE + CORPUS input conditions, with per-corpus-signal-unit decile decomposition reported separately. Single-signal baselines remain necessary to show how each signal's AUROC range varies across corpus-decile bins.

## 6. PC Probe guardrail

PC Probe is reference-only and not implemented. Hidden-state / probe features are excluded.

Phillips / PC Probe may be cited only for conceptual framing around entropy-only failure and confidently wrong errors. It must not appear as an implemented method, baseline, feature source, or training input.

## 7. Metrics contract

Report these metrics overall, per dataset, and per corpus-signal-unit decile.

- AUROC for hallucination detection (`is_correct` 의 negation 을 양성 클래스로 둔 ranking).
- AUPRC for hallucination detection under class imbalance.
- AURAC (Area Under Rejection-Accuracy Curve, Farquhar Nature 2024 main metric).
- Brier score, ECE for calibration of fusion.
- Per-corpus-signal-unit decile AUROC; report range $\Delta = \mathrm{AUROC}_{\max} - \mathrm{AUROC}_{\min}$ for each (signal, axis) pair.
- Prompt-grouped bootstrap confidence intervals (B = 500 default; B = 1,000 for headline) on AUROC and on Δ AUROC differences across axes.
- Spearman $\rho$ of decile rank vs per-decile AUROC for monotonicity.
- Leave-one-dataset-out (LODO) and per-dataset AUROC for robustness.
- Threshold calibration sensitivity for fusion classifiers.

SE bins may be used as secondary analysis bins, but the main thesis axis is corpus-signal-unit decile. Fine-grained bins are analysis strata only; they do not create new correctness labels.

## 8. Output schema contract

Every result artifact must be reproducible from a manifest and include enough metadata to trace formulas, features, datasets, corpus snapshots, and splits.

Required output elements per row of `generation_features.parquet`:

- `run_id`: stable run identifier.
- `method_name`: one of the baseline names or `Corpus-Conditioned Hallucination Metric Reliability Study`.
- `dataset`: source dataset name (TriviaQA / SQuAD-1.1 / BioASQ / NQ-Open / SVAMP).
- `prompt_id`, `sample_index`: stable per-generation identifiers (prompt_id used as the GroupKFold group).
- `is_correct`, `nli_max_prob`, `match_method`, `token_overlap_match`: NLI-derived correctness label and audit columns.
- `features`: object containing the required feature families used by the method (paper-faithful uncertainty, free-sample diagnostics, corpus-axis).
- `corpus_axis`: object containing raw counts, transformed continuous scores, decile bin ids, and corpus provenance.
- `prediction_score`, `prediction_label`: fusion model output and decision at the selected threshold.
- `metrics`: aggregate object with AUROC, AUPRC, AURAC, Brier, ECE, decile range, prompt-grouped bootstrap CI when applicable.
- `feature_alignment`: paper-faithful / adapted / diagnostic status.
- `feature_importance`: coefficient or importance object for learned fusion methods.
- `formula_manifest_ref`, `dataset_manifest_ref`, `model_name`, `tokenizer_name`, `generation_config`, `logits_schema_version`, `created_at`.

Generation runs preserve per-token full logits via `full_logits_ref` pointing to a same-stem `.full_logits.parquet` sidecar. They also preserve per-step `logsumexp`, generated token ids, and provenance fields.

## 9. Hexagonal architecture convention

The experiment implementation must use backend-style typed hexagonal architecture, not notebook-style scripts.

- `domain`: typed dataclasses, enums, value objects, labels, feature records, predictions, metrics, and provenance objects.
- `ports`: abstract interfaces for datasets, formula specs, corpus statistics, model logits, storage, and evaluation output.
- `adapters`: concrete readers and clients for upstream artifacts, Hugging Face datasets, corpus-stat services, model / logit sources, local storage, and serialized manifests.
- `application`: use-case services that orchestrate labeling, feature extraction, fusion training, evaluation, and export without depending on concrete adapters.
- `scripts`: thin CLI entrypoints that parse arguments and call application services.
- `configs`: dataset, literature, formula, feature, fusion, and evaluation configuration files.
- `manifests`: derived dataset, corpus, feature, formula, and run manifests with hashes or stable references.
- `results`: generated metrics, row-level predictions, tables, figures, and final reports.
- `literature`: saved papers, checksums, extracted formula notes, and evidence notes used by thesis prose.

## 10. Reference-status caveats

- QuCo-RAG is used as corpus frequency and entity-pair co-occurrence inspiration. Cite with a preprint or status caveat.
- Ma Semantic Energy is used as the Semantic Energy source. Cite with a preprint or submission caveat. Eq. (12) SUM + Eq. (8) likelihood weighting is the paper-faithful interpretation adopted here.
- Phillips / PC Probe is reference-only. PC Probe is not implemented and hidden-state / probe features are excluded.

## 11. Environment and execution contract

- Create or refresh the repo-local virtual environment with `uv sync`.
- Install heavyweight generation dependencies only when needed with `uv sync --group generation`.
- Run experiment CLIs through `uv run`.
- The checked-in `experiments/configs/generation_se_qwen.yaml` live path uses `Qwen/Qwen2.5-3B` as both `model.model_name` and `model.tokenizer_name` on CUDA by default.
- Free sampling uses temperature=1.0, top_p=0.9, top_k=50, max_new_tokens=64, free_sample_count=10. Sentence-length generation (`answer_only.enabled=false`).
- Truncated samples (12.6% under max_new_tokens=64) are re-generated at max_new_tokens=128 for the affected sample subset; AUROC change in re-generation lies within ±0.001.
- Live generation uses `runtime.free_sample_batch_size` and `runtime.candidate_score_batch_size` to batch Qwen forward passes for throughput while preserving checkpoint-backed resume semantics.
- `run_generation.py --resume` is the default S2' policy, with `consolidate_checkpoints_se.py` re-assembling `free_sample_rows.json` from per-shard checkpoints.
- Thesis claims require the SE 5-dataset pipeline (트랙 B, S1' → S11') plus all gates in `experiments/PIPELINE.md`.

## Legacy / 시행착오

Phase 1 (candidate-level paired, TruthfulQA + HaluEval-QA) 와 Phase 2 (prompt-level `is_hard` proxy) 는 폐기되었다. 자세한 사유와 폐기 근거는 `HISTORY.md` 참조. Legacy paired-track 산출물 (`datasets.yaml`, S0–S13) 은 트랙 A 로 `PIPELINE.md` 에 보존되어 있으며, 향후 archive 디렉터리로 이동 예정이다.
