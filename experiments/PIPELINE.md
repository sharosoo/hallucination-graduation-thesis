# 논문 실험 파이프라인

이 문서는 처음 보는 연구자가 같은 산출물을 다시 만들 수 있도록 고정한 실행 계약이다. 이 repo에는 논문용 **실험 데이터셋**이 하나만 있다. 실험 데이터셋은 TruthfulQA와 HaluEval-QA에서 제공되는 (정답, 환각) 후보 쌍을 prompt 단위로 정렬한 paired discriminative dataset이다.

검증 기준은 단순하다. **Full logits must be repo-owned**, **Corpus statistics must be repo-owned**, **Infini-gram-compatible count backend**를 사용한다. **Elasticsearch/BM25 is used for retrieval** only, and **Elasticsearch/BM25 may be used for retrieval evidence**; count 대체물로 쓰지 않는다. 학습기는 **custom stdlib L2 logistic regression**이다. correctness는 dataset annotation에서 직접 온다. heuristic 문자열 매칭, generated-answer 사후 판정, LLM-as-judge fallback은 thesis-valid label source가 아니다. 문서-구현 정렬은 `run_pipeline.py`와 코드 리뷰가 책임진다.

이 파이프라인의 연구 질문은 RAG 시스템 구축이 아니다. QuCo-RAG에서 온 entity frequency와 entity-pair co-occurrence를 **continuous corpus-support axis**로 만들고, 그 axis의 bin마다 hallucination metric reliability가 어떻게 달라지는지 검증한다.

## 1. 전체 방법론 다이어그램

```mermaid
flowchart TD
    A[1. HF paired dataset 저장
TruthfulQA + HaluEval-QA]
    B[2. prompt group 고정
prompt_groups.jsonl
dataset_manifest.json]
    C[3. candidate row 확장
candidate_rows.jsonl
right + hallucinated]
    D[4. model scoring
teacher-forced candidate logits
answer-only free samples N=10]
    E[5. generation/logits 검증
full-vocab logits + logsumexp + sample count]
    F[6. annotation label artifact
annotation_labels.jsonl]
    G[7. NLI likelihood Semantic Entropy
prompt-level clusters]
    H[8. QuCo-style corpus axis
entity frequency + pair co-occurrence]
    I[9. paper-faithful Semantic Energy
semantic clusters + selected-token energy]
    J[10. candidate diagnostics
NLL + margin + variance]
    K[11. feature table
features + corpus bins + labels]
    L[12. global and condition-aware fusion]
    M[13. corpus-bin reliability robustness]

    A --> B --> C --> D --> E
    C --> F
    D --> G
    C --> H
    D --> I
    D --> J
    F --> K
    G --> K
    H --> K
    I --> K
    J --> K
    K --> L --> M
```

## 2. 전체 실행 명령

manifest만 만들 때:

```bash
uv run python experiments/scripts/run_pipeline.py --dry-run --out experiments/results/runs
```

실제로 산출물을 만들 때:

```bash
uv sync --group generation
uv run python experiments/scripts/run_pipeline.py --execute --out experiments/results/runs
```

기본 live generation 설정은 `experiments/configs/generation.yaml`의 `model.model_name` / `model.tokenizer_name`에 고정된 Qwen2.5 계열 causal LM과 같은 계열 tokenizer를 CUDA에서 사용한다. `run_pipeline.py`는 stage별 command, expected input/output, validator command, planned stdout/stderr log, git commit, Python version, platform, UTC timestamp를 manifest에 저장한다. 또한 manifest top-level `script_execution_log`에 primary/validation script와 인자, command string, planned stdout/stderr log path, 실행했다면 return code를 사람이 바로 읽을 수 있는 형태로 중복 기록한다.

## 3. 단계별 해야 할 일

### S0. 구조 검증

```bash
uv run python experiments/scripts/validate_architecture.py
```

목적: 도메인/포트/어댑터/애플리케이션 패키지 구조와 핵심 dataclass·port가 hexagonal 계약을 지키는지 확인한다. 문서-구현 정렬은 사람의 변경 검토와 코드 리뷰가 책임진다.

### S1. paired prompt group 및 candidate row 생성

```bash
uv run python experiments/scripts/prepare_datasets.py --config experiments/configs/datasets.yaml --out experiments/results/datasets
```

- 입력: `experiments/configs/datasets.yaml`
- 모듈: `experiments/scripts/prepare_datasets.py`, `experiments/adapters/hf_datasets.py`
- 출력:
  - `experiments/results/datasets/prompt_groups.jsonl`
  - `experiments/results/datasets/candidate_rows.jsonl`
  - `experiments/results/datasets/dataset_manifest.json`
  - `experiments/results/datasets/dataset_preparation_report.json`

| Dataset | Prompt unit | Candidate rows | Label source |
| --- | --- | --- | --- |
| TruthfulQA | `question` | one deterministic `right` candidate and one deterministic `hallucinated` candidate per prompt | `correct_answers[]` and `incorrect_answers[]` annotation |
| HaluEval-QA | `knowledge + question` | dataset `right_answer` and `hallucinated_answer` | HaluEval-QA annotation |

`candidate_rows.jsonl` stores exactly two rows per `pair_id`: `candidate_label=right` with `is_correct=true`, and `candidate_label=hallucinated` with `is_correct=false`.

### S2. model scoring: candidate logits와 prompt free samples

```bash
uv run python experiments/scripts/run_generation.py --config experiments/configs/generation.yaml --prompt-groups experiments/results/datasets/prompt_groups.jsonl --candidates experiments/results/datasets/candidate_rows.jsonl --out-free-samples experiments/results/generation/free_sample_rows.json --out-candidate-scores experiments/results/generation/candidate_scores.json --resume
uv run python experiments/scripts/validate_generation_logits.py experiments/results/generation/free_sample_rows.json
uv run python experiments/scripts/validate_generation_logits.py experiments/results/generation/candidate_scores.json
```

- 입력: `prompt_groups.jsonl`, `candidate_rows.jsonl`, `experiments/configs/generation.yaml`
- 모듈: `experiments/scripts/run_generation.py`, `experiments/adapters/model_generation.py`
- 출력:
  - `experiments/results/generation/free_sample_rows.json`
  - `experiments/results/generation/candidate_scores.json`
- 저장 필드:
  - candidate-level teacher-forced token ids, selected token logits, per-step `logsumexp`, per-step full-vocabulary logits reference, model/tokenizer provenance
  - prompt-level answer-only free samples with `sample_index` from 0 to 9, response text, generated token ids, sequence log-likelihood, generation provenance, and full-vocabulary logits reference

모델은 candidate 답을 자유 생성하지 않는다. candidate feature는 `prompt + candidate_answer`를 teacher-forced로 scoring해서 만든다. Semantic Entropy와 paper-faithful Semantic Energy를 위해서만 prompt당 N=10 answer-only free samples를 별도로 만든다.

증분 실행 정책: 기존 N=5 artifact가 있으면 sample indexes 0--4는 보존하고 5--9만 추가 생성한다. 최종 thesis-valid artifact는 prompt마다 sample indexes `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}`가 모두 있어야 한다.

S2 validation gate: 각 candidate token position의 inline `full_logits` 또는 `full_logits_ref`가 가리키는 Parquet sidecar vector 길이가 tokenizer vocab size와 같아야 한다. top-k-only logits는 실패다. prompt free samples는 prompt별 10개 group이 모두 있어야 한다. answer-only artifact는 모든 `response_text`가 non-empty single-line answer span이어야 하며 invalid final sample rate가 0%여야 한다. generation validation은 별도 numbered stage가 아니라 S2 validator set으로 manifest에 기록한다.

### S3. annotation label artifact 생성

```bash
uv run python experiments/scripts/build_correctness_dataset.py --candidates experiments/results/datasets/candidate_rows.jsonl --out experiments/results/correctness
```

- 입력: `candidate_rows.jsonl`
- 모듈: `experiments/scripts/build_correctness_dataset.py`, `experiments/adapters/correctness_dataset.py`
- 출력:
  - `experiments/results/correctness/data/correctness_judgments.jsonl`
  - `experiments/results/correctness/dataset_manifest.json`
  - `experiments/results/correctness/README.md`

이 artifact는 공개 가능한 label-source dataset이다. correctness 정보는 label-only이며 trainable feature가 아니다.

### S4. NLI likelihood Semantic Entropy feature 생성

```bash
uv run python experiments/scripts/compute_semantic_entropy.py --free-samples experiments/results/generation/free_sample_rows.json --out experiments/results/semantic_entropy_features.parquet
```

- 입력: prompt-level N=10 free samples in `free_sample_rows.json`
- 모듈: `experiments/scripts/compute_semantic_entropy.py`, NLI clustering adapter
- 출력: `experiments/results/semantic_entropy_features.parquet`
- 계산:
  - DeBERTa-family NLI model로 sampled answer 간 entailment를 계산한다.
  - semantic equivalence는 bidirectional entailment 또는 사전 고정 relaxed entailment rule로 정의한다.
  - 각 sample의 sequence likelihood에서 cluster probability mass를 log-sum-exp로 계산한다.
  - 최종 필드는 `semantic_entropy_nli_likelihood`, `semantic_entropy_cluster_count`, `semantic_entropy_discrete_cluster_entropy`, `nli_model_ref`, `sample_log_likelihoods`, `cluster_likelihoods`를 포함한다.

SE는 prompt-level 신호다. 같은 `pair_id`에서 나온 right/hallucinated candidate row는 같은 SE를 공유한다.

### S5. QuCo-style corpus axis 생성

```bash
uv run python experiments/scripts/compute_corpus_features.py --candidates experiments/results/datasets/candidate_rows.jsonl --out experiments/results/corpus_features.parquet
uv run python experiments/scripts/validate_feature_provenance.py experiments/results/corpus_features.parquet
```

- 입력: `candidate_rows.jsonl`
- 모듈: `experiments/scripts/compute_corpus_features.py`, corpus count adapter
- 출력: `experiments/results/corpus_features.parquet`
- 계산:
  - candidate answer text에서 entity를 추출한다.
  - Infini-gram-compatible count backend 또는 고정 corpus snapshot에서 entity frequency와 `head AND tail` co-occurrence를 조회한다.
  - raw counts, log-transformed continuous axis scores, low/zero flags, bin ids, corpus provenance를 저장한다.

Elasticsearch/BM25는 retrieval evidence 용도다. entity frequency와 entity-pair co-occurrence는 Infini-gram-compatible count backend의 direct count semantics를 따른다. Corpus feature는 hallucination label이 아니라 reliability conditioning axis이다.

### S6. paper-faithful Semantic Energy 및 logit diagnostics 생성

```bash
uv run python experiments/scripts/compute_energy_features.py --candidate-scores experiments/results/generation/candidate_scores.json --out experiments/results/energy_features.parquet
uv run python experiments/scripts/validate_energy_features.py experiments/results/energy_features.parquet
```

- 입력: `candidate_scores.json`, `free_sample_rows.json`, `semantic_entropy_features.parquet`의 semantic cluster metadata
- 모듈: `experiments/scripts/compute_energy_features.py`, Semantic Energy adapter
- 출력: `experiments/results/energy_features.parquet`
- 계산:
  - paper-faithful path: sampled responses의 selected-token logits에서 sample energy를 계산하고, S4 semantic clusters로 cluster-level uncertainty를 집계한다.
  - diagnostic path: teacher-forced candidate token window에서 `mean_negative_log_probability`, `logit_variance`, `confidence_margin`, `semantic_energy_boltzmann_diagnostic`을 계산한다.

`semantic_energy_boltzmann_diagnostic`은 token-level `-logZ`를 후보 답 길이로 평균낸 기존 diagnostic이다. multi-generation semantic clustering과 cluster-level energy aggregation이 없으면 이를 paper-faithful Semantic Energy라고 부르지 않는다.

### S7. feature table 결합 및 axis/bin metadata 확정

```bash
uv run python experiments/scripts/build_feature_table.py --inputs experiments/results --out experiments/results/features.parquet
uv run python experiments/scripts/validate_type_labels.py experiments/results/features.parquet
```

- 입력:
  - `correctness/data/correctness_judgments.jsonl`
  - `semantic_entropy_features.parquet`
  - `corpus_features.parquet`
  - `energy_features.parquet`
- 모듈: `experiments/scripts/build_feature_table.py`, `experiments/application/labeling.py`
- 출력: `experiments/results/features.parquet`

`annotation_labels.jsonl`의 correctness는 label-only source로만 저장한다. feature table에는 prompt-level NLI SE, paper-faithful Semantic Energy, candidate-level diagnostics, candidate-level corpus axis scores, corpus bin ids, dataset/prompt/pair/candidate identities가 함께 들어간다.

### S8. global fusion 및 condition-aware fusion 평가

```bash
uv run python experiments/scripts/run_fusion.py --features experiments/results/features.parquet --config experiments/configs/fusion.yaml --out experiments/results/fusion
```

평가 대상: SE-only, Energy-only, logit-diagnostic-only, corpus-axis-only, global learned fusion without corpus axis, global learned fusion with corpus axis, corpus-bin feature selection, corpus-bin weighted fusion, axis-interaction logistic fusion.

중요: headline은 learned fusion with corpus의 aggregate AUROC 하나가 아니다. S8은 global fusion과 condition-aware fusion을 전체 및 corpus-bin별로 비교한다. 각 feature가 어떤 corpus-support bin에서 reliable한지 함께 보고해야 한다.

출력: `experiments/results/fusion/summary.json`, `experiments/results/fusion/predictions.jsonl`, condition-aware fusion artifacts.

### S9. corpus-bin reliability 및 robustness 검증

```bash
uv run python experiments/scripts/run_robustness.py --features experiments/results/features.parquet --fusion experiments/results/fusion --out experiments/results/robustness
```

검증 항목: prompt-grouped bootstrap confidence interval, leave-one-dataset-out, within-dataset checks, calibration checks, corpus-bin metric reliability, binning sensitivity, and condition-aware fusion deltas.

각 corpus-axis bin에서 AUROC, AUPRC, paired win rate, hallucinated-minus-normal delta, confidence interval을 보고한다. confidence interval이 0을 가로지르면 안정적 개선이라고 쓰지 않는다. robustness split은 prompt 단위로 묶는다. 같은 prompt에서 나온 두 candidate row가 train/test로 갈라지면 누수다.

## 4. 최종 산출물 체크리스트

| Artifact | 생성 stage | 역할 |
| --- | --- | --- |
| `prompt_groups.jsonl` | S1 | prompt-level unit, SE grouping, robustness grouping |
| `candidate_rows.jsonl` | S1 | paired right/hallucinated candidate rows |
| `dataset_manifest.json` | S1 | dataset provenance |
| `free_sample_rows.json` | S2 | prompt-level N=10 free samples for SE and Semantic Energy |
| `candidate_scores.json` | S2 | teacher-forced candidate logits and token diagnostics |
| `correctness_judgments.jsonl` | S3 | annotation-derived correctness labels |
| `semantic_entropy_features.parquet` | S4 | NLI likelihood Semantic Entropy and cluster metadata |
| `corpus_features.parquet` | S5 | corpus support axes, bins, and provenance |
| `energy_features.parquet` | S6 | paper-faithful Semantic Energy + candidate diagnostics |
| `features.parquet` | S7 | fusion input table with corpus bins |
| `fusion/summary.json` | S8 | global and condition-aware fusion results |
| `robustness/summary.json` | S9 | corpus-bin reliability and robustness results |

논문 claim은 위 산출물이 모두 생성되고 각 validator가 통과한 run manifest에서만 작성한다.

## 5. Paper-derived feature alignment

| Feature family | Source | Required implementation check |
| --- | --- | --- |
| `semantic_entropy_nli_likelihood`, `semantic_entropy_cluster_count` | Farquhar Semantic Entropy | prompt-level N=10 free samples, NLI semantic clustering, likelihood-based cluster probability distribution에서 계산해야 하며 correctness/gold 정보를 feature로 쓰지 않는다. |
| `semantic_energy_cluster_uncertainty` | Ma Semantic Energy | multiple generated answers, semantic clusters, selected-token logit-derived energy, cluster-level aggregation이 모두 있어야 paper-faithful Semantic Energy로 표기한다. |
| `semantic_energy_boltzmann_diagnostic`, `mean_negative_log_probability`, `logit_variance`, `confidence_margin` | candidate token likelihood/logit diagnostics | candidate-level diagnostic으로 명명한다. Ma Semantic Energy 공식 자체라고 쓰지 않는다. |
| `entity_frequency_axis`, `entity_frequency_min` | QuCo-RAG | Infini-gram-compatible count backend의 entity frequency semantics를 따르며 continuous corpus-support axis로 저장한다. |
| `entity_pair_cooccurrence_axis` | QuCo-RAG | `head AND tail` pair co-occurrence semantics를 따르며 relation-level corpus-support axis로 저장한다. |
| condition-aware fusion | reliability analysis framing | corpus-axis bin 또는 axis interaction term을 사용해 global fusion과 비교한다. |

## 6. Guardrails

- No heuristic matching. String match, substring match, alias match, and normalized gold/reference match cannot create correctness labels.
- No generated-answer correctness labeling. Candidate labels come from dataset annotations before scoring.
- No LLM-as-judge fallback. A judge may be discussed only as an excluded path.
- `semantic_entropy_nli_likelihood` is prompt-level. Broadcasting it to both candidate rows in a pair is expected.
- Current exact-string N=5 SE artifacts are preliminary diagnostics, not final paper-faithful SE evidence.
- Paper-faithful Semantic Energy requires multi-generation semantic clusters and cluster-level selected-token-logit energy.
- Paper-faithful Semantic Energy requires multi-generation semantic clustering and cluster-level energy aggregation.
- Corpus axis values are not direct correctness labels.
- This experiment is not a RAG system; QuCo-RAG supplies corpus-support axis motivation only.
- Candidate-level Energy/logit diagnostics must use per-token mean length normalization.
- Corpus features are corpus-support axes and must come from direct count semantics, not retrieval scores.
- Robustness evaluation is grouped by prompt. Candidate siblings cannot be split across train/test.
