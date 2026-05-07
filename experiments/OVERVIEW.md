# 실험 전체 개관

이 문서는 처음 저장소를 보는 사람이 실험 목표, 데이터셋, 단계별 산출물, 검증 모듈을 한 번에 이해하도록 만든 안내서이다. 실행 계약의 source of truth는 `experiments/PIPELINE.md`이다. 이 repo에는 논문용 실험 데이터셋이 하나만 있다.

## 1. 문제와 논지

이 논문은 RAG 시스템 논문이 아니라 **환각 탐지 metric의 조건부 신뢰성**을 분석하는 논문이다. 질문은 다음과 같다.

> corpus에서 entity가 얼마나 자주 등장하는지, 그리고 entity pair가 함께 등장하는지에 따라 Semantic Entropy, Semantic Energy, likelihood/logit diagnostics 같은 hallucination metric의 신뢰도가 달라지는가?

따라서 corpus feature는 정답성의 직접 증거가 아니다. QuCo-RAG에서 entity frequency와 co-occurrence가 high-uncertainty retrieval 상황을 찾는 데 쓰였다는 직관을 가져오되, 여기서는 retrieval 정책이 아니라 **metric reliability를 나누는 continuous corpus-support axis**로 사용한다.

핵심 논지는 세 가지다.

1. Corpus entity frequency와 entity-pair co-occurrence는 hallucination metric의 reliability를 조건화할 수 있는 연속 축이다.
2. Semantic Entropy와 Semantic Energy는 원 논문 방식에 맞춰 구현한 뒤, corpus-axis bin마다 성능과 안정성을 따져야 한다.
3. Global fusion 하나만 보는 대신 corpus-bin-aware fusion 또는 axis interaction fusion이 더 해석 가능하거나 안정적인지 검증한다.

## 2. 평가 프로토콜: discriminative pair-based

휴리스틱 매칭과 LLM-as-judge를 모두 배제하기 위해 **discriminative 프로토콜**을 사용한다. 모델이 자유 생성한 답을 사후 판정하지 않고, **데이터셋이 미리 제공한 (정답, 환각) 후보 답 자체를 평가 대상**으로 삼는다.

- 한 prompt당 두 개의 평가 row가 생성된다.
  - `(question, right_answer)` → `is_correct=true`
  - `(question, hallucinated_answer)` → `is_correct=false`
- 라벨은 dataset annotation에서 직접 온다. 매칭도 judge도 없다.
- 모델은 후보 답을 새로 생성하지 않는다. teacher-forced로 후보 답에 대한 full-vocab logits를 계산해서 candidate-level diagnostics를 만든다.
- Semantic Entropy와 paper-faithful Semantic Energy를 위해서는 prompt에 대해 별도로 **N=10 answer-only free sampling**을 수행한다.

## 3. 실험 데이터셋

| Dataset | HF id | Prompts | Eval rows | (정답, 환각) 쌍 출처 |
| --- | --- | ---: | ---: | --- |
| TruthfulQA | `truthfulqa/truthful_qa` | 815 | 1,630 | dataset의 `correct_answers[]`와 `incorrect_answers[]`를 직접 짝지음. 정제 후 중복/overlap 2개 prompt는 제외 |
| HaluEval-QA | `pminervini/HaluEval` | 5,000 | 10,000 | dataset annotation의 `right_answer` / `hallucinated_answer` |

총 prompt 5,815 / 평가 row 11,630.

TriviaQA, Natural Questions, HotpotQA, FEVER, BioASQ는 사용하지 않는다. 이 데이터셋들은 (정답, 환각) 쌍을 dataset 차원에서 제공하지 않아 hallucinated 후보를 깨끗하게 만들 방법이 없다.

## 4. 한눈에 보는 파이프라인

```text
S0 계약 검증
  ↓
S1 HF dataset → prompt_groups.jsonl + candidate_rows.jsonl
  ↓
S2 model scoring
   ├─ teacher-forced candidate logits        (candidate-level diagnostics)
   └─ answer-only free samples N=10          (SE / Semantic Energy용)
  ↓
S2b full logits + sample-count 검증
  ↓
S3 annotation-driven correctness dataset
  ↓
S4 NLI likelihood Semantic Entropy           (prompt-level)
  ↓
S5 QuCo-style corpus axis                    (candidate/pair-level continuous bins)
  ↓
S6 paper-faithful Semantic Energy + diagnostics
  ↓
S7 feature table + axis/bin metadata
  ↓
S8 global fusion vs condition-aware fusion
  ↓
S9 corpus-bin reliability + robustness
```

실행:

```bash
uv sync --group generation
uv run python experiments/scripts/run_pipeline.py --execute --out experiments/results/runs
```

## 5. feature 계산 단위

| Feature | level | 두 candidate row 간 | 계산 방식 |
| --- | --- | --- | --- |
| `semantic_entropy_nli_likelihood` | prompt | 공유(broadcast) | prompt에서 N=10 answer-only sample → DeBERTa-family NLI clustering → likelihood-based cluster entropy |
| `semantic_entropy_discrete_cluster_entropy`, `semantic_entropy_cluster_count` | prompt | 공유 | 같은 NLI cluster에서 count-based 보조 통계 |
| `semantic_energy_cluster_uncertainty` | prompt/cluster | 공유 또는 cluster-linked | 여러 generated answers의 semantic clusters와 selected-token logit-derived energy를 cluster level로 집계 |
| `mean_negative_log_probability` | candidate | 다름 | 후보 토큰의 −log p 평균 |
| `logit_variance`, `confidence_margin` | candidate | 다름 | 후보 토큰 위치 logits에서 계산하는 diagnostic |
| `semantic_energy_boltzmann_diagnostic` | candidate | 다름 | 기존 candidate-level `-logsumexp` 평균. paper-faithful Semantic Energy가 아니라 energy-inspired diagnostic으로 표기 |
| `entity_frequency_axis`, `entity_pair_cooccurrence_axis`, `corpus_axis_bin` | candidate/pair | 다름 | candidate 답 텍스트에서 entity를 추출하고 Infini-gram-compatible count backend로 raw count, log score, bin을 계산 |

## 6. corpus-axis 분석 원칙

- Low entity frequency는 오답 라벨이 아니라 long-tail corpus support 부족 조건이다.
- Zero entity-pair co-occurrence는 관계적 support 부족을 나타내는 위험 조건이지만, non-zero co-occurrence가 정답을 보장하지 않는다.
- Corpus axis는 continuous score로 보존하고, train split quantile 또는 사전 고정 threshold로 binning한다.
- 각 bin에서 AUROC, AUPRC, paired win-rate, hallucinated-minus-normal delta, prompt-grouped bootstrap CI를 보고한다.
- SE는 보조 축으로 쓸 수 있지만, redesigned thesis의 기본 축은 corpus support이다.

## 7. 단계별 모듈과 검증

| Stage | Module | Output | Validation |
| --- | --- | --- | --- |
| S0 | `validate_architecture.py` | pass/fail | hexagonal 패키지 구조와 핵심 dataclass·port 강제 |
| S1 | `prepare_datasets.py`, `adapters/hf_datasets.py` | `prompt_groups.jsonl`, `candidate_rows.jsonl`, `dataset_manifest.json` | pair_id 짝의 (right, hallucinated) 1:1 검증 |
| S2 | `run_generation.py`, `adapters/model_generation.py` | `free_sample_rows.json`, `candidate_scores.json` | N=10 free samples와 candidate logits schema 검사 |
| S3 | `build_correctness_dataset.py`, `adapters/correctness_dataset.py` | `correctness_judgments.jsonl` | annotation source presence, `heuristic_matching_used=false`, `llm_as_judge_used=false` |
| S4 | `compute_semantic_entropy.py`, NLI clustering adapter | `semantic_entropy_features.parquet` | cluster prob 합 1, NLI model provenance, likelihood-based fields |
| S5 | `compute_corpus_features.py`, corpus count adapter | `corpus_features.parquet` | raw count provenance, axis score, bin assignment |
| S6 | `compute_energy_features.py`, Semantic Energy adapter | `energy_features.parquet` | selected-token logit energy, cluster-level aggregation, diagnostic separation |
| S7 | `build_feature_table.py`, `application/labeling.py` | `features.parquet` | prompt-level broadcast와 corpus-bin join 무결성 |
| S8 | `run_fusion.py`, `application/fusion.py` | `fusion/summary.json`, `predictions.jsonl` | global vs condition-aware comparison |
| S9 | `run_robustness.py`, `application/robustness.py` | `robustness/summary.json` | prompt-grouped bootstrap, corpus-bin CI, binning sensitivity |

## 8. 주의점

- 휴리스틱 매칭, LLM-as-judge fallback은 모두 금지. 정답성은 dataset annotation에서만 온다.
- 현재 N=5 exact-string SE artifact는 preliminary diagnostic이다. 최종 논문용 SE는 N=10, NLI clustering, likelihood-based cluster probability가 필요하다.
- 현재 candidate-level `semantic_energy_boltzmann`은 useful diagnostic일 수 있지만, multi-generation semantic clustering과 cluster-level energy aggregation이 없으면 paper-faithful Semantic Energy라고 쓰지 않는다.
- Corpus feature는 direct count semantics에서 와야 한다. Elasticsearch/BM25는 retrieval evidence 용도이며 entity frequency/co-occurrence count 대체물이 아니다.
- Robustness evaluation은 prompt 단위로 묶는다. 같은 prompt에서 나온 두 candidate row가 train/test로 갈라지면 누수다.
