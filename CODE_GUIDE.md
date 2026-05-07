# 코드 모듈 가이드

이 문서는 새로 합류하는 사람이 30분 안에 코드 구조를 이해할 수 있도록 만든 가이드다.
실행 계약(reproducible run contract)은 `experiments/PIPELINE.md`, 식 ↔ 코드 매핑은
`experiments/literature/formula_notes.md`, 논지 개관은 `experiments/OVERVIEW.md`에 있다.

## 용어

- **paired discriminative protocol** — 데이터셋이 미리 제공한 `(정답 후보, 환각 후보)`
  쌍을 모델이 *생성하지 않고* teacher-forced로 scoring만 해서 환각 탐지기를 평가하는 방식.
  LLM-as-judge나 휴리스틱 매칭을 피하려는 본 논문의 평가 contract.
- **corpus axis** — 답에 등장하는 entity가 corpus에 얼마나 자주 등장하는지(entity
  frequency)와 entity pair가 같이 등장한 횟수(pair co-occurrence)를 사용해 만든 연속
  변수. 본 논문에서는 정답성의 직접 라벨이 아니라 *환각 탐지 신호의 신뢰도가 어떤
  조건에서 달라지는지* 측정하는 conditioning 축이다.
- **TypeLabel (= 진단용 4-class 라벨)** — 초기 실험에서 Semantic Entropy 값을 임계로
  환각 후보를 `NORMAL` / `HIGH_DIVERSITY` / `LOW_DIVERSITY` / `AMBIGUOUS_INCORRECT`
  네 등급으로 나누었던 라벨. 본 논문의 main 분석 축은 corpus axis이므로, 이 라벨은
  학습 target이 아니라 *진단용 메타데이터* 로만 보존한다 ("archived" = 보존만 됨,
  학습/평가 target 아님 — 코드/문서에서 자주 진단용 TypeLabel (NORMAL/HIGH_DIVERSITY/LOW_DIVERSITY/AMBIGUOUS_INCORRECT, 학습 target 아님)로 표기).
- **N=10 free sampling** — 같은 prompt에 대해 모델에게 10개의 짧은 답을 sampling하게
  하는 절차. Semantic Entropy와 paper-faithful Semantic Energy의 입력으로 쓴다.
- **teacher-forced scoring** — 모델에게 답을 *주고*, 그 답의 각 토큰 위치에서 logit /
  log-probability만 뽑는 방식. 모델이 답을 생성하지 않으므로 정답/환각 candidate 후보를
  *동일한 비교 기준* 으로 점수화할 수 있다.

## 전체 그림

```
experiments/
├── domain/        ← 데이터 모델 (frozen dataclass / Enum)
├── ports/         ← 추상 인터페이스 (ABC)
├── adapters/      ← 외부 system 연결 구현체
├── application/   ← 비즈니스 로직 (use case)
├── scripts/       ← CLI 진입점
├── configs/       ← YAML 설정
├── literature/    ← 참고 논문 + 식 매핑
└── results/       ← 산출물 (parquet/JSON/MD)
thesis/            ← LaTeX 본문 + 매크로 분리된 결과 placeholder
```

---

## 1. `domain/` — 도메인 모델

순수 데이터 형태 정의. 외부 라이브러리/IO 호출 금지. 다른 어떤 모듈에도 의존하지 않는다.
모두 `@dataclass(frozen=True)` 또는 `Enum`이다 (`validate_architecture.py`가 강제).

| 파일 | 핵심 타입 | 역할 |
|---|---|---|
| `records.py` | `QuestionExample`, `PromptGroup`, `CandidateRow`, `ModelResponse`, `SampleEnergyRecord` | paired discriminative dataset의 한 row, prompt group, 모델 응답 단위 |
| `labels.py` | `TypeLabel` Enum, `FeatureRole` Enum | archived diagnostic 라벨 + feature 역할 (TRAINABLE / LABEL_ONLY / ANALYSIS_ONLY) |
| `features.py` | `FeatureVector`, `FeatureProvenance`, `AnalysisBin` | feature table 한 row + 출처 메타 + corpus axis bin 정의 |
| `metrics.py` | `MetricResult`, `EvaluationSummary` | AUROC/AUPRC/F1 결과 컨테이너 |
| `manifests.py` | `ExperimentManifest`, `ArtifactRef` | 실행 manifest 스키마 |

---

## 2. `ports/` — 추상 인터페이스

application 계층이 구현을 모르고도 동작할 수 있게 하는 경계. 모두 `ABC`.

| 파일 | 인터페이스 | 무엇을 입력→출력 |
|---|---|---|
| `dataset_loader.py` | `DatasetLoaderPort` | config → `(PromptGroup, CandidateRow)` |
| `corpus_counts.py` | `CorpusCountBackendPort` | entity / pair → count + provenance |
| `corpus_stats.py` | `CorpusStatsPort` | candidate → `CorpusStats` (frequency, co-occurrence, flags) |
| `feature_extractor.py` | `FeatureExtractorPort` | candidate row → `FeatureVector` |
| `fusion_strategy.py` | `FusionStrategyPort` | feature vector → 환각 risk score |
| `evaluator.py` | `EvaluatorPort` | predictions + labels → `EvaluationSummary` |
| `artifact_store.py` | `ArtifactStorePort` | feature table / manifest / metric → 영속화 |

**왜 port를 둠**: corpus count backend를 REST API → local engine으로 바꿀 때
application 계층은 한 줄도 안 바꾼다. 테스트가 쉽고 신호 backend 교체가 안전하다.

---

## 3. `adapters/` — 구현체 (외부 system 닿는 곳)

| 파일 | 어느 port 구현 | 외부 의존 |
|---|---|---|
| `hf_datasets.py` | `DatasetLoaderPort` | `datasets` 라이브러리, HuggingFace Hub |
| `corpus_counts.py` | `CorpusCountBackendPort` (3종 backend) | REST API or local `infini_gram` package |
| `corpus_features.py` | `CorpusStatsPort` + 자체 adapter | entity 추출 → backend 호출 → axis score/bin |
| `model_generation.py` | (자체 service) | HuggingFace `transformers`, CUDA. teacher-forced scoring + N=10 free sampling. checkpoint shard 저장 |
| `correctness_dataset.py` | annotation → `CorrectnessJudgment` | dataset annotation 파싱 |
| `semantic_entropy_features.py` | (자체 adapter) | DeBERTa NLI. N=10 free samples → bidirectional entailment → semantic cluster + likelihood mass |
| `energy_features.py` | (자체 adapter) | Ma 2025 Eq.(11)–(14) 구현. selected_token_logits → sample energy → cluster total energy(SUM) → cluster_uncertainty |

**핵심 책임 분리**: adapter는 *형식 변환*만 담당. 학습/평가 로직은 application/.

### 3.1 corpus_counts.py 상세

3개 backend 구현체:

- `InfinigramApiBackend` — REST + on-disk JSON cache, retry-with-backoff. public 서버
  rate-limit (~3 q/s)이라 small-scale 디버그용.
- `InfinigramLocalEngineBackend` — 67 GB Dolma sample 인덱스를 메모리 매핑한 뒤
  Python `infini_gram` 패키지의 `count` / `count_cnf` AND query 직접 호출. ~1290 q/s.
  thesis-valid run의 primary backend.
- `FixtureCorpusCountBackend` — 단위 테스트용. JSON sidecar에서 미리 정의된 count 반환.

backend 선택은 우선순위: fixture sidecar > sidecar config (`corpus_backend.json`) > 환경
변수 (`THESIS_CORPUS_BACKEND`, `INFINIGRAM_LOCAL_INDEX_DIR`) > 기본 REST.

### 3.2 model_generation.py 상세

- teacher-forced candidate scoring + answer-only N=10 free sampling을 한 runtime에서.
- 산출물: candidate token-level full-vocab logits (parquet sidecar), free sample
  response text + selected-token logit/logprob.
- checkpoint shard 단위 저장 — restart 시 metadata 호환성 검사로 기존 shard 보존
  (`ESSENTIAL_METADATA_FIELDS`만 비교, batch/cap 변경에 영향 없음).
- answer-only protocol: bounded resampling (max_invalid_attempts), forbidden pattern
  필터 (예: "step 1", "stream of consciousness"), short answer span 강제.

### 3.3 semantic_entropy_features.py 상세

- 입력: prompt당 N=10 sample의 response text + sequence log-likelihood
- DeBERTa-large-MNLI로 sample i, j 사이 양방향 entailment 판정 → semantic equivalence
- greedy clustering: sample-index 결정적 순서로 cluster 합병
- cluster probability = `log-sum-exp(member sample log-likelihoods)` → normalize
- output: `semantic_entropy_nli_likelihood` (Shannon entropy over cluster probs),
  `semantic_entropy_cluster_count`, sample/cluster log-likelihood, cluster id 매핑

### 3.4 energy_features.py 상세

- paper-faithful path (Ma 2025 Eq.(11)–(14)):
  1. token energy `Ẽ(x_t) = -z_θ(x_t)` (Eq. 13)
  2. sample energy `E(x^(i)) = mean(token energies)` (Eq. 11)
  3. cluster total energy `E_Bolt(C_k) = sum(member sample energies)` (Eq. 12 SUM)
  4. final `U = sum_k p(C_k) · E_Bolt(C_k)` with Eq. (8) likelihood weighting
- diagnostic path (Ma 식 아닌 별도 진단 — candidate level):
  - `mean_negative_log_probability`, `logit_variance`, `confidence_margin`,
    `semantic_energy_boltzmann_diagnostic` (token-level `-logsumexp` 평균)
- 두 path는 같은 column으로 섞지 않는다.

---

## 4. `application/` — 비즈니스 로직

| 파일 | 책임 |
|---|---|
| `architecture_validation.py` | hexagonal 구조 강제 (REQUIRED_DIRECTORIES, frozen dataclass 검사, scripts에 dataclass 정의 금지) |
| `labeling.py` | feature table join. prompt-level Semantic Entropy를 두 candidate row에 broadcast, candidate-level Energy/corpus 결합. 진단용 TypeLabel (NORMAL/HIGH_DIVERSITY/LOW_DIVERSITY/AMBIGUOUS_INCORRECT, 학습 target 아님) 부여 (학습 target 아님, 진단용) |
| `fusion.py` | 환각 탐지기 학습/평가. leave-one-dataset-out fold 관리. logistic regression + sklearn variants 다수 baseline |
| `robustness.py` | prompt-grouped bootstrap (2,000 iter), leave-one-dataset-out, threshold sensitivity, 3-bin vs 5-bin corpus axis sensitivity, selective risk, calibration |
| `type_analysis.py` | feature/signal 분포를 slice별로 분석 (corpus-axis bin이 main, 진단용 TypeLabel (NORMAL/HIGH_DIVERSITY/LOW_DIVERSITY/AMBIGUOUS_INCORRECT, 학습 target 아님)은 보조 진단) |
| `thesis_evidence.py` | fusion + robustness + type_analysis → `thesis_evidence_summary.json` + `thesis_evidence_table.tex` 자동 생성 |

### 4.1 fusion.py 상세

Baseline 9종 + sklearn variants:
- **단일 신호**: SE-only, Energy-only, corpus-only, logit-diagnostic-only
- **조합 baseline**: hard cascade (SE 임계값 위면 Energy 사용), fixed linear blend (3가지 ratio)
- **학습 fusion**: learned fusion without/with corpus, corpus-bin feature selection,
  axis-interaction logistic
- **학습기**: 기본은 stdlib L2 logistic regression (의존성 없음 + 결정적). sklearn 통합으로
  RandomForest / GradientBoosting / SVM variants도 같은 feature_set에 plug in 가능.

학습 target은 무조건 `is_hallucination` (annotation-backed). fold 분할은 prompt 단위로
묶어 같은 prompt의 두 candidate row가 train/test fold에 갈라지지 않도록 강제.

### 4.2 type_analysis.py 상세

각 신호(`semantic_entropy_score`, `semantic_energy_score`, `mean_negative_log_probability`
등) 가 다음 slice에서 어떻게 분포되는지 보고:

- **corpus-axis bin slice (main)**: low / mid / high corpus support 별로 AUROC, AUPRC,
  paired delta, win rate. 본 논문의 핵심 분석축.
- **진단용 TypeLabel (NORMAL/HIGH_DIVERSITY/LOW_DIVERSITY/AMBIGUOUS_INCORRECT, 학습 target 아님) slice (보조)**: NORMAL vs HIGH_DIVERSITY vs LOW_DIVERSITY 분포 —
  thesis-valid analysis의 main이 아니라 진단용.
- **dataset slice**: TruthfulQA vs HaluEval-QA 별 분포.

---

## 5. `scripts/` — CLI 진입점

세 그룹:

### 5.1 단계별 데이터/특징 생성
- `prepare_datasets.py` — paired dataset materialization
- `run_generation.py` — model scoring (candidate logits + N=10 free samples)
- `build_correctness_dataset.py` — annotation correctness artifact
- `compute_semantic_entropy.py` — NLI clustering + likelihood SE
- `compute_corpus_features.py` — Infini-gram count → corpus axis / bin
- `compute_energy_features.py` — paper-faithful Energy + candidate-level diagnostics
- `build_feature_table.py` — 모든 feature join + leakage 검사

### 5.2 평가 / orchestration
- `run_fusion.py` — fusion baseline 평가
- `run_robustness.py` — robustness 보고서
- `run_type_analysis.py` — corpus-axis bin과 진단용 TypeLabel별 신호 분포 분석
- `run_pipeline.py` — 전체 orchestrator (manifest + script_execution_log 기록)
- `stage_control.py` — phase checkpoint 정합 helper

### 5.3 validator + util
- `validate_architecture.py` — hexagonal 구조 강제
- `validate_datasets.py`, `validate_energy_features.py`, `validate_feature_provenance.py`,
  `validate_generation_logits.py`, `validate_type_labels.py` — 산출물 schema/leakage 검사
- `setup_local_corpus_backend.py`, `prefetch_infinigram_counts.py` — Infini-gram 운영
- `export_thesis_evidence.py`, `inventory_artifacts.py`, `build_manifests.py`,
  `fetch_literature.py` — meta utility

---

## 6. `configs/` — YAML 설정

| 파일 | 내용 |
|---|---|
| `datasets.yaml` | TruthfulQA + HaluEval-QA HF id, prompt unit, label policy, 진단용 TypeLabel (NORMAL/HIGH_DIVERSITY/LOW_DIVERSITY/AMBIGUOUS_INCORRECT, 학습 target 아님) 정의 |
| `generation.yaml` | 모델 ref, N=10 sampling, answer-only protocol caps, batch size, full-logits sidecar 정책 |
| `formulas.yaml` | feature 식 등록. 각 feature에 source paper id, computed_features, formula 인용 명시 |
| `fusion.yaml` | baseline 정의, feature_sets, hyperparameter, forbidden_features (gold/correctness 누수 차단) |

`forbidden_features`는 fusion 학습기에 절대 넣어서는 안 되는 column 명시:
`label`, `is_hallucination`, `is_correct`, `candidate_label`, `gold_answer`,
`correctness_judgment`, `hidden_state` 등. 이 누수 가드는 `validate_type_labels.py`가
강제한다.

---

## 7. `literature/` — 참고 논문 + 식 매핑

- 4편 PDF 사본: Farquhar 2024 (Semantic Entropy), Ma 2025 (Semantic Energy),
  QuCo-RAG 2025 (entity frequency RAG), Phillips 2026 (PC Probe — 사용 안 함, framing reference)
- `formula_notes.md` — 각 paper의 어느 식이 어느 코드에서 어떻게 구현됐는지 단일 매핑
  문서. 예: Ma 2025 Eq.(11)–(14)가 `experiments/adapters/energy_features.py`의 어느 줄에
  구현되어 있는지 명시.

---

## 8. `thesis/` — LaTeX 본문

- `main.tex` — 본문 + bibliography
- `sections/experiment_method.tex` — 실험 방법 (paper-faithful 식 inline)
- `figures_tikz.tex` — TikZ 다이어그램 정의
- `results_macros.tex` — 본문에 들어가는 모든 headline 수치를 매크로로 분리. 새 결과
  도착 시 이 파일 9줄만 swap → 본문 prose는 손대지 않고 `pdflatex` 두 번으로 갱신
- `thesis_evidence_table.tex` — `experiments/results/`에서 자동 export된 표를 thesis가
  `\input` 으로 가져옴

---

## 9. 산출물 정책 (`.gitignore`)

- **commit**: `summary.json`, `report.md`, manifest, evidence_table.tex, baseline metrics CSV
  — 작은 텍스트 산출물 (재현 검증용)
- **ignore**: `*.parquet`, `predictions.jsonl` (~270 MB), `prompt_groups.jsonl`,
  `candidate_rows.jsonl`, full-logits sidecar parquet, `paired-datasets-qwen/`,
  `qwen-live-subset/`, `runs/`, `generation/`
- 정책: "텍스트 evidence는 commit, heavy raw data는 ignore + 재실행으로 복구 가능."

---

## 10. 자주 보는 문서 위치

| 무엇을 알고 싶을 때 | 어느 문서 |
|---|---|
| 실행 명령 / 단계별 contract | `experiments/PIPELINE.md` |
| 논지 + feature 표 | `experiments/OVERVIEW.md` |
| 식 ↔ 코드 매핑 | `experiments/literature/formula_notes.md` |
| 코드 구조 | 본 문서 (`CODE_GUIDE.md`) |
| 운영 가이드 (uv, build) | `experiments/README.md` |
| 작업 history (decisions, learnings) | `.sisyphus/notepads/` (gitignore) |
