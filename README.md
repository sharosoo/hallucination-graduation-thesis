# 환각 탐지 졸업논문 아카이브

이 repo 는 환각 (hallucination) 탐지 주제 졸업논문과, 논문 수치를 재생성하기 위한
`uv` 기반 실험 파이프라인을 함께 관리한다.

## 한 문장 요약

> **LLM 이 답을 지어낼 때 (환각), Semantic Entropy 와 Semantic Energy 같은 신호의
> 분리력은 답변에 등장하는 entity 가 corpus 안에서 얼마나 등장하는지에 따라
> 달라진다 — 이 변동 폭을 단위가 다른 corpus 신호별로 비교한다.**

본 논문은 새 환각 탐지기를 제안하는 것이 아니라, Semantic Entropy / Semantic
Energy / 답변 단위 logit 통계 같은 기존 신호의 **조건부 신뢰도** 를 단위가 다른
corpus 신호 (entity 빈도, entity 쌍 동시 등장, question-answer bridge, n-gram) 위에서
정량화하는 평가 framework 을 제안한다.

## 핵심 결정 (논지)

- **평가 단위**: 답변 단위 NLI 정답성. 한 sample 마다 자유 생성 답변 10개를 산출한
  뒤, 각 답변과 데이터셋 정답 후보 (정답 표현, 동의 표현, 별칭 목록) 사이의 NLI
  양방향 함의 확률 $\max(p(c \to s), p(s \to c)) \geq 0.5$ 로 `is_correct` 를 판정한다.
  Farquhar 등 (2024) 와 Ma 등 (2025) 의 평가 단위와 동일.
- **데이터셋 (5종)**: TriviaQA 800 + SQuAD-1.1 800 + BioASQ 800 + NQ-Open 800 + SVAMP
  300 = 3,500 sample. Farquhar (Nature 2024) Semantic Entropy 와 동일한 5 datasets.
- **모델**: Qwen2.5-3B (base, fp16, RTX 5090 32 GB).
- **생성 설정**: 온도 1.0, top-p 0.9, top-k 50, 최대 생성 토큰 수 64. sample 당 자유
  생성 답변 10개 → 총 35,000 generation. 절단된 답변이 발생한 12.6% sample 은 최대
  생성 토큰 수 128로 재생성하였고, 재생성 전후 AUROC 변화는 ±0.001 범위에 머물렀다.
- **신호**: Semantic Entropy (NLI 의미 cluster + 시퀀스 로그우도 기반 cluster
  probability), Semantic Energy (Ma 2025 Eq.11–14, paper-faithful), 답변 단위 token
  logit 통계 4종 (평균 NLL, 시퀀스 로그우도, logit 분산, 평균 로그 분배함수), corpus
  신호 7종 (entity 빈도, entity 쌍 동시 등장, baseline 평균, question-answer bridge,
  3-gram / 5-gram 등장 빈도, 3-gram 미등장 개수).
- **Corpus axis**: 답변에 등장하는 entity 의 corpus 통계를 Infini-gram local backend
  (현재 `v4_dolmasample_olmo`, 16B Dolma sample tokens) 로 query 해 sample 단위
  연속 신호로 산출하고, 단위별로 sample 을 10분위로 나눠 신호별 AUROC range 를 비교한다.
- **분석 축**: corpus 신호 단위 (entity, entity 쌍, qa-bridge, n-gram) 별 AUROC range 가
  main 결과. fusion 평가는 prompt 단위 5-fold GroupKFold 로 수행해 학습-평가 사이의
  sample 단위 정보 누출을 차단한다.

## 실험 파이프라인 (요약)

`experiments/PIPELINE.md` 의 트랙 B (SE 5-dataset Single-candidate) 가 메인 평가의
source 이다. 13 단계 (S1' → S13') 로 구성된다.

```text
S1' 데이터셋 준비 (5 SE datasets, 3,500 prompt)
  ↓
S2' Generation (Qwen2.5-3B, N=10 free samples, sentence-length)
  ↓
S3' Checkpoint consolidate (free_sample_rows.json 재조립)
  ↓
S4' Semantic Entropy (NLI cluster + likelihood)
  ↓
S5' Semantic Energy (paper-faithful, Ma 2025 Eq. 11–14)
  ↓
S6' Free-sample diagnostics (답변 단위 token 통계)
  ↓
S7' Generation-level NLI correctness (is_correct 라벨)
  ↓
S8' Corpus features (entity 수준)
  ↓
S9' QA Bridge co-occurrence (corpus 다양화 1)
  ↓
S10' N-gram coverage (corpus 다양화 2)
  ↓
S11' Generation-level fusion + robustness (fusion summary + per-decile + per-dataset + calibration)
  ↓
S12' Review-driven 부트스트랩 + Spearman ρ + SVAMP 민감도
  ↓
S13' thesis/results_macros.tex 자동 생성 (30개 macro)
```

## 현재 결과 해석

- **단일 신호**: Semantic Entropy AUROC 0.759, Semantic Energy AUROC 0.774. Farquhar
  Nature 2024 (0.75–0.85) 및 Ma 등 (2025) (0.74–0.85) 범위와 일치.
- **Fusion (5-fold GroupKFold(prompt_id) OOF)**: Logistic Regression (CORE+CORPUS)
  0.795, Random Forest 0.800, Gradient Boosting 0.808. 대표 수치는 GBM with corpus
  AUROC 0.808.
- **AURAC** (Farquhar Nature 2024 main metric): SE 0.526, Energy 0.533, Fusion 0.559.
- **Main 발견 — corpus 신호 단위별 AUROC range 비교**: entity 쌍 동시 등장 신호로
  sample 을 10분위로 나누면 SE AUROC 가 0.643에서 0.793 까지 변동 (Δ = 0.150).
  entity 빈도 신호 단위에서는 같은 SE 의 변동 폭이 Δ = 0.080에 그친다. 두 변동 폭
  비율 약 1.88배. SVAMP 제외 시 1.68배 (SE) / 1.54배 (Energy), 비율 범위 1.5–1.9배.
  prompt 단위 bootstrap (B = 500) 95% 신뢰구간 [+0.002, +0.117] 로 0을 가로지르지
  않는다.

## 빠른 시작

```bash
uv sync --group generation
uv run python experiments/scripts/validate_architecture.py
```

실제 모델 실행 (CUDA GPU 권장, full run 은 큰 NVMe 필요). 각 명령은 `$RUN` 을
산출물 root 로 가정한다.

```bash
# S1' 데이터셋 준비 (5 SE datasets)
uv run python experiments/scripts/prepare_datasets_se.py \
  --config experiments/configs/datasets_se.yaml \
  --out-dir $RUN/results/datasets

# S2' Generation (Qwen2.5-3B)
uv run python experiments/scripts/run_generation.py \
  --config experiments/configs/generation_se_qwen.yaml \
  --prompt-groups $RUN/results/datasets/prompt_groups.jsonl \
  --candidates $RUN/results/datasets/candidate_rows.jsonl \
  --out-free-samples $RUN/qwen/results/generation/free_sample_rows.json \
  --out-candidate-scores $RUN/qwen/results/generation/candidate_scores.json

# S3' Checkpoint consolidate
uv run python experiments/scripts/consolidate_checkpoints_se.py \
  --checkpoint-dir $RUN/qwen/results/generation/free_sample_rows.json.checkpoint \
  --out $RUN/qwen/results/generation/free_sample_rows.json

# S4' Semantic Entropy
uv run python experiments/scripts/compute_semantic_entropy.py \
  --free-samples $RUN/qwen/results/generation/free_sample_rows.json \
  --out $RUN/qwen/results/semantic_entropy_features.parquet

# S5' Semantic Energy (paper-faithful)
uv run python experiments/scripts/compute_energy_se_minimal.py \
  --free-samples $RUN/qwen/results/generation/free_sample_rows.json \
  --semantic-entropy $RUN/qwen/results/semantic_entropy_features.parquet \
  --out $RUN/qwen/results/energy_features.parquet

# S8' Corpus features
uv run python experiments/scripts/compute_corpus_features.py \
  --candidates $RUN/results/datasets/candidate_rows.jsonl \
  --out $RUN/qwen/results/corpus_features.parquet \
  --entity-extractor spacy

# S11' Generation-level fusion + robustness (메인)
uv run python experiments/scripts/run_generation_se_analysis.py \
  --run-dir $RUN/qwen --bootstrap-n 1000

# S12' Review-driven 부트스트랩 + Spearman ρ + SVAMP 민감도
uv run python experiments/scripts/review_ablations.py \
  --run-dir $RUN/qwen --n-boot 500

# S13' thesis/results_macros.tex 자동 생성 (30개 \providecommand)
uv run python experiments/scripts/build_results_macros.py \
  --run-dir $RUN/qwen --out thesis/results_macros.tex
```

상세 단계별 명령 및 산출물 schema 는 `experiments/PIPELINE.md` 의 트랙 B 절 (S1' → S13') 을 참조.

## 어디부터 읽어야 하나

| 무엇을 알고 싶을 때 | 어느 문서 |
|---|---|
| 본 논문 본문 | [`thesis/main.pdf`](./thesis/main.pdf) |
| 단계별 실행 contract (필수) | [`experiments/PIPELINE.md`](./experiments/PIPELINE.md) |
| Phase 1→2→3 평가 단위 pivot 의 사유 | [`HISTORY.md`](./HISTORY.md) |
| 30분 안에 코드 구조 파악 | [`CODE_GUIDE.md`](./CODE_GUIDE.md) |
| 논지 + feature 표 | [`experiments/OVERVIEW.md`](./experiments/OVERVIEW.md) |
| 식 ↔ 코드 매핑 (Ma 2025 Eq.11–14, Farquhar Eq.8 등) | [`experiments/literature/formula_notes.md`](./experiments/literature/formula_notes.md) |
| 운영 / uv / build 가이드 | [`experiments/README.md`](./experiments/README.md) |
| 연구 동기 + 차별점 | [`experiments/RESEARCH_PLAN.md`](./experiments/RESEARCH_PLAN.md) |

## repo 구조

```text
hallucination-graduation-thesis/
├── README.md                ← 본 문서 (entry point)
├── HISTORY.md               ← Phase 1→2→3 평가 단위 pivot 사유
├── CODE_GUIDE.md            ← 모듈별 코드 가이드 (헥사고날 layer 책임)
├── experiments/
│   ├── PIPELINE.md          ← 실행 contract source of truth (트랙 B 메인)
│   ├── OVERVIEW.md          ← 논지 + feature 표
│   ├── README.md            ← 운영 가이드
│   ├── RESEARCH_PLAN.md     ← 연구 동기
│   ├── configs/             ← YAML (datasets_se, generation_se_qwen, fusion 등)
│   ├── domain/              ← frozen dataclass / Enum
│   ├── ports/               ← 추상 인터페이스 (ABC)
│   ├── adapters/            ← 외부 system 구현체 (Infini-gram, HF, NLI, free-sample diagnostics 등)
│   ├── application/         ← 비즈니스 로직 (generation_level_eval, prompt_accuracy 등)
│   ├── scripts/             ← CLI 진입점 (run_generation_se_analysis 등)
│   ├── literature/          ← 참고 논문 PDF + formula_notes.md
│   ├── manifests/           ← upstream artifact 매니페스트
│   └── results/             ← 산출물 (parquet + JSON 요약 + report)
└── thesis/
    ├── main.tex             ← 논문 본문
    ├── main.pdf             ← 컴파일된 PDF
    ├── results_macros.tex   ← headline 수치를 매크로로 분리 (결과 swap 즉시 가능)
    ├── thesis_evidence_table.tex  ← 자동 export 된 결과 표
    ├── figures_tikz.tex
    ├── sections/experiment_method.tex
    ├── snuthesis.cls
    └── snutocstyle.tex
```

## 산출물 정책 (`.gitignore`)

- **commit**: 작은 텍스트 산출물 — `summary.json`, `report.md`, manifest,
  `thesis_evidence_table.tex`, baseline metrics CSV. 재현 검증을 빠르게 할 수 있도록.
- **ignore**: 큰 raw data — `*.parquet`, `predictions.jsonl`,
  `prompt_groups.jsonl`, `candidate_rows.jsonl`, full-logits sidecar parquet,
  `qwen/`, `gemma/`, `runs/`, `generation/`, graphify 빌드 디렉터리, 로컬 에이전트 설정.
- 정책: "텍스트 evidence 는 commit, heavy raw data 는 ignore + 재실행으로 복구 가능."

## 참고 논문 6편

| 논문 | 본 repo 에서 어떻게 쓰는가 |
|---|---|
| Farquhar 2024, Nature — *Detecting hallucinations using semantic entropy* | S4' Semantic Entropy 토대. Eq.(8) likelihood-based cluster probability 를 paper-faithful Semantic Energy 에서도 그대로 상속. 본 논문 평가 단위 (답변 단위 NLI 정답성) 와 동일. |
| Ma 2025 (preprint) — *Semantic Energy* | S5' paper-faithful Semantic Energy 식 (Eq.11–14) 그대로 구현. cluster total energy = SUM (Eq.12), U = Σ p(C_k)·E_Bolt(C_k). |
| QuCo-RAG 2025 — *Query-Corpus uncertainty for RAG* | entity frequency / pair co-occurrence 를 *RAG 검색 trigger* 가 아니라 **신뢰도 conditioning 축** 으로 재해석. |
| Phillips 2026 — *PC Probe / selective prediction* | hidden-state probe 없이 외부 corpus + selected-token logit 만 쓰는 본 논문 contract 와 대비되는 framing reference. baseline 에 포함하지 않음. |
| Valentin et al. 2024 (arXiv:2407.21424) — *Cost-Effective Hallucination Detection* | conditional calibration framework. 본 연구와 직교: 그쪽은 *내부* score attribute 조건화, 본 연구는 *외부* corpus statistics 조건화. Black-box compatibility 비교 시 인용. |
| Simhi et al. 2025 (arXiv:2502.12964) — *Trust Me, I'm Wrong (CHOKE)* | 모델이 정답 지식을 가지고도 high-certainty hallucination 생성. SE 의 low-diversity wrong answer 한계의 외부 evidence. corpus 단위 분해 분석의 동기 강화. |

## 논문 PDF 다시 빌드

```bash
cd thesis
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

새 실험 결과가 도착하면 `thesis/results_macros.tex` 매크로만 swap 한 뒤 위
명령으로 빌드하면 본문 prose 는 손대지 않고 모든 headline 수치가 갱신된다.

## 시행착오 (요약)

본 논문은 candidate-level paired (Phase 1) → is_hard proxy (Phase 2) → generation-level
NLI (Phase 3) 의 평가 단위 pivot 을 거쳤다. 자세한 사유와 폐기 근거는
[`HISTORY.md`](./HISTORY.md) 참조.

## 현재 진행 상황 (요약)

- 트랙 B (SE 5-dataset, generation-level) full pipeline (Qwen2.5-3B): 완료
- 답변 단위 (35,000 generation) NLI correctness 라벨링 + 단일 신호 / fusion / 단위별
  AUROC range 분석: 완료
- prompt 단위 bootstrap (B = 500) 95% 신뢰구간 + per-decile + Spearman 단조성 검정: 완료
- thesis 본문 (Phase 3 기준) 재작성: 완료

상세 작업 history 는 `git log` 와 `HISTORY.md` 참조.
