# 환각 탐지 졸업논문 아카이브

이 repo는 환각(hallucination) 탐지 주제 졸업논문과, 논문 수치를 재생성하기 위한
`uv` 기반 실험 파이프라인을 함께 관리합니다.

## 한 문장 요약

> **LLM이 답을 지어낼 때(환각), 그 답이 "신뢰할 만한 답"인지 판별하는 신호가
> *얼마나 잘 작동하는지*는 corpus 안에 그 답이 얼마나 자주 등장하는지에 따라
> 달라진다 — 이것을 측정한다.**

본 논문은 새 환각 탐지기를 제안하는 것이 아니라, Semantic Entropy / Semantic
Energy / logit diagnostics 같은 기존 신호의 **조건부 신뢰도**를 corpus support
축에서 정량화하는 평가 framework를 제안합니다.

## 핵심 결정 (논지)

- **평가 protocol**: paired discriminative — 데이터셋이 미리 제공한 `(정답 후보,
  환각 후보)` 쌍을 모델이 *생성하지 않고* teacher-forced로 점수화만 한다. LLM-as-judge
  와 휴리스틱 매칭을 모두 배제. 학습 target은 dataset annotation의 `is_hallucination`
  (이진).
- **데이터셋**: TruthfulQA (815 prompts × 2 후보) + HaluEval-QA (5,000 prompts × 2
  후보) = 5,815 prompts / 11,630 candidate rows.
- **모델**: Qwen2.5-3B (causal LM, vocab 151,936). full-vocabulary logits는 JSON에
  inline하지 않고 같은 stem의 `.full_logits.parquet` sidecar로 저장.
- **신호**: 8종 — Semantic Entropy (N=10 + DeBERTa-large NLI clustering + likelihood
  cluster mass), paper-faithful Semantic Energy (Ma 2025 Eq.11–14: 토큰 energy =
  −logit, sample energy = mean, cluster total = SUM, U = Σ p(C_k)·E_Bolt(C_k)),
  cluster-prompt sample energy, candidate-level logit diagnostics 4종 (NLL, logit
  variance, confidence margin, boltzmann diagnostic), corpus risk score.
- **Corpus axis**: 답에 등장하는 entity의 corpus frequency + entity pair
  co-occurrence를 Infini-gram-compatible count backend (현재 local
  `v4_dolmasample_olmo`, 16B Dolma sample tokens) 로 query해 continuous axis로
  변환. 3-bin primary + 5-bin sensitivity + 10-bin (decile) sensitivity 산출.
- **분석 축**: corpus-axis bin이 main. 각 bin에서 모든 신호의 AUROC, AUPRC,
  prompt-grouped paired delta, paired win rate, paired tie rate를 보고.

## 빠른 시작

```bash
uv sync --group generation
uv run python experiments/scripts/validate_architecture.py
uv run python experiments/scripts/run_pipeline.py --dry-run --out experiments/results/runs
```

실제 모델 실행 (CUDA GPU 권장, full run은 큰 NVMe 필요):

```bash
uv run python experiments/scripts/run_pipeline.py --execute --out <large-nvme-run-root>
```

상세 단계별 명령은 `experiments/PIPELINE.md` 참조.

## 어디부터 읽어야 하나

| 무엇을 알고 싶을 때 | 어느 문서 |
|---|---|
| 30분 안에 코드 구조 파악 | [`CODE_GUIDE.md`](./CODE_GUIDE.md) |
| 논지 + feature 표 | [`experiments/OVERVIEW.md`](./experiments/OVERVIEW.md) |
| 단계별 실행 contract (필수) | [`experiments/PIPELINE.md`](./experiments/PIPELINE.md) |
| 식 ↔ 코드 매핑 (Ma 2025 Eq.11–14, Farquhar Eq.8 등) | [`experiments/literature/formula_notes.md`](./experiments/literature/formula_notes.md) |
| 운영 / uv / build 가이드 | [`experiments/README.md`](./experiments/README.md) |
| 연구 동기 + 차별점 | [`experiments/RESEARCH_PLAN.md`](./experiments/RESEARCH_PLAN.md) |

## repo 구조

```text
hallucination-graduation-thesis/
├── README.md                ← 본 문서 (entry point)
├── CODE_GUIDE.md            ← 모듈별 코드 가이드 (헥사고날 layer 책임)
├── experiments/
│   ├── PIPELINE.md          ← 실행 contract source of truth
│   ├── OVERVIEW.md          ← 논지 + feature 표
│   ├── README.md            ← 운영 가이드
│   ├── RESEARCH_PLAN.md     ← 연구 동기
│   ├── configs/             ← YAML (datasets, generation, fusion, formulas)
│   ├── domain/              ← frozen dataclass / Enum
│   ├── ports/               ← 추상 인터페이스 (ABC)
│   ├── adapters/            ← 외부 system 구현체 (Infini-gram, HF, NLI 등)
│   ├── application/         ← 비즈니스 로직 (fusion, robustness, type_analysis)
│   ├── scripts/             ← CLI 진입점
│   ├── literature/          ← 참고 논문 PDF + formula_notes.md
│   ├── manifests/           ← upstream artifact 매니페스트
│   └── results/             ← 산출물 (parquet + JSON 요약 + report)
└── thesis/
    ├── main.tex             ← 논문 본문
    ├── main.pdf             ← 컴파일된 PDF
    ├── results_macros.tex   ← headline 수치를 매크로로 분리 (결과 swap 즉시 가능)
    ├── thesis_evidence_table.tex  ← 자동 export된 결과 표
    ├── figures_tikz.tex
    ├── sections/experiment_method.tex
    ├── snuthesis.cls
    └── snutocstyle.tex
```

## 산출물 정책 (`.gitignore`)

- **commit**: 작은 텍스트 산출물 — `summary.json`, `report.md`, manifest,
  `thesis_evidence_table.tex`, baseline metrics CSV. 재현 검증을 빠르게 할 수 있도록.
- **ignore**: 큰 raw data — `*.parquet`, `predictions.jsonl` (~270 MB),
  `prompt_groups.jsonl`, `candidate_rows.jsonl`, full-logits sidecar parquet,
  `paired-datasets-qwen/`, `qwen-live-subset/`, `runs/`, `generation/`,
  graphify 빌드 디렉터리, 로컬 에이전트 설정.
- 정책: "텍스트 evidence는 commit, heavy raw data는 ignore + 재실행으로 복구 가능."

## 참고 논문 6편

| 논문 | 본 repo에서 어떻게 쓰는가 |
|---|---|
| Farquhar 2024, Nature — *Detecting hallucinations using semantic entropy* | S4 Semantic Entropy 토대. Eq.(8) likelihood-based cluster probability를 paper-faithful Semantic Energy에서도 그대로 상속. |
| Ma 2025 (preprint) — *Semantic Energy* | S6 paper-faithful Semantic Energy 식 (Eq.11–14) 그대로 구현. cluster total energy = SUM (Eq.12), U = Σ p(C_k)·E_Bolt(C_k). |
| QuCo-RAG 2025 — *Query-Corpus uncertainty for RAG* | entity frequency / pair co-occurrence를 *RAG 검색 trigger* 가 아니라 **신뢰도 conditioning 축**으로 재해석. |
| Phillips 2026 — *PC Probe / selective prediction* | hidden-state probe 없이 외부 corpus + selected-token logit만 쓰는 본 논문 contract와 대비되는 framing reference. baseline에 포함하지 않음. |
| **Valentin et al. 2024 (arXiv:2407.21424) — *Cost-Effective Hallucination Detection*** | conditional calibration framework. 본 연구와 직교: 그쪽은 *내부* score attribute 조건화, 본 연구는 *외부* corpus statistics 조건화. Black-box compatibility 비교 시 인용. |
| **Simhi et al. 2025 (arXiv:2502.12964) — *Trust Me, I'm Wrong (CHOKE)*** | 모델이 정답 지식을 가지고도 high-certainty hallucination 생성. SE의 low-diversity wrong answer 한계의 외부 evidence. corpus-axis conditioning 동기 강화. |

## 논문 PDF 다시 빌드

```bash
cd thesis
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

새 실험 결과가 도착하면 `thesis/results_macros.tex` 9개 매크로만 swap한 뒤 위
명령으로 빌드하면 본문 prose는 손대지 않고 모든 headline 수치가 갱신됩니다.

## 현재 진행 상황 (요약)

- 코드 구조 / 헥사고날 layer / contract: 안정 (`CODE_GUIDE.md` 참조)
- corpus-axis bin 분석 main 전환 + 진단용 4-class TypeLabel ontology 제거: 완료
- corpus axis 10-bin 추가 (3 / 5 / 10 모두): 완료
- paper-faithful Semantic Energy (Ma 2025 Eq.11–14): 완료
- NLI batched inference (GPU + FP16, batch=64) — S4 시간 1시간+ 절감: 완료
- Qwen2.5-3B full pipeline 실행: 진행 중

상세 작업 history는 `.sisyphus/notepads/` 와 git log 참조.
