# 시행착오 기록 — 평가 단위 Pivot 3 단계

본 문서는 본 졸업논문 (\"LLM 환각 탐지 신호의 Corpus Support 조건부 분해\") 이
**평가 단위 (evaluation unit)** 를 세 차례 갈아끼운 과정을 기록한다.
README / experiments/PIPELINE.md / thesis/main.tex 는 모두 최종 단계 (Phase 3)
기준으로 작성되어 있고, 폐기된 단계의 코드와 산출물은 이미 삭제되었다.
본 문서는 이 시행착오의 narrative 만 남긴다.

---

## 한 문장 요약

| Phase | 평가 단위 | 라벨 정의 | 결정적 한계 | 폐기 시점 |
|---|---|---|---|---|
| 1 | candidate row (paired) | 데이터셋 annotation 의 `is_hallucination` | SE / Energy 가 prompt-broadcast 신호이므로 paired candidate 사이를 분리 못 함 (AUROC ≈ 0.5) | 2026-04 |
| 2 | prompt | `is_hard = (free-sample 매칭률 < 0.5)` proxy | 라벨과 SE 신호가 같은 N=10 free-sample 에서 산출되어 부분적 결합 (label-signal coupling) | 2026-05-01 |
| 3 (본 논문) | generation (free-sample $s_i$) | 답변마다 NLI 양방향 함의 ≥ 0.5 → `is_correct` | 본 논문의 메인 평가. Farquhar 2024 / Ma 2025 와 평가 단위 일치 | — |

---

## Phase 1 — Candidate-Level Paired Discriminative

**시기**: 2026-02 ~ 2026-04
**데이터셋**: TruthfulQA (815 prompts) + HaluEval-QA (5{,}000 prompts) = **5{,}815 prompts × 2 candidates = 11{,}630 rows**.
**라벨**: 데이터셋이 제공한 정답 / 환각 후보 쌍.
**스코어링**: teacher-forced log-prob (right candidate vs hallucinated candidate).
**탐지 신호**: Semantic Entropy, Semantic Energy, candidate-level token logit diagnostics (NLL, logit variance, log-likelihood).
**Fusion**: GroupKFold(prompt_id) 5-fold, 두 candidate row 위에서 ranking.

### 동기

Farquhar 등 (2024) 의 free-sample 평가는 N=10 자유 생성 답변이 필요한데, RTX 5090
1대로 5{,}815 prompt × 10 sample = 58{,}150 generation 을 시드 고정으로 재현하기에
generation 비용이 컸다. paired discriminative 는 N=2 의 teacher-forced scoring
만 필요하므로 \"같은 GPU 예산으로 더 큰 표본\" 을 얻는 절충안이었다. 또한
`is_hallucination` annotation 이 데이터셋에서 제공되므로 라벨 신뢰도가 높다는
장점이 있었다.

### 폐기 사유

Semantic Entropy 와 Semantic Energy 는 본질적으로 **prompt 단위 신호** 다. 한
prompt 에 대해 산출한 답변 cluster 분포의 엔트로피이므로, 같은 prompt 에서
나온 두 candidate (right, hallucinated) 에 동일한 SE 값이 broadcast 된다.
candidate-level ranking AUROC 는 한 prompt 안의 두 row 를 분리해야 하는데,
SE / Energy 는 이 분리에 기여할 수 없으므로 **AUROC 는 0.5 부근에 수렴**
한다 (broadcasting bug 가 아니라 신호 구조의 자연스러운 귀결).

candidate-level token logit diagnostics (NLL, logit variance) 만이 두 candidate
사이를 분리하는 신호로 작동했고, fusion AUROC 0.6–0.7 수준에 머물렀다.
**Farquhar / Ma 와 평가 단위가 다르므로 직접 비교가 불가능했다.**

### 잔존 흔적

- `experiments/PIPELINE.md` 의 \"트랙 A — Paired Candidate-Level (legacy)\" 섹션.
- `experiments/literature/evidence_notes/pair_cooccurrence_choke_evidence.md` 등
  paired 시기 evidence note (2026-05-08 retracted 표시 포함).
- 산출물 (`experiments/results/{fusion,robustness,type_analysis,correctness,
  datasets,paired-datasets-qwen}/...`) 모두 삭제. SE-only AUROC ≈ 0.5 같은
  Phase 1 broadcasting 한계가 결과처럼 보고되어 fresh reader 의 혼동을
  방지하기 위함.

---

## Phase 2 — Prompt-Level `is_hard` Proxy

**시기**: 2026-04 ~ 2026-05-01
**데이터셋**: 동일 paired 셋 (TruthfulQA + HaluEval-QA, 5{,}815 prompts).
**라벨**: 각 prompt 의 free-sample N=10 답변과 데이터셋 정답 후보 사이 NLI
양방향 함의 매칭. 매칭률 = matches / 10. **`is_hard = (매칭률 < 0.5)`** binary
proxy.
**평가 단위**: prompt (한 prompt 에 하나의 is_hard 라벨).
**탐지 신호**: prompt 단위 SE / Energy / 토큰 logit 통계 집계.
**Fusion**: prompt 단위 GroupKFold 5-fold. AUROC 0.85+ 까지 올라옴.

### 동기

Phase 1 의 candidate-level AUROC ≈ 0.5 문제를 우회하기 위해 평가 단위를
prompt 로 올렸다. SE / Energy 가 prompt 단위 신호이므로 평가 단위와 신호
단위가 정합되어 분리력이 회복되었고 (AUROC 0.85–0.89), \"어려운 prompt 에서
SE / Energy 가 비정상적으로 높다\" 는 직관적 결과를 얻었다.

### 폐기 사유

두 가지 결합 (coupling) 문제가 발견되었다.

1. **Label-signal coupling**: 라벨 (`is_hard`) 과 신호 (SE / Energy) 가 모두
   동일한 N=10 free-sample 에서 산출된다. NLI 매칭률이 낮은 prompt 는 자유
   생성 답변이 의미적으로 산만하다는 뜻이고, 같은 답변들의 cluster 엔트로피
   (SE) 도 자연스럽게 높다. 두 양은 동일 source 의 **분산** 을 일부 공유한다.
   따라서 본 분석의 AUROC 는 \"모델 출력의 어려움 (라벨)\" 과 \"모델 출력의
   불확실성 (신호)\" 사이의 결합도를 측정하며, **모델 출력과 외부 정답 텍스트
   사이의 일치도에 대한 독립 측정이 아니다**.
2. **선행 연구 비교 단절**: Farquhar 2024 / Ma 2025 의 평가 단위는 generation
   (free-sample $s_i$) 단위이지 prompt 단위가 아니다. prompt-level AUROC 0.85
   는 generation-level AUROC 와 산술적으로 직접 비교할 수 없다.

### 잔존 흔적

- 본 코드베이스에는 남아 있지 않음 (해당 스크립트 `run_prompt_level_analysis.py`
  / `relabel_is_hard_nli.py` 모두 Phase 3 정리 단계에서 삭제).
- `experiments/PIPELINE.md` 의 S10 단계는 \"DELETED\" 마커로 축소되어 본
  HISTORY.md 를 가리킨다.

---

## Phase 3 — Generation-Level NLI Correctness (본 논문)

**시기**: 2026-05-01 ~ 현재
**데이터셋**: TriviaQA 800 + SQuAD-1.1 800 + BioASQ 800 + NQ-Open 800 +
SVAMP 300 = **3{,}500 prompts**. Farquhar 2024 와 동일.
**모델**: Qwen2.5-3B base. T=1.0, top-p 0.9, top-k 50, max_new_tokens 64
(중간 절단 12.6\% sample 은 max_new_tokens=128 로 재생성, AUROC ±0.001
변화 확인).
**평가 단위**: 생성 답변 (free-sample $s_i$). prompt 당 N=10. 총 **35{,}000
generations**.
**라벨**: $s_i$ 와 데이터셋 정답 후보 $c$ 사이 NLI 양방향 함의 확률
$\\max(p(c \\to s_i), p(s_i \\to c)) \\geq 0.5$ → `is_correct = 1`.
\\texttt{microsoft/deberta-large-mnli} 사용.
**탐지 신호**:
- prompt 단위 broadcast: Semantic Entropy, Semantic Energy.
- generation 단위 token logit 통계: 평균 NLL, 시퀀스 로그우도, logit 분산,
  평균 로그 분배함수.
- prompt 단위 corpus 신호 broadcast: entity 빈도, entity co-occurrence,
  question-answer bridge, 답변 3-gram / 5-gram 등장 빈도, 3-gram 미등장 개수.
**Fusion**: GroupKFold(prompt_id) 5-fold. logistic regression / random forest /
gradient boosting 비교.

### 동기

Phase 2 의 두 결합 문제를 동시에 해결한다.
1. 평가 단위가 generation 이므로 **Farquhar 2024 / Ma 2025 와 평가 단위가
   일치** 한다. AUROC 직접 비교 가능.
2. 라벨이 \"한 답변과 외부 정답 텍스트 사이의 일치도\" 이므로, 신호 (같은 답변
   집합의 cluster 엔트로피) 와 라벨이 가리키는 양이 명확히 분리된다 (완전한
   독립은 아니나 Phase 2 의 직접 결합은 해소).

### 헤드라인 결과

- 단일 신호 AUROC: SE 0.759, Semantic Energy 0.774. Farquhar 보고치
  (TriviaQA 0.79, SQuAD 0.83, Llama-2 7B) 와 비슷한 범위.
- Fusion AUROC: gradient boosting (corpus 미포함) 0.800, gradient boosting
  (corpus 포함) **0.808**.
- corpus 신호 fusion 추가 효과: $+0.008 \\sim +0.010$ (95\\% CI
  $[+0.005, +0.011]$, 모든 부트스트랩 반복에서 양수).
- **본 논문의 핵심 비대칭**: 동일 corpus 신호를 분해 신호로 사용하면 SE AUROC
  range 가 entity co-occurrence 기준 $\\Delta 0.150$, entity 빈도 기준
  $\\Delta 0.080$ 으로 약 **1.88배** (95\\% CI $[+0.002, +0.117]$) 차이.
  fusion 입력 변수로서의 기여 (+0.008) 와 분해 신호로서의 효용 (range $\\Delta
  0.150$) 의 비대칭이 본 논문의 메인 발견.

### 잔존 한계 (thesis §5.2 발췌)

- proxy corpus index (Dolma 16B sample) 사용 — Qwen 의 실제 사전학습 corpus
  와 다름. 본 결과는 corpus 빈도와 환각 사이의 **상관 관찰** 에 한정되며 인과
  진술 아님.
- 단일 모델 평가 (Qwen2.5-3B). Llama / Gemma / Mistral 일반화 미검증.
- Phase 2 와 동일한 \"라벨 / 신호 부분 결합\" — 라벨과 신호 모두 동일 free-sample
  에서 산출됨 (단 Phase 2 처럼 동일 `match-rate` 는 아니므로 결합 약함).

