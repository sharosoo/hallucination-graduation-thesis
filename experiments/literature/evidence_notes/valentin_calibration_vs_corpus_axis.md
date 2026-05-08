# Valentin et al. 2024 (Cost-Effective Hallucination Detection) vs 본 연구

## Scope

Valentin et al. 2024는 hallucination detection score를 *input/response attribute*에
조건화해 calibrate하는 multi-scoring framework를 제안한다. 본 연구의 corpus-axis
conditioning과 비교 대상이 될 가능성이 높아, 차별점을 명확히 정리한다.

## Valentin et al. 2024 핵심

- **Source**: arXiv:2407.21424 (Amazon, 2024).
- **Pipeline**: (1) generate confidence scores, (2) calibrate based on input/response
  characteristics, (3) apply threshold for detection.
- **주장**: "calibrating individual scoring methods is critical for risk-aware
  downstream decision making". 단일 method가 universal하게 우수하지 않으므로
  multi-scoring framework로 cost를 줄이면서 정확도를 유지/향상한다.
- **Tasks**: question answering, fact-checking, summarization 모두에서 평가.

## 본 연구와의 핵심 차별점

| 차원 | Valentin et al. 2024 | 본 연구 |
|---|---|---|
| **Conditioning 대상** | input/response의 *내부* attribute (응답 길이, 모델 confidence 분포, scoring method 자체의 metadata 등) | *외부 corpus statistics* (entity frequency, pair co-occurrence) — 모델 내부와 분리된 객관적 reference |
| **Black-box compatibility** | 일부 scoring method가 internal probability 의존 → closed-source LLM 적용에 제한 가능 | external corpus + selected-token logit만 사용 → closed-source LLM에도 적용 가능 (logit API만 있으면) |
| **목표** | *Cost-effective production detection* — multi-scoring 조합으로 비용 절감 | *Meta-evaluation* — 어떤 신호가 *언제* (어느 corpus support 조건에서) 작동하는지 조건부 reliability map 산출 |
| **Framing** | 새 detection framework 제안 | 새 detector 제안 ❌ — 기존 detector(SE/Energy/logit)의 신뢰도 조건성 정량화 |
| **Calibration의 의미** | score → probability mapping을 input attribute로 조정 | bin별 reliability를 *보고* — calibration보다는 *분석 axis* |

## 본 논문 본문에서 다룰 positioning

서론/관련 연구에서 Valentin et al. 2024를 인용하며 다음을 명시:

> "Recent work (Valentin et al. 2024) shows the importance of calibrating
> hallucination scores conditional on input/response attributes. Our framework
> shares this conditional view but uses *external corpus statistics* (entity
> frequency, pair co-occurrence) instead of internal score attributes, making
> it black-box compatible and orthogonal to internal-attribute calibration."

## Guardrails

- Valentin et al.의 multi-scoring framework를 본 연구가 *대체*한다고 주장하지 않는다.
  서로 직교적(orthogonal). 둘을 결합 가능 (corpus-axis conditioning + multi-scoring).
- "calibration"이라는 용어를 본 연구에서 사용할 때는 Valentin et al.과 다른 의미
  (조건부 reliability 분석)임을 본문에서 명시.
- production cost 절감은 본 연구의 contribution이 아님. 본 연구는 evaluation
  framework이지 deployment-cost framework가 아니다.

## Citation status

- arXiv preprint. 2024년 등록. 인용 시 preprint 표기 유지.
