# 졸업논문 수정본 재평가 보고서

> 재평가 대상: 수정 후 `thesis/main.tex`, `thesis/main.pdf`, `thesis/sections/experiment_method.tex`,
> `thesis/thesis_evidence_table.tex`, `thesis/results_macros.tex`, 주요 그림 파일
>
> 기준 문서: `thesis_evaluation_report.md`의 P0/P1 체크리스트

---

## 1. 종합 재판정

수정본은 이전 평가에서 가장 위험했던 부분을 상당히 잘 반영했다. 특히 `is_hard`가 실제 환각 라벨이 아니라
free-sample 정답 매칭률 기반 hard-question proxy라는 점을 초록과 본문에 명시했고, “단조 상승”과
“Fusion이 항상 최선은 아니다” 계열의 과한 표현을 많이 낮췄다. 그림 4.2도 두 패널 구조로 바뀌어
이전보다 훨씬 읽기 쉬워졌다.

현재 상태는 **이전보다 훨씬 방어 가능한 학사논문**이다. 다만 아직 남은 문제는 mostly 큰 실험 부족이 아니라
**앞뒤 표현 정합성, 표/목차 조판, 일부 metric 표기** 쪽이다. 이 남은 항목을 고치면 제출본으로 매우 안정적이다.

### 상태 요약

| 구분 | 상태 | 판단 |
|---|---|---|
| `is_hard` proxy 명시 | 거의 해결 | 초록, 결과 해석, 한계에 반영됨 |
| “단조 상승” 완화 | 해결 | “전반적 상승 경향”으로 수정됨 |
| Fusion 주장 수위 조정 | 거의 해결 | 유의성 미검정 단서가 붙음 |
| Brier/ECE 해석 | 대부분 해결 | 표 4.3은 수정됨. 표 4.1 caption/bold는 잔여 |
| LODO/generalization 표현 | 부분 해결 | §4.6은 정직해졌지만 §1.3에 잔여 표현 있음 |
| corpus support 단위 통일 | 부분 해결 | “질문” vs “질문과 후보 답변” 표현이 아직 혼재 |
| 그림 4.2 | 크게 개선 | overlap/scale 문제 대부분 해소 |
| 표목차/영문 줄바꿈 | 미해결 | 긴 표 caption과 soft-hyphen artifact 잔여 |

---

## 2. 잘 고쳐진 점

### 2.1 `is_hard` proxy 한정이 들어갔다

초록에 다음 취지의 문장이 추가되었다.

> `is_hard`는 실제 환각 여부를 새로 라벨링한 것이 아니라 free-sample 정답 매칭률 기반 hard question proxy이며,
> 결론은 이 proxy 기준의 영역별 패턴에 한정된다.

이 수정은 매우 중요하다. 이전에는 “환각 탐지 난이도”와 “hard-question proxy 판별 난이도”가 섞여 보였는데,
수정본은 방어력이 확실히 올라갔다.

### 2.2 “단조” 표현이 완화되었다

기존의 “단조 상승” 대신 “전반적 상승 경향”이라는 표현을 사용한다. 표 4.2의 값이 완전한 strict monotonic은 아니므로,
현재 표현이 훨씬 정확하다.

### 2.3 Fusion 주장이 더 안전해졌다

수정본은 decile 40--50의 `-0.014` 차이에 대해 통계적 유의성을 검정하지 않았다고 명시하고,
결론을 “Fusion 우위 폭이 균일하지 않다” 쪽으로 낮췄다. 이는 심사 방어에 유리하다.

### 2.4 Brier/ECE 해석이 개선되었다

표 4.3에서는 Random Forest의 Brier `0.112`가 bold 처리되고, Fusion은 ECE 기준 1위라는 식으로 분리되었다.
본문도 “Brier는 RF가 근소 우위, ECE는 Fusion이 우위”라는 방향으로 고쳐졌다.

### 2.5 그림 4.2가 크게 개선되었다

이전 그림 4.2는 큰 `vs logit-diagnostic` 막대 때문에 `vs Energy` 차이가 묻혔고, 붉은 주석이 x축과 겹쳤다.
수정본은 Panel A/B로 나뉘어 작은 delta와 큰 delta를 별도 scale에서 보여준다. `-0.014`도 직접 표시되어 핵심 메시지가 훨씬 잘 보인다.

### 2.6 TruthfulQA skew 설명이 강화되었다

본문은 TruthfulQA의 `is_hard=0.97`이 token-overlap proxy 한계 때문임을 더 직접적으로 설명하고,
주요 해석은 HaluEval-QA 기준으로 제한한다고 썼다. 이전보다 훨씬 안전하다.

---

## 3. 아직 남은 중요 이슈

## 3.1 §1.3의 “leave-one-dataset-out 일반화” 표현

`main.tex`의 논문 구성 문단에는 아직 다음 취지의 문장이 남아 있다.

> leave-one-dataset-out 일반화와 threshold / calibration 진단을 추가한다.

하지만 §4.6에서는 다음처럼 정직하게 설명한다.

> 진정한 leave-one-dataset-out이 아니라 pooled 5-fold CV 결과를 데이터셋별로 분해한 것이다.

따라서 §1.3 문장은 반드시 바꿔야 한다.

권장 수정:

> 제4장에서는 종합 baseline 위에 영역별 비교 결과 세 가지를 차례로 보고하고,
> 데이터셋별 분해 결과와 calibration 진단을 추가한다.

---

## 3.2 corpus support 정의 단위가 아직 완전히 통일되지 않았다

현재 문장들은 다음처럼 섞여 있다.

- `main.tex` 서론: corpus support는 **질문에 등장하는 entity**의 frequency/co-occurrence를 결합한 점수
- 방법/실험: **질문과 후보 답변에서 추출한 entity**의 corpus frequency/co-occurrence를 결합

이 둘은 의미가 다르다. 반드시 하나로 통일해야 한다.

가장 안전한 권장 정의:

> 본 논문에서 corpus support는 질문과 후보 답변에서 추출한 entity의 corpus frequency 및 entity-pair co-occurrence를 결합한,
> 질문 단위의 operational support score이다.

만약 실제 구현이 후보 답변 entity를 포함한다면, 서론의 “질문에 등장하는 entity” 표현을 위 문장으로 바꾸는 것이 좋다.

---

## 3.3 초록 마지막 문장의 “환각 탐지 난이도”는 조금 강하다

초록에서 proxy 한정 문장을 넣은 것은 좋다. 하지만 마지막에는 여전히 다음 취지의 표현이 남아 있다.

> 환각 탐지 난이도와 외부 corpus support의 관계를 영역별로 분해한다.

더 안전한 표현:

> `is_hard` proxy 기준 탐지 신호 판별력과 외부 corpus support의 관계를 영역별로 분해한다.

또는 조금 덜 딱딱하게:

> hard-question proxy 기준에서 탐지 신호의 영역별 판별력과 외부 corpus support의 관계를 분해한다.

---

## 3.4 표 4.1의 Brier bold/caption은 아직 모호하다

표 4.3은 개선되었지만, `thesis_evidence_table.tex`의 표 4.1은 아직 caption에 “굵은 글씨 = 1위”라고 쓰고,
gradient boosting 행의 AUROC/AUPRC/ECE만 bold 처리되어 있다. 그런데 Brier 최저값은 Random Forest `0.112`이다.

해결책 중 하나를 택하면 된다.

1. Brier `0.112`도 bold 처리한다.
2. caption을 다음처럼 바꾼다.

> 굵은 글씨 = AUROC/AUPRC/ECE 기준 1위. Brier는 낮을수록 좋으며 Random Forest가 근소 우위.

추천은 1번이다. 독자가 표만 봐도 바로 이해할 수 있다.

---

## 3.5 단일 신호 “결정” 표현은 약간 강하다

현재 본문에는 다음 취지의 표현이 있다.

> 단일 신호 선택은 corpus support 영역에 따라 달라지지 않는 결정이다.

데이터상 Energy > SE가 모든 decile에서 관찰되므로 방향은 맞지만, 신뢰구간이나 검정 없이 “결정”이라고 말하면 약간 강하다.

권장 수정:

> 본 표본에서는 단일 신호 비교의 결론이 corpus support 영역에 따라 바뀌지 않았다.

---

## 4. PDF/시각자료 재평가

### 4.1 해결된 시각 문제

- 그림 4.2의 이전 overlap 문제는 거의 해소되었다.
- Panel A/B 분리로 scale 문제가 크게 줄었다.
- `-0.014` 주석이 명확히 보인다.
- 그림 목록은 optional short caption 덕분에 깔끔해졌다.

### 4.2 남은 시각 문제

| 문제 | 상태 | 권장 조치 |
|---|---|---|
| 표 목록 과밀 | 남음 | 표 4.1/4.3에 optional short caption 적용 |
| 표 4.1 Brier bold | 남음 | RF Brier 0.112 bold 처리 또는 caption 수정 |
| 영어 줄바꿈 artifact | 남음 | 주요 복합어에 `\mbox{}` 또는 비분리 처리 적용 |
| 그림 목록 `그림4.1` 간격 | 경미 | `tocloft` spacing 조정 |
| 그림 4.1 범례 | 경미 | 우하단 범례가 일부 영역을 덮지만 핵심 정보는 보임 |

특히 PDF 검색/복사 품질에서 다음 artifact가 남아 있다.

- `logit variance`
- `entity-pair`
- `rank-quantile`
- `self-conditioning`
- `logit-diagnostic`
- 참고문헌의 `Challenges`

시각적으로 치명적이지는 않지만, 최종 제출본이라면 고치는 편이 좋다.

---

## 5. 통계/방법론 남은 리스크

큰 실험을 새로 하지 않아도 방어 가능성은 많이 올라갔다. 다만 P1 통계 보강은 아직 없다.

아직 추가되지 않은 항목:

- decile별 AUROC 95% CI
- Fusion-Energy delta CI
- decile별 `is_hard` prevalence
- decile별 AUPRC
- HaluEval-only decile 분석

이 항목들은 “필수 수정”은 아니지만, 있으면 논문의 방법론 점수가 크게 오른다. 시간이 없다면 최소한 본문에 “점추정치 기반 탐색적 분석”이라는 표현을 추가해도 방어력이 올라간다.

---

## 6. 수정 우선순위

### P0: 바로 고치면 좋은 문장/표 수정

1. §1.3의 “leave-one-dataset-out 일반화” 삭제 또는 완화
2. corpus support 정의를 “질문과 후보 답변 entity 기반 질문 단위 score”로 통일
3. 초록 마지막의 “환각 탐지 난이도”를 `is_hard` proxy 기준 표현으로 완화
4. 표 4.1에서 Brier 0.112 bold 처리 또는 caption 수정
5. “단일 신호 선택은 ... 결정”을 “본 표본에서는 ... 바뀌지 않았다”로 완화

### P1: 조판 polish

1. 표 4.1/4.3 optional short caption 추가
2. 영어 복합어 줄바꿈 방지
3. 그림/표 목록 spacing 조정

### P2: 있으면 강해지는 통계 보강

1. decile별 CI
2. Fusion-Energy delta CI
3. HaluEval-only decile 분석
4. decile별 `is_hard` prevalence / AUPRC

---

## 7. 재평가 결론

수정본은 이전보다 훨씬 좋다. 특히 핵심 방어 포인트였던 `is_hard` proxy, Fusion claim 수위,
Brier/ECE 해석, 그림 4.2가 눈에 띄게 개선되었다.

현재 상태를 한 문장으로 평가하면 다음과 같다.

> 학사논문으로는 이미 충분히 방어 가능한 수준이며, 남은 위험은 대규모 실험 부족보다
> 몇 개의 앞뒤 표현 불일치와 조판/표기 문제에 가깝다.

가장 먼저 고칠 것은 딱 세 가지다.

1. `leave-one-dataset-out 일반화` 표현 제거
2. corpus support 정의 통일
3. 표 4.1 Brier bold/caption 수정

이 세 가지만 고쳐도 심사장에서 공격받을 가능성이 크게 줄어든다.
