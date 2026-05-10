# 졸업논문 최신 수정본 재분석 보고서

> 분석 대상: 최신 `thesis/main.tex`, `thesis/main.pdf`, `thesis/sections/experiment_method.tex`,
> `thesis/thesis_evidence_table.tex`, `thesis/results_macros.tex`, 주요 그림 파일
>
> 분석 방식: 텍스트 재검토, 방법론 재검토, PDF/시각 재검토, skeptic 심사위원 리뷰를 병렬로 수행한 뒤 종합

---

## 1. 최종 종합 판정

최신 수정본은 **학사 졸업논문으로 충분히 방어 가능한 수준**이다. 이전에 가장 위험했던 항목들,
특히 `leave-one-dataset-out` 표현, corpus support 정의 불일치, 표 4.1 Brier 표기, 초록의 과한 주장,
Fusion-Energy 불확실성 문제가 대부분 실질적으로 개선되었다.

현재 논문은 다음 프레이밍으로 말하면 가장 안전하다.

> 본 논문은 실제 환각 여부를 새로 판정하는 SOTA detector 논문이 아니라,
> free-sample 기반 `is_hard` hard-question proxy 하에서 기존 환각 탐지 신호들의 조건부 신뢰도가
> 외부 corpus support 구간에 따라 어떻게 달라지는지 보인 평가 연구이다.

이 프레이밍을 유지하면 심사장에서 방어 가능하다. 반대로 “환각 탐지 일반 법칙”이나
“새 detector 성능 입증”처럼 말하면 아직 공격받을 수 있다.

---

## 2. 이전 핵심 이슈 해결 여부

| 이전 이슈 | 최신 상태 | 판단 |
|---|---|---|
| §1.3의 LODO/generalization 표현 | “데이터셋별 분해 결과와 calibration 진단”으로 수정 | **해결** |
| corpus support 정의 불일치 | “질문과 후보 답변에서 추출한 entity 기반 질문 단위 score”로 통일 | **해결에 가까움** |
| 초록의 “환각 탐지 난이도” 과장 | `is_hard proxy 기준 탐지 신호 판별력`으로 완화 | **해결** |
| 표 4.1 Brier/caption | RF Brier `0.112` bold + caption에서 Brier는 RF 근소 우위 명시 | **해결** |
| 표목차 과밀 | optional short caption 적용 | **대체로 해결** |
| Fusion-Energy 불확실성 | prompt bootstrap CI 표 추가 | **크게 개선** |
| 그림 4.2 overlap/scale | Panel A/B 구조 유지 | **해결에 가까움** |
| 영어 line-break artifact | 일부 남음 | **잔여** |
| figure/list spacing | `그림4.1` 공백 문제 일부 남음 | **경미한 잔여** |

---

## 3. 크게 좋아진 부분

### 3.1 초록과 결론의 claim 수위가 안정화됨

초록은 이제 `is_hard`가 실제 환각 여부가 아니라 proxy임을 분명히 밝히고,
결론도 이 proxy 기준의 영역별 패턴으로 제한한다. 이 수정은 방어력에 가장 크게 기여한다.

### 3.2 corpus support 정의가 정리됨

이전에는 “질문 entity”인지 “후보 답변 entity”인지 혼재했지만, 최신본은
“질문과 후보 답변에서 추출한 entity의 corpus frequency 및 entity-pair co-occurrence를 결합한 질문 단위 score”로 정리되었다.

남는 작은 질문은 “두 후보 답변의 entity를 질문 단위로 어떻게 aggregate했는가”이지만,
큰 개념 혼선은 해결되었다.

### 3.3 Fusion-Energy 비교에 CI가 추가됨

이전에는 `-0.014` 역전이 noise일 수 있다는 단서만 있었지만, 최신본은 decile별 bootstrap CI를 표로 제공한다.
특히 decile 40--50의 Energy 우위 점추정치가 CI에 0을 포함한다고 명시해, 과대해석 위험이 크게 줄었다.

### 3.4 Brier/ECE 표기 오류가 해결됨

표 4.1과 표 4.3에서 Brier는 RF가 근소 우위, ECE는 GBM이 우위라는 해석이 맞게 반영되었다.
이전처럼 “calibration도 Fusion 1위”로 오해될 가능성이 줄었다.

### 3.5 PDF의 핵심 시각자료가 더 읽기 쉬워짐

그림 4.2는 Panel A/B로 분리되어 scale 문제가 많이 줄었다. 표/그림 목차도 optional caption 덕분에 개선되었다.

---

## 4. 아직 남은 마지막 리스크

## 4.1 숫자 불일치: 그림 4.2 caption vs CI 표

최신 텍스트에서 눈에 띄는 새 문제다.

- 그림 4.2 caption: decile 70--80에서 `+0.050`, decile 40--50에서 `-0.014`
- bootstrap CI 표: decile 70--80에서 `+0.041`, decile 40--50에서 `-0.016`

이 둘은 아마 서로 다른 산출물 또는 이전 수치가 섞인 것으로 보인다. 심사자가 표와 그림 caption을 같이 보면 바로 보일 수 있다.

권장 수정:

> 그림 4.2 caption의 수치를 CI 표와 같은 값으로 맞추거나, caption에서 구체 수치를 빼고
> “70--80에서 최대 우위, 40--50에서 Energy 우위 점추정치”처럼 정성적으로 표현한다.

추천은 **caption에서 구체 수치를 빼는 것**이다. 수치는 표가 담당하는 편이 안전하다.

---

## 4.2 `main.tex:286` 오탈자

현재 문장에 다음 표현이 남아 있다.

> 환각 답이 정답 답보다 더 자신있게 평가되는

권장 수정:

> 환각 답이 정답보다 더 자신있게 평가되는

작은 오탈자지만 최종본에서는 눈에 띈다.

---

## 4.3 “통계 유의”의 범위

bootstrap CI가 추가된 것은 큰 개선이다. 다만 decile 10개를 동시에 본 것이므로,
심사위원이 multiple comparison 보정을 물을 수 있다.

권장 완화 문장:

> 본 유의성 표시는 다중비교 보정 전의 prompt-level bootstrap CI 기준이며,
> decile별 차이의 탐색적 불확실성 진단으로 해석한다.

이 한 문장을 표 caption이나 본문에 추가하면 훨씬 안전하다.

---

## 4.4 `is_hard`와 SE/Energy의 순환성 한계

skeptic 리뷰에서 가장 강하게 나온 지점이다. `is_hard` 라벨도 free-sample N=10에서 나오고,
SE/Energy도 같은 free-sample 기반 신호이므로 방법론적 순환성 또는 내생성 질문이 가능하다.

현재 proxy 한계는 들어가 있지만, 이 순환성까지 정면으로 쓰면 더 단단하다.

권장 한계 문장:

> 또한 `is_hard` 라벨과 SE/Energy 신호가 모두 동일한 free-sampling 절차에서 파생되므로,
> 본 결과는 실제 독립 라벨에 대한 일반 환각 탐지 성능이라기보다 sampling 기반 difficulty proxy와
> sample-consistency 신호 사이의 조건부 관계를 측정한 것으로 해석해야 한다.

이 문장은 한계 절에 넣는 것이 좋다.

---

## 4.5 corpus support aggregation 세부

정의 단위는 정리되었지만, 질문과 두 후보 답변에서 entity를 뽑은 뒤 질문 단위 score로 만드는 집계 방식은
심사에서 질문받을 수 있다.

권장 보강:

- 질문 entity와 후보 entity를 합쳐 중복 제거했는가?
- 정답 후보와 환각 후보 entity가 모두 들어가는가?
- pair co-occurrence는 어떤 entity pair 조합에 대해 계산되는가?
- 후보별 score를 질문 단위로 평균/최소/최대 중 무엇으로 집계했는가?

이 중 실제 구현과 일치하는 한두 문장을 `experiment_method.tex`에 추가하면 충분하다.

---

## 4.6 PDF 조판 잔여

시각 평가 결과, 핵심 표/그림은 읽히지만 최종 제출 polish 관점에서 다음이 남아 있다.

| 문제 | 심각도 | 조치 |
|---|---|---|
| 영어 line-break artifact (`variance`, `entity-pair`, `rank-quantile`, `self-conditioning`, `logit-diagnostic`, `Challenges`) | 중간~높음 | `\mbox{}` 또는 LaTeX hyphenation 조정 |
| 그림 목차의 `그림4.1` 공백 문제 | 낮음 | `tocloft` spacing 조정 |
| 긴 caption으로 페이지 밀도 증가 | 중간 | caption 1--2문장으로 축소하고 해석은 본문으로 이동 |
| 일부 번호 리스트/그림 주변 여백 | 낮음~중간 | 제출 전 polish 수준에서 조정 |

---

## 5. 현재 방어 가능성

### 방어 가능한 주장

- `is_hard` proxy 기준에서 sample-consistency 계열 신호의 판별력은 corpus support 영역별로 달라진다.
- Energy는 본 표본에서 SE보다 모든 decile에서 높게 관찰된다.
- Fusion의 Energy 대비 평균 이득은 영역별로 균일하지 않다.
- Fusion-Energy 차이는 양끝 decile에서 더 분명하고, 중간 decile에서는 통계적으로 구별되지 않는 구간이 있다.
- corpus support는 detector input이 아니라 평가 축으로 사용되었다.

### 피해야 할 주장

- corpus support가 낮으면 일반적으로 환각 탐지가 어렵다.
- Fusion은 어떤 영역에서는 Energy보다 확실히 나쁘다.
- 본 결과가 모델/데이터셋/코퍼스 전반에 일반화된다.
- corpus statistic의 직접 한계 효용을 feature로 검증했다.
- TruthfulQA 결과가 독립적으로 강한 근거다.

---

## 6. 심사장에서 나올 가능성이 높은 질문

### Q1. `is_hard`가 실제 환각 라벨이 아닌데 왜 환각 탐지 논문인가?

안전한 답변:

> 본 논문은 새로운 환각 라벨을 구축하거나 SOTA detector를 제안하는 논문이 아니라,
> 기존 환각 탐지 신호가 free-sample 기반 hard-question proxy 기준에서 어떤 조건에서 안정적인지 평가하는 논문입니다.
> 따라서 결론은 `is_hard` proxy 기준의 조건부 신뢰도 분석으로 한정합니다.

### Q2. SE/Energy와 `is_hard`가 같은 sampling에서 나오면 순환적인 것 아닌가?

안전한 답변:

> 맞습니다. 그래서 본 논문은 이를 독립 라벨에 대한 일반 환각 탐지 성능으로 주장하지 않습니다.
> 동일 sampling 절차에서 파생된 difficulty proxy와 sample-consistency 신호의 조건부 관계를 분석한 것으로 제한합니다.

### Q3. corpus support를 feature로 넣지 않았는데 왜 한계 효용이라고 하나?

안전한 답변:

> 본 연구에서 corpus support는 detector feature가 아니라 평가 축입니다. 따라서 직접 feature 추가 효과를 주장하지 않고,
> 기존 신호들의 조건부 성능 변동을 드러내는 데 목적이 있습니다.

### Q4. Fusion-Energy 차이는 유의한가?

안전한 답변:

> prompt-level bootstrap CI를 보면 양끝 decile에서는 Fusion 우위가 0을 넘지만,
> 중간 decile에서는 0을 포함합니다. 따라서 결론은 Fusion 평균 우위가 모든 영역에서 균일하지 않다는 것이지,
> 모든 decile에서 유의한 차이가 있다는 주장은 아닙니다.

---

## 7. 마지막 수정 우선순위

### 제출 전 반드시 고칠 것

1. 그림 4.2 caption의 `+0.050`, `-0.014` 수치를 CI 표와 맞추거나 caption에서 삭제
2. `정답 답보다` 오탈자 수정
3. bootstrap CI의 “통계 유의”가 uncorrected exploratory CI임을 한 문장 추가
4. `is_hard`와 SE/Energy가 같은 free-sampling에서 나온다는 순환성 한계를 한계 절에 추가

### 가능하면 고칠 것

1. corpus support 질문 단위 score의 집계 방식 1--2문장 추가
2. 긴 caption 일부를 본문으로 이동
3. 영어 line-break artifact 정리
4. 그림 목차 `그림4.1` spacing 조정

---

## 8. 최종 결론

최신 수정본은 이전보다 확실히 좋아졌다. 특히 bootstrap CI 표 추가는 논문의 약했던 RQ3를 크게 보강했다.
이제 논문은 “좋은 아이디어지만 과장된 원고”에서 **“범위를 잘 한정한 방어 가능한 학사논문”**으로 이동했다.

남은 문제는 주로 마지막 정합성이다.

- figure caption과 CI 표의 수치 일치
- free-sampling 순환성 한계 명시
- 통계 유의성의 탐색적 성격 명시
- PDF 조판 polish

이 네 가지를 정리하면 제출본으로 매우 안정적이다.
