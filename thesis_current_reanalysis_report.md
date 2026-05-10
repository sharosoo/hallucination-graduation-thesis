# 최신 재분석 보고서: 학사 졸업논문 제출/방어 준비도 점검

## 1. 종합 판정

현재 원고는 **학사 졸업논문으로 방어 가능(defensible)** 한 수준에 도달했다. 다만 이는 “독립적인 SOTA 환각 탐지기 제안”이 아니라, **free-sampling 기반 `is_hard` hard-question proxy를 기준으로 SE/Energy/Fusion 신호의 corpus-support 조건부 행태를 분석한 연구**로 해석할 때의 판정이다.

핵심 약점이었던 proxy 라벨, TruthfulQA skew, HaluEval-QA 보조 분석, bootstrap 해석, free-sampling 순환성은 본문과 한계 절에서 대부분 선제적으로 인정되어 있다. 따라서 남은 문제는 큰 방법론 결함보다는 **수치 대표값 통일, 과강한 표현 완화, PDF/문장 polish**에 가깝다.

## 2. 이번 라운드에서 확인된 개선점

### 2.1 주장 범위 통제

- `is_hard`가 실제 환각 라벨이 아니라 free-sample 기반 hard-question proxy라는 점이 초록, 결과 해석, 한계에서 반복적으로 드러난다.
- HaluEval-QA는 주 분석이 아니라 보조 검증으로 분리되어 있어, AGG 0.889 또는 top-decile 0.978을 일반 환각 탐지 성능으로 과대해석할 위험이 줄었다.
- TruthfulQA의 token-overlap 기반 라벨 노이즈와 극단적 class skew를 한계로 인정하고 있다.
- 단일 모델, 단일 corpus, free-sampling 기반 proxy라는 제한을 결론부에서 방어적으로 처리하고 있다.

### 2.2 방법론 설명

- corpus support aggregation 수식과 정의가 이전보다 명확하다.
- corpus support를 fusion 입력 feature로 사용하지 않고 평가 축으로만 사용했다는 점은 self-conditioning 비판을 완화한다.
- prompt-grouped bootstrap, 95% CI, 다중비교 보정 전 탐색적 진단이라는 설명은 방어 가능한 수준이다.

### 2.3 표와 그림

- Table 4.2/4.3의 delta 차이에 대해 full-precision 또는 bootstrap 산출 기준이 다를 수 있다는 설명 장치가 추가되었다.
- Figure 4.1/4.2는 축, 범례, panel 구분이 읽히는 수준이며, 이전의 심각한 overlap 문제는 크게 완화된 것으로 보인다.
- PDF 전체 구조는 목차, 본문, 표/그림, 참고문헌 흐름이 완결되어 있다.

## 3. 제출 전 필수 수정 권장 항목

### 3.1 “95% CI가 0 위” 표현 수정

초록과 결론부에 남아 있는 **“95% CI가 0 위”** 표현은 한국어로 부자연스럽고 통계적 의미도 모호하다.

권장 표현:

- “95% CI가 0을 포함하지 않는다”
- “95% CI의 하한이 0보다 크다”
- “bootstrap 95% CI 기준으로 0과 분리된다”

이 항목은 작은 문장 문제이지만, 심사위원이 바로 발견할 수 있는 비문이므로 최우선으로 고치는 편이 좋다.

### 3.2 Table 4.2/4.3 delta 대표값 통일

수치 일관성은 대부분 개선됐지만, 아직 다음과 같은 잔여 불일치가 보고되었다.

- decile 40--50의 Fusion--Energy 차이가 `-0.016`과 `-0.014`로 혼재한다.
- 일부 위치에서 `+0.050`과 `+0.047`처럼 같은 delta를 가리키는 듯한 수치가 다르게 나타난다.
- Table 4.2의 rounded AUROC 차이와 Table 4.3의 bootstrap/full-precision delta가 단순 반올림 차이 이상으로 보일 수 있다.

권장 조치:

1. 본문, 매크로, Table 4.3의 대표 delta 값을 하나의 source of truth로 통일한다.
2. Table 4.3 caption에는 “rounded AUROC의 단순 차”가 아니라 “full-precision AUROC 또는 bootstrap estimate 기준 delta”라는 점을 더 분명히 적는다.
3. 발표/방어에서는 Table 4.2는 decile별 성능 요약, Table 4.3은 prompt-grouped bootstrap 기반 delta 검정이라는 역할 차이를 명시한다.

### 3.3 PDF soft-hyphen / 인코딩 artifact 제거

PDF 텍스트 또는 시각 검사에서 다음 형태의 하이픈 artifact가 남아 있는 것으로 보고되었다.

- `rank￾quantile`
- `self￾conditioning`
- `Chal￾lenges`

실제 눈으로 보이는 결함인지 텍스트 추출상의 discretionary hyphen인지 확인이 필요하지만, 제출본 품질 관점에서는 제거하는 것이 안전하다. TeX 소스에서 수동 하이픈, discretionary hyphen, 줄바꿈 처리, bibliography entry의 특수문자를 점검하는 것을 권장한다.

### 3.4 “동일 수준에서 재현된다” 표현 완화

Farquhar 등(2024)의 TriviaQA / Llama-2 7B AUROC와 비교하는 문장에서 **“동일 수준에서 재현된다”**는 표현은 다소 강하다. 데이터셋, 모델, 라벨/proxy 조건이 다르므로 재현(replication)처럼 읽히면 방어 부담이 커진다.

권장 표현:

- “비슷한 수준의 AUROC를 보인다”
- “선행연구에서 보고된 수치와 대략 같은 범위에 있다”
- “직접 재현이라기보다 조건이 다른 참고 비교로 해석해야 한다”

### 3.5 강한 인과/일관성 표현 완화

다음 계열의 문장은 현재 연구 설계보다 약간 강하게 읽힐 수 있다.

- “corpus exposure가 신뢰도 분포를 형성한다”
- “직관이 corpus 영역에 무관하게 일관되게 깨진다”
- “효과는 분명하다”
- “외부 Corpus 신호의 조건부 한계 효용”

권장 방향은 인과 표현을 피하고, **관찰된 조건부 패턴**으로 낮추는 것이다.

예시:

- “corpus support decile에 따라 신뢰도 지표의 분포가 달라지는 양상이 관찰된다”
- “본 실험 조건에서는 naive corpus-support 직관과 다른 패턴이 반복적으로 나타난다”
- “corpus feature를 추가한 성능 향상”이 아니라 “기존 탐지 신호의 corpus-support 조건부 reliability 분석”임을 명시한다.

### 3.6 그림목차 spacing polish

PDF 그림목차에서 `그림4.1`, `그림4.2`처럼 “그림”과 번호 사이 공백이 없는 형식이 보고되었다. 본문 캡션의 `그림 4.1` 형식과 맞추는 것이 좋다.

## 4. 방법론 방어 포인트

### 4.1 가장 중요한 방어 문장

방어 발표에서는 다음 취지를 먼저 말하는 것이 안전하다.

> 본 논문은 독립적인 환각 판별 benchmark에서 새로운 탐지기를 제안하는 연구가 아니라, Qwen2.5-3B와 Dolma sample corpus 조건에서 free-sampling 기반 `is_hard` proxy를 사용해 SE/Energy/Fusion의 조건부 신뢰도 변화를 분석한 연구이다.

이 문장을 먼저 확정하면 proxy 라벨, sampling 순환성, 일반화 한계에 대한 질문을 방어적으로 흡수할 수 있다.

### 4.2 예상 질문과 답변 방향

#### Q1. `is_hard`는 실제 hallucination label이 아닌데 왜 환각 탐지라고 부르는가?

답변 방향: 맞다. 그래서 본 논문은 독립 환각 판별 성능을 주장하지 않고, free-sampling에서 정답 도달이 어려운 prompt를 hard-question proxy로 정의해 기존 uncertainty 신호가 이 proxy를 얼마나 구분하는지 본다.

#### Q2. SE/Energy와 `is_hard`가 같은 free-sampling에서 나오면 순환적이지 않은가?

답변 방향: 순환성 위험이 있다. 따라서 결과는 일반 환각 탐지 성능이 아니라 sampling 기반 조건부 관계 측정으로 제한한다. 이 한계는 결론과 한계 절에 명시했다.

#### Q3. corpus support가 정답+환각 후보 entity 합집합으로 정의되면 실제 탐지 상황에서 관찰 가능한가?

답변 방향: 본 연구에서 corpus support는 모델 입력 feature가 아니라 사후 분석 축이다. 실제 deployable detector의 입력으로 제안하는 것이 아니므로, 관찰 가능성보다는 corpus-conditioned diagnostic axis로 해석해야 한다.

#### Q4. 다중비교 보정 전인데 통계적으로 유의하다고 말해도 되는가?

답변 방향: decile별 bootstrap은 탐색적 진단이며 family-wise error 보정 전 결과다. 따라서 개별 decile의 확정적 유의성보다 전체 패턴과 불확실성 범위를 함께 해석한다.

#### Q5. 단일 모델과 단일 corpus로 일반화할 수 있는가?

답변 방향: 일반화는 제한된다. 본 논문은 일반 법칙을 주장하지 않고, 한 조건에서 관찰된 패턴과 그 해석상 한계를 제시한다.

## 5. Repository-level 주의점

논문 `.tex`와 PDF는 최신 프레임으로 정리되었지만, 일부 `experiments/results/*.json` 또는 `fusion/summary.json` 계열 산출물이 구형 candidate-row/TypeLabel 결과를 담고 있을 수 있다는 지적이 있었다. 심사자가 repository까지 함께 보면 혼란이 생길 수 있으므로, 현재 논문의 source of truth가 `thesis/` 쪽 표, 매크로, 본문이라는 점을 README나 appendix 수준에서 분명히 하는 것이 좋다.

## 6. 최종 제출 준비도

현재 준비도는 다음과 같이 평가한다.

- **내용/방법론:** 제출 가능에 가까움
- **주장 범위 통제:** 이전보다 크게 개선됨
- **표/그림:** 대체로 사용 가능하나 delta 설명과 PDF artifact 확인 필요
- **문장 polish:** 몇몇 비문과 강한 표현 수정 필요
- **방어 가능성:** proxy 기반 조건부 분석으로 프레이밍하면 방어 가능

최종 결론: **큰 구조나 실험을 다시 할 필요는 없어 보이며, 제출 전에는 1--2시간 내에 처리 가능한 수치/문장/PDF polish를 우선 해결하는 것이 가장 효율적이다.**
