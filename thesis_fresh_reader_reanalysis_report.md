# 처음 보는 독자 관점 재분석 보고서

## 1. 종합 판정

현재 논문은 **학사 졸업논문으로는 방어 가능**하다. 연구 질문, 관련 연구 연결, 실험 절차, 결과 해석, 한계 인식이 모두 존재하며, 특히 `is_hard` proxy와 free-sampling 순환성 한계를 본문에서 숨기지 않고 밝히는 점은 강점이다.

다만 처음 보는 독자에게는 아직 **“일반 환각 탐지 성능 논문”인지, “sampling 기반 hard-question proxy에 대한 조건부 신뢰도 분석 논문”인지**가 초반에 흔들릴 수 있다. 이 논문은 후자로 방어해야 안전하다. 제목, 초록 첫 문단, 연구 질문, 결론의 표현을 이 프레이밍에 더 강하게 맞추면 방어 가능성이 크게 올라간다.

한 문장으로 정리하면 다음과 같다.

> 본 논문은 독립적인 후보별 환각 라벨을 맞히는 새 탐지기 성능 논문이 아니라, free-sample 정답 매칭률로 정의한 `is_hard` hard-question proxy에 대해 SE/Energy/Fusion 신호의 corpus-support 조건부 판별력이 어떻게 달라지는지를 분석한 학사논문이다.

## 2. 처음 보는 독자가 이해한 논문

논문은 TruthfulQA 815문항과 HaluEval-QA 5,000문항, 총 5,815개 질문을 대상으로 한다. 각 질문에 대해 Qwen2.5-3B의 free-sampling 답변 10개를 만들고, 정답 token-overlap 매칭률이 50% 미만이면 질문 단위 `is_hard=1`로 둔다. 이후 Semantic Entropy, Semantic Energy, logit diagnostic, Fusion이 이 `is_hard` proxy를 얼마나 잘 구분하는지 AUROC로 평가한다.

외부 corpus support는 질문과 정답 후보, 환각 후보에서 추출한 entity의 frequency 및 entity-pair co-occurrence로 만든다. 이 값은 Fusion 입력 feature로 넣지 않고, 질문들을 decile로 나누는 **평가 축**으로만 사용한다. 핵심 주장은 평균 AUROC 하나로는 보이지 않는 영역별 변동, 특히 Fusion 우위가 corpus support 영역에 따라 균일하지 않다는 점이다.

## 3. 강점

### 3.1 학사논문 기준의 완성도

- 연구 질문이 세 개로 명확하게 제시된다.
- 관련 연구가 SE, Energy, fusion, corpus statistic 활용으로 자연스럽게 연결된다.
- 실험 방법이 데이터셋, 모델 출력, feature 계산, corpus support, 평가 지표 순서로 비교적 투명하게 설명된다.
- 결론에서 연구 질문에 다시 답하는 구조가 안정적이다.
- 한계 절이 형식적이지 않고 실제 약점을 인정한다.

### 3.2 방법론적 장점

- corpus support를 Fusion 입력 feature로 사용하지 않고 decile 평가 축으로만 사용해 self-conditioning artifact를 피하려 한 점은 방어에 유리하다.
- 평균 AUROC만 보고하지 않고 decile별 AUROC와 Fusion--Energy bootstrap CI를 제시한 점은 학부 논문으로 충분히 의미 있는 분석이다.
- TruthfulQA의 token-overlap 라벨 노이즈, `is_hard`와 SE/Energy의 free-sampling 순환성, 단일 모델/단일 corpus 한계를 명시한 점은 정직하다.

### 3.3 최근 개선 확인

- `95% CI가 0 위` 표현은 현재 `main.tex` 기준으로 “0을 포함하지 않음” 계열 표현으로 정리되어 있다.
- Farquhar 등(2024) 비교도 “동일 수준에서 재현”이 아니라 “비슷한 수준 / 직접 재현이 아닌 참고 비교”로 완화되어 있다.
- `results_macros.tex`의 `\HeadlineFusionVsEnergyMinDelta{-0.016}` 및 `\HeadlineFusionVsEnergyMaxDelta{0.047}`는 Table 4.3의 `-0.016`, `+0.047`과 맞는다.

## 4. 가장 큰 리스크: 프레이밍

모든 reviewer가 공통으로 지적한 가장 큰 위험은 **프레이밍의 크기**다. 제목과 초록, 서론의 큰 간판은 “LLM 환각 탐지”인데, 실제 주요 AUROC target은 후보별 환각 여부가 아니라 질문 단위 `is_hard` proxy이다.

방어에서 흔들리면 안 되는 답은 다음이다.

> Positive label은 독립 환각 라벨이 아니라 질문 단위 `is_hard` proxy이다. 본 논문은 환각 여부를 새로 판정하는 benchmark가 아니라, 환각 탐지에 쓰이는 uncertainty 신호가 sampling 기반 hard-question proxy를 corpus support 조건별로 얼마나 잘 구분하는지 분석한다.

이 문장을 제목, 초록, 연구 질문, 결론에 더 일관되게 반영해야 한다.

## 5. 제출 전 우선 수정 항목

### 5.1 제목 낮추기

현재 제목의 “LLM 환각 탐지”와 “외부 Corpus 신호의 조건부 한계 효용”은 실제 분석보다 강하게 읽힌다. 특히 “한계 효용”은 corpus feature를 detector 입력으로 넣어 marginal utility를 검증한 것처럼 오해될 수 있다.

권장 방향:

- `LLM 환각 탐지 신호의 Corpus Support 조건부 신뢰도 분석: Sampling 기반 Hard-Question Proxy를 중심으로`
- 또는 `Sampling 기반 Hard-Question Proxy에서 LLM 환각 탐지 신호의 Corpus-conditioned Reliability 분석`

핵심은 “새 환각 탐지기”가 아니라 “환각 탐지 신호의 조건부 reliability 분석”임을 제목에서 드러내는 것이다.

### 5.2 초록 첫 문단에서 proxy 정체성 먼저 밝히기

현재 초록은 “LLM 환각 탐지에서…”로 시작한 뒤 다음 문단에서 `is_hard`가 proxy라고 수습한다. 처음 보는 독자는 첫 문단의 AUROC 설명을 독립 환각 탐지 성능으로 읽을 수 있다.

권장 수정 방향:

- 첫 2--3문장 안에 “본 논문은 독립 환각 라벨이 아니라 free-sample 정답 매칭률 기반 `is_hard` proxy를 사용한다”고 명시한다.
- AGG 0.889나 top decile 수치보다 “proxy 기준 조건부 패턴”이라는 범위를 먼저 세운다.

### 5.3 연구 질문 1의 표현 수정

“탐지 난이도”는 일반 환각 탐지 난이도로 읽힐 수 있다. “hard-question proxy 판별 난이도” 또는 “`is_hard` proxy 판별력”으로 낮추는 것이 안전하다.

예시:

- `탐지 난이도와 corpus support의 관계` → ``is_hard` proxy 판별력과 corpus support의 관계`

### 5.4 후보 라벨과 질문 라벨의 관계 명확화

실험 설정에서 후보 답변의 정답/환각 annotation과 질문 단위 `is_hard` proxy가 모두 등장한다. 처음 보는 독자는 “후보별 환각을 맞힌 것인가, 질문별 어려움을 맞힌 것인가?”를 헷갈릴 수 있다.

권장 설명:

- 후보 annotation은 teacher-forced scoring 및 정답 매칭 기준을 구성하는 데 쓰인다.
- 주요 AUROC 라벨은 후보 row가 아니라 prompt-level `is_hard`이다.
- 따라서 결과 단위는 11,630 candidate rows가 아니라 5,815 prompt-level examples임을 더 앞에서 강조한다.

### 5.5 Corpus support 정의의 오해 방지

corpus support는 질문만으로 계산된 조건 변수가 아니라, 질문과 두 후보 답변의 entity 합집합에서 계산된다. 이는 paired discriminative analysis에서는 가능하지만, 일반 inference-time detector feature처럼 보이면 공격받는다.

권장 표현:

- `질문 단위 corpus support` → `question + candidate-pair entity 기반 corpus support`
- 초록 또는 서론 기여 문단에도 “정답 후보와 환각 후보 entity를 포함한 사후 분석 축”이라는 단서를 한 번 넣는다.

### 5.6 Fusion 유의성 표현 낮추기

Table 4.3은 다중비교 보정 전 bootstrap CI이다. 본문에서는 대체로 조심스럽지만, 초록과 결론에서는 “양 끝 5개 영역에서 통계 유의”가 강하게 읽힐 수 있다.

권장 표현:

- `통계 유의` → `다중비교 보정 전 exploratory bootstrap 기준으로 CI가 0을 포함하지 않음`
- `Fusion이 약해질 수 있음` → `Energy와 통계적으로 구별되지 않는 영역이 있음`

## 6. 새로 확인된 수치/표기 이슈

### 6.1 Fusion highest decile 매크로 불일치

방법론 리뷰에서 중요한 수치 불일치가 새로 확인되었다.

- `results_macros.tex`: `\HeadlineFusionDecileHighest{0.978}`
- `main.tex` Table 4.2의 90--100 decile Fusion: `0.981`
- Table 4.3의 90--100 delta `+0.027`은 `0.981 - 0.954 = 0.027`과 맞는다.

따라서 `\HeadlineFusionDecileHighest`는 표 기준으로는 `0.981`이어야 할 가능성이 높다. 이 수치는 초록과 결론에 노출되는 headline 값이므로 반드시 통일해야 한다.

### 6.2 Table 4.3 caption의 표 참조 오류 가능성

PDF 리뷰에서 Table 4.3 caption에 “표 4.4의 반올림된 AUROC 단순 차이…”라는 식의 문구가 보인다고 지적되었다. 문맥상 decile AUROC matrix인 Table 4.2를 가리켜야 할 가능성이 높다. Table 4.4는 데이터셋별 AUROC/calibration 표이므로 독자가 혼란스러울 수 있다.

### 6.3 ECE 단위 혼동

표에서는 ECE가 `0.014` 비율 단위이고, 매크로는 `\HeadlineECE{1.4}`이며 본문에서 `\HeadlineECE\%`로 사용된다. 의도는 맞지만 비율과 퍼센트 단위가 섞여 보일 수 있다. 표와 본문에서 `0.014 (=1.4%)`처럼 한 번 병기하면 안전하다.

## 7. PDF/조판 리스크

PDF는 구조와 표/그림 가독성 면에서 거의 제출 가능하지만, 다음 polish defect는 제출 전 수정해야 한다.

### 7.1 soft-hyphen/control-character artifact

PDF에서 다음 형태의 깨짐이 반복 보고되었다.

- `rank￾quantile`
- `self￾conditioning`
- `Chal￾lenges`

이는 `\mbox{rank-quantile}`, `\mbox{self-conditioning}` 같은 하이픈 포함 단어와 긴 영어 참고문헌 제목 줄바꿈에서 비롯된 것으로 보인다. 제출본에서 보이면 완성도를 크게 낮춘다.

### 7.2 그림 번호 spacing

그림 목차와 그림 캡션에서 `그림4.1`, `그림4.2`처럼 “그림”과 번호 사이 공백이 없어 보인다. 표는 `표 4.1`처럼 자연스러워서 그림 쪽 불일치가 더 눈에 띈다.

### 7.3 캡션 길이

Figure 4.1/4.2와 Table 4.3 caption은 정확하지만 길고 빽빽하다. 학사논문으로 허용 가능하나, 첫 독자에게는 캡션이 본문 문단처럼 느껴질 수 있다. 핵심 문장만 남기고 상세 해석은 본문으로 옮기면 더 깔끔하다.

## 8. 심사위원 예상 질문

1. 이 논문의 positive label은 환각인가, 어려운 질문인가?
2. SE/Energy와 `is_hard`가 같은 N=10 free samples에서 나오는데, 이것이 순환적 자기예측 아닌가?
3. TruthfulQA의 `is_hard=0.97`이면 AGG 0.889와 top decile 0.981을 얼마나 믿을 수 있는가?
4. HaluEval-only에서는 최고 decile 수치가 더 낮다면서 왜 AGG 결과가 headline인가?
5. corpus support에 정답 후보와 환각 후보 entity를 모두 넣으면 실제 배포 시점 조건 변수라고 볼 수 있는가?
6. rank-quantile decile이 절대적인 corpus-poor/rich 의미를 갖는가, 아니면 데이터셋 내부 상대 순위인가?
7. 다중비교 보정 전인데 “양 끝 영역 유의”를 초록에서 강조해도 되는가?
8. teacher-forced candidate-level logit diagnostics를 prompt-level feature로 정확히 어떻게 집계했는가?
9. CHOKE 사례를 직접 식별하지 않았는데 “CHOKE 패턴”이라고 부를 수 있는가?

## 9. 방어 전략

방어 발표 첫머리에 다음 문장을 확정적으로 말하는 것이 좋다.

> 본 논문은 독립 환각 판별 benchmark에서 새 탐지기의 성능을 주장하는 연구가 아닙니다. free-sample 정답 매칭률로 정의한 질문 단위 `is_hard` proxy를 사용해, 기존 환각 탐지 신호인 SE/Energy/Fusion이 corpus support 조건별로 얼마나 안정적으로 작동하는지 분석한 연구입니다.

이 문장 이후에는 다음 순서로 방어한다.

1. 왜 proxy를 썼는지 설명한다: SE 원논문 평가 단위와 맞추고, 질문 단위 difficulty를 보기 위함.
2. 순환성은 인정한다: 그래서 일반 환각 탐지 성능이 아니라 조건부 관계 측정이라고 제한한다.
3. TruthfulQA skew를 인정한다: 주요 방어는 HaluEval-QA와 제한적 해석에 둔다.
4. corpus support는 deployable feature가 아니라 사후 분석 축이라고 설명한다.
5. Fusion 결과는 “항상 우월”이 아니라 “평균 우위가 영역별로 균일하지 않음”이라고 말한다.

## 10. 최종 제출 준비도

### Go / No-Go

**조건부 Go.** 큰 실험을 새로 할 필요는 없어 보인다. 학사논문 기준으로 연구 질문과 분석은 충분하다. 그러나 제출 전 다음 네 가지는 반드시 고치는 것이 좋다.

1. 제목/초록/연구질문에서 `hard-question proxy` 프레이밍을 전면화한다.
2. `\HeadlineFusionDecileHighest{0.978}` vs Table 4.2 `0.981` 수치를 통일한다.
3. Table 4.3 caption의 표 참조 오류 가능성을 확인한다.
4. PDF soft-hyphen artifact와 `그림4.1` spacing 문제를 수정한다.

이 네 가지를 고치면 처음 보는 독자에게도 “정직한 proxy 기반 조건부 분석 논문”으로 읽힐 가능성이 높고, 학사 졸업논문 방어 가능성은 충분하다.
