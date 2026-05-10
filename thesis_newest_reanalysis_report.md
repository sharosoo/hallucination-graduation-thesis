# 졸업논문 최신 재분석 보고서

> 분석 대상: 최신 `thesis/main.tex`, `thesis/main.pdf`, `thesis/sections/experiment_method.tex`,
> `thesis/thesis_evidence_table.tex`, `thesis/results_macros.tex`, 주요 figure asset
>
> 분석 방식: 텍스트, 방법론, PDF/시각, 심사위원 방어 관점 subagent 결과와 직접 검토를 종합

---

## 1. 최종 판정

최신본은 **학사 졸업논문으로 방어 가능하며, 조건부 제출 가능 상태**에 가깝다. 직전 보고서의 must-fix 대부분은 반영되었다.

해결된 것:

- §4.6의 “Fusion이 두 데이터셋 모두에서 1위” 문장 수정
- `experiment_method.tex`의 “정답 답” 수정
- “두 흐름” vs 세 항목 구조 수정
- “사후 판정 절차 없음” 표현을 후보 라벨과 `is_hard` proxy로 분리 설명
- corpus support 집계 방식 구체화
- bootstrap CI의 탐색적/다중비교 미보정 단서 명시
- free-sampling 순환성 한계 명시

남은 핵심은 **표 4.2와 표 4.3의 Fusion-Energy delta 정합성**이다. 이 항목은 논문 핵심 주장과 직접 연결되므로 제출 전 재확인이 필요하다.

---

## 2. 가장 중요한 잔여 이슈

## 2.1 표 4.2와 표 4.3 delta 차이

표 4.3 caption은 표 4.2가 소수 셋째 자리로 반올림된 AUROC라서 full-precision bootstrap 점추정치와 일부 decile에서 차이가 날 수 있다고 설명한다. 이 설명 방향은 좋다.

다만 실제 보이는 값 기준으로 일부 차이가 단순 반올림 오차보다 커 보인다.

예시:

| decile | 표 4.2 단순 차이 | 표 4.3 점추정치 | 차이 |
|---|---:|---:|---:|
| 30--40 | `0.826 - 0.812 = +0.014` | `+0.024` | `+0.010` |
| 40--50 | `0.850 - 0.864 = -0.014` | `-0.016` | `-0.002` |
| 70--80 | `0.918 - 0.868 = +0.050` | `+0.041` | `-0.009` |

`-0.002` 정도는 반올림으로 설명 가능하지만, `0.009--0.010` 차이는 독자가 “다른 산출물인가?”라고 물을 수 있다.

### 권장 해결

가장 좋은 해결책은 **표 4.2와 표 4.3을 같은 full-precision source에서 다시 생성**하는 것이다.

만약 두 표가 의도적으로 다른 산출 단위를 사용한다면, caption을 다음처럼 더 명확히 써야 한다.

> 표 4.2는 decile별 AUROC를 소수 셋째 자리로 표시한 요약표이고, 표 4.3은 bootstrap resampling 과정에서 각 반복의 AUROC 차이를 직접 재계산한 점추정치와 CI이다. 따라서 표 4.2의 표시값을 단순 차이로 계산한 값과 표 4.3의 bootstrap 점추정치는 일부 decile에서 다를 수 있다.

하지만 가능하면 수치를 통일하는 편이 낫다.

### 위험도

**중간~높음.** 핵심 주장인 “Fusion-Energy 영역별 변동”과 직접 연결된다.

---

## 3. 해결된 이전 이슈들

### 3.1 데이터셋별 1위 표현

이전에는 “Fusion이 두 데이터셋 모두에서 1위”라고 읽혔지만, 최신본은 다음처럼 정확해졌다.

> AGG와 HaluEval-QA에서는 Fusion이 가장 높고, TruthfulQA에서는 Random Forest가 Fusion을 0.002 차로 근소하게 앞선다.

이제 표 4.4와 본문이 일치한다.

### 3.2 `정답 답` 오탈자

`experiment_method.tex`의 표현이 “정답 답변과 환각 답변”으로 수정되었다. 해결됐다.

### 3.3 “두 흐름” vs 세 항목

관련 연구 문장이 “세 흐름”으로 수정되었다. 해결됐다.

### 3.4 사후 판정 절차 설명

방법 절은 이제 후보 답변의 정답/환각 라벨은 dataset annotation을 쓰고, 질문 단위 `is_hard` proxy는 free-sample N=10의 token-overlap으로 별도 산출한다고 분리 설명한다. 해결됐다.

### 3.5 corpus support 집계 방식

질문과 두 후보 답변의 entity를 합집합으로 모으고, unordered pair를 구성하며, `f_min`, `p_mean`, `coverage = 1/2(f_axis + p_axis)`를 산출한다고 명시되었다. 이전보다 훨씬 명확하다.

---

## 4. 방법론 방어 가능성

현재 방법론은 **학사논문 기준으로 방어 가능**하다.

방어력이 올라간 이유:

- `is_hard`를 실제 환각 라벨이 아닌 hard-question proxy로 반복 제한한다.
- SE/Energy와 `is_hard`가 같은 free-sampling에서 파생된다는 순환성 한계를 명시한다.
- TruthfulQA skew를 인정하고 HaluEval-QA only 보조 분해를 추가했다.
- corpus support를 detector input이 아니라 평가 축으로만 사용한다고 명시한다.
- bootstrap CI를 다중비교 미보정 탐색적 진단으로 설명한다.

남는 본질적 한계:

- `is_hard`는 독립 환각 라벨이 아니다.
- corpus support는 질문과 후보 답변 entity 합집합에서 산출되므로 prompt-pure condition은 아니다.
- TruthfulQA는 label skew가 크다.
- 레포 내부 구형 산출물과 본문 최신 계약이 완전히 정리되어 있지 않으면 혼란이 생길 수 있다.

이 한계들은 논문에 어느 정도 명시되어 있으므로, 방어 시 claim만 제한하면 충분히 대응 가능하다.

---

## 5. 문장/표현 잔여 polish

### 5.1 강한 해석 표현

다음 표현은 방어적 문체로 더 낮출 수 있다.

| 현재 표현 | 권장 표현 |
|---|---|
| “직관이 corpus 영역에 무관하게 일관되게 깨지는 결과” | “해당 직관이 본 표본에서는 일관되게 지지되지 않는 결과” |
| “CHOKE 패턴이 corpus 영역에 무관하게 유지” | “CHOKE 패턴과 일관된 약한 판별력이 관찰” |
| “효과는 분명하다” | “보완 효과가 관찰된다” |

치명적이지는 않지만, 최종본에서는 조금 더 학술적으로 보인다.

### 5.2 “Farquhar 2024와 동일 수준에서 재현”

본문은 SE/Energy baseline이 Farquhar 2024의 AUROC와 동일 수준에서 재현된다고 말한다. 데이터셋, 라벨, 모델, 평가 단위가 다르므로 “재현”은 약간 강하다.

권장 표현:

> Farquhar 등 (2024)이 보고한 TriviaQA / Llama-2 7B 환경의 AUROC 0.79와 비슷한 수치 범위이다.

---

## 6. PDF/시각 polish

현재 PDF는 구조, 표, 그림 배치가 전반적으로 제출 가능한 수준이다. 다만 polish 항목이 남아 있다.

남은 항목:

- `rank-quantile`, `self-conditioning`, 참고문헌 `Challenges` 등 soft-hyphen artifact
- “95% CI 가 0 위” 표현
- 일부 PDF 추출상 공백 누락: “환각 답이정답보다”, “후보답변의평균음의로그확률”
- 그림 목차의 `그림4.1`, `그림4.2` spacing
- 긴 caption으로 인한 페이지 밀도

우선순위:

1. soft-hyphen artifact 제거
2. “95% CI가 0을 포함하지 않음”으로 표현 수정
3. 조사/띄어쓰기 polish
4. 그림 목차 spacing 정리

---

## 7. 방어용 최종 프레이밍

심사에서 가장 안전한 한 문장:

> 본 논문은 독립 환각 라벨에 대한 일반 탐지 성능을 주장하는 것이 아니라,
> free-sampling 기반 `is_hard` proxy 조건에서 기존 환각 탐지 신호의 조건부 신뢰도가
> corpus support 구간에 따라 어떻게 달라지는지 분석한 평가 연구이다.

피해야 할 말:

- “환각 탐지 성능을 일반적으로 입증했다.”
- “Fusion이 특정 영역에서 Energy보다 확실히 나쁘다.”
- “corpus support를 feature로 넣어 한계 효용을 검증했다.”
- “TruthfulQA 결과도 독립적으로 강한 근거다.”

---

## 8. 마지막 체크리스트

### 반드시 확인

- [ ] 표 4.2와 표 4.3의 Fusion-Energy delta source 통일 또는 설명 강화
- [ ] “95% CI 가 0 위” 표현 수정
- [ ] “Farquhar 2024와 동일 수준에서 재현” 표현 완화 여부 검토

### 가능하면 수정

- [ ] soft-hyphen artifact 제거
- [ ] 강한 해석 표현 완화
- [ ] 그림 목차 spacing 정리
- [ ] 긴 caption 일부 축소

---

## 9. 최종 결론

최신본은 이제 **방어 가능한 학사논문**이다. 이전 must-fix는 대부분 해결되었고,
방법론적 한계도 정직하게 드러냈다.

최종 제출 전 가장 중요한 것은 하나다.

> **표 4.2와 표 4.3의 Fusion-Energy delta 정합성 확인**

이 문제를 해결하거나 충분히 설명하면, 나머지는 대부분 최종 polish 영역이다.
