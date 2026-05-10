# 졸업논문 최종 후속 재분석 보고서

> 분석 대상: 최신 `thesis/main.tex`, `thesis/main.pdf`, `thesis/sections/experiment_method.tex`,
> `thesis/thesis_evidence_table.tex`, `thesis/results_macros.tex`, 주요 figure asset
>
> 분석 방식: 텍스트, 방법론, PDF/시각, 심사위원 방어 관점 subagent 결과와 직접 검토를 종합

---

## 1. 최종 판정

최신 원고는 **학사 졸업논문으로 방어 가능**하다. 이전에 남아 있던 네 가지 핵심 문제는 대부분 반영되었다.

- 그림 4.2 caption의 직접 수치 불일치: **대체로 해결**
- `정답 답보다` 오탈자: **해결**
- bootstrap CI의 탐색적/다중비교 미보정 단서: **해결**
- `is_hard`와 SE/Energy의 free-sampling 순환성 한계: **해결**

현재 논문은 다음 프레이밍으로 말할 때 가장 안전하다.

> 실제 독립 환각 라벨에 대한 최종 탐지기 성능 논문이 아니라,
> sampling-derived `is_hard` hard-question proxy 하에서 기존 신호들의 조건부 reliability를
> corpus support 축에서 분해한 평가 연구이다.

이 프레이밍을 유지하면 충분히 설득력 있다.

---

## 2. 이번 수정에서 좋아진 점

### 2.1 그림 4.2 caption이 안전해짐

이전에는 caption에 `+0.050`, `-0.014` 같은 수치가 직접 들어가 CI 표와 충돌했다. 최신본은 구체 수치를 빼고
“정확한 수치와 95% bootstrap CI는 표 참조”로 바뀌었다. 이 방향은 안전하다.

### 2.2 free-sampling 순환성 한계가 명시됨

한계 절에 `is_hard` 라벨과 SE/Energy 신호가 모두 동일한 free-sampling 절차에서 파생된다는 점이 추가되었다.
이는 심사장에서 나올 가능성이 높은 공격을 선제적으로 막는다.

### 2.3 bootstrap CI 단서가 들어감

CI 표 caption에 다중비교 보정 전의 탐색적 불확실성 진단이라는 설명이 들어갔다. 이제 “통계 유의” 표현이
무리한 확증 주장으로 읽힐 가능성이 줄었다.

### 2.4 HaluEval-QA only 보조 분해가 추가됨

TruthfulQA 라벨 노이즈를 제거한 HaluEval-QA only 분해가 들어가, pooled 결과가 dataset mixture artifact일 수 있다는
비판에 대한 방어력이 좋아졌다.

---

## 3. 아직 남은 중요 리스크

## 3.1 표 4.2와 표 4.3의 Fusion-Energy delta 불일치

현재 가장 중요한 남은 이슈다.

예시:

- 표 4.2 기준 decile 70--80: Fusion `0.918`, Energy `0.868` → 차이 `+0.050`
- 표 4.3 기준 decile 70--80: Fusion - Energy 점추정치 `+0.041`

또 다른 예:

- 표 4.2 기준 decile 40--50: Fusion `0.850`, Energy `0.864` → 차이 `-0.014`
- 표 4.3 기준 decile 40--50: `-0.016`

반올림, bootstrap 재계산, full-precision AUROC, 다른 fold 산출물 등 이유가 있을 수 있다. 하지만 독자는 두 표를 단순 비교한다.

권장 해결:

1. 표 4.3 caption 또는 본문에 다음 설명 추가:

   > 표 4.2는 소수 셋째 자리로 반올림한 AUROC를 표시하므로, 표 4.3의 full-precision bootstrap 점추정치와
   > 단순 차이가 일부 decile에서 다를 수 있다.

2. 가능하면 표 4.2와 표 4.3이 같은 full-precision source에서 나오도록 재생성.

### 실전 위험도

**중간 이상.** 심사자가 표를 대조하면 바로 물어볼 수 있다. 가장 먼저 정리할 항목이다.

---

## 3.2 §4.6 “Fusion이 두 데이터셋 모두에서 1위” 문장

표 4.4에서 TruthfulQA는 Random Forest `0.953`, Fusion `0.951`이다. 그런데 본문은 “Fusion이 두 데이터셋 모두에서 1위”라고 되어 있다.

권장 수정:

> AGG와 HaluEval-QA에서는 Fusion이 가장 높고, TruthfulQA에서는 Random Forest가 근소하게 높다.

또는:

> Fusion은 AGG와 HaluEval-QA에서 1위이며, TruthfulQA에서는 Random Forest와 근소한 차이로 경쟁한다.

### 실전 위험도

**중간.** 작은 차이지만 표와 본문이 다르면 신뢰도가 깎인다.

---

## 3.3 `experiment_method.tex`의 “정답 답” 오탈자

`main.tex`의 `정답 답보다`는 고쳐졌지만, `experiment_method.tex`에는 다음 표현이 남아 있다.

> 제공되는 정답 답과 환각 답

권장 수정:

> 제공되는 정답과 환각 답변

또는:

> 제공되는 정답 답변과 환각 답변

### 실전 위험도

낮음. 다만 최종 polish에서는 고치는 것이 좋다.

---

## 3.4 관련 연구의 “두 흐름” vs 세 항목

관련 연구에서 “corpus statistic을 환각 탐지 신호로 직접 사용한 두 흐름”이라고 한 뒤,
QuCo-RAG, Zhang, WildHallucinations 세 항목을 제시한다.

권장 수정:

> corpus statistic을 환각 탐지 또는 factuality 평가에 활용한 세 흐름이 있다.

### 실전 위험도

낮음~중간. 독자가 꼼꼼하면 구조적 어색함을 느낄 수 있다.

---

## 3.5 `experiment_method.tex`의 “사후 판정 절차 없음” 표현

방법 절에서 “새 답을 생성한 뒤 정오를 사후 판정하는 절차 없이 paired 비교”라고 설명하지만,
동시에 free sampling N=10과 정답 매칭률로 `is_hard`를 정의한다. 의도는 “후보 답변의 환각 여부를 LLM-as-judge로 새 판정하지 않는다”일 가능성이 크다.

권장 수정:

> 후보 답변의 정답/환각 라벨은 데이터셋 annotation을 사용하며, LLM-as-judge로 후보의 정오를 새로 판정하지 않는다.
> 다만 질문 단위 `is_hard` proxy는 free-sample N=10의 정답 매칭률로 별도 산출한다.

### 실전 위험도

중간. 라벨 정의와 직접 관련되어 있어 명확히 하는 것이 좋다.

---

## 4. 방법론 방어 가능성

### 방어 가능해진 이유

- `is_hard` proxy라는 범위 제한이 초록, 본문, 한계에 반복된다.
- TruthfulQA skew를 숨기지 않고 HaluEval-QA only 보조 결과를 추가했다.
- Fusion-Energy 비교에 bootstrap CI와 다중비교 미보정 단서를 넣었다.
- free-sampling 순환성 한계를 한계 절에 추가했다.
- corpus support를 detector input이 아니라 평가 축으로만 쓴다는 점을 유지했다.

### 여전히 방어 시 조심할 점

- “환각 탐지 일반 성능”이라고 말하지 말 것.
- “Fusion이 Energy보다 항상 낫다/나쁘다”라고 말하지 말 것.
- TruthfulQA 결과를 독립적인 강한 근거처럼 말하지 말 것.
- corpus support를 feature로 넣어 성능 향상을 검증했다고 말하지 말 것.

---

## 5. PDF/시각 polish

### 현재 상태

핵심 표와 그림은 읽힌다. Figure 4.2는 Panel A/B 구조로 분리되어 좋고, Table 4.3 CI도 읽을 수 있다.

### 남은 문제

- `rank-quantile`, `self-conditioning`, 참고문헌 `Challenges` 등 soft-hyphen artifact가 남아 있다.
- 그림 목차에서 `그림4.1`, `그림4.2`처럼 공백이 빠져 보인다.
- 긴 caption이 일부 페이지의 밀도를 높인다.
- `0.889으로` 같은 조사 polish가 보인다. `0.889로`가 자연스럽다.

### 우선순위

1. soft-hyphen artifact 제거
2. 그림 목차 spacing 정리
3. 조사/띄어쓰기 polish
4. 긴 caption 일부 축소

---

## 6. 심사 방어용 핵심 답변

### Q. 이 논문은 환각 탐지 논문인가, proxy 분석 논문인가?

> 본 논문은 새로운 독립 환각 라벨을 구축하거나 SOTA detector를 제안하는 논문이 아니라,
> free-sampling 기반 `is_hard` proxy 하에서 기존 환각 탐지 신호들의 조건부 신뢰도 패턴을 분석한 평가 연구입니다.

### Q. `is_hard`와 SE/Energy가 같은 sampling에서 나오면 순환적이지 않나?

> 맞습니다. 그래서 본 논문은 이를 독립 라벨에 대한 일반 환각 탐지 성능으로 주장하지 않고,
> sampling 기반 difficulty proxy와 sample-consistency 신호 사이의 조건부 관계로 제한합니다.

### Q. Fusion-Energy 차이는 유의한가?

> prompt-level bootstrap CI 기준으로 양 끝 decile에서는 Fusion 우위가 관찰되지만,
> 중간 decile에서는 CI가 0을 포함합니다. 또한 이 표시는 다중비교 보정 전 탐색적 진단입니다.

### Q. TruthfulQA 결과는 믿을 수 있나?

> TruthfulQA는 paraphrase 정답이 많아 token-overlap proxy가 hard 쪽으로 치우칩니다.
> 따라서 주요 해석은 HaluEval-QA only 보조 분석과 함께 제한적으로 읽어야 합니다.

---

## 7. 마지막 체크리스트

### 반드시 고칠 것

- [ ] 표 4.2와 표 4.3의 Fusion-Energy delta 차이 설명 또는 통일
- [ ] §4.6의 “Fusion이 두 데이터셋 모두에서 1위” 문장 수정
- [ ] `experiment_method.tex`의 “정답 답” 수정
- [ ] “두 흐름” vs 세 항목 구조 수정
- [ ] “사후 판정 절차 없음” 표현을 후보 라벨과 `is_hard` proxy로 분리 설명

### 가능하면 고칠 것

- [ ] soft-hyphen artifact 제거
- [ ] `0.889으로` → `0.889로` 등 조사 polish
- [ ] 그림 목차 spacing 조정
- [ ] 긴 caption 축소

---

## 8. 최종 결론

최신본은 이제 **방어 가능한 학사논문**이다. 새 실험이 반드시 필요한 단계는 지났다.

남은 핵심은 세 가지다.

1. 표와 본문 수치 정합성
2. 라벨/평가 절차 설명의 정확성
3. PDF 최종 polish

이 세 가지만 정리하면 최종 제출본으로 매우 안정적이다.
