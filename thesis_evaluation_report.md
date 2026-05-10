# 졸업논문 심층 평가 보고서

> 평가 대상: `thesis/main.tex`, `thesis/main.pdf`, `thesis/sections/experiment_method.tex`,
> `thesis/results_macros.tex`, `thesis/thesis_evidence_table.tex`, 주요 표/그림 산출물
>
> 평가 기준: `evaluation.md`
>
> 평가 방식: 기준 평가자, 세부 기준 평가자, 방법론 평가자, PDF/시각 평가자,
> joker 평가자의 병렬 검토 결과를 종합

---

## 1. 종합 판정

현재 논문은 **학사 졸업논문으로는 충분히 통과 가능한 우수권 원고**이다. 연구 질문이 분명하고,
평균 AUROC 하나가 가리는 영역별 성능 변동을 보겠다는 문제의식이 논문 전체를 관통한다.
특히 “새 탐지기 제안”이 아니라 **기존 환각 탐지 신호의 조건부 신뢰도 지형을 corpus support 축에서 분해**한다는 포지셔닝은 학사논문 기준으로 성숙하다.

다만 제출 전 반드시 다듬어야 할 핵심 리스크가 있다.

1. `is_hard` 라벨이 실제 환각 여부가 아니라 **free-sample 정답 매칭률 기반 proxy**이다.
2. TruthfulQA에서 `is_hard` 비율이 0.97로 치우쳐, pooled 결과 해석을 약하게 만든다.
3. decile별 AUROC와 Fusion-Energy 차이에 **신뢰구간이나 검정이 없다**.
4. “단조 상승”, “Fusion이 항상 최선은 아니다”, “calibration도 1위” 같은 일부 문장이 증거보다 강하다.
5. PDF의 그림 4.2, 표/그림 목차, 영문 줄바꿈, Brier bold 처리 등 시각/형식 문제가 보인다.

### 최종 점수 해석

| 평가 관점 | 점수 | 해석 |
|---|---:|---|
| 전체 기준 평가자 | 86 / 100 | 학사논문 기준 우수, 통과 가능성 높음 |
| 세부 역할 평가자 합산 | 약 76 / 100 | 라벨·방법론·통계·서술 정합성을 엄격히 반영한 보수 점수 |
| 종합 조정 판정 | **80 / 100 내외** | 현재도 제출 가능권이나, P0 수정 후 85점 이상 가능 |

> 한 줄 판단: **아이디어와 프레이밍은 좋고, 실험도 성실하지만, 결론 문장이 증거보다 약간 앞서간다.**

---

## 2. 항목별 평가

| 항목 | 배점 | 종합 점수 | 평가 |
|---|---:|---:|---|
| 연구 주제와 문제 설정 | 20 | **17-18** | 세 연구 질문이 명확하고 범위도 학사논문에 적절하다. 다만 “환각 탐지”와 “어려운 질문 탐지”의 차이를 초반부터 더 분명히 해야 한다. |
| 선행연구 검토 | 15 | **12** | SE, Semantic Energy, SelfCheckGPT, CHOKE, QuCo-RAG, WildHallucinations, Valentin 등을 본 논문 위치와 연결한다. 그러나 문헌 간 긴장 관계와 비교표가 더 필요하다. |
| 방법론의 적절성 | 15 | **10-13** | 데이터, 모델, 신호, decile, fusion CV 설명은 충실하다. 그러나 `is_hard` proxy, TruthfulQA skew, corpus support 결합식 미명시, true LODO 부재가 큰 감점 요인이다. |
| 분석과 논의의 깊이 | 20 | **13-17** | RQ별 표/그림 대응은 좋다. Energy > SE 전 decile 결과는 강하다. 그러나 단조성, decile 역전, 난이도 집중 해석은 CI 없이 다소 강하다. |
| 논문 구성과 글쓰기 | 15 | **12-13** | 장 흐름은 자연스럽고 학술논문 구조를 갖췄다. 다만 한영 혼용, 긴 캡션, 일부 구어적 표현, PDF 줄바꿈 문제가 있다. |
| 인용, 참고문헌, 연구윤리 | 10 | **8** | 주요 출처는 대체로 잘 붙어 있고 한계도 명시한다. 다만 참고문헌 형식, arXiv 번호, 미사용 FActScore 처리, 데이터/윤리 문단 보강이 필요하다. |
| 발표 및 질의응답 readiness | 5 | **4-5** | 세 질문, 세 결과, 세 한계 구조가 발표에 유리하다. 예상 질문도 논문 내부에서 답할 수 있으나 라벨과 통계 질문에는 보수적으로 답해야 한다. |

---

## 3. 가장 강한 장점

### 3.1 연구 질문이 선명하다

논문은 다음 세 질문을 중심으로 잘 조직되어 있다.

1. corpus support decile에 따라 탐지 신호의 AUROC가 어떻게 달라지는가?
2. 단일 신호 간 우열은 corpus support 영역에 따라 달라지는가?
3. Fusion은 모든 영역에서 단일 신호보다 우수한가?

이 구조는 `main.tex`의 서론, 실험 장, 결론에서 반복 회수된다. 학사논문에서 흔히 나타나는 “주제는 크지만 결론은 흐린” 문제가 상대적으로 적다.

### 3.2 “평균 metric의 한계”라는 문제의식이 좋다

기존 연구가 평균 AUROC 하나로 신호나 fusion의 우월성을 보고하는 관행을 비판하고,
이를 corpus support decile별로 분해한다는 접근은 설득력 있다.

### 3.3 Energy가 SE를 모든 decile에서 이긴다는 결과는 강하다

표 4.2에서 Energy가 10개 decile 모두에서 SE보다 높다. 이 결과는 논문 내 핵심 주장 중 가장 방어 가능하다.

### 3.4 한계를 숨기지 않는다

TruthfulQA 라벨 노이즈, 단일 corpus snapshot, 단일 모델, 인과 주장 불가, decile별 유의성 미검정 등을 명시한 점은 좋다. 다만 이 한계가 초록과 결론의 주장 수위에도 더 강하게 반영되어야 한다.

---

## 4. 핵심 리스크

## 4.1 `is_hard` 라벨은 환각 라벨이 아니다

현재 라벨은 다음과 같이 정의된다.

> free-sample N=10의 정답 매칭 비율이 0.5 미만이면 `is_hard=1`

이 라벨은 실제 “환각 여부”라기보다 **모델이 정답 문자열 또는 정답 표현을 안정적으로 생성하지 못하는 정도**에 가깝다.

문제는 SE/Energy도 같은 free-sampling 행동에서 파생된다는 점이다. 따라서 예측 신호와 타깃이 동일한 생성행동을 공유하는 내생성 위험이 있다.

### 안전한 표현

현재 원고의 일부 표현:

> corpus 부족 영역에서 환각 탐지 자체가 더 어렵다

권장 표현:

> 현재 `is_hard` proxy 기준에서 sample-consistency 계열 신호의 판별력이 corpus 부족 영역에서 낮아진다.

---

## 4.2 TruthfulQA skew가 매우 크다

본문은 TruthfulQA의 `is_hard` 비율이 0.97이라고 인정한다. 이 정도면 단순 노이즈가 아니라 **라벨 체계와 데이터셋 특성이 구조적으로 충돌**하는 수준이다.

따라서 다음 조정이 필요하다.

- HaluEval-QA를 주 분석으로 승격한다.
- AGG와 TruthfulQA 결과는 보조 결과로 둔다.
- “TruthfulQA 결과는 token-overlap proxy의 한계를 보여주는 사례”로 재배치한다.

---

## 4.3 “단조 상승”은 과하다

그림 4.1과 표 4.2는 high support 영역에서 AUROC가 높아지는 **전반적 경향**을 보여준다. 그러나 decile 값이 모든 구간에서 엄밀하게 증가하지는 않는다.

권장 수정:

- “단조 상승” → “전반적 상승 경향”
- “corpus 부족 영역에 난이도가 집중된다” → “low support 영역에서 상대적으로 낮은 AUROC가 관찰된다”

---

## 4.4 Fusion 주장은 재미있지만 가장 취약하다

“Fusion이 항상 최선은 아니다”는 흥미로운 문장이지만, 핵심 근거는 decile 40-50에서 Energy가 Fusion보다 0.014 높다는 점추정치이다. 신뢰구간이 없으므로 강하게 주장하면 방어가 어렵다.

권장 표현:

> Fusion의 평균 우위는 corpus 영역별로 균일하지 않으며, 일부 중간 support 영역에서는 Energy 단독과 동률 또는 열세에 가까워진다.

---

## 4.5 corpus support 정의 단위가 흔들린다

본문 일부는 “질문에 등장하는 entity”라고 읽히고, 방법 절은 “후보 답변에 등장하는 entity” 기반으로 읽힌다. 질문 entity support와 후보 답변 entity support는 의미가 다르다.

반드시 다음 중 하나로 통일해야 한다.

1. 질문 entity support
2. 후보 답변 entity support
3. 질문과 후보 답변 entity를 결합한 support

그리고 실험 방법 절에 coverage score 결합식을 명시해야 한다.

---

## 4.6 Brier / calibration 서술 오류

`thesis_evidence_table.tex`와 표 4.3 계열 수치상:

- Random Forest Brier = 0.112
- Gradient Boosting Brier = 0.113

Brier는 낮을수록 좋다. 그런데 gradient boosting 행이 bold 처리되어 있고, 본문은 “calibration도 1위”라고 읽힌다.

권장 수정:

- ECE는 GBM이 가장 낮다.
- Brier는 RF가 근소 우위이다.
- “calibration도 1위” → “ECE 기준 가장 낮고, Brier도 RF와 근접한 수준”

---

## 5. 방법론 보강 권장

### P0: 반드시 수정

1. **`is_hard`의 의미를 초록, 서론, 실험 설정에 더 명확히 쓰기**
   - 환각 라벨이 아니라 hard question proxy임을 앞에서 고정한다.
2. **TruthfulQA/AGG headline 낮추기**
   - HaluEval-QA를 주 결과로, TruthfulQA는 라벨 proxy 한계 사례로 둔다.
3. **Brier bold와 calibration 문장 정정**
4. **“단조”, “항상 최선은 아니다”, “환각 탐지 자체” 같은 강한 표현 완화**
5. **corpus support 정의 단위 통일**

### P1: 가능하면 꼭 추가

1. **decile별 AUROC 95% bootstrap CI**
2. **Fusion-Energy delta의 decile별 CI**
3. **decile별 `is_hard` prevalence**
4. **decile별 AUPRC**
5. **HaluEval-only decile 분석**

### P2: 있으면 논문이 훨씬 단단해짐

1. TruthfulQA 일부 샘플에 대한 수작업 또는 NLI 기반 정답 매칭 검증
2. corpus support coverage score의 수식과 재현 절차 추가
3. fusion GBM의 하이퍼파라미터, 튜닝 여부, fold stratification 명시
4. 관련 연구 비교표 추가

---

## 6. 시각 자료와 PDF 평가

전체 PDF는 26쪽 규모이며, 표 4개와 그림 2개가 들어간다. 기본 조판은 안정적이고 표는 대체로 읽기 쉽다. 그러나 몇 가지 시각적 문제는 제출 전 수정하는 것이 좋다.

### 6.1 표

| 표 | 평가 |
|---|---|
| 표 2.1 | 단일 신호 비교 표. 간결하고 읽기 좋음. |
| 표 4.1 | 종합 baseline 표. 열 구성은 좋지만 Brier bold 기준 오류 가능성이 있음. |
| 표 4.2 | decile별 AUROC 표. 핵심 결과 전달력이 좋음. CI가 없다는 점은 한계. |
| 표 4.3 | dataset/calibration 표. 유용하지만 “generalization”처럼 읽히지 않게 주의 필요. |

### 6.2 그림

| 그림 | 평가 |
|---|---|
| 그림 4.1 | Per-decile AUROC 경향을 잘 보여준다. 다만 “단조” 표현은 완화해야 한다. |
| 그림 4.2 | Fusion delta 핵심 그림이지만, 붉은 주석과 x축 라벨이 겹치고 scale 때문에 Energy 비교가 묻힌다. |

### 6.3 그림 4.2 개선안

현재 그림 4.2는 `vs logit-diagnostic` 막대가 너무 커서 `vs Energy`의 작은 차이가 잘 보이지 않는다.

권장 재구성:

- Panel A: Fusion - SE, Fusion - Energy
- Panel B: Fusion - logit-diagnostic
- decile 40-50 음수 막대에는 `-0.014` 직접 라벨 표시
- 붉은 주석은 그래프 밖이나 캡션으로 이동

---

## 7. 글쓰기와 형식 수정

### 7.1 문체

다음 표현은 학술문체로 다듬는 것이 좋다.

| 현재 표현 | 권장 표현 |
|---|---|
| 세 가지 발견. | 본 논문의 주요 발견은 세 가지이다. |
| 좋은 케이스 | 우수한 경우 |
| hard 쪽으로 치우친다 | 어려운 질문으로 과다 분류된다 |
| baseline | 기준 성능 또는 baseline, 첫 등장 시 정의 |
| proxy | 대리 지표 또는 proxy, 첫 등장 시 정의 |

### 7.2 용어 통일

다음 용어는 표준형을 정하고 끝까지 유지해야 한다.

- `fusion` / `Fusion`
- `logit diagnostic` / `logit-diagnostic`
- `corpus statistic` / `corpus support` / `corpus signal` / `corpus-axis`
- `hard question` / `어려운 질문`

### 7.3 PDF 조판

확인된 문제:

- `variance`, `WildHallucinations`, `self-conditioning`, `co-occurrence`, `rank-quantile` 등 영문 단어 줄바꿈 아티팩트
- 표 목차에서 “미만인경우어려운질문”, “단일신호는”처럼 붙어 보이는 문제
- 그림 목차에서 `그림4.1`처럼 간격이 어색한 문제
- 긴 캡션이 표목차에 그대로 들어가 목록이 과밀해지는 문제

권장 조치:

- 표에도 optional short caption 사용: `\caption[짧은 표 제목]{긴 설명}`
- 핵심 고유명사는 `\mbox{WildHallucinations}` 등으로 줄바꿈 방지
- `tocloft`의 figure/table number spacing 조정

---

## 8. 참고문헌과 연구윤리

### 좋은 점

- 주요 방법과 데이터셋에 출처가 붙어 있다.
- LLM-as-judge를 배제한 점을 명시한다.
- corpus support를 라벨이나 detector input으로 쓰지 않는다고 밝힌다.
- 상관과 인과를 구분한다.

### 보완할 점

1. arXiv 문헌의 번호와 출판 상태를 정리한다.
2. `[5] FActScore`는 본문에서 정식 인용하거나 참고문헌에서 삭제한다.
3. 데이터셋 사용, 개인정보/인간대상 연구 해당 없음, 재현 가능한 코드/산출물 위치를 짧은 연구윤리/재현성 문단으로 추가한다.
4. 참고문헌 형식을 통일한다.

---

## 9. 예상 심사 질문과 답변 방향

### Q1. 이 논문은 환각 탐지 논문인가, 어려운 질문 탐지 논문인가?

안전한 답변:

> 본 논문은 실제 환각 여부를 직접 새로 라벨링한 논문이라기보다, free-sample 정답 매칭률로 정의한 `is_hard` proxy를 기준으로 기존 환각 탐지 신호의 조건부 판별력을 분석한 평가 연구입니다. 따라서 결론은 `is_hard` proxy 기준의 조건부 성능 패턴으로 제한됩니다.

### Q2. 왜 corpus support를 detector feature로 넣지 않았나?

안전한 답변:

> corpus statistic을 feature로 넣으면 평가 축과 입력 feature가 같은 source에서 나와 self-conditioning artifact가 생길 수 있습니다. 본 논문은 직접 성능 향상보다, 기존 신호의 신뢰도가 입력 조건에 따라 어떻게 변하는지 평가하는 데 목적이 있습니다.

### Q3. Fusion이 Energy보다 못한 decile이 정말 의미 있나?

안전한 답변:

> 현재 원고에서는 점추정치만 제시했기 때문에 통계적으로 유의한 역전이라고 단정하지 않습니다. 핵심 결론은 Fusion의 평균 우위가 영역별로 균일하지 않고, 일부 영역에서는 Energy와 거의 동률 또는 열세에 가까워진다는 점입니다.

### Q4. TruthfulQA `is_hard=0.97`이면 결과가 무너지는 것 아닌가?

안전한 답변:

> TruthfulQA는 paraphrase 정답이 많아 token-overlap proxy가 부적절하다는 한계를 보여줍니다. 따라서 주요 해석은 HaluEval-QA 중심으로 제한하고, TruthfulQA는 proxy 라벨의 한계를 드러내는 보조 결과로 다루는 것이 타당합니다.

---

## 10. 최종 수정 체크리스트

### 제출 전 필수

- [ ] `is_hard`가 환각 라벨이 아니라 proxy임을 초록/서론/실험 설정에 명확히 쓰기
- [ ] “단조 상승”을 “전반적 상승 경향”으로 수정
- [ ] “Fusion이 항상 최선은 아니다”를 “Fusion 우위 폭이 균일하지 않다”로 완화
- [ ] Brier bold와 “calibration 1위” 문장 수정
- [ ] `§4.6`의 generalization / leave-one-dataset-out 뉘앙스 제거
- [ ] corpus support가 질문 기준인지 후보 답변 기준인지 통일
- [ ] 그림 4.2 주석 겹침과 scale 문제 수정
- [ ] 표 optional short caption 추가
- [ ] 영어 줄바꿈 아티팩트와 목차 spacing 수정
- [ ] 참고문헌 형식과 arXiv 번호 정리

### 가능하면 추가

- [ ] decile별 AUROC CI 추가
- [ ] Fusion-Energy delta CI 추가
- [ ] decile별 `is_hard` prevalence 추가
- [ ] HaluEval-only decile 분석 추가
- [ ] 관련 연구 비교표 추가

---

## 11. 최종 결론

이 논문은 **틀린 논문이 아니라, 좋은 관찰을 가지고 있지만 결론 수위를 조금 낮춰야 하는 논문**이다.

현재 상태에서도 학사 졸업논문으로는 충분히 설득력이 있다. 다만 심사에서 공격받을 지점은 명확하다.

- `is_hard` 라벨이 실제 환각 라벨인가?
- TruthfulQA 왜곡을 어떻게 처리할 것인가?
- decile 차이에 신뢰구간이 있는가?
- Fusion이 정말 Energy보다 항상 낫지 않은가?
- calibration 1위라는 말이 표와 일치하는가?

이 다섯 가지를 정리하면, 논문은 훨씬 단단해진다. 가장 안전한 최종 논지는 다음과 같다.

> 본 논문은 Qwen2.5-3B와 두 QA 데이터셋, token-overlap 기반 `is_hard` proxy 조건에서,
> sample-consistency 계열 신호의 AUROC가 corpus support 축에 따라 크게 달라지고,
> Energy는 SE보다 안정적으로 강하며,
> Fusion의 평균 이득은 영역별로 균일하지 않음을 보였다.

이렇게 말하면 방어 가능하다. 이보다 강하게 말하면 심사에서 공격받을 가능성이 크다.
