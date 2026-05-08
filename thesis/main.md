# 서론

<span id="ch:intro" label="ch:intro"></span>

## 연구 배경

대규모 언어모델(Large Language Model, LLM)은 텍스트 생성, 질의응답, 요약, 번역 등 다양한 자연어 처리 태스크에서 널리 사용되고 있다. 그러나 최신 LLM도 사실과 다른 내용을 그럴듯하게 생성하는 환각(hallucination) 문제를 보이며, 이는 LLM 신뢰성과 안전성의 핵심 과제로 논의되어 왔다. 생성 텍스트의 신뢰성은 사실성(factuality)과 충실성(faithfulness) 관점에서 구분해 다룰 수 있으며, 본 연구는 외부 사실과 충돌하거나 검증 가능한 정답과 맞지 않는 응답을 탐지하는 문제에 초점을 둔다.

환각 문제는 LLM의 실용적 적용에서 중요한 장애물이다. 의료 분야에서 잘못된 진단 정보를 제공하거나, 법률 분야에서 존재하지 않는 판례를 인용하거나, 금융 분야에서 부정확한 수치를 제시하는 경우 큰 피해를 초래할 수 있다. TruthfulQA는 모델이 인간이 흔히 믿는 거짓 정보를 얼마나 그럴듯하게 모방하는지 평가하기 위해 제안된 벤치마크이며, 이러한 설정은 LLM 응답을 사후에 검증하고 환각을 자동으로 탐지하는 방법의 필요성을 보여준다.

환각 탐지를 위한 대표적인 불확실성 기반 방법으로 Semantic Entropy(SE)가 있다. SE는 하나의 질문에 대해 여러 응답을 샘플링한 뒤, 자연어 추론(Natural Language Inference, NLI) 모델을 사용하여 의미적으로 클러스터링하고, 클러스터 분포의 엔트로피를 계산한다. 반복 샘플링과 응답 간 일관성은 SelfCheckGPT와 같은 선행 연구에서도 환각 탐지를 위한 중요한 단서로 사용되었다. SE가 높으면 모델 응답이 여러 의미로 갈라지므로 의미적 불확실성이 크다는 신호가 된다.

그러나 SE에는 알려진 한계가 있다. SE는 응답 간 의미적 다양성에 의존하므로, 모델이 모든 응답에서 같은 의미의 틀린 답변을 반복하면 판별력이 약해진다. 같은 질문을 여러 번 샘플링했는데 모든 응답이 동일한 잘못된 인물명이나 수치로 수렴한다면, 응답들은 하나의 클러스터에 모이고 SE는 0에 가까워진다. 이런 *고확신 환각(high-certainty hallucination)* 사례에서는 SE만으로는 정답 여부를 가리기 어렵다. Simhi 등(2025)은 이러한 사례를 CHOKE라고 부르고, 모델이 정답 지식을 가지고도 높은 확신으로 환각을 만들 수 있음을 실증하였다.

이 한계는 SE만의 문제가 아니다. 후보 답변의 평균 음의 로그 확률(NLL), confidence margin 같은 *후보 단위 logit 진단* 신호도 “환각 답이 정답보다 더 자신없게 평가될 것”이라는 직관에 기댄다. 그러나 환각 답이 오히려 더 자신있게 평가되는 사례가 흔하다면, 이런 신호는 단일 점수로 전 영역에서 잘 작동한다고 보기 어렵다.

본 연구는 이런 한계를 *외부 corpus 신호* 로 조건화하여 해결한다. 후보 답의 entity frequency 와 entity-pair co-occurrence 를 외부 corpus 에서 측정한 뒤, 각 *corpus support 구간* 에서 어떤 신호가 안정적으로 작동하는지 분리해 평가한다. corpus 빈도는 환각 라벨도 아니고 retrieval trigger도 아니며, 어디까지나 *지표 신뢰성을 나누는 외부 조건 변수*이다.

## 연구 목적

본 연구는 LLM 환각 탐지 신호의 신뢰성을 외부 corpus 조건 위에서 평가한다는 관점에서 다음 세 가지 질문을 다룬다.

1.  **Corpus support에 따른 신호 신뢰성 변화 측정**. SE, Semantic Energy, logit diagnostic이 corpus support 구간에 따라 AUROC, AUPRC, paired win-rate에서 어떻게 달라지는지 측정한다.

2.  **Conditional fusion의 효용 검증**. corpus 신호를 global fusion에 그대로 더하는 방식과, corpus support bin마다 별도 가중치를 학습하는 conditional fusion 중 어느 쪽이 더 우수한지 prompt-grouped bootstrap으로 통계적으로 검정한다.

3.  **CHOKE 패턴의 외부 신호 분리**. Simhi 등(2025)이 보고한 high-certainty hallucination 패턴이 candidate-level logit 신호뿐 아니라 외부 corpus 신호(특히 entity-pair co-occurrence)에서도 동시에 관측되는지 확인한다.

## 논문 구성

본 논문은 다음과 같이 구성된다. 제 <a href="#ch:related" data-reference-type="ref" data-reference="ch:related">[ch:related]</a>장에서는 LLM 환각 탐지 관련 연구와 본 연구의 위치를 정리한다. 제 <a href="#ch:method" data-reference-type="ref" data-reference="ch:method">[ch:method]</a>장에서는 단일 전역 점수 가정의 한계, corpus support를 조건 축으로 쓰는 근거, 평가 단위 분리 원칙, 그리고 본 논문이 제안하는 corpus-conditioned reliability 분석 절차를 제시한다. 제 <a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>장에서는 baseline 결과, prompt-grouped bootstrap 신뢰구간, corpus support bin별 신뢰성, decile 정밀 분해, CHOKE 패턴 재현, leave-one-dataset-out, threshold/calibration 진단을 보고한다. 제 <a href="#ch:conclusion" data-reference-type="ref" data-reference="ch:conclusion">[ch:conclusion]</a>장에서 결론과 향후 연구 방향을 제시한다.

# 관련 연구

<span id="ch:related" label="ch:related"></span>

본 장에서는 LLM 환각 연구를 두 흐름으로 구분해 검토한다. 첫째는 환각의 정의, 사실성 평가, 벤치마크를 다루는 연구이며, 둘째는 생성 모델의 불확실성 또는 내부 점수를 이용해 환각을 탐지하려는 방법 연구이다. 이 구분은 본 연구가 기존 사실성 평가 자체를 대체하려는 것이 아니라, 반복 샘플링 기반 SE의 한계가 드러나는 영역에서 보조 탐지 신호와 결합 구조를 검토한다는 점을 명확히 한다.

## 환각 분류, 사실성, 벤치마크

환각은 LLM이 입력, 문맥, 또는 외부 사실과 맞지 않는 내용을 생성하는 현상으로 다루어진다. Huang 등은 LLM 환각의 원인, 유형, 탐지 과제를 폭넓게 정리하며, 환각 탐지가 LLM 신뢰성 확보를 위한 핵심 문제임을 보인다. Maynez 등은 생성 텍스트 평가에서 사실성(factuality)과 충실성(faithfulness)을 구분해 논의했으며, 이 구분은 환각 탐지 문장을 작성할 때 단순한 “틀림”보다 어떤 기준에서 맞지 않는지 명확히 해야 함을 보여준다.

벤치마크와 평가 연구는 환각 탐지의 대상과 평가 기준을 제공한다. TruthfulQA는 모델이 인간의 오해나 거짓 믿음을 모방하는지를 측정하는 질의응답 벤치마크이다. FActScore는 장문 생성에서 원자적 사실 단위의 정밀도를 평가하는 방법을 제안한다. 이러한 연구들은 환각을 평가하고 분류하는 기준을 제공하지만, 본 연구가 다루는 “*어느 corpus support 구간에서 어느 신호의 신뢰도가 높은가*”라는 조건부 신뢰성 문제는 직접 다루지 않는다.

## 불확실성 및 반복 샘플링 기반 탐지

SelfCheckGPT는 외부 지식 베이스 없이 여러 응답을 샘플링하고 응답 간 일관성을 비교해 환각을 탐지하는 접근을 제안했다. 이 흐름은 하나의 응답만 보는 대신, 같은 질문에 대한 여러 생성 결과의 분산과 일관성을 환각 탐지 신호로 본다. Semantic Entropy도 이 관점과 연결되지만, 응답을 의미 클러스터로 묶고 클러스터 분포의 엔트로피를 사용한다는 점에서 더 명시적인 의미 단위 불확실성 측정 방법이다.

표 <a href="#tab:method_comparison" data-reference-type="ref" data-reference="tab:method_comparison">2.1</a>은 본 연구에서 다루는 신호들의 정의 단위와 한계를 요약한다.

<div id="tab:method_comparison">

| 신호 | 정의 단위 | 핵심 아이디어 | 한계 |
|:---|:---|:---|:---|
| Semantic Entropy | prompt-level | NLI cluster + Shannon entropy | 단일 cluster 사례 판별 약함 |
| Semantic Energy | cluster-level | cluster 가중 token logit 합 | cluster 정의 의존, 다양성 가정 |
| logit diagnostic | candidate-level | NLL, logit variance, margin | CHOKE 사례에서 방향 역전 |
| corpus support | candidate-level | entity frequency / pair co-occurrence | 외부 corpus snapshot 의존, proxy 성격 |

본 연구에서 비교하는 환각 탐지 신호.

</div>

## Semantic Entropy (SE)

Semantic Entropy는 Farquhar 등이 2024년 Nature에 발표한 LLM 불확실성 측정 방법이다. 기존의 토큰 단위 확률 기반 불확실성 측정 방법과 달리, SE는 응답의 의미적 내용을 기반으로 불확실성을 측정한다.

### 작동 원리

SE의 계산 과정은 다음과 같다:

1.  **응답 샘플링**: 하나의 질문 $`q`$에 대해 LLM으로부터 $`K`$개의 응답 $`\{r_1, r_2, ..., r_K\}`$를 샘플링한다.

2.  **의미적 클러스터링**: NLI 모델을 사용하여 응답들을 의미적으로 클러스터링한다. 두 응답이 서로 entailment 관계에 있으면 같은 클러스터로 분류한다.

3.  **엔트로피 계산**: 클러스터 분포의 Shannon entropy를 계산한다.

수식으로 표현하면:
``` math
\begin{equation}
    SE = -\sum_{c \in C} p(c) \log p(c)
\end{equation}
```
여기서 $`C`$는 의미 클러스터 집합이고, $`p(c)`$는 클러스터 $`c`$에 속하는 응답의 비율이다.

### 한계점

SE 자체의 주요 한계점은 다음과 같다.

- **Hard clustering**: 응답을 이진적(동일/다름)으로 묶어 세밀한 유사도를 반영하지 못한다.

- **단일 cluster 사례에서의 판별력 한계**: 모든 응답이 하나의 cluster로 수렴하면 SE 값이 작아져 prompt 단위 순위 정보가 줄어든다. 이런 사례가 곧 정답을 의미하지는 않지만, SE만으로는 환각/정답을 가리기 어렵다.

- **응답 길이와 클러스터링 의존성**: 긴 응답이나 복잡한 답변에서는 NLI 기반 클러스터링 품질에 따라 결과가 달라질 수 있다.

한편 SE는 정의상 prompt 단위 값이므로, 본 연구가 사용하는 paired 평가 설정에서 같은 prompt의 두 candidate row(정답/환각)에 동일 값이 복사된다. 이는 SE 자체의 한계라기보다 *paired candidate-row 평가에 prompt-level 신호를 적용한 구조적 결과*이며, 본 논문은 이 점을 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>에서 분리해 다룬다.

## Semantic Energy

Ma 등(2025)이 제안한 Semantic Energy는 여러 generated answer와 semantic cluster를 사용해 토큰 logit 기반 uncertainty를 해석하려는 방법이다. 현재 문헌 상태에서는 기존 방법 맥락과 구현 목표로만 사용하며, 본 연구의 핵심 주장은 아직 검증되지 않은 Energy 성능이 아니라 corpus 조건 축에 따라 hallucination metric의 신뢰도를 평가해야 한다는 문제 설정에 둔다.

### 작동 원리

Energy는 LLM이 각 토큰을 생성할 때 부여한 **softmax 전 raw logit** 값을 사용한다:
``` math
\begin{equation}
    Energy = \frac{1}{nT} \sum_{i=1}^{n} \sum_{t=1}^{T_i} -z_\theta(x_t^{(i)})
\end{equation}
```
여기서 $`z_\theta(x_t)`$는 토큰 $`x_t`$의 logit 값이다.

### SE 대비 차별점

<div id="tab:se_vs_energy">

| 항목      | SE              | Semantic Energy            |
|:----------|:----------------|:---------------------------|
| 사용 값   | softmax 후 확률 | softmax 전 logit           |
| 측정 대상 | 응답 간 다양성  | cluster 단위 토큰 logit 합 |
| 정보 손실 | 정규화로 손실   | logit 크기 보존            |
| 정의 단위 | prompt-level    | cluster-level              |

SE와 Semantic Energy의 비교.

</div>

### Energy 신호의 motivation과 본 연구의 평가

Energy의 문헌적 motivation은 응답 다양성이 작은 사례에서도 토큰 단위 점수가 추가 정보를 줄 수 있다는 것이다. 다만 본 연구의 결과(§<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>)는 후보 단위 logit 진단(NLL, confidence margin)이 candidate-row 평가에서 *역방향*(0.5 미만, 환각 답이 더 자신있게 평가되는 CHOKE 패턴)으로 작동함을 보인다. 이는 Energy 계열 신호를 단일 전역 점수로 쓸 수 없음을 뜻하며, 본 연구는 같은 신호를 corpus support 구간별로 분리 평가해 어떤 구간에서 어떤 신호가 안정적으로 작동하는지 살핀다.

## QuCo-RAG: 외부 corpus 기반 신호

QuCo-RAG는 Qiu 등(2025)이 retrieval-augmented generation(RAG) 의 query-aware corpus grounding 절차에서 사용한 외부 corpus statistic 계열 신호이다. 핵심 아이디어는 입력 query 나 후보 답변에 등장하는 entity 의 corpus 내 frequency 와 entity-pair co-occurrence 를 corpus count 로 측정해, 그 답변이 외부 자료로 얼마나 “뒷받침될 수 있는지” 를 정량화하는 것이다.

QuCo-RAG의 원래 사용 맥락은 retrieval grounding이지만, 같은 corpus statistic은 다음 두 측면에서 본 연구와 직접 닿는다.

- **모델 출력에 의존하지 않는 외부 신호**. corpus 카운트는 모델의 logit, hidden state, 응답 다양성 등과 독립적으로 산출된다. 따라서 모델 신호의 신뢰도를 *직교적으로* 조건화하는 데 쓸 수 있다.

- **모델 “익숙함” 의 외부 측정**. entity 의 corpus frequency 는 모델이 사전 학습 단계에서 그 entity 를 얼마나 마주쳤을지에 대한 *간접 지표* 이다. 다만 이는 사전 학습 노출의 직접 증거나 hidden-state 신뢰도와 동일한 것이 아니므로 어디까지나 proxy 로 다뤄야 한다.

본 연구는 QuCo-RAG의 통계 자체를 그대로 사용하되, 그 *용도*를 retrieval trigger가 아니라 *지표 신뢰도를 나누는 외부 조건 변수*로 재해석한다. 즉 corpus 빈도가 작다고 환각 라벨로 쓰지 않고, 빈도가 크다고 정답 라벨로 쓰지 않으며, 어디까지나 *어떤 corpus support 구간에서 어떤 신호가 더 안정적으로 작동하는지*를 평가하기 위한 binning 변수로 사용한다. 이 재해석은 corpus-grounded 신호와 모델-내부 hidden-state 신호를 명확히 구분하는 본 연구의 guardrail과 일치한다.

## 기존 연구의 한계 및 연구 공백

기존 연구들의 한계를 본 연구의 문제 설정 관점에서 정리하면 다음과 같다.

1.  **환각 분류 및 평가 연구**. 환각의 정의, 사실성 평가, 벤치마크를 제공하지만, “*어느 입력 조건에서 어느 탐지 신호의 신뢰도가 높은가*” 라는 조건부 신뢰성 문제는 직접 다루지 않는다.

2.  **SE 계열 방법**. 반복 샘플링과 의미 cluster 분포를 활용하지만, 모든 응답이 하나의 cluster로 수렴하는 사례에서는 prompt 단위 정보가 줄어든다.

3.  **Energy / logit 계열 신호**. 토큰 단위 확신도 정보를 제공하지만, “환각 답이 정답보다 더 자신없을 것”이라는 직관에 기댄다. 본 데이터에서는 환각 답이 오히려 더 자신있게 평가되는 CHOKE 사례가 흔해서, 단일 전역 점수로 쓰기 어렵다.

4.  **외부 조건에 따른 결합 구조의 필요성**. 단일 글로벌 점수가 모든 데이터에서 가장 좋다는 가정 대신, 외부 조건 축에 따라 신호 신뢰성을 분리 평가하는 구조가 필요하다. 본 연구는 그 외부 조건 축으로 corpus support를 사용한다.

##### Pretraining corpus coverage 기반 직접 탐지 (Zhang 등 2025).

Zhang 등(2025)은 본 연구와 독립적으로 유사한 직관에서 출발한 연구를 수행하였다. 이들은 RedPajama 1.3조 토큰 pretraining corpus 위에 suffix array 를 구축하고, prompt 와 생성 답변의 n-gram 빈도($`1 \le n \le 5`$)를 환각 탐지 신호로 평가하였다. 핵심 결과는 “occurrence 기반 feature 는 log-probability 보다 약한 단독 신호이지만 함께 쓸 경우 특정 artifact 패턴 탐지에 보완적 가치가 있다”는 것이다. 본 연구와의 차별점은 세 층위에 있다.

- **신호의 역할**. Zhang 등은 corpus n-gram 통계를 환각 탐지 신호 *자체* 로 평가하는 반면, 본 연구는 corpus 빈도를 탐지 신호로 사용하지 않고 *어느 corpus support 구간에서 어느 신호의 신뢰도가 높은가* 를 나누는 조건 축으로 재해석한다. corpus 빈도가 낮다고 환각 라벨로, 높다고 정답 라벨로 쓰지 않는다.

- **신호의 단위**. Zhang 등은 token surface 수준의 n-gram lexical coverage 를 측정하나, 본 연구는 entity frequency 와 entity-pair co-occurrence 라는 *관계 수준* 의 신호를 사용한다. 동일 entity 가 등장하더라도 entity 간 관계가 corpus 에 얼마나 공출현하는지를 측정함으로써, “익숙한 entity 를 잘못된 관계로 조합하는” CHOKE 패턴과 직접 연결된다.

- **평가 패러다임**. Zhang 등은 단일 생성 답변의 이진 분류로 평가하는 반면, 본 연구는 같은 prompt 의 정답 후보와 환각 후보를 paired 로 비교하는 matched-pair 설정을 사용한다. 이 설정은 CHOKE 패턴(환각 답이 정답 답보다 더 높은 확신으로 평가됨) 을 paired win-rate 로 직접 검출할 수 있게 한다.

Zhang 등이 명시한 “결과는 상관관계이며 인과 관계를 확립하지 않는다”는 caveat 은 본 연구의 corpus-level 해석에도 동일하게 적용된다 — corpus 카운트 backend 의 외부 corpus snapshot 이 모델의 실제 pretraining corpus 와 다를 수 있으므로, corpus support 와 환각의 연결은 *상관* 이지 *인과* 가 아니다.

##### High-certainty hallucination (CHOKE).

Simhi 등(2025)은 모델이 정답 지식을 가지고도 높은 확신으로 환각을 생성하는 사례를 CHOKE로 정의하고, 이런 사례가 logit-only / sample-consistency 기반 detector의 본질적 한계임을 보였다. 본 연구는 Simhi가 제안한 hidden-state probing 대신, corpus support를 조건 축으로 사용해 같은 현상을 외부 신호로 분리해 낼 수 있는지 검증한다. 본 실험에서 CHOKE 패턴은 candidate-level(NLL/margin의 방향 역전)과 corpus-level(entity-pair co-occurrence의 양의 paired win-rate) 두 직교 scope에서 동시에 재현되며, 자세한 결과는 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>에서 보고한다.

##### Corpus 기반 환각 경험 증거 (WildHallucinations).

Zhao 등(2024)은 “*Wikipedia 페이지가 없는 entity*에 대한 질문에서 LLM이 더 자주 환각한다”는 관찰을 보고하였다. 이는 본 연구가 corpus 빈도를 조건 축으로 쓰는 가설의 *선행 경험적 근거*이다. 차이점은 두 가지다. (i) WildHallucinations 는 Wikipedia 존재 여부 라는 *이진* 조건을 사용하는 반면, 본 연구는 entity frequency / pair co-occurrence 기반 *연속 corpus support 축*으로 일반화한다. (ii) WildHallucinations 는 환각률 자체를 corpus 조건으로 설명하는 *evaluation* 연구이고, 본 연구는 이미 환각이 일어난 후보 행에 대해 *어떤 탐지 신호가 corpus 조건에 따라 신뢰로운가*를 평가하는 *detection* 연구이다.

##### Conditional calibration framework (Valentin et al.).

Valentin 등(2024)은 환각 탐지 점수를 *입력/응답 attribute에 conditional 하게 calibrate* 하는 multi-scoring framework를 제안하였다. 본 연구와의 핵심 차이는 *조건의 출처*이다. Valentin은 모델 *내부 score attribute* 에 conditional하게 calibrate 하는 반면, 본 연구는 모델 출력과 독립적인 *외부 corpus statistic* 으로 조건화한다. 두 framework는 직교적이며, 본 연구의 corpus 조건 축은 Valentin의 internal-attribute conditioning과 결합 가능한 보완 신호로 위치할 수 있다.

# 제안 방법

<span id="ch:method" label="ch:method"></span>

본 장에서는 본 논문이 제안하는 *corpus-conditioned reliability 분석 절차*와 그 이론적 동기를 정리한다. 제 <a href="#ch:related" data-reference-type="ref" data-reference="ch:related">[ch:related]</a>장에서 정리한 SE / Semantic Energy / corpus 신호의 한계가 출발점이고, 결과 보고 형식 (reliability map) 까지가 결과 chapter (§<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>) 와 자연스럽게 연결되도록 절차를 정의한다.

## 단일 전역 점수 가정의 한계

기존 환각 탐지 연구는 SE, Semantic Energy, logit-기반 점수 등을 *전역 단일 점수*로 사용해 한 번에 모든 데이터에 대해 환각을 가려 내려고 한다. 이 가정은 두 가지 사례에서 깨진다.

- **단일 cluster 사례**. 모든 N개의 응답이 의미적으로 같은 답으로 수렴하면 SE는 작아져 prompt 단위 정보가 줄어든다. 이때 같은 prompt의 정답 후보와 환각 후보 사이 순위는 SE만으로는 결정할 수 없다.

- **고확신 환각 (CHOKE) 사례**. Simhi 등(2025)이 보고한 대로, 환각 답이 정답 답보다 모델 입장에서 *더 자신있게* 평가되는 사례가 흔하다. NLL이 작거나 confidence margin이 큰 사례를 “정답일 것”이라고 가정하는 logit-based detector는 이 영역에서 방향이 역전된다.

두 사례는 신호 자체가 무용하다는 뜻이 아니라, *어느 입력 조건에서 어느 신호의 신뢰도가 높은지를 분리해서 봐야 한다*는 뜻이다. 본 논문은 그 외부 입력 조건으로 corpus support를 사용한다.

## Corpus support를 조건 축으로 사용하는 근거

후보 답변의 entity frequency 와 entity-pair co-occurrence 는, 모델이 “익숙한” entity 를 다루고 있는지를 외부 자료로 측정한 지표이다. corpus support 를 조건 축으로 쓰는 근거는 다음과 같다.

- **모델 출력에 의존하지 않는 외부 신호**. corpus support는 모델의 logit, hidden state, 응답 다양성과 독립적으로 산출된다. 따라서 같은 모델 신호가 어느 외부 조건에서 안정적인지를 *직교적으로* 평가할 수 있다.

- **CHOKE와의 자연스러운 연결**. CHOKE는 “익숙한 entity를 잘못 결합해 생기는 고확신 환각”이라는 직관과 닿아 있다. 모델 입장에서 “익숙함”을 외부 corpus 빈도로 측정하면, CHOKE 사례가 corpus support의 어떤 영역에 분포하는지 데이터로 확인할 수 있다.

- **환각 라벨이나 retrieval trigger와의 분리**. corpus 빈도가 작다고 해서 곧 환각이라는 뜻은 아니고, corpus 빈도가 크다고 해서 곧 정답이라는 뜻도 아니다. corpus support는 *지표 신뢰성을 조건화*하는 변수일 뿐이다.

## 평가 단위 차이가 만드는 해석 함정

본 논문이 다루는 신호들은 정의 단위가 모두 다르다.

- Semantic Entropy는 **prompt 단위** 신호이다. 같은 prompt의 두 candidate row(정답/환각)에 같은 값이 복사된다.

- 원논문 정의의 Semantic Energy는 **cluster 단위** 신호이지만, 최종 출력은 한 prompt의 N=10 표본 전체에서 cluster 확률 가중 집계로 얻은 *prompt 단위 scalar* $`U(\mathbf{x})=\sum_k p(\mathbb{C}_k)E_{\text{Bolt}}(\mathbb{C}_k)`$ 이다. 따라서 candidate-row 평가에서는 같은 prompt의 두 후보 행에 같은 값이 복사되어, SE 와 같은 이유로 candidate-row AUROC가 구조적으로 0.5에 묶인다.

- NLL, logit variance, confidence margin, entity frequency / pair co-occurrence는 **candidate 단위** 신호이다. 후보마다 값이 다르다.

세 단위를 같은 candidate-row AUROC 표에 합치면, prompt 단위 신호는 구조적으로 0.5에 묶이고 cluster 단위 신호도 정의에 맞지 않는 평가를 받는다. 이 점을 먼저 명시하지 않으면 “SE가 무용하다”거나 “Energy가 좋다”는 식의 잘못된 결론으로 이어질 수 있다. 본 논문은 이 단위 차이를 본문, 표, 그림 전체에 일관되게 명시한다.

## 본 논문의 분석 방향

위 세 가지 관찰을 합치면 본 논문의 분석 방향이 자연스럽게 도출된다.

1.  후보 단위 신호(NLL, logit variance, margin, entity frequency / pair co-occurrence)와 prompt/cluster 단위 신호(SE, Semantic Energy)를 *단위별로 분리해서* 보고한다.

2.  후보 단위 신호 위에서 corpus support 구간(3-bin 기본, 5/10-bin sensitivity)별로 “어떤 신호의 신뢰도가 높은가”를 측정한다.

3.  corpus 신호를 단순히 global fusion에 더하는 방식과, corpus support bin마다 가중치를 다르게 학습하는 conditional fusion 방식을 정량 비교하고, prompt-grouped bootstrap으로 통계적 유의성을 검정한다.

이러한 분석 절차는 단일 우열 비교가 아니라, “어느 corpus support 영역에서 어느 신호 또는 fusion 의 신뢰도가 가장 높은가”를 보여주는 *reliability map* 을 산출한다. 실험 결과는 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>에서 보고한다.

## Semantic Energy와 candidate-level diagnostic의 분리

본 실험은 Ma 등(2025) 식 (11)–(14)를 그대로 따른 Semantic Energy와, candidate-level Boltzmann-style energy diagnostic을 분리해 보고한다. 전자는 token energy $`\tilde{E}(x_t)=-z_\theta(x_t)`$, sample energy $`E(x^{(i)})=\frac{1}{T_i}\sum_t \tilde{E}(x_t^{(i)})`$, cluster total energy $`E_{\text{Bolt}}(\mathbb{C}_k)=\sum_{x^{(i)} \in \mathbb{C}_k} E(x^{(i)})`$, 최종 uncertainty $`U(\mathbf{x})=\sum_k p(\mathbb{C}_k) E_{\text{Bolt}}(\mathbb{C}_k)`$로 계산하며, cluster 확률 $`p(\mathbb{C}_k)`$는 SE 단계의 likelihood-based cluster probability를 그대로 상속한다.

후자는 teacher-forced 방식으로 후보 답을 채점할 때 얻은 token 단위 $`-\log Z`$의 평균, NLL의 평균, logit variance, confidence margin을 candidate 단위 한 줄로 묶은 진단 지표이다. 정의 단위가 cluster vs candidate로 다르므로 두 신호는 같은 열에 합치지 않는다.

# 실험

<span id="ch:experiment" label="ch:experiment"></span>

## 실험 설정

본 절에서는 환각 탐지 실험 절차를 설명한다. 실험의 핵심은 모델이 새 답변을 생성한 뒤 그 정오를 사후에 판정하는 방식이 아니라, 데이터셋이 이미 정답 답과 환각 답을 명시적으로 구분해 제공한다는 점을 활용해 두 후보를 같은 질문 아래에서 paired로 비교하는 데 있다. 이렇게 하면 (i) 새 답이 정답인지를 판정하기 위한 문자열 일치 heuristic, (ii) 별도 LLM에게 정오를 묻는 LLM-as-judge 방식 모두에 의존하지 않고, 이미 annotation으로 구분된 후보 쌍을 대상으로 uncertainty와 corpus-grounded feature가 환각 탐지에 어떤 신호를 주는지 분석할 수 있다.

### 데이터셋 구성

실험에는 TruthfulQA와 HaluEval-QA 두 데이터셋을 사용하였다. 두 데이터셋 모두 질문마다 정답 후보와 잘못된(혹은 환각) 후보를 명시적으로 제공하므로 paired 평가에 적합하다. TruthfulQA에서는 각 질문의 `correct_answers`와 `incorrect_answers`에서 하나의 정답 후보와 하나의 오답 후보를 결정론적으로 선택하였다. HaluEval-QA에서는 데이터셋이 제공하는 `right_answer`와 `hallucinated_answer`를 그대로 사용하였다.

각 질문은 하나의 prompt 단위로 다루며, 한 prompt에서 정확히 두 개의 candidate row를 만든다. 하나는 정답 후보, 다른 하나는 환각 후보이다. 최종 실험 데이터는 prompt 5,815개, candidate row 11,630개로 구성된다. HaluEval-QA가 prompt 5,000개와 row 10,000개, TruthfulQA가 정제 후 prompt 815개와 row 1,630개를 제공한다.

<div id="tab:dataset_composition">

| 데이터셋    | Prompt 수 | Candidate row 수 | 후보 쌍 출처              |
|:------------|----------:|-----------------:|:--------------------------|
| TruthfulQA  |       815 |            1,630 | 정답/오답 answer list     |
| HaluEval-QA |     5,000 |           10,000 | right/hallucinated answer |
| 합계        |     5,815 |           11,630 | paired candidates         |

실험 데이터셋 구성

</div>

### 환각 여부 라벨링

환각 여부 라벨은 모델 출력이나 후처리 judge에서 오지 않는다. 각 candidate row의 주 라벨은 데이터셋 annotation이 제공하는 `is_hallucination` 이진 값이며, 이 값이 최종 환각 탐지 평가의 기준이다. 즉 본 연구는 데이터셋 annotation을 ground truth로 받아들이고, 모델 출력 자체에서 라벨을 유도하지 않는다. 이때 Semantic Entropy는 prompt 단위 값이므로 같은 질문의 정답 후보와 환각 후보가 같은 SE 값을 공유한다.

### 모델 scoring 절차

모델은 Qwen2.5-3B 계열 causal LM을 사용한다. 후속 분석에서 token logit을 그대로 활용하기 위해 모든 후보의 full-vocabulary logits을 함께 기록한다. 실험은 두 종류의 모델 출력을 사용한다.

- **Free sampling**. Semantic Entropy 계산을 위해 각 prompt에 대해 모델로 N=10개의 짧은 답변을 자유 sampling한다 (아래 “Answer-only sampling 절차” 참조). 이는 prompt 단위 신호의 입력이다.

- **Teacher-forced scoring**. candidate 단위 feature 계산을 위해 각 정답/환각 후보 답을 모델에게 입력으로 주입하면서 점수만 측정한다 (아래 “Teacher-forced scoring 구현” 참조). 모델이 후보 답을 새로 만들지는 않으며, 주어진 답을 평가만 한다.

이 구분은 이후 결과 해석에서 중요하다. Semantic Entropy는 같은 prompt의 두 candidate row에 같은 값이 복사되지만, candidate 단위 diagnostic(NLL, logit variance, confidence margin)은 후보마다 값이 다르고, Semantic Energy는 cluster 단위로 정의된다.

##### Answer-only sampling 절차.

모델이 질문에 대해 설명을 덧붙이지 않고 답만 짧게 출력하도록 sampling을 제약하는 절차이다. 답변 1개당 최대 64 token까지 생성하되 그보다 길어지면 잘라내며, 답변 형식이 맞지 않은 표본은 일정 횟수까지 다시 sampling한다. 이렇게 한 prompt 당 짧은 답변을 N=10개 모은 뒤 NLI semantic clustering을 수행한다. 답변 길이를 짧게 제한하는 이유는 (i) NLI 모델이 짧은 entity-중심 답변에서 cluster 판단을 더 안정적으로 하고, (ii) 후보 답변과 단위가 맞아 비교가 깔끔하기 때문이다.

##### Teacher-forced scoring 구현.

모델로 후보 답을 *새로 생성*하지 않고, 주어진 후보 답이 그 prompt 뒤에 등장할 때 모델이 어떤 점수를 부여하는지를 측정한다. 구체적으로 입력은 “prompt + 정답 후보” 또는 “prompt + 환각 후보”이며, 후보 답에 해당하는 token 구간만 평가 대상이 된다. 한 token 위치 $`t`$ 에 대해 모델은 vocabulary 전체에 대한 logit vector $`z_t \in \mathbb{R}^{|V|}`$ 를 산출한다. 본 실험은 그 위치에서 다음 값을 모두 기록한다.

- 선택된 token의 logit $`z_t(x_t)`$ 와 log probability $`\log p_\theta(x_t \mid \text{prompt}, x_{<t})`$.

- 분배 함수 값 $`\log Z_t = \log \sum_{v \in V} \exp(z_t(v))`$.

- 같은 위치에서 vocabulary 전체에 대한 logit의 분산.

- 가장 높은 logit을 받은 token과 두 번째로 높은 logit을 받은 token 사이의 차이(confidence margin).

후보 단위 점수는 위 token-level 값을 후보 답 길이 $`T`$ 에 대해 평균하여 만든다. NLL(평균 음의 로그 확률)과 token-level $`-\log Z`$의 평균이 candidate 단위 한 줄로 묶이는 진단 지표이다. 이 절차는 한 prompt 의 두 후보(정답/환각)를 동일한 채점 기준으로 비교할 수 있게 해 준다는 점에서 본 연구의 paired 평가 설정에 핵심적이다.

### Feature 계산

본 연구는 세 종류의 feature군을 사용한다.

##### Semantic Entropy.

Semantic Entropy는 한 prompt에서 얻은 N=10 답변을 `microsoft/deberta-large-mnli` 기반 bidirectional NLI entailment로 semantic cluster에 묶은 뒤, 표본 sequence log-likelihood를 cluster 단위로 log-sum-exp로 집계해 cluster probability mass를 만들고 그 위에서 entropy를 계산한다. 산출물에는 `semantic_entropy_nli_likelihood`, `semantic_entropy_cluster_count`, NLI model reference, 표본 / cluster log-likelihood가 함께 기록된다. SE는 prompt-level feature이고, paired 설정에서 정답 후보와 환각 후보가 같은 SE 값을 받는다.

##### Semantic Energy와 candidate-level diagnostic.

Semantic Energy는 Ma 등(2025) 식 (11)–(14)를 그대로 따른다. 동일한 N=10 답변과 SE 단계의 NLI semantic cluster를 재사용하며, cluster를 다시 계산하지 않고 `(prompt_id, sample_index)`로 join한다. 식은 다음과 같다.
``` math
\begin{align}
\tilde{E}(x_t^{(i)}) &= -z_\theta(x_t^{(i)}) \quad \text{(식 13: token energy = $-$ selected-token logit)} \\
E(x^{(i)}) &= \frac{1}{T_i}\sum_{t=1}^{T_i} \tilde{E}(x_t^{(i)}) \quad \text{(식 11: sample energy)} \\
E_{\text{Bolt}}(\mathbb{C}_k) &= \sum_{x^{(i)} \in \mathbb{C}_k} E(x^{(i)}) \quad \text{(식 12: cluster total energy)} \\
U(\mathbf{x}) &= \sum_{k=1}^{K} p(\mathbb{C}_k)\, E_{\text{Bolt}}(\mathbb{C}_k) \quad \text{(식 14: 최종 uncertainty)}
\end{align}
```
여기서 $`p(\mathbb{C}_k) = \sum_{x^{(i)} \in \mathbb{C}_k} \bar{p}(x^{(i)} \mid q)`$는 식 (8)의 likelihood-based cluster probability로, SE 단계에서 산출된 값을 그대로 상속한다. 이는 Ma 등(2025) §2.2가 인용하는 Farquhar 등(2024)의 cluster-likelihood weighting과 일치한다. 부호 규약은 lower energy = higher reliability(높은 energy = 높은 uncertainty)이다.

한편 teacher-forced candidate scoring에서 얻은 token-level $`-\log Z`$의 평균, NLL의 평균, logit variance, confidence margin은 candidate-level Boltzmann-style energy diagnostic으로 별도 보존하며, Semantic Energy와 같은 열에 합치지 않는다.

##### Corpus-grounded feature.

후보 답변에 등장하는 entity 의 corpus 내 frequency 와 두 entity 사이의 co-occurrence 를 측정해 corpus support 신호를 만든다. Entity 추출은 외부 NER 모델 대신 *경량 규칙 기반 phrase 후보 추출기* 를 사용한다 — 따옴표로 감싼 3 글자 이상 문자열, 1–4 단어 길이의 capitalized n-gram (각 단어 첫 글자 대문자), 그리고 stopword 가 아닌 5 글자 이상의 token 을 후보로 모아 정규화 후 중복 제거한다. Entity-pair 는 한 후보 답변 내에서 추출된 entity 의 unordered pair 로 정의하며, entity 가 2개 미만 추출된 답변은 co-occurrence 계산에서 제외된다.

corpus count backend 로는 Liu 등(2024) 이 제안한 Infini-gram 을 사용하며, 본 실험은 외부 corpus 로 16B token 규모의 web-scale 인덱스를 사용한다. 한 후보 답변에 대해 (i) 추출된 entity 의 frequency, (ii) 두 entity 의 co-occurrence, (iii) 이 두 값을 분위로 나눈 corpus support 구간 label (낮음 / 중간 / 높음 3-bin 과 더 세분화된 5-bin / 10-bin sensitivity) 을 산출한다.

이 신호는 외부 자료에서 후보 답변이 지지될 가능성을 *간접적으로* 나타내는 proxy이며, 모델의 hidden-state나 환각 여부 라벨을 사용하지 않는다. 낮은 frequency나 zero co-occurrence 자체는 환각 라벨이 아니라, 어디까지나 *지표 신뢰도를 조건화하는 binning 변수*이다.

### 실험 단계

실험은 다음 순서로 진행된다.

1.  Paired 데이터셋을 구성한다.

2.  모델 출력을 수집한다 (free sampling과 teacher-forced scoring).

3.  Semantic Entropy를 계산한다.

4.  Corpus support 축을 계산한다.

5.  Semantic Energy와 candidate 단위 logit diagnostic을 계산한다.

6.  위 신호를 한 feature table로 결합한다.

7.  단일 신호 baseline과 여러 fusion 방식을 비교한다.

8.  Robustness 분석(prompt-grouped bootstrap, leave-one-dataset-out, corpus support 구간별 신뢰도, binning sensitivity, threshold / calibration)을 수행한다.

### 평가 지표와 해석 원칙

공통 baseline 표에는 AUROC, AUPRC, F1 등 표준 분류 지표를 보고한다. 학습 라벨은 데이터셋 annotation의 `is_hallucination` 이진 값이다. 주 분석 축은 corpus support bin(3-bin 기본, 5-bin / 10-bin sensitivity)이며, 각 bin 안에서 SE / Energy / logit-diagnostic의 AUROC와 paired win-rate를 보고한다.

다만 이 candidate-row 종합 지표는 모든 feature에서 같은 의미를 갖지 않는다. Semantic Entropy는 prompt-level feature이므로, 같은 prompt의 정답 후보와 환각 후보가 같은 SE 값을 가진다. SE-only의 candidate-row AUROC가 0.5에 묶이는 것은 SE가 무의미하다는 뜻이 아니라, prompt-level feature를 candidate-row 순위 평가에 적용한 구조적 결과이다.

Semantic Energy와 logit diagnostic은 candidate-level 또는 cluster-level 신호이지만, 그 신뢰도가 corpus support 조건에 따라 달라질 수 있다. 따라서 핵심 평가는 각 corpus support bin에서 환각 후보가 짝지어진 정답 후보보다 더 높은 위험 점수를 받는지 확인하는 matched-pair metric이다. 구체적으로 (i) hallucinated $`-`$ correct paired delta, (ii) paired win-rate, (iii) prompt-grouped bootstrap 95% 신뢰구간, (iv) leave-one-dataset-out 일반화, (v) threshold / calibration sensitivity, (vi) 3/5/10-bin binning sensitivity를 모두 보고한다. Train/test 분할은 prompt 단위로 묶어 같은 prompt의 두 candidate row가 fold를 가로지르지 않게 강제한다.

## 종합 베이스라인 결과

표 <a href="#tab:current_thesis_evidence" data-reference-type="ref" data-reference="tab:current_thesis_evidence">4.2</a>는 본 실험의 종합 베이스라인 결과이다. SE-only / Energy-only는 candidate-row AUROC가 구조적으로 0.5에 묶인다 — SE는 정의상 prompt 단위, Semantic Energy는 cluster 단위이지만 cluster 확률 가중 집계로 *prompt 단위 scalar* 가 되어 한 prompt의 두 candidate row(정답/환각)에 같은 값이 복사되며, 결과적으로 candidate row 사이 순위에 영향을 주지 못한다. 본 실험에서 두 신호 모두 candidate-row AUROC = 0.500 (SE-only) / 0.495 (Energy-only) 로 측정되었으며, 후자가 정확히 0.5가 아닌 이유는 매우 작은 cluster 확률 차이로 인해 같은 prompt의 두 후보가 서로 다른 cluster 에 join되어 미세한 변화가 생기기 때문이다. 두 값 모두 prompt 단위 신호의 broadcast 결과이므로 단일 비교 수치로 해석하지 않고 결과 표에서는 제외한다.

<div id="tab:current_thesis_evidence">

<table>
<caption>종합 baseline 결과 (단일 자동화 실행 산출). candidate row 11<span>,</span>630개, 학습 라벨은 데이터셋 annotation의 <code>is_hallucination</code>. learned fusion 계열은 prompt-grouped fold 분리로 평가하여 같은 prompt의 두 후보가 fold를 가로지르지 않게 강제하였다.</caption>
<thead>
<tr>
<th style="text-align: left;">Method</th>
<th style="text-align: center;">AUROC</th>
<th style="text-align: center;">AUPRC</th>
<th style="text-align: center;">F1</th>
<th style="text-align: left;">비고</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Single-signal baseline</em></td>
</tr>
<tr>
<td style="text-align: left;">logit-diagnostic-only</td>
<td style="text-align: center;">0.375</td>
<td style="text-align: center;">0.423</td>
<td style="text-align: center;">0.659</td>
<td style="text-align: left;">역방향 (CHOKE)</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Global fusion (no corpus)</em></td>
</tr>
<tr>
<td style="text-align: left;">logistic regression</td>
<td style="text-align: center;">0.372</td>
<td style="text-align: center;">0.417</td>
<td style="text-align: center;">0.668</td>
<td style="text-align: left;">선형</td>
</tr>
<tr>
<td style="text-align: left;">random forest</td>
<td style="text-align: center;">0.525</td>
<td style="text-align: center;">0.517</td>
<td style="text-align: center;">0.601</td>
<td style="text-align: left;">비선형</td>
</tr>
<tr>
<td style="text-align: left;">gradient boosting</td>
<td style="text-align: center;"><strong>0.541</strong></td>
<td style="text-align: center;">0.521</td>
<td style="text-align: center;">0.616</td>
<td style="text-align: left;">비선형 1위</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Corpus-conditioned fusion (제안)</em></td>
</tr>
<tr>
<td style="text-align: left;">corpus-bin feature selection</td>
<td style="text-align: center;">0.482</td>
<td style="text-align: center;">0.475</td>
<td style="text-align: center;">0.585</td>
<td style="text-align: left;">bin마다 single feature</td>
</tr>
<tr>
<td style="text-align: left;"><strong>corpus-bin weighted fusion</strong></td>
<td style="text-align: center;"><strong>0.538</strong></td>
<td style="text-align: center;"><strong>0.512</strong></td>
<td style="text-align: center;">0.666</td>
<td style="text-align: left;"><strong>본 논문 제안</strong></td>
</tr>
</tbody>
</table>

</div>

표에서 세 가지 사실이 두드러진다.

- Logit-diagnostic-only는 AUROC 0.375로 무작위(0.5)보다 못한 역방향이다. 이는 환각 답이 정답 답보다 모델 입장에서 더 자신있는 답으로 평가됨을 의미하며, Simhi 등(2025)이 보고한 CHOKE 패턴과 일치한다. 같은 candidate-level 신호를 그대로 받는 선형 logistic regression(no corpus)도 AUROC 0.372로 같은 역전을 흡수한다.

- 비선형 결합기인 sklearn gradient boosting(no corpus)은 AUROC 0.541로, 동일한 candidate-level 신호 위에서 비선형 결합을 통해 역전을 부분 회복한다.

- 본 논문이 제안하는 corpus-bin weighted fusion은 AUROC 0.538로 비선형 결합과 같은 수준에 도달한다. 정의상 단순한 선형 구조 위에서 corpus support bin마다 다른 가중치를 학습하는 것만으로 같은 회복을 달성한다는 점이 핵심이다.

이는 본 논문의 주장을 직접 뒷받침한다. 단일 global fusion으로는 가려지는 신호 가치가, 외부 조건 축에 따라 분리해 평가하면 같은 데이터에서 복원된다.

## Prompt-grouped bootstrap 신뢰구간

corpus-bin weighted fusion이 *각 corpus support 구간에서* 환각 후보를 정답 후보보다 더 위험한 점수로 평가하는지 통계적으로 검정하기 위해, prompt-grouped bootstrap(같은 prompt의 두 후보를 한 단위로 묶어 resample)을 2,000회 수행하여 95% 신뢰구간을 산출하였다. 표 <a href="#tab:bootstrap_ci" data-reference-type="ref" data-reference="tab:bootstrap_ci">4.3</a>는 corpus support decile별 paired win-rate와 mean paired delta(환각 후보 점수 $`-`$ 정답 후보 점수)의 신뢰구간이다.

<div id="tab:bootstrap_ci">

| decile | n | paired win-rate \[95% CI\] | mean delta \[95% CI\] |
|:---|:--:|:--:|:--:|
| 00–10 | 549 | 0.417 \[0.366, 0.468\] | $`-0.0382`$ \[$`-0.0472`$, $`-0.0292`$\] |
| 10–20 | 1,020 | 0.540 \[0.496, 0.581\] | $`-0.0022`$ \[$`-0.0055`$, $`+0.0010`$\] |
| 20–30 | 2,601 | **0.558 \[0.534, 0.586\]** | **$`+0.0034`$ \[$`+0.0018`$, $`+0.0049`$\]** |
| 30–40 | 3,465 | **0.635 \[0.611, 0.658\]** | **$`+0.0126`$ \[$`+0.0108`$, $`+0.0145`$\]** |
| 40–50 | 1,617 | **0.620 \[0.587, 0.654\]** | **$`+0.0130`$ \[$`+0.0097`$, $`+0.0165`$\]** |
| 50–60 | 802 | **0.559 \[0.512, 0.608\]** | **$`+0.0096`$ \[$`+0.0041`$, $`+0.0151`$\]** |
| 60–70 | 587 | 0.493 \[0.432, 0.550\] | $`+0.0040`$ \[$`-0.0039`$, $`+0.0123`$\] |
| 70–80 | 507 | **0.589 \[0.532, 0.646\]** | **$`+0.0090`$ \[$`+0.0017`$, $`+0.0163`$\]** |
| 80–90 | 356 | **0.616 \[0.547, 0.680\]** | **$`+0.0116`$ \[$`+0.0011`$, $`+0.0216`$\]** |
| 90–100 | 125 | 0.580 \[0.464, 0.710\] | $`+0.0054`$ \[$`-0.0117`$, $`+0.0222`$\] |

Corpus support decile별 corpus-bin weighted fusion의 paired 통계 95% 신뢰구간 (prompt-grouped bootstrap, n=2,000). win-rate $`> 0.5`$ 또는 mean delta $`> 0`$ 이 “환각 후보가 더 위험하게 평가됨”을 의미.

</div>

세 가지 패턴이 두드러진다.

- **Decile 20–50, 70–90 (n=9,948, 86%)**: 6개 decile에서 paired win-rate가 0.5보다 통계적으로 유의미하게 높다 (CI가 0.5를 가로지르지 않음). corpus-bin weighted fusion이 본 데이터의 대부분 영역에서 환각 후보를 정답보다 위험하게 평가한다. 한편 decile 60–70 (n=587, CI \[0.432, 0.550\])과 decile 90–100 (n=125, CI \[0.464, 0.710\])은 0.5를 포함하므로 통계적으로 유의미하지 않다. decile 60–70 의 비유의는 표 <a href="#tab:bin10_method_matrix" data-reference-type="ref" data-reference="tab:bin10_method_matrix">4.4</a>와 표 <a href="#tab:bin10_winrate" data-reference-type="ref" data-reference="tab:bin10_winrate">4.5</a>에서 corpus-bin weighted fusion (0.493), gradient boosting (0.534), random forest (0.525) 가 모두 0.5 근처에 모이는 *어떤 신호도 분명한 우위를 갖지 못하는 전이 영역*임을 반영하며, decile 90–100 의 비유의는 표본 수(n=125) 자체가 작아 신뢰구간이 넓어진 결과이다.

- **Decile 30–50 sweet spot (n=5,082)**: paired win-rate 신뢰구간이 \[0.587, 0.658\] 까지 올라가며, mean delta도 +0.01 이상 양수로 robust하다. 본 결합기의 핵심 작동 영역이다.

- **Decile 00–10 (n=549)**: paired win-rate 신뢰구간 \[0.366, 0.468\] 이 *0.5보다 낮은 영역에서* 통계적으로 유의미하게 머무른다. corpus가 거의 모르는 답에서는 corpus-bin weighted fusion 이 오히려 잘못된 방향으로 작동함을 명확히 보여준다 (이 영역에서는 corpus를 쓰지 않는 gradient boosting이 1위라는 §결과와 일치).

종합 수준의 비교에서 corpus-bin weighted fusion 은 선형 global fusion (logistic regression, AUROC 0.372 — 무작위 0.5 이하의 역방향 결합기) 대비 AUROC **+0.166 \[+0.157, +0.176\]**, AUPRC +0.096 \[+0.089, +0.102\] 로 신뢰구간이 0을 포함하지 않는다. 다만 이 비교는 *역방향으로 작동하는 baseline* 을 기준으로 한다는 점에 유의해야 한다. 실질적 경쟁 baseline 인 비선형 sklearn gradient boosting (no corpus, AUROC 0.541) 과는 AUROC 차이가 $`-0.003`$ 으로 사실상 동급이다. 본 논문 제안의 우위는 전체 AUROC 1위에 있는 것이 아니라 (i) decile 별 paired 분해에서 다수 구간 통계적 우위 (본 절 표 <a href="#tab:bootstrap_ci" data-reference-type="ref" data-reference="tab:bootstrap_ci">4.3</a>), (ii) calibration 지표의 약 2.6배 개선 (Brier 0.253 vs 0.278, ECE 4.7% vs 12.1%), (iii) 데이터셋 전이 격차 최소 (§<a href="#sec:lodo" data-reference-type="ref" data-reference="sec:lodo">4.6</a>) 의 conditional / 보정 측면에 있다. 본 절의 decile 별 분해는 그 조건부 우위가 *어느 corpus support 구간에서 발생하는지* 를 정밀하게 보여준다.

## Corpus support 구간별 신뢰도 (10-bin 정밀 분해)

corpus support 축을 10분위(decile)로 분해해 각 구간에서 어떤 방법이 가장 잘 작동하는지 본다. 표 <a href="#tab:bin10_method_matrix" data-reference-type="ref" data-reference="tab:bin10_method_matrix">4.4</a>는 각 decile에서 핵심 4개 방법의 AUROC를, 표 <a href="#tab:bin10_winrate" data-reference-type="ref" data-reference="tab:bin10_winrate">4.5</a>는 같은 decile의 paired win-rate(0.5가 무작위, 1에 가까울수록 환각 후보를 더 위쪽에 배치)를 보여준다. 그림 <a href="#fig:money" data-reference-type="ref" data-reference="fig:money">4.1</a>는 같은 데이터를 색으로 보여주는 reliability map이다.

<div id="tab:bin10_method_matrix">

| decile | n | corpus-bin weighted fusion | gradient boosting (no corpus) | random forest (no corpus) | logit-diagnostic |
|:---|:--:|:--:|:--:|:--:|:--:|
| 00–10 | 549 | 0.534 | **0.605** | **0.618** | 0.494 |
| 10–20 | 1,020 | **0.563** | 0.552 | 0.510 | 0.390 |
| 20–30 | 2,601 | 0.561 | **0.576** | 0.552 | 0.347 |
| 30–40 | 3,465 | **0.602** | 0.556 | 0.536 | 0.357 |
| 40–50 | 1,617 | **0.612** | 0.515 | 0.495 | 0.326 |
| 50–60 | 802 | **0.554** | 0.496 | 0.472 | 0.343 |
| 60–70 | 587 | **0.552** | 0.531 | 0.521 | 0.442 |
| 70–80 | 507 | **0.544** | 0.521 | 0.524 | 0.444 |
| 80–90 | 356 | **0.543** | 0.533 | 0.505 | 0.450 |
| 90–100 | 125 | 0.500 | 0.488 | 0.498 | **0.503** |

Method $`\times`$ corpus support decile AUROC. 굵은 글씨는 decile별 1위.

</div>

<div id="tab:bin10_winrate">

| decile | n | corpus-bin weighted fusion | gradient boosting (no corpus) | random forest (no corpus) | logit-diagnostic |
|:---|:--:|:--:|:--:|:--:|:--:|
| 00–10 | 549 | 0.417 | **0.580** | 0.592 | 0.474 |
| 10–20 | 1,020 | 0.540 | **0.549** | 0.511 | 0.358 |
| 20–30 | 2,601 | **0.558** | 0.558 | 0.549 | 0.330 |
| 30–40 | 3,465 | **0.635** | 0.546 | 0.551 | 0.334 |
| 40–50 | 1,617 | **0.620** | 0.520 | 0.509 | 0.284 |
| 50–60 | 802 | **0.559** | 0.534 | 0.474 | 0.316 |
| 60–70 | 587 | 0.493 | **0.534** | 0.525 | 0.339 |
| 70–80 | 507 | **0.589** | 0.576 | 0.567 | 0.430 |
| 80–90 | 356 | **0.616** | 0.571 | 0.507 | 0.409 |
| 90–100 | 125 | **0.580** | 0.464 | 0.493 | 0.507 |

Method $`\times`$ corpus support decile paired win-rate. 0.5가 무작위. 굵은 글씨는 decile별 1위.

</div>

<figure id="fig:money" data-latex-placement="htbp">

<figcaption>Signal <span class="math inline">×</span> corpus support decile reliability map (10-bin). 위 panel은 candidate-level 신호의 paired win-rate(0.5 중심, 빨강이면 환각 후보가 정답보다 더 위험하게 평가됨, 파랑이면 그 반대), 아래 panel은 prompt-level 신호의 bin 평균을 신호 내 z-score로 표준화한 값(cividis colormap)이다. 셀 안의 숫자는 표본 수와 win-rate / 평균을 함께 보여준다.</figcaption>
</figure>

핵심 발견은 다음 네 가지이다.

- **Decile 30–50: 최적 구간** (n=5,082). corpus-bin weighted fusion이 AUROC 0.602–0.612, paired win-rate 0.620–0.635로 모든 방법 중 1위이다. 너무 희귀하지도 너무 흔하지도 않은 entity 조합 영역에서 corpus-conditioned 결합이 가장 강하게 작동한다.

- **Decile 00–10** (n=549). gradient boosting (no corpus)만이 1위(AUROC 0.605, win-rate 0.580). corpus가 거의 모르는 답에서는 corpus 신호가 잡음이 되고, 모델 내부 logit의 비선형 결합이 더 안정적이다.

- **Decile 20–50: CHOKE 심화** (n=7,683). logit-diagnostic의 paired win-rate가 0.284–0.358까지 떨어져 가장 강한 역전을 보인다. 환각 답이 정답 답보다 더 자신있게 생성되는 현상은 corpus가 어느 정도 지지하는 답에서 가장 두드러지며, 이는 모델이 익숙한 entity에 대해 잘못된 답을 confident하게 만드는 경향과 일치한다.

- **Decile 90–100** (n=125). 표본이 작아 paired win-rate 신뢰구간 \[0.464, 0.710\] 이 0.5를 포함하므로 통계적 우열을 단정할 수 없는 영역이다. 관측 win-rate 0.580은 추세 참고용이다.

종합하면, corpus support 축은 단순한 “희귀 vs 빈출” 이분법이 아니라, 어느 신호의 신뢰도가 높은가가 지지도에 따라 비단조적으로 바뀌는 축이다. 본 논문의 corpus 조건부 reliability 분석이 단일 global fusion으로 환원되지 않는다는 직접 증거이다. 더 거친 3-bin 분할(낮음/중간/높음)이나 5-bin 분할에서도 동일한 sweet-spot 패턴이 나타나며, sensitivity 결과는 부록 산출물에 함께 보고된다.

## CHOKE 패턴의 두 직교 scope 재현

Simhi 등(2025)의 CHOKE 패턴, 즉 모델이 정답 지식을 가지고도 high-certainty로 환각을 생성하는 현상을, 본 데이터에서 서로 직교적인 두 평가 scope에서 동시에 관측하였다.

- **Candidate-level 증거**. 환각 답이 정답 답보다 NLL이 작고(paired win-rate 0.36), confidence margin이 크다(win-rate 0.34, n$`\approx`$<!-- -->5,200, ties 제외). 두 신호 모두 “정답이 더 자신있는 답이어야 한다”는 직관과 반대 방향이며, 환각 답이 모델 입장에서 더 자신있는 답으로 평가된다는 뜻이다.

- **Corpus-level 증거**. 후보 답에서 추출한 entity-pair co-occurrence 축에서 환각 후보가 정답 후보보다 큰 값을 가질 paired win-rate 가 0.551 이다(n=2,499). 즉 환각 답이 corpus 에서 co-occurrence 가 더 큰 entity pair 를 사용한다. 이 효과 크기는 candidate-level 증거(NLL win-rate 0.36, margin 0.34) 보다 현저히 작으므로 단독 신호보다는 *corroborating evidence* 로 해석한다.

두 증거는 같은 그림을 가리킨다. 모델은 익숙한 entity를 잘못된 관계로 confident하게 조합해 환각을 만든다. logit이나 sample-consistency 기반 detector의 본질적 한계를 외부 corpus 신호로 분리해 낼 수 있음을 의미한다. 본 논문은 Simhi 등이 제안한 hidden-state probing 대신 corpus 축으로 같은 현상을 분리해 낸다.

## 데이터셋 간 일반화

한 데이터셋만 학습한 결합기가 다른 데이터셋에서도 우위를 유지하는지 확인하기 위해 *한 데이터셋 빼고 학습 (leave-one-dataset-out)* 분할을 적용한다. 절차는 다음과 같다. 두 데이터셋(HaluEval-QA / TruthfulQA) 중 하나를 *평가 전용*으로 따로 빼고, 나머지 데이터셋으로만 결합기를 학습한 뒤 빠진 데이터셋에서 평가한다. 이를 두 데이터셋에 대해 한 번씩 수행하면 (i) HaluEval-QA로 학습하고 TruthfulQA에서 평가, (ii) TruthfulQA로 학습하고 HaluEval-QA에서 평가, 두 결과가 나온다. 한 결합기가 *학습 데이터셋과 다른 종류의 데이터셋에서도 잘 작동하는지*를 보는 도메인 전이 검증이다.

표 <a href="#tab:lodo" data-reference-type="ref" data-reference="tab:lodo">4.6</a>는 그 결과이다. 표 헤더 “Eval HaluEval-QA”는 *HaluEval-QA를 평가 전용으로 따로 빼고* TruthfulQA만으로 결합기를 학습한 뒤 빠진 HaluEval-QA(n=10,000) 에서 평가한 결과이다. 마찬가지로 “Eval TruthfulQA”는 HaluEval-QA만으로 학습하고 TruthfulQA(n=1,630) 에서 평가한 결과이다. AGG는 학습 / 평가 모두 두 데이터셋을 합쳐 사용한 종합 결과(prompt 단위 fold 분리) 이다.

<div id="tab:lodo">

| Method                         |    AGG    | Eval HaluEval-QA | Eval TruthfulQA |
|:-------------------------------|:---------:|:----------------:|:---------------:|
| **corpus-bin weighted fusion** | **0.538** |    **0.536**     |    **0.556**    |
| gradient boosting (no corpus)  |   0.541   |      0.550       |      0.523      |
| random forest (no corpus)      |   0.525   |      0.528       |      0.510      |
| corpus-bin feature selection   |   0.482   |      0.478       |      0.428      |
| logit-diagnostic-only          |   0.375   |      0.340       |      0.462      |

데이터셋 간 일반화 (leave-one-dataset-out) AUROC. AGG: 두 데이터셋을 모두 학습 / 평가에 포함한 종합 결과 (prompt 단위 fold 분리). “Eval HaluEval-QA” / “Eval TruthfulQA”: 해당 데이터셋을 평가 전용으로 빼고 *나머지 한 데이터셋만으로* 학습한 결합기의 결과. 두 분할 사이 격차가 작을수록 데이터셋 전이에 robust.

</div>

corpus-bin weighted fusion은 두 분할에서 0.536 / 0.556으로 격차가 0.020에 그친다. 반면 gradient boosting은 HaluEval(0.550)과 TruthfulQA(0.523) 격차가 0.027로 더 크고, logit-diagnostic-only는 격차가 0.122로 데이터셋 의존성이 크다. logit-diagnostic 의 큰 격차는 CHOKE 패턴 (logit 신호의 방향 역전) 이 두 데이터셋에서 강도가 다르게 나타남을 시사한다 — HaluEval-QA 의 환각 후보가 더 강하게 confident하게 평가되어 logit 신호의 역전이 심하고 (AUROC 0.340), TruthfulQA 는 그 정도가 약하다 (0.462). corpus-bin weighted fusion은 corpus 조건에 따라 신호 가중치를 달리 적용하므로 이 dataset-specific CHOKE 강도 차이에 덜 휘둘린다.

## Threshold / calibration 진단

분류 정확도뿐 아니라 출력 점수를 확률로 해석할 수 있는지도 함께 점검하였다(표 <a href="#tab:calibration" data-reference-type="ref" data-reference="tab:calibration">4.7</a>). Brier score와 ECE(expected calibration error)는 둘 다 작을수록 calibration이 좋다.

<div id="tab:calibration">

| Method                         |   Brier   |    ECE    |
|:-------------------------------|:---------:|:---------:|
| **corpus-bin weighted fusion** | **0.253** | **0.047** |
| random forest (no corpus)      |   0.267   |   0.094   |
| gradient boosting (no corpus)  |   0.278   |   0.121   |
| logit-diagnostic-only          |   0.280   |   0.190   |
| global fusion (no corpus)      |   0.279   |   0.196   |

Threshold / calibration 진단. Brier와 ECE 모두 낮을수록 좋다.

</div>

corpus-bin weighted fusion 은 분류 정확도뿐 아니라 calibration 에서도 1위이다 (Brier 0.253, ECE 4.7%). 다른 방법 대비 ECE 가 1/2–1/4 수준으로 작다. 다만 calibration 우위의 실용적 의미는 downstream use case 에 따라 달라진다 — 본 실험의 모든 방법의 AUROC 는 0.5–0.54 범위로 discriminability 자체가 제한적이므로, 탐지 점수를 단순 순위(ranking) 로 쓰는 응용에서는 calibration 의 직접적 이점이 작다. 반면 점수를 확률로 해석해 임계값 기반 선별 (예: 점수 $`\ge 0.7`$ 인 응답만 사용자에게 표시) 이나 downstream 리스크 계산에 사용하는 경우, calibration 이 좋은 점수는 실제 오류율과의 일치도가 높아 의사결정 품질이 개선된다. 본 실험의 ECE 4.7% 는 실제 오류율과의 오차가 약 4.7%p 이내임을 의미한다. 단, AUROC 0.54 수준의 탐지기를 실제 배포할 경우 false positive / negative 율이 높다는 근본적 한계는 calibration 개선만으로 해소되지 않으며, 이는 본 연구의 주요 한계점 중 하나이다.

# 결론

<span id="ch:conclusion" label="ch:conclusion"></span>

## 연구 요약

본 논문은 LLM 환각 탐지에서 Semantic Entropy(SE), Semantic Energy, corpus-grounded 신호를 corpus support라는 조건 축 관점에서 재정리하고, paired 설정 위에서 비교 평가하였다. 제 <a href="#ch:intro" data-reference-type="ref" data-reference="ch:intro">[ch:intro]</a>장에서 제시한 세 연구 질문에 대한 답을 정리하면 다음과 같다.

##### RQ1: Corpus support에 따른 신호 신뢰성 변화 측정.

SE / Semantic Energy / logit diagnostic 의 신뢰도는 corpus support 구간에 따라 단조롭지 않은 패턴을 보인다. logit diagnostic 의 paired win-rate 는 중간 decile (20–50) 에서 0.28–0.36까지 떨어져 CHOKE 방향 역전이 가장 심하고, 동일 영역에서 corpus-bin weighted fusion 은 win-rate 0.62–0.64 로 가장 강하게 작동한다. 다만 decile 60–70 의 일시적 dip 후 70–80 회복 패턴은 해당 decile 의 신뢰구간이 0.5 를 포함하는 비유의 영역을 포함하므로, 잔차 noise 와 실제 구조적 비단조성을 구별하려면 독립 데이터셋 검증이 필요하다. 현재 데이터에서 확인 가능한 사실은 “corpus support 가 높을수록 탐지 신호가 단조롭게 개선되지는 않는다” 는 점이다. SE / Semantic Energy 자체는 prompt 또는 cluster 단위 신호이므로 candidate-row AUROC 로 직접 비교할 수 없으며, 이를 별도 평가 단위로 분리하지 않으면 신호 가치가 가려진다.

##### RQ2: Conditional fusion의 효용 검증.

선형 global fusion (logistic regression, AUROC 0.372) 은 무작위 0.5 이하의 *역방향* 결합기이다. corpus-bin weighted fusion (0.538) 은 이 역방향 baseline 대비 AUROC 0.166 \[0.157, 0.176\] 의 유의미한 회복을 달성한다 (AUPRC 차이도 양수, 절대값 0.096). 그러나 실질적 경쟁 baseline 인 비선형 sklearn gradient boosting (no corpus, AUROC 0.541) 과는 종합 AUROC 가 사실상 동급 (차이 $`-0.003`$) 이다. 본 논문 제안 결합기의 독립적 가치는 종합 AUROC 우위가 아니라 (i) decile 30–80 의 6 개 구간에서 paired win-rate 1위, (ii) 데이터셋 전이 격차 최소 (0.020 vs gradient boosting 0.027), (iii) calibration 1위 (Brier 0.253, ECE 4.7% — gradient boosting 의 1/2.6 수준) 라는 conditional / 보정 측면에 있다.

##### RQ3: CHOKE 패턴의 외부 신호 분리.

Simhi 등(2025)의 CHOKE 패턴이 본 데이터에서 두 직교 scope에서 동시 재현되었다. candidate-level 에서는 NLL paired win-rate 0.36, confidence margin 0.34 로 환각 답이 더 자신있게 평가됨이 일관 관측되고, corpus-level 에서는 entity-pair co-occurrence paired win-rate 0.551 로 환각 답이 corpus 에서 co-occurrence 가 더 큰 entity pair 를 사용한다는 *상관 관계* 가 관측된다. 두 증거를 종합하면 corpus 에서 고-co-occurrence entity pair 가 환각 답변에 더 자주 사용된다는 패턴이 외부 corpus 신호로 분리 가능함을 보여준다. 이는 모델이 corpus 노출 빈도가 높은 entity 조합을 더 confident 하게 생성하는 경향과 일관되나, 본 실험 corpus(Infini-gram 16B token) 와 모델의 실제 pretraining 데이터가 다르므로 *인과 관계* 로 해석하는 데는 주의가 필요하다 (Zhang 등(2025) 에서도 동일한 caveat 명시).

## 학술적 기여

본 논문의 학술적 기여는 다음과 같다.

1.  **CHOKE 패턴의 두 직교 scope 동시 재현**. Simhi 등(2025)이 hidden-state probing으로 보고한 high-certainty hallucination 패턴을 candidate-level(NLL / confidence margin paired win-rate 0.36 / 0.34) 과 corpus-level(entity-pair co-occurrence paired win-rate 0.551) 에서 동시에 재현하였다. 저자들이 검토한 범위에서, entity frequency / co-occurrence 를 직접 탐지 신호가 아닌 *조건 축*으로 재해석하고 CHOKE 패턴을 candidate-level 과 corpus-level 두 scope 에서 동시 정량 검증한 연구는 보고되지 않았다. 다만 Zhang 등(2025)은 독립적으로 pretraining corpus 의 n-gram 통계가 환각 탐지에 보완 신호를 제공함을 보였으며, 본 연구와의 구체적 차이는 §<a href="#ch:related" data-reference-type="ref" data-reference="ch:related">[ch:related]</a>에 정리하였다.

2.  **Conditional reliability 분석 절차의 통계적 검증**. corpus support bin마다 어떤 신호가 안정적으로 작동하는지 분리 평가하는 절차를 제안하고, corpus-bin weighted fusion이 선형 global fusion 대비 AUROC 0.166 \[+0.157, +0.176\] 의 유의미한 우위를 보임을 95% prompt-grouped bootstrap으로 검증하였다. 전체 AUROC에서는 비선형 gradient boosting과 동급이나, calibration(ECE 4.7% vs 12.1%) 측면에서 corpus 조건부 구조의 독립적 가치를 확인하였다.

3.  **평가 단위 분리의 명시적 적용**. SE는 prompt-level broadcast, Semantic Energy는 cluster $`\to`$ prompt 단위 collapse, logit diagnostic은 candidate-level 신호임을 본문, 표, 그림 전체에 일관되게 적용하여, 단위 혼동에서 비롯된 잘못된 비교를 차단하였다.

## 한계

본 연구는 다음 한계를 가진다.

1.  **단일 corpus snapshot 의존**. corpus support 축은 `v4_dolmasample_olmo`(16B Dolma sample tokens, OLMo tokenizer) 단일 index 위에서 산출되므로, 규모나 도메인이 다른 corpus index에서는 entity frequency 분포와 zero co-occurrence 비율이 달라질 수 있다.

2.  **Corpus 신호의 proxy 성격**. corpus-grounded 신호는 외부 근거 가능성을 나타내는 proxy일 뿐, 모델의 내부 지식을 직접 증명하지 않는다.

3.  **단일 모델 평가**. 본 실험은 Qwen2.5-3B 한 종류의 causal LM에서만 수행되었고, 모델 규모나 계열에 따른 일반화는 본 논문의 범위를 넘는다.

4.  **모델 내부 상태 해석 불가**. SE, Semantic Energy, corpus feature 모두 출력에서 계산한 관측 지표이며, 모델의 의도나 내부 원인을 직접 추론하지는 않는다.

5.  **Corpus support 구간 간 표본 불균형**. decile 30–40 (n=3,465) 과 decile 90–100 (n=125) 의 표본 수 차이가 약 28배에 달한다. 종합 AUROC는 표본이 많은 중간 decile 의 성능에 더 크게 영향을 받으므로, 희귀 구간(decile 00–10, 90–100) 의 신뢰도 추정치는 통계적 불확실성이 크다. 본 논문이 “sweet spot” 으로 보고한 decile 30–50 영역도 전체의 약 44% 를 차지하기 때문에, 종합 결과가 이 영역에 편향될 수 있음을 함께 명시한다. 희귀 corpus support 구간에서의 결론은 추가 표본 확보 후 재검증이 필요하다.

## 향후 연구

향후 다음 방향을 검토할 수 있다.

1.  **여러 corpus snapshot 비교**. `v4_dolmasample_olmo` 외에 더 큰 Infini-gram index(예: `v4_pileval_llama`)나 도메인 특화 corpus에서 corpus support bin 분포가 어떻게 달라지는지 비교한다.

2.  **모델 일반화**. 동일 평가 절차를 Llama, Mistral 계열 등 다른 causal LM에 적용해 SE와 Semantic Energy의 corpus-bin reliability가 모델에 의존하는지 본다.

3.  **Conditional fusion 구조 확장**. 본 논문의 corpus-bin weighted fusion 외에 axis interaction term, monotonic fusion, hierarchical fusion으로 확장한다.

4.  **Selective prediction 평가**. prompt-level abstention과 selective-risk 관점을 candidate-row 평가와 분리해 검증한다.

<div class="thebibliography">

99

Huang et al.,
“A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions,”
*ACM Transactions on Information Systems*, vol. 43, no. 2, pp. 1–55, 2025.

J. Maynez et al.,
“On Faithfulness and Factuality in Abstractive Summarization,”
*Proceedings of ACL*, pp. 1906–1919, 2020.

S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal,
“Detecting hallucinations in large language models using semantic entropy,”
*Nature*, vol. 630, pp. 625–630, 2024.

P. Manakul, A. Liusie, and M. J. F. Gales,
“SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models,”
*Proceedings of EMNLP*, pp. 9004–9017, 2023.

S. Min et al.,
“FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation,”
*Proceedings of EMNLP*, pp. 12076–12100, 2023.

A. Ravichander, S. Ghela, D. Wadden, and Y. Choi,
“HALoGEN: Fantastic LLM Hallucinations and Where to Find Them,”
*Proceedings of ACL*, pp. 1402–1425, 2025.

Z. Ma et al.,
“Semantic Energy: A novel approach for detecting confabulation in language models,”
*arXiv preprint arXiv:2412.07965*, 2025.

S. Lin, J. Hilton, and O. Evans,
“TruthfulQA: Measuring how models mimic human falsehoods,”
*Proceedings of ACL*, 2022.

Z. Qiu et al.,
“QuCo-RAG: Query-aware Corpus Grounding for Retrieval-Augmented Generation,”
*arXiv preprint arXiv:2512.19134*, 2025.

J. Phillips et al.,
“Learning to Predict When Language Models Are Wrong,”
*arXiv preprint arXiv:2603.21172*, 2026.

A. Simhi et al.,
“Trust Me, I’m Wrong: High-Certainty Hallucinations in Language Models (CHOKE),”
*arXiv preprint*, 2025.

J. Li, X. Cheng, W. X. Zhao, J.-Y. Nie, and J.-R. Wen,
“HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models,”
*Proceedings of EMNLP*, pp. 6449–6464, 2023.

J. Liu et al.,
“Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens,”
*arXiv preprint arXiv:2401.17377*, 2024.

W. Zhao et al.,
“WildHallucinations: Evaluating Long-form Factuality in LLMs with Real-World Entity Queries,”
*arXiv preprint arXiv:2407.17468*, 2024.

S. Valentin et al.,
“Cost-Effective Hallucination Detection for LLMs,”
*arXiv preprint arXiv:2407.21424*, 2024.

Y. Zhang et al.,
“Probing Hallucination via Pretraining-corpus Coverage: A Suffix-Array N-gram Study,”
*arXiv preprint arXiv:2511.17946*, 2025.

</div>
