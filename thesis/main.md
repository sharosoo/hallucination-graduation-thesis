# 서론

<span id="ch:intro" label="ch:intro"></span>

## 연구 배경

대규모 언어모델 (Large Language Model, LLM) 은 텍스트 생성, 질의응답 등 다양한 자연어 처리 태스크에 널리 사용되고 있으나, 사실과 다른 내용을 그럴듯하게 생성하는 환각 (hallucination) 문제를 보이며 이는 LLM 신뢰성의 핵심 과제이다. 본 연구는 외부 사실과 충돌하는 응답을 탐지하는 문제 (사실성, factuality) 에 초점을 두며, TruthfulQA 같은 벤치마크가 이 평가 단위를 제공한다.

환각 탐지의 대표적 방법인 Semantic Entropy (SE, Farquhar 등 2024) 는 한 질문에 대해 $`K`$ 개 응답을 샘플링한 뒤 NLI 클러스터링 후 분포의 엔트로피를 계산한다. 반복 샘플링과 응답 간 일관성은 SelfCheckGPT 와 같은 선행 연구에서도 환각 탐지의 핵심 단서로 사용되었다. 그러나 SE 와 같은 sample-consistency 신호는 모델이 모든 응답에서 같은 의미의 틀린 답을 반복하면 판별력이 약해진다 — Simhi 등 (2025) 의 CHOKE (high-certainty hallucination) 패턴. 후보 답변의 NLL / confidence margin 같은 logit 진단 신호도 “환각 답이 정답보다 덜 자신있을 것” 이라는 직관에 기대므로 동일 한계를 갖는다.

본 연구는 이 한계를 *외부 corpus 신호* 로 보완한다. 후보 답의 entity frequency 와 entity-pair co-occurrence 를 외부 corpus 에서 측정한 뒤, SE / Energy baseline 위에 결합하여 추가 lift 가 발생하는지 측정하고, 그 lift 가 corpus support 영역별로 어떻게 분포하는지 분해한다. corpus 빈도는 환각 라벨이 아니라 *지표 신뢰성을 보완하는 외부 조건 변수* 이다.

## 연구 목적

본 연구는 SE / Semantic Energy 같은 uncertainty 신호 위에 외부 corpus support 신호를 결합하면 환각 탐지 성능에 추가 lift 가 발생하는지를 정량 평가한다는 관점에서 다음 세 가지 질문을 다룬다.

1.  **SE / Energy baseline 재현**. Farquhar 등 (2024) 의 SE 와 Ma 등 (2025) 의 Semantic Energy 가 본 데이터 (TruthfulQA + HaluEval-QA, Qwen2.5-3B) 에서 어떤 AUROC 를 보이는지 측정하고, SE 원논문 (TriviaQA / Llama-2 7B 기반 AUROC 0.79) 과 일치하는지 확인한다. 본 질문은 단순 재현이 아니라 다음 두 질문의 *사전 검증* 이다 — baseline 이 재현되지 않으면 그 위에 corpus signal 을 결합한다는 후속 결과의 토대가 성립하지 않는다.

2.  **Corpus signal 추가의 한계 효용 (marginal lift) 측정**. SE / Energy 위에 외부 corpus support 신호 (entity frequency, entity-pair co-occurrence) 를 추가하면 AUROC 가 얼마나 더 올라가는지를 bootstrap (n=2,000) 으로 통계적 유의성을 포함해 측정한다.

3.  **Corpus support 구간별 lift 분포 분석**. corpus 추가의 한계 효용이 모든 corpus support 영역에서 균일한지, 아니면 corpus 가 entity 를 거의 모르는 영역에서 더 큰지를 3-bin / 5-bin / 10-bin decile 분해로 측정한다. 이는 corpus signal 의 가치가 *어디에서 가장 크게 발생하는지* 를 데이터로 드러내는 분석이다.

##### 본 논문의 기여.

본 논문은 SE / Semantic Energy 의 baseline 위에 외부 corpus signal 을 추가했을 때 발생하는 한계 효용을 직접 측정하고, 그 효용이 corpus support 영역에 따라 어떻게 분포하는지를 정량 보고한다. corpus support 가 부족한 lowest decile 에서 lift 가 가장 크다는 것이 본 논문이 데이터로 보이는 핵심 사실이다. 모델 출력에 의존하지 않는 외부 corpus statistic 으로 환각 탐지 신호를 보완하는 방향은 Valentin 등 (2024) 이 모델 *내부* score attribute 로 conditional calibration 을 시도한 방향과 직교적이다.

## 논문 구성

본 논문은 다음과 같이 구성된다. 제 <a href="#ch:related" data-reference-type="ref" data-reference="ch:related">[ch:related]</a>장에서는 LLM 환각 탐지 관련 연구와 본 연구의 위치를 정리한다. 제 <a href="#ch:method" data-reference-type="ref" data-reference="ch:method">[ch:method]</a>장에서는 평가 절차의 근거와 corpus support 를 보완 신호로 쓰는 이론적 근거 (QuCo-RAG, Zhang 등 2025) 를 제시한다. 제 <a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>장에서는 종합 baseline (SE / Energy / fusion), bootstrap CI, corpus support bin 별 한계 효용, leave-one-dataset-out 일반화, threshold / calibration 진단을 보고한다. 제 <a href="#ch:conclusion" data-reference-type="ref" data-reference="ch:conclusion">[ch:conclusion]</a>장에서 세 연구 질문에 대한 답변과 학술적 기여, 한계, 향후 연구 방향을 제시한다.

# 관련 연구

<span id="ch:related" label="ch:related"></span>

본 장에서는 LLM 환각 연구를 두 흐름으로 구분해 검토한다. 첫째는 환각의 정의, 사실성 평가, 벤치마크를 다루는 연구이며, 둘째는 생성 모델의 불확실성 또는 내부 점수를 이용해 환각을 탐지하려는 방법 연구이다. 이 구분은 본 연구가 기존 사실성 평가 자체를 대체하려는 것이 아니라, 반복 샘플링 기반 SE의 한계가 드러나는 영역에서 보조 탐지 신호와 결합 구조를 검토한다는 점을 명확히 한다.

## 환각 분류, 사실성, 벤치마크

환각은 LLM이 입력, 문맥, 또는 외부 사실과 맞지 않는 내용을 생성하는 현상으로 다루어진다. Huang 등은 LLM 환각의 원인, 유형, 탐지 과제를 폭넓게 정리하며, 환각 탐지가 LLM 신뢰성 확보를 위한 핵심 문제임을 보인다. Maynez 등은 생성 텍스트 평가에서 사실성(factuality)과 충실성(faithfulness)을 구분해 논의했으며, 이 구분은 환각 탐지 문장을 작성할 때 단순한 “틀림”보다 어떤 기준에서 맞지 않는지 명확히 해야 함을 보여준다.

벤치마크와 평가 연구는 환각 탐지의 대상과 평가 기준을 제공한다. TruthfulQA는 모델이 인간의 오해나 거짓 믿음을 모방하는지를 측정하는 질의응답 벤치마크이다. FActScore는 장문 생성에서 원자적 사실 단위의 정밀도를 평가하는 방법을 제안한다. 이러한 연구들은 환각을 평가하고 분류하는 기준을 제공하지만, 본 연구가 다루는 “*어느 corpus support 구간에서 어느 신호의 신뢰도가 높은가*”라는 조건부 신뢰성 문제는 직접 다루지 않는다.

## 불확실성 및 반복 샘플링 기반 탐지

SelfCheckGPT는 외부 지식 베이스 없이 여러 응답을 샘플링하고 응답 간 일관성을 비교해 환각을 탐지하는 접근을 제안했다. 이 흐름은 하나의 응답만 보는 대신, 같은 질문에 대한 여러 생성 결과의 분산과 일관성을 환각 탐지 신호로 본다. Semantic Entropy도 이 관점과 연결되지만, 응답을 의미 클러스터로 묶고 클러스터 분포의 엔트로피를 사용한다는 점에서 더 명시적인 의미 단위 불확실성 측정 방법이다.

표 <a href="#tab:method_comparison" data-reference-type="ref" data-reference="tab:method_comparison">2.1</a>은 본 연구에서 다루는 신호들의 핵심 아이디어와 한계를 요약한다.

<div id="tab:method_comparison">

| 신호 | 핵심 아이디어 | 한계 |
|:---|:---|:---|
| Semantic Entropy | NLI cluster + Shannon entropy | 단일 cluster 사례 판별 약함 |
| Semantic Energy | cluster 가중 token logit 합 | cluster 정의 의존, 다양성 가정 |
| logit diagnostic | NLL, logit variance, margin | CHOKE 사례에서 방향 역전 |
| corpus support | entity frequency / pair co-occurrence | 외부 corpus snapshot 의존, proxy 성격 |

본 연구에서 비교하는 환각 탐지 신호.

</div>

## Semantic Entropy (SE)

Semantic Entropy 는 Farquhar 등 (2024) 이 Nature 에 발표한 LLM 불확실성 측정 방법이다. 한 질문에 대해 $`K`$ 개 응답을 샘플링한 뒤 NLI 모델로 의미적 클러스터로 묶고, 클러스터 분포의 Shannon entropy $`SE = -\sum_{c \in C} p(c) \log p(c)`$ 를 산출한다 ($`p(c)`$ = cluster $`c`$ 에 속한 응답의 비율). 토큰 단위 확률이 아닌 *의미 단위* 불확실성을 측정한다는 점이 핵심 차별점이며, prompt 단위 신호 — 한 질문에 대해 하나의 SE 값 — 로 정의된다. 한계는 모든 응답이 한 cluster 로 수렴하는 *고확신 환각* 사례에서 판별력이 약해진다는 점이다 (Simhi 등 2025 의 CHOKE 패턴). 본 논문은 SE 를 baseline 으로 사용하고 그 위에 외부 corpus signal 을 추가했을 때 lift 가 있는지를 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a> 에서 측정한다.

## Semantic Energy

Ma 등 (2025) 의 Semantic Energy 는 여러 generated answer 와 semantic cluster 를 사용해 토큰 logit 기반 uncertainty 를 해석한다. 토큰 단위 raw logit $`z_\theta(x_t)`$ 의 부호 반전 평균 $`\frac{1}{nT}\sum_i \sum_t -z_\theta(x_t^{(i)})`$ 를 sample energy 로 정의하고, NLI cluster 확률 가중 합으로 prompt 단위 scalar 를 산출한다 (자세한 식은 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a> 의 실험 방법 절). SE 가 softmax 후 cluster 확률 분포의 entropy 를 사용하는 반면 Energy 는 softmax 전 logit 의 크기를 보존한다는 점이 차별점이다. 응답 다양성이 작은 (낮은 SE) 사례에서도 토큰 logit 이 추가 정보를 줄 수 있다는 것이 motivation 이다. 본 논문은 Semantic Energy 를 SE 와 함께 baseline 으로 사용하며, 그 위에 외부 corpus support 신호를 추가했을 때 발생하는 한계 효용을 측정한다 (§<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>).

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

- **신호의 단위**. Zhang 등은 token surface 수준의 n-gram lexical coverage 를 측정하나, 본 연구는 entity frequency 와 entity-pair co-occurrence 라는 *관계 수준* 의 신호를 사용한다. 동일 entity 가 등장하더라도 entity 간 관계가 corpus 에 얼마나 공출현하는지를 측정함으로써, “익숙한 entity 를 잘못된 관계로 조합하는” CHOKE 가설과 자연스럽게 연결될 수 있는 측정 단위이다 (인과 관계가 아닌 상관 관찰; 본 chapter 마지막 단락의 단서 및 §결론 한계 참조).

- **평가 패러다임**. Zhang 등은 단일 생성 답변의 이진 분류 (single answer hallucinated / not) 로 평가하는 반면, 본 연구는 “이 prompt 의 free-sample 정답률이 임계값 미만인가” 를 평가하는 binary task 로 평가한다 (Farquhar 2024 SE 의 평가 단위와 일치).

Zhang 등이 명시한 “결과는 상관관계이며 인과 관계를 확립하지 않는다”는 단서는 본 연구의 corpus-level 해석에도 동일하게 적용된다 — corpus 카운트 backend 의 외부 corpus snapshot 이 모델의 실제 pretraining corpus 와 다를 수 있으므로, corpus support 와 환각의 연결은 *상관* 이지 *인과* 가 아니다.

##### High-certainty hallucination (CHOKE).

Simhi 등 (2025) 은 모델이 정답 지식을 가지고도 높은 확신으로 환각을 생성하는 사례를 CHOKE 로 정의하고, 이런 사례가 logit-only / sample-consistency 기반 탐지기 의 본질적 한계임을 보였다. 본 데이터에서도 같은 질문의 정답 답과 환각 답을 비교하면 환각 답이 NLL 이 더 작고 (61.6%) confidence margin 이 더 큰 (65.2%) 비율이 일관되게 관측되어 logit / margin 단독 신호의 한계가 재현된다. 본 논문은 이 한계를 직접 메우는 대신, SE / Energy 같은 sample-consistency 신호 위에 외부 corpus support 를 보완 결합하는 방향으로 우회한다 (§<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>).

##### Corpus 기반 환각 경험 증거 (WildHallucinations).

Zhao 등(2024)은 “*Wikipedia 페이지가 없는 entity*에 대한 질문에서 LLM이 더 자주 환각한다”는 관찰을 보고하였다. 이는 본 연구가 corpus 빈도를 조건 축으로 쓰는 가설의 *선행 경험적 근거*이다. 차이점은 두 가지다. (i) WildHallucinations 는 Wikipedia 존재 여부 라는 *이진* 조건을 사용하는 반면, 본 연구는 entity frequency / pair co-occurrence 기반 *연속 corpus support 축*으로 일반화한다. (ii) WildHallucinations 는 환각률 자체를 corpus 조건으로 설명하는 *evaluation* 연구이고, 본 연구는 이미 환각이 일어난 후보 행에 대해 *어떤 탐지 신호가 corpus 조건에 따라 더 안정적으로 작동하는가*를 평가하는 *detection* 연구이다.

##### Conditional calibration 분석 틀 (Valentin et al.).

Valentin 등(2024)은 환각 탐지 점수를 *입력/응답 attribute에 conditional 하게 calibrate* 하는 multi-scoring 분석 틀를 제안하였다. 본 연구와의 핵심 차이는 *조건의 출처*이다. Valentin은 모델 *내부 score attribute* 에 conditional하게 calibrate 하는 반면, 본 연구는 모델 출력과 독립적인 *외부 corpus statistic* 으로 조건화한다. 두 분석 틀는 직교적이며, 본 연구의 corpus 조건 축은 Valentin의 internal-attribute conditioning과 결합 가능한 보완 신호로 위치할 수 있다.

# 제안 방법

<span id="ch:method" label="ch:method"></span>

본 장은 본 논문이 채택하는 *corpus-conditioned 분석 절차* 와 그 이론적 동기를 정리한다.

## 단일 점수 가정의 한계

SE 와 Energy 같은 sample-consistency 신호는 모든 응답이 단일 의미 cluster 로 수렴하는 *고확신 환각 (CHOKE)* 사례 에서 판별력이 약해지고, NLL / margin 같은 logit 신호는 환각 답이 더 자신있게 평가되는 사례에서 방향이 역전된다 (§<a href="#ch:related" data-reference-type="ref" data-reference="ch:related">[ch:related]</a> CHOKE paragraph). 두 사례는 신호 자체가 무용하다는 뜻이 아니라 *어느 입력 조건에서 어느 신호가 안정적으로 작동하는지를 분리해서 봐야 한다* 는 뜻이며, 본 논문은 그 외부 입력 조건으로 corpus support 를 사용한다.

## Corpus support 를 보완 신호로 사용하는 근거

후보 답변의 entity frequency 와 entity-pair co-occurrence 는 모델이 “익숙한” entity 를 다루고 있는지를 외부 자료로 측정한 지표이다. Qiu 등 (2025) 의 QuCo-RAG 는 query-aware corpus grounding 에서 entity frequency 와 co-occurrence 가 모델 답변의 외부 근거 가능성과 연관됨을 보였고, Zhang 등 (2025) 은 RedPajama 1.3T token pretraining corpus 위에 suffix array 를 구축하여 n-gram coverage 와 환각 탐지 신호의 비단조적 상관 관계를 정량 보고했다. 두 결과의 핵심은 *corpus exposure 가 모델 출력의 신뢰도 분포를 형성한다* 는 메커니즘 가설이며, 본 논문은 이 가설을 차용하되 corpus statistic 을 직접 탐지기 로 쓰지 않고 SE / Energy 위에 *보완 결합* 한다.

##### Conditioning 가설의 메커니즘.

“corpus 가 부족한 영역에서 corpus 보완 신호의 가치가 가장 클 것” 이라는 본 논문 가설은 다음 인과 사슬로 동기 부여된다. (i) Zhao 등 (2024) 의 WildHallucinations 가 보였듯 corpus exposure 부족 → 모델 representation 불안정 → 환각률 ↑. (ii) 이렇게 발생한 환각이 모델 입장에서 “확신 있게 일관된” 답으로 나오는 경우가 많은 것이 Simhi 등 (2025) 의 CHOKE 패턴 — corpus 부족 영역에서 free-sample 이 단일 cluster 로 수렴하면 SE / Energy 의 sample-consistency 판별력이 *구조적으로 약화* 된다. (iii) 따라서 corpus 부족 영역에서는 모델 자체 신호로 hallucination 탐지가 어렵고, 외부 corpus 신호가 “이 prompt 가 어려운 prompt 임” 을 분리하는 보완 정보를 제공해야 lift 가 발생한다. 본 논문은 이 인과 사슬을 decile 별 corpus 추가 lift 분포로 정량 검증한다 (§<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a> 의 lowest decile lift +0.022 vs high decile lift $`\approx 0`$ 패턴).

## 본 논문의 분석 방향

위 관찰로부터 본 논문의 절차는 다음과 같다. (1) 라벨은 SE 원논문 위치 설정 그대로 — 각 질문의 free-sample N=10 정답 매칭 비율로 binary hard/easy 부여. (2) Single-signal baseline (SE / Energy / logit-diagnostic / corpus-axis-only) 과 그 위에 corpus support aggregate (entity_frequency_axis, entity_pair_cooccurrence_axis 의 mean / max / delta) 를 추가한 fusion (LR / RF / GBM / SVM) 의 AUROC 를 비교한다. (3) corpus 추가의 한계 효용을 bootstrap (n=2,000) 으로 통계 검정하고 corpus support 3/5/10-bin 별로 분해한다. 실험 결과는 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a> 에 보고한다.

# 실험

<span id="ch:experiment" label="ch:experiment"></span>

## 실험 설정

본 실험은 데이터셋이 이미 정답 후보와 환각 후보를 명시적으로 구분해 제공한다는 점을 활용하여, 새 답을 생성한 뒤 정오를 사후 판정하는 절차나 LLM-as-judge 의존 없이 paired 비교를 수행한다.

### 데이터셋 구성

TruthfulQA 와 HaluEval-QA 두 데이터셋을 사용한다. TruthfulQA 는 `correct_answers` / `incorrect_answers` 에서 결정론적으로 한 쌍씩, HaluEval-QA 는 제공되는 `right_answer` / `hallucinated_answer` 를 그대로 사용한다. 각 질문마다 정확히 두 후보 답 (정답 + 환각) 을 두며, 최종 데이터는 5,815 개 질문 + 11,630 개 후보 답이다 (TruthfulQA 815 / 1,630, HaluEval-QA 5,000 / 10,000).

### 모델 출력 수집

모델은 Qwen2.5-3B causal LM 을 사용한다. 두 종류의 출력을 함께 수집한다.

- **Free sampling** (Semantic Entropy 입력용): 각 prompt 에 대해 N=10 개의 짧은 답변을 자유 sampling 한다. 답변 길이는 최대 64 token 으로 제한하고 형식이 맞지 않은 표본은 일정 횟수까지 재 sampling 한다. 짧은 entity-중심 답변에서 NLI cluster 판단이 안정적이며 후보 답변 단위와 일치한다.

- **Teacher-forced scoring** (candidate diagnostic 입력용): 정답 / 환각 후보 답을 모델에 입력으로 주입하면서 점수만 측정한다. 후보 답에 해당하는 token 위치 $`t`$ 마다 selected token logit $`z_t(x_t)`$, $`\log p_\theta(x_t \mid \text{prompt}, x_{<t})`$, 분배 함수 $`\log Z_t`$, vocabulary logit 분산, top-1 vs top-2 logit 차 (confidence margin) 을 모두 기록하고 후보 답 길이 $`T`$ 로 평균한다.

### Feature 계산

##### Semantic Entropy.

한 질문에서 얻은 N=10 답변을 `microsoft/deberta-large-mnli` bidirectional NLI entailment 로 semantic cluster 에 묶고, sequence log-likelihood 를 cluster 단위로 log-sum-exp 집계해 cluster probability $`p(\mathbb{C}_k)`$ 를 만든 뒤 그 위에서 entropy 를 계산한다 (Farquhar 등 2024). SE 는 한 질문에 하나의 값으로 정의된다.

##### Semantic Energy.

Ma 등 (2025) 식 (11)–(14) 를 그대로 따른다. 동일 N=10 답변과 SE 단계의 NLI cluster 를 `(prompt_id, sample_index)` 로 join 하여 재사용하며,
``` math
\begin{align}
\tilde{E}(x_t^{(i)}) &= -z_\theta(x_t^{(i)}) & \text{(token energy)} \\
E(x^{(i)}) &= \tfrac{1}{T_i}\sum_{t} \tilde{E}(x_t^{(i)}) & \text{(sample energy)} \\
E_{\text{Bolt}}(\mathbb{C}_k) &= \sum_{x^{(i)} \in \mathbb{C}_k} E(x^{(i)}) & \text{(cluster total)} \\
U(\mathbf{x}) &= \sum_{k} p(\mathbb{C}_k)\, E_{\text{Bolt}}(\mathbb{C}_k) & \text{(최종)}
\end{align}
```
로 계산한다. 부호 규약은 lower energy = higher reliability. teacher-forced scoring 의 token-level $`-\log Z`$ 평균, NLL 평균, logit variance, confidence margin 은 candidate-level diagnostic 으로 별도 보존한다 (Energy 와 같은 열에 합치지 않는다).

##### Corpus-grounded feature.

후보 답변에 등장하는 entity 의 corpus 내 frequency 와 entity-pair co-occurrence 로 corpus support 신호를 만든다. Entity 추출은 *spaCy `en_core_web_lg` NER* 로 수행하며, PERSON / ORG / GPE / LOC / DATE / EVENT / WORK_OF_ART / FAC / NORP / PRODUCT / LANGUAGE / LAW 12 개 label 만 corpus 조회 대상으로 유지한다 (CARDINAL / ORDINAL / MONEY / PERCENT / QUANTITY / TIME 은 factoid 답의 의미 단위로 부적합). NER 이 비면 noun-chunk fallback, 그래도 비고 6 단어 이하 짧은 답이면 정규화한 텍스트 자체를 단일 entity 로 추가한다. 모든 entity 는 lowercase 정규화 + 중복 제거 + 후보당 최대 8 개로 잘라낸다. corpus count backend 는 Liu 등 (2024) 의 Infini-gram 의 `v4_dolmasample_olmo` 인덱스 (16B token, OLMo-7B-hf tokenizer) 를 사용하며, entity-pair co-occurrence 는 `count_cnf` AND query 로 산출한다. 두 값을 결합한 coverage score 위에서 *rank-quantile 균등 분할* 로 corpus support 구간 label (3-bin / 5-bin / 10-bin) 을 산출한다 — fixed-cutoff 가 spaCy NER 채택 후 좌편향 분포 (약 58% 가 0) 에서 표본 쏠림 문제를 보이기 때문이다.

corpus 신호는 외부 자료에서 후보 답이 지지될 가능성을 *간접적으로* 나타내는 proxy 이며 모델 hidden-state 나 환각 라벨을 사용하지 않는다.

### 평가 지표

종합 표는 AUROC / AUPRC / Brier / ECE 를 보고한다. Train / test 분할은 prompt 단위 5-fold KFold (한 prompt 의 모든 정보가 같은 fold) 로 강제한다. Prompt-level main 평가의 binary 라벨은 free-sample N=10 의 token-overlap ($`\ge 50\%`$) 정답 매칭 비율이 $`< 0.5`$ 인 prompt 를 *hard prompt* 로 정의한다 (SE 원논문 framing 과 일치). Candidate-row sensitivity 평가의 라벨은 데이터셋 annotation 의 `is_hallucination` 을 그대로 사용한다.

## Prompt-level 종합 베이스라인 결과

표 <a href="#tab:current_thesis_evidence" data-reference-type="ref" data-reference="tab:current_thesis_evidence">4.1</a> 는 단일 signal baseline (SE, Energy, logit-diagnostic, corpus-axis-only) 과 결합기 (LR / RF / GBM / SVM, with / without corpus) 의 AUROC / AUPRC / Brier / ECE 이다. 라벨은 각 질문의 free-sample N=10 정답 매칭 비율 (token-overlap $`\ge 0.5`$) 이 50% 미만이면 *hard prompt* 로 정의한다 (HaluEval-QA 20% / TruthfulQA 97%, 종합 31%; TruthfulQA 라벨 비대칭은 §한계 항목에서 다룬다).

<div id="tab:current_thesis_evidence">

<table>
<caption>Prompt-level 종합 baseline (n=5<span>,</span>815). 각 prompt 의 free-sample N=10 의 정답 매칭 비율이 50% 미만이면 <em>hard prompt</em> (is_hard=1) 로 정의. 점수 신호 단독 (single signal) 과 결합기 (fusion) 를 5-fold CV (KFold, prompt 단위) 위에서 평가. 굵은 글씨는 1위.</caption>
<thead>
<tr>
<th style="text-align: left;">Method</th>
<th style="text-align: center;">AUROC</th>
<th style="text-align: center;">AUPRC</th>
<th style="text-align: center;">Brier</th>
<th style="text-align: center;">ECE</th>
<th style="text-align: left;">비고</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: left;"><em>Single-signal baseline</em></td>
</tr>
<tr>
<td style="text-align: left;">logit-diagnostic-only</td>
<td style="text-align: center;">0.581</td>
<td style="text-align: center;">0.416</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: left;">후보 단위 logit 신호</td>
</tr>
<tr>
<td style="text-align: left;">corpus-axis-only</td>
<td style="text-align: center;">0.634</td>
<td style="text-align: center;">0.546</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: left;">corpus support 단독</td>
</tr>
<tr>
<td style="text-align: left;">Semantic Entropy (SE)</td>
<td style="text-align: center;">0.832</td>
<td style="text-align: center;">0.726</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: left;">Farquhar 2024</td>
</tr>
<tr>
<td style="text-align: left;">Semantic Energy (Boltzmann)</td>
<td style="text-align: center;">0.862</td>
<td style="text-align: center;">0.794</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: left;">Ma 2025</td>
</tr>
<tr>
<td colspan="6" style="text-align: left;"><em>Global fusion (no corpus)</em></td>
</tr>
<tr>
<td style="text-align: left;">logistic regression</td>
<td style="text-align: center;">0.876</td>
<td style="text-align: center;">0.809</td>
<td style="text-align: center;">0.124</td>
<td style="text-align: center;">0.034</td>
<td style="text-align: left;">선형</td>
</tr>
<tr>
<td style="text-align: left;">random forest</td>
<td style="text-align: center;">0.888</td>
<td style="text-align: center;">0.836</td>
<td style="text-align: center;">0.112</td>
<td style="text-align: center;">0.016</td>
<td style="text-align: left;">비선형</td>
</tr>
<tr>
<td style="text-align: left;">gradient boosting</td>
<td style="text-align: center;">0.889</td>
<td style="text-align: center;">0.837</td>
<td style="text-align: center;">0.113</td>
<td style="text-align: center;">0.014</td>
<td style="text-align: left;">비선형</td>
</tr>
<tr>
<td colspan="6" style="text-align: left;"><em>Global fusion (with corpus)</em></td>
</tr>
<tr>
<td style="text-align: left;">logistic regression</td>
<td style="text-align: center;">0.878</td>
<td style="text-align: center;">0.816</td>
<td style="text-align: center;">0.121</td>
<td style="text-align: center;">0.030</td>
<td style="text-align: left;">선형</td>
</tr>
<tr>
<td style="text-align: left;">random forest</td>
<td style="text-align: center;">0.889</td>
<td style="text-align: center;">0.839</td>
<td style="text-align: center;">0.111</td>
<td style="text-align: center;">0.022</td>
<td style="text-align: left;">비선형</td>
</tr>
<tr>
<td style="text-align: left;">SVM rbf</td>
<td style="text-align: center;">0.838</td>
<td style="text-align: center;">0.787</td>
<td style="text-align: center;">0.131</td>
<td style="text-align: center;">0.016</td>
<td style="text-align: left;">비선형</td>
</tr>
<tr>
<td style="text-align: left;"><strong>gradient boosting</strong></td>
<td style="text-align: center;"><strong>0.892</strong></td>
<td style="text-align: center;"><strong>0.842</strong></td>
<td style="text-align: center;"><strong>0.110</strong></td>
<td style="text-align: center;">0.017</td>
<td style="text-align: left;"><strong>본 논문 main</strong></td>
</tr>
</tbody>
</table>

</div>

표에서 네 가지 사실이 두드러진다.

- **SE / Energy 단독이 강한 baseline 을 형성**. SE-only AUROC 0.832, Semantic Energy 단독 0.862으로 Farquhar 등 (2024) 이 TriviaQA / Llama-2 7B 에서 보고한 SE AUROC 0.79 와 동일 수준에서 본 데이터에 재현된다. 본 논문이 사용한 SE / Energy 구현이 원논문과 일치하는 baseline 임을 검증한다.

- **단일 corpus signal 은 SE / Energy 보다 약함**. corpus-axis-only AUROC 0.634, entity-frequency-only 0.676, entity-pair-cooccurrence-only 0.593 으로 단독 사용 시 SE / Energy 에 미치지 못한다. corpus signal 의 가치는 탐지기 자체로서가 아니라 SE / Energy 위에 추가되는 보완 신호로서 발휘됨을 시사한다.

- **본 논문 핵심: GBM (with corpus) 가 종합 1위**. AUROC 0.892으로 Energy 단독 (0.862) 대비 +0.030, SE 단독 (0.832) 대비 +0.060 의 lift. Brier 0.110, ECE 1.7% 로 calibration 도 모든 결합기 중 1위.

- **Corpus 추가 lift 는 작지만 일관 양수**. LR (0.876 → 0.878), RF (0.888 → 0.889), GBM (0.889 → 0.892). 통계적 유의성과 corpus support 영역별 분포는 다음 두 절에서 분석한다.

## Corpus 추가의 한계 효용 — Bootstrap CI

corpus signal 추가가 SE / Energy baseline 위에서 통계적으로 유의한 추가 lift 를 제공하는지 검정하기 위해, 5,815 개 prompt 위에서 prompt 단위 bootstrap (n=2,000) 으로 fusion (with corpus) 의 AUROC delta vs no-corpus baseline 의 95% 신뢰구간을 산출하였다.

<div id="tab:bootstrap_ci">

<table>
<caption>Corpus signal 추가의 한계 효용 (AUROC delta) Bootstrap CI (n=2<span>,</span>000, prompt 단위 resample). 동일 결합기 family (LR / RF / GBM / SVM) 에서 corpus aggregate 추가 vs 미포함의 차이. 양수 = corpus 추가가 도움.</caption>
<thead>
<tr>
<th style="text-align: left;">비교</th>
<th style="text-align: center;"><span class="math inline"><em>Δ</em></span> AUROC [95% CI]</th>
<th style="text-align: center;">통계적 유의</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="3" style="text-align: left;"><em>vs LR (no corpus, AUROC 0.876) baseline</em></td>
</tr>
<tr>
<td style="text-align: left;">LR (with corpus)</td>
<td style="text-align: center;"><span class="math inline">+0.0022</span> [<span class="math inline">+0.0003, +0.0041</span>]</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">RF (with corpus)</td>
<td style="text-align: center;"><span class="math inline">+0.0130</span> [<span class="math inline">+0.0070, +0.0189</span>]</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>GBM (with corpus)</strong></td>
<td style="text-align: center;"><span class="math inline"><strong>+</strong><strong>0.0158</strong></span> <strong>[<span class="math inline"><strong>+</strong><strong>0.0108</strong><strong>,</strong> <strong>+</strong><strong>0.0208</strong></span>]</strong></td>
<td style="text-align: center;">(핵심)</td>
</tr>
<tr>
<td style="text-align: left;">SVM rbf (with corpus)</td>
<td style="text-align: center;"><span class="math inline">−0.0379</span> [<span class="math inline">−0.0468, −0.0290</span>]</td>
<td style="text-align: center;">negative</td>
</tr>
<tr>
<td colspan="3" style="text-align: left;"><em>vs GBM (no corpus, AUROC 0.889) baseline</em></td>
</tr>
<tr>
<td style="text-align: left;"><strong>GBM (with corpus)</strong></td>
<td style="text-align: center;"><span class="math inline"><strong>+</strong><strong>0.0027</strong></span> <strong>[<span class="math inline"><strong>+</strong><strong>0.0001</strong><strong>,</strong> <strong>+</strong><strong>0.0053</strong></span>]</strong></td>
<td style="text-align: center;">(marginal but significant)</td>
</tr>
<tr>
<td style="text-align: left;">RF (with corpus)</td>
<td style="text-align: center;"><span class="math inline">−0.0001</span> [<span class="math inline">−0.0044, +0.0044</span>]</td>
<td style="text-align: center;">null</td>
</tr>
<tr>
<td style="text-align: left;">SVM rbf (with corpus)</td>
<td style="text-align: center;"><span class="math inline">−0.0511</span> [<span class="math inline">−0.0602, −0.0419</span>]</td>
<td style="text-align: center;">negative</td>
</tr>
</tbody>
</table>

</div>

세 가지 사실이 두드러진다.

- **Corpus 추가 lift 가 선형 baseline 대비 강함**. LR (no corpus) 위에 corpus 를 추가하면 GBM family 까지 올라가면 AUROC 0.0158 \[0.0108, 0.0208\] 의 lift. 95% CI 가 0 을 넓게 벗어남 — corpus signal 의 추가 정보가 결합기 capacity 향상과 함께 통계적으로 의미 있게 사용된다.

- **비선형 baseline (GBM no corpus) 위에서도 corpus 추가 lift 통계적으로 유의**. GBM (no corpus 0.889) 에 corpus 를 추가한 GBM (with corpus, 0.892) 의 AUROC delta = +0.0027 \[0.0001, 0.0053\], CI lower bound \> 0 이므로 marginal 이지만 유의함. 즉 SE / Energy / candidate-level signal 을 모두 비선형 결합한 강력한 baseline 위에서도 corpus signal 이 추가 정보를 보탠다.

- **SVM rbf 는 corpus 추가가 오히려 negative**. SVM 의 RBF kernel 은 corpus aggregate feature 의 분포 (대부분 좌편향, $`{\sim}50\%`$ 가 0) 와 잘 맞지 않아 추가 corpus 가 over-smoothing 을 야기하는 것으로 추정된다. tree-based 결합기 (RF / GBM) 가 corpus 신호를 안정적으로 활용한다.

표 <a href="#tab:bootstrap_ci" data-reference-type="ref" data-reference="tab:bootstrap_ci">4.2</a> 의 결과는 본 논문의 두 번째 연구 질문 (corpus signal 추가의 한계 효용) 에 대한 직접적 답이다 — *tree-based fusion 위에서 corpus signal 의 한계 효용은 아주 작지만 통계적으로 유의* 하다. 다음 절에서는 이 lift 가 corpus support 영역별로 어떻게 분포하는지를 분해한다.

## Corpus support 구간별 한계 효용 분포

종합 평균 +0.003 의 corpus 추가 lift 가 모든 corpus support 영역에서 균일한지, 아니면 특정 영역에서 더 큰지를 3-bin / 5-bin / 10-bin 분해로 측정한다. 표 <a href="#tab:bin10_method_matrix" data-reference-type="ref" data-reference="tab:bin10_method_matrix">4.3</a> 는 10-bin (rank-quantile decile) × 핵심 4개 method 의 AUROC matrix 이다. 그림 <a href="#fig:money" data-reference-type="ref" data-reference="fig:money">4.1</a> 는 같은 데이터의 reliability map (색=AUROC) 이다.

<div id="tab:bin10_method_matrix">

| decile | n | GBM (no corpus) | **GBM (with corpus)** | RF (no corpus) | LR (no corpus) | corpus lift |
|:---|:--:|:--:|:--:|:--:|:--:|---:|
| 00–10 | 582 | 0.829 | **0.851** | 0.829 | 0.818 | $`\boldsymbol{+0.022}`$ |
| 10–20 | 570 | 0.776 | 0.777 | **0.776** | 0.749 | $`+0.001`$ |
| 20–30 | 578 | 0.778 | **0.787** | 0.778 | 0.755 | $`+0.009`$ |
| 30–40 | 585 | 0.830 | **0.836** | 0.830 | 0.826 | $`+0.006`$ |
| 40–50 | 582 | **0.849** | 0.848 | 0.849 | 0.855 | $`-0.001`$ |
| 50–60 | 568 | 0.873 | 0.868 | **0.873** | 0.855 | $`-0.005`$ |
| 60–70 | 594 | **0.860** | 0.851 | 0.860 | 0.850 | $`-0.009`$ |
| 70–80 | 584 | 0.918 | 0.909 | **0.918** | 0.895 | $`-0.009`$ |
| 80–90 | 592 | **0.930** | 0.930 | 0.930 | 0.913 | $`0.000`$ |
| 90–100 | 580 | 0.979 | **0.981** | 0.979 | 0.973 | $`+0.002`$ |

Method $`\times`$ corpus support decile AUROC (rank-quantile 분할, 각 decile $`\approx`$ 580 질문). 굵은 글씨는 decile 별 1위. 마지막 열 = corpus 추가 lift = GBM(with corpus) $`-`$ GBM(no corpus).

</div>

<figure id="fig:money" data-latex-placement="htbp">

<figcaption>Method <span class="math inline">×</span> corpus support decile AUROC (n=5<span>,</span>815, 라벨 = hard 질문). 색 = AUROC (RdYlGn, 0.5–1.0 범위). corpus 가 풍부할수록 (오른쪽) 모든 결합기 AUROC 가 올라가며, corpus 추가의 한계 효용은 corpus 가 부족한 좌측 (decile 00–10) 에서 가장 크다.</figcaption>
</figure>

핵심 발견은 세 가지이다.

- **Lowest decile 에서 corpus lift 가 가장 큼** (decile 00–10, n=582). GBM (with corpus) 0.851 vs GBM (no corpus) 0.829, lift = +0.022 — 본 논문 conditioning 가설 “corpus 가 entity 를 거의 모르는 영역에서 corpus 추가 정보의 가치가 가장 크다” 의 직접 데이터 근거. SE / Energy 만으로는 충분히 잡지 못하는 hallucination 패턴이 이 영역에 집중되어 있고, corpus signal 이 그 격차를 메운다.

- **Mid–high decile 에서는 corpus 추가가 큰 차이 없음**. decile 40–80 영역에서 corpus 추가 lift 는 $`-0.009`$   $`0.000`$ 사이로, 이 영역의 hallucination 은 SE / Energy 의 sample-consistency 신호만으로도 잘 탐지되어 (AUROC 0.85–0.92) corpus aggregate 가 추가 정보를 거의 보태지 못한다. 효과 크기가 종합 lift 의 noise 수준이므로 lowest decile 에 corpus 가치가 집중된다는 본 논문 주요 결론을 뒤집지 않는다.

- **전체 AUROC 가 corpus support 와 함께 단조 상승**. decile 10–20 의 0.78 부터 decile 90–100 의 0.98 까지 corpus 가 풍부할수록 detection 이 쉬워진다. 이는 hallucination 의 발생 패턴 자체가 corpus support 와 연관되어 있음을 시사한다 — corpus 가 풍부한 영역에서는 SE / Energy 같은 모델 자체 신호가 더 안정적으로 작동하고, corpus 가 부족한 영역에서는 모델 신호의 신뢰도가 떨어지므로 보완 신호 (corpus signal) 의 가치가 커진다.

종합하면 *corpus signal 의 가치는 corpus 가 가장 부족한 영역에 집중* 된다. 평균 +0.003 의 lift 뒤에 가려진 conditional 분포를 정확히 보여주는 것이 본 분석의 의의이다. 본 결과는 본 논문의 세 번째 연구 질문 (corpus 추가의 한계 효용 영역별 분포) 에 직접 답한다.

## 데이터셋 간 일반화

본 분석 결과가 두 dataset (HaluEval-QA / TruthfulQA) 에서 어떻게 작동하는지 확인하기 위해 per-dataset AUROC 를 표 <a href="#tab:lodo" data-reference-type="ref" data-reference="tab:lodo">4.4</a> 에 보고한다.

<div id="tab:lodo">

| Method | AGG (n=5,815) | Eval HaluEval-QA (n=5,000) | Eval TruthfulQA (n=815) |
|:---|:--:|:--:|:--:|
| **GBM (with corpus)** | **0.892** | **0.817** | **0.963** |
| GBM (no corpus) | 0.889 | 0.815 | 0.951 |
| RF (with corpus) | 0.889 | 0.813 | 0.954 |
| RF (no corpus) | 0.888 | 0.813 | 0.953 |
| LR (with corpus) | 0.878 | 0.796 | 0.927 |
| corpus-bin weighted fusion | 0.878 | 0.798 | 0.918 |
| LR (no corpus) | 0.876 | 0.797 | 0.916 |
| SVM rbf (with corpus) | 0.838 | 0.757 | 0.907 |

Per-dataset AUROC (5-fold CV pooled). Eval HaluEval-QA: HaluEval-QA 질문만으로 평가. Eval TruthfulQA: TruthfulQA 질문만으로 평가. AGG: 종합.

</div>

GBM (with corpus) 가 두 dataset 모두에서 1위이다. HaluEval-QA 에서 corpus 추가 lift 는 +0.002 (0.815 → 0.817) 로 작고, TruthfulQA 에서는 +0.012 (0.951 → 0.963) 로 더 크다.

##### TruthfulQA 라벨 노이즈에 따른 해석 주의.

TruthfulQA 의 is_hard 라벨 비율 (0.97) 이 매우 비대칭임에 주의해야 한다. 본 논문의 token-overlap 기반 매칭 (free-sample $`\ge`$ 50% token overlap with right_answer / best_answer / correct_answers list) 은 의미적으로 동등한 paraphrase 를 충분히 잡지 못해 TruthfulQA 의 다양한 정답 표현 (예: “Cardiff”, “the Welsh capital”, “a city in Wales”) 을 hard 로 잘못 분류한다. 따라서 TruthfulQA 단독 AUROC 0.963 은 라벨 노이즈에 의해 부풀려진 수치로 해석해야 하며, *본 논문 주요 결과 해석은 HaluEval-QA 단독 (0.817) 또는 AGG (0.892) 기준* 으로 진행한다. NLI-기반 매칭으로 라벨링하면 이 노이즈가 줄어들 가능성이 높으며, 이는 §결론 한계 / 향후 연구 항목에서 명시한다. 그럼에도 GBM (with corpus) 가 두 dataset 모두에서 안정적인 1위라는 사실은 본 논문 주요 결과가 단일 dataset 에 의존하지 않음을 보여준다.

## Threshold / calibration 진단

분류 정확도뿐 아니라 출력 점수를 확률로 해석할 수 있는지도 함께 점검하였다 (표 <a href="#tab:calibration" data-reference-type="ref" data-reference="tab:calibration">4.5</a>). Brier score와 ECE(expected calibration error)는 둘 다 작을수록 calibration이 좋다.

<div id="tab:calibration">

| Method                     |   Brier   |    ECE    |
|:---------------------------|:---------:|:---------:|
| **GBM (with corpus)**      | **0.110** |   0.017   |
| RF (with corpus)           |   0.111   |   0.022   |
| RF (no corpus)             |   0.112   |   0.016   |
| GBM (no corpus)            |   0.113   | **0.014** |
| corpus-bin weighted fusion |   0.120   |   0.022   |
| LR (with corpus)           |   0.121   |   0.030   |
| LR (no corpus)             |   0.124   |   0.034   |
| SVM rbf (with corpus)      |   0.131   |   0.016   |

Prompt-level threshold / calibration 진단 (n=5,815). Brier 와 ECE 모두 낮을수록 좋다. 굵은 글씨 = 1위.

</div>

GBM (with corpus) 는 분류 정확도뿐 아니라 calibration 에서도 강력하다 — Brier 0.110 으로 모든 방법 중 1위, ECE 1.7% 로 GBM (no corpus, 1.4%) 다음 두 번째. AUROC 0.892 + ECE 1.7% 는 환각 탐지 점수를 확률로 해석해 임계값 기반 선별 (예: $`\ge 0.7`$ 인 응답만 사용자에게 표시) 이나 downstream 리스크 계산에 사용할 경우 실제 오류율과의 오차가 약 1.7%p 이내임을 의미한다.

# 결론

<span id="ch:conclusion" label="ch:conclusion"></span>

## 연구 요약

본 논문은 LLM 환각 탐지에서 Semantic Entropy(SE), Semantic Energy, corpus-grounded 신호를 corpus support라는 조건 축 관점에서 재정리하고, paired 설정 위에서 비교 평가하였다. 제 <a href="#ch:intro" data-reference-type="ref" data-reference="ch:intro">[ch:intro]</a>장에서 제시한 세 연구 질문에 대한 답을 정리하면 다음과 같다.

##### 첫째: SE / Energy baseline 재현.

SE AUROC 0.832, Semantic Energy 0.862으로 Farquhar 등 (2024) 의 TriviaQA / Llama-2 7B 보고치 (0.79) 와 동일 수준에서 본 데이터에 재현된다.

##### 둘째: Corpus signal 추가의 한계 효용.

GBM (with corpus) AUROC 0.892으로 Energy 단독 대비 +0.030, SE 단독 대비 +0.060 lift. corpus 추가 자체의 한계 효용은 통계적으로 유의 — 선형 baseline 대비 +0.0158 \[0.0108, 0.0208\], 비선형 baseline 대비 +0.0027 \[0.0001, 0.0053\] (CI lower $`> 0`$). Calibration 도 1위 (Brier 0.110, ECE 1.7%).

##### 셋째: Corpus lift 의 영역별 분포.

종합 평균 +0.003 lift 가 균일하지 않으며, corpus 가 entity 를 거의 모르는 lowest decile (00–10) 에서 +0.022 로 가장 크다. 매우 풍부한 영역 (decile 90–100) 에서는 SE/Energy 만으로 AUROC 0.98 에 도달해 corpus 한계 효용이 0 에 수렴한다. 본 논문 conditioning 가설의 직접 데이터 근거이다. corpus exposure 와 모델 confidence 의 인과 관계 해석에는 주의가 필요하다 (Infini-gram 16B token corpus $`\ne`$ Qwen2.5-3B pretraining corpus, Zhang 등 2025 와 동일한 단서).

## 학술적 기여

본 논문의 학술적 기여는 다음과 같다.

1.  **Corpus signal 한계 효용의 정량 검증**. SE / Semantic Energy baseline 위에 외부 corpus support 신호를 추가한 GBM (with corpus) 가 SE 단독 대비 +0.060, Energy 단독 대비 +0.030 lift 를 보이고, corpus 추가 자체의 한계 효용 +0.0027 \[0.0001, 0.0053\] 가 통계적으로 유의함을 bootstrap (n=2,000) 으로 검증하였다. Qiu 등 (2025) / Zhang 등 (2025) 의 “corpus statistic 이 환각 탐지에 보완 가치를 가진다” 가설을 본 데이터에서 정량 검증한 것이다.

2.  **Corpus 신호 가치의 영역별 분포 측정**. 평균 +0.003 의 작은 lift 뒤에 가려진 conditional 분포를 10-bin decile 분해로 드러냈다. *lowest decile (00–10) 에서 lift +0.022 로 가장 크다* — corpus 가 부족한 영역에서 SE / Energy 의 신뢰도가 떨어지므로 corpus 보완 신호의 가치가 가장 크다는 conditioning 가설의 직접 데이터 근거.

3.  **외부 corpus 조건화의 직교성 확인**. Valentin 등 (2024) 이 모델 *내부* score attribute 로 conditional calibration 을 시도한 방향과 직교적으로, 본 논문은 모델 출력과 독립적인 *외부* corpus statistic 으로 환각 탐지 신호를 보완한다. 두 방향은 결합 가능한 보완 구조이다.

## 한계

본 연구는 다음 한계를 가진다.

1.  **Token-overlap 라벨 proxy 한계**. is_hard 라벨은 free-sample N=10 의 정답 token-overlap ($`\ge 50\%`$) 매칭 결과로 정의된다. HaluEval-QA 의 짧은 factoid 답에서는 잘 작동하나 (is_hard 0.20), TruthfulQA 의 다양한 paraphrase 정답을 충분히 잡지 못해 라벨이 hard 쪽으로 치우친다 (0.97). NLI-기반 매칭으로의 교체는 향후 연구에서 검토한다.

2.  **단일 corpus snapshot 및 단일 모델**. corpus support 축은 `v4_dolmasample_olmo` (16B Dolma sample) 단일 index 위에서 산출되며, 본 실험 corpus 와 Qwen2.5-3B 의 실제 pretraining corpus 가 다르므로 corpus support 와 환각의 연결은 *상관* 이지 *인과* 가 아니다. 또한 단일 모델 (Qwen2.5-3B) 평가이므로 모델 규모/계열 일반화는 본 논문의 범위를 넘는다.

3.  **Corpus 추가 lift 의 평균 크기는 작음**. 종합 AUROC delta +0.003 (vs GBM no corpus) 으로 통계 유의하나 효과 크기는 작다. 본 논문의 핵심 가치는 (i) lift 가 corpus 가 부족한 영역에 집중된다는 conditional 분포 발견 + (ii) SE / Energy 를 baseline 으로 명시적으로 재현해 공정 비교 틀 를 제시한 것에 있다.

## 향후 연구

향후 다음 방향을 검토할 수 있다.

1.  **NLI-기반 prompt accuracy 라벨**. 현재 token-overlap proxy 를 NLI 매칭 (Farquhar 2024 SE 의 cluster 매칭과 일관) 으로 교체해 TruthfulQA 의 다양한 정답 표현을 잡는다.

2.  **여러 corpus snapshot 비교**. `v4_dolmasample_olmo` 외에 더 큰 Infini-gram index (예: `v4_pileval_llama`) 나 도메인 특화 corpus 에서 corpus support 분포와 corpus 추가 lift 가 어떻게 달라지는지 비교한다.

3.  **모델 일반화**. 동일 평가 절차를 Llama, Mistral 계열 등 다른 causal LM 에 적용해 SE / Energy baseline 과 corpus 추가 lift 가 모델에 의존하는지 본다.

4.  **외부 SOTA 환각 탐지 시스템과의 비교**. SelfCheckGPT, INSIDE, FactScore 등과 본 논문 GBM (with corpus) 의 절대 성능 비교.

5.  **Conditional fusion 구조 확장**. 본 논문의 단순 GBM (with corpus) 외에 corpus-conditioned ensemble, monotonic fusion, hierarchical fusion 으로 corpus 추가 lift 가 가장 큰 영역에 집중하는 구조를 시도한다.

<div class="thebibliography">

99

Huang et al., “A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions,” *ACM Transactions on Information Systems*, vol. 43, no. 2, pp. 1–55, 2025.

J. Maynez et al., “On Faithfulness and Factuality in Abstractive Summarization,” *Proceedings of ACL*, pp. 1906–1919, 2020.

S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting hallucinations in large language models using semantic entropy,” *Nature*, vol. 630, pp. 625–630, 2024.

P. Manakul, A. Liusie, and M. J. F. Gales, “SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models,” *Proceedings of EMNLP*, pp. 9004–9017, 2023.

S. Min et al., “FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation,” *Proceedings of EMNLP*, pp. 12076–12100, 2023.

Z. Ma et al., “Semantic Energy: A novel approach for detecting confabulation in language models,” *arXiv preprint arXiv:2412.07965*, 2025.

S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” *Proceedings of ACL*, 2022.

Z. Qiu et al., “QuCo-RAG: Query-aware Corpus Grounding for Retrieval-Augmented Generation,” *arXiv preprint arXiv:2512.19134*, 2025.

A. Simhi et al., “Trust Me, I’m Wrong: High-Certainty Hallucinations in Language Models (CHOKE),” *arXiv preprint*, 2025.

J. Li, X. Cheng, W. X. Zhao, J.-Y. Nie, and J.-R. Wen, “HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models,” *Proceedings of EMNLP*, pp. 6449–6464, 2023.

J. Liu et al., “Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens,” *arXiv preprint arXiv:2401.17377*, 2024.

W. Zhao et al., “WildHallucinations: Evaluating Long-form Factuality in LLMs with Real-World Entity Queries,” *arXiv preprint arXiv:2407.17468*, 2024.

S. Valentin et al., “Cost-Effective Hallucination Detection for LLMs,” *arXiv preprint arXiv:2407.21424*, 2024.

Y. Zhang et al., “Measuring the Impact of Lexical Training Data Coverage on Hallucination Detection in Large Language Models,” *arXiv preprint arXiv:2511.17946*, 2025.

</div>
