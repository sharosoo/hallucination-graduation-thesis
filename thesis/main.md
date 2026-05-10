# 서론

<span id="ch:intro" label="ch:intro"></span>

## 연구 배경

대규모 언어모델 (Large Language Model, LLM) 은 텍스트 생성, 질의응답 등 다양한 자연어 처리 태스크에 널리 사용되고 있으나, 사실과 다른 내용을 그럴듯하게 생성하는 환각 (hallucination) 문제를 보이며 이는 LLM 신뢰성의 핵심 과제이다. 본 논문은 외부 사실과 충돌하는 응답을 탐지하는 문제 (사실성, factuality) 에 초점을 두며, TruthfulQA 같은 벤치마크가 이 평가 단위를 제공한다.

환각 탐지의 대표적 방법인 Semantic Entropy (SE, Farquhar 등 2024) 는 한 질문에 대해 $`K`$ 개 응답을 샘플링한 뒤 NLI 클러스터링 후 분포의 엔트로피를 계산한다. 반복 샘플링과 응답 간 일관성은 SelfCheckGPT 와 같은 선행 연구에서도 환각 탐지의 핵심 단서로 사용되었다. 그러나 SE 와 같은 sample-consistency 신호는 모델이 모든 응답에서 같은 의미의 틀린 답을 반복하면 판별력이 약해진다. 이는 Simhi 등 (2025) 이 CHOKE (high-certainty hallucination) 로 명명한 패턴이다. 후보 답변의 NLL / confidence margin 같은 logit 진단 신호도 “환각 답이 정답보다 덜 자신있을 것” 이라는 직관에 기대므로 동일 한계를 갖는다.

본 논문은 이 한계를 외부 *corpus support* 신호로 분해한다. 본 논문에서 corpus support 는 질문과 후보 답변에서 추출한 entity 의 corpus frequency 및 entity-pair co-occurrence 를 결합한 *질문 단위 score* 이다 (자세한 정의는 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a> 실험 설정). corpus support 는 환각 / 정답 라벨이 아니라 *탐지 신호의 영역별 행태를 분해하는 외부 조건 변수* 이며, 본 논문은 corpus support 를 평가 단위로만 사용하고 탐지기 입력 feature 로는 사용하지 않는다.

## 연구 목적

본 논문은 다음 세 질문을 다룬다.

1.  **탐지 난이도와 corpus support 의 관계**. 본 논문에서 *탐지 난이도* 는 한 영역 안에서 탐지기의 AUROC 가 낮은 정도를 가리킨다 (AUROC 가 낮을수록 어려운 질문과 쉬운 질문을 가르기 어려움). SE, Semantic Energy, logit-diagnostic, fusion 의 AUROC 가 corpus support 영역 (10-decile) 에 따라 어떻게 변하는가? 모든 신호가 공통 패턴을 보이는가, 아니면 신호별로 다른가?

2.  **영역별 단일 신호 우열**. 단일 신호 비교 (SE / Energy / logit-diagnostic) 의 결론이 corpus support 영역에 따라 달라지는가, 아니면 일관된가?

3.  **Fusion 의 영역별 변동**. Fusion 결합기가 단일 신호보다 항상 우수한가, 아니면 어떤 영역에서는 단일 신호가 더 좋은 케이스가 있는가?

##### 본 논문의 기여.

본 논문은 외부 corpus statistic 을 평가 단위로 사용하여 환각 탐지 신호의 영역별 행태를 분해한다. 평균 AUROC 한 수치로 신호와 fusion 의 우열을 보고하는 기존 관행이 가리는 영역별 변동을 데이터로 보인다. 외부 corpus 를 탐지기 입력 feature 가 아니라 평가 축으로 사용하는 방향은 Valentin 등 (2024) 이 모델 *내부* score attribute 로 conditional calibration 을 시도한 방향과 직교적이다.

## 논문 구성

본 논문은 다음과 같이 구성된다. 제 <a href="#ch:related" data-reference-type="ref" data-reference="ch:related">[ch:related]</a>장에서는 LLM 환각 탐지 관련 연구와 본 논문의 위치를 정리한다. 제 <a href="#ch:method" data-reference-type="ref" data-reference="ch:method">[ch:method]</a>장에서는 단일 점수 가정의 한계, corpus support 를 외부 conditional 축으로 사용하는 근거, 본 논문의 분석 절차를 제시한다. 제 <a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>장에서는 종합 baseline 위에 영역별 비교 결과 세 가지를 차례로 보고하고 데이터셋별 분해 결과와 calibration 진단을 추가한다. 제 <a href="#ch:conclusion" data-reference-type="ref" data-reference="ch:conclusion">[ch:conclusion]</a>장에서 세 연구 질문에 대한 답변과 학술적 기여, 한계, 향후 연구 방향을 제시한다.

# 관련 연구

<span id="ch:related" label="ch:related"></span>

본 장은 LLM 환각 탐지 연구를 (a) 단일 신호, (b) 신호 결합 (fusion), (c) corpus statistic 활용 시도의 세 흐름으로 정리하고, 마지막에 공통 한계와 본 논문의 위치를 명시한다.

## 단일 신호 기반 탐지

환각 탐지의 가장 기본적인 접근은 단일 신호로 모델의 답변 신뢰도를 평가하는 것이다. 본 논문이 baseline 으로 사용하는 세 신호를 정리한다 (표 <a href="#tab:method_comparison" data-reference-type="ref" data-reference="tab:method_comparison">2.1</a>).

##### Semantic Entropy (SE).

Farquhar 등 (2024) 이 Nature 에 발표한 방법으로, 한 질문에 대해 $`K`$ 개 응답을 샘플링한 뒤 NLI 모델로 의미 클러스터에 묶고 클러스터 분포의 Shannon entropy $`SE = -\sum_{c} p(c) \log p(c)`$ 를 산출한다. SelfCheckGPT 가 시작한 sample-consistency 접근의 의미-단위 일반화이며, TriviaQA / Llama-2 7B 환경에서 AUROC 0.79 를 보고하였다.

##### Semantic Energy.

Ma 등 (2025) 은 SE 가 softmax 후 cluster 확률 분포의 entropy 만 사용해 token logit 의 크기 정보를 잃는다는 점을 지적하며, softmax 전 raw logit $`z_\theta(x_t)`$ 의 부호 반전 평균 $`\tilde{E}(x_t) = -z_\theta(x_t)`$ 을 sample energy 로 정의하고 NLI cluster 확률 가중 합으로 질문 단위 scalar 를 산출한다. 응답 다양성이 작은 사례에서도 token logit 차이가 추가 정보를 줄 수 있다는 점이 이 방법의 동기이다.

##### Logit diagnostic + CHOKE.

후보 답변의 평균 음의 로그 확률 (NLL), confidence margin, logit variance 같은 token-level 신호는 “환각 답이 정답보다 덜 자신있을 것” 이라는 직관에 기댄다. 그러나 Simhi 등 (2025) 의 CHOKE (high-certainty hallucination) 패턴이 보였듯, 모델이 정답을 알면서도 일관된 오답을 더 확신 있게 생성하는 경우가 흔해 logit 단독 신호는 영역에 따라 방향이 역전된다.

<div id="tab:method_comparison">

| 신호 | 핵심 아이디어 | 한계 |
|:---|:---|:---|
| SE (Semantic Entropy) | NLI cluster + Shannon entropy | 단일 cluster 수렴 사례 약함 |
| Energy (Semantic Energy) | cluster 가중 token logit 합 | 다양성 작은 사례 의존 |
| logit-diagnostic | NLL, logit variance, margin | CHOKE 사례에서 역전 |

본 논문이 비교하는 단일 신호.

</div>

## Multi-signal Fusion

단일 신호의 한계를 보완하기 위해 여러 신호를 결합하는 fusion 접근이 표준화되었다. Valentin 등 (2024) 은 환각 탐지 점수를 모델 내부 score attribute (예: top-k logit margin) 에 conditional 하게 calibrate 하는 multi-scoring framework 를 제안하였다. SelfCheckGPT 의 multiple consistency check (BERTScore, NLI entailment, n-gram match 등) 도 단일 prompt 의 여러 일관성 측정을 결합하는 fusion 의 일종이다.

이러한 fusion 연구는 공통적으로 *평균 AUROC / Brier 한 수치로* fusion 의 우월성을 보고한다. 그러나 fusion 결합기의 우월성이 입력 조건 (예: 질문의 entity 가 corpus 에 얼마나 등장하는지) 에 따라 어떻게 변하는지, 어떤 영역에서는 단일 신호보다 약한지를 분해해 보고한 연구는 본 저자들이 검토한 범위에서 보고되지 않았다.

## Corpus statistic 의 환각 탐지 활용 시도

corpus statistic 을 환각 탐지 또는 factuality 평가에 활용한 세 흐름이 있다.

##### QuCo-RAG.

Qiu 등 (2025) 은 retrieval-augmented generation 의 query-aware corpus grounding 절차에서 entity frequency 와 entity-pair co-occurrence 를 corpus count 로 측정해 답변이 외부 자료로 얼마나 “뒷받침될 수 있는지” 를 정량화하였다. 본 논문은 같은 corpus statistic 을 사용하되 그 *용도* 를 retrieval trigger 가 아니라 *탐지 신호 신뢰도를 분해하는 외부 조건 축* 으로 재해석한다.

##### Pretraining corpus coverage (Zhang 등 2025).

Zhang 등 (2025) 은 RedPajama 1.3 조 토큰 pretraining corpus 위에 suffix array 를 구축하고 prompt / 답변의 n-gram 빈도를 환각 탐지 신호로 평가하였다. “occurrence 기반 feature 는 log-probability 보다 약한 단독 신호이지만 함께 쓰면 특정 artifact 패턴 탐지에 보완 가치가 있다” 는 결론을 보고하면서 “결과는 상관관계이며 인과 관계를 확립하지 않는다” 는 단서를 명시하였다.

##### WildHallucinations.

Zhao 등 (2024) 은 *Wikipedia 페이지가 없는 entity* 에 대한 질문에서 LLM 환각률이 유의미하게 증가함을 관찰하여, corpus exposure 부족과 환각 발생률의 경험적 연관을 보고하였다.

본 논문은 위 세 연구가 corpus statistic 을 환각 탐지 *신호 자체* 로 사용한 것과 달리, corpus 를 *탐지 신호의 영역별 행태를 분해하는 외부 조건 축* 으로만 사용한다. corpus 빈도가 낮다고 환각 라벨로 쓰지 않고, 높다고 정답 라벨로 쓰지 않으며, 탐지기 입력 feature 로도 사용하지 않는다.

## 공통 한계와 본 논문의 위치

위 세 흐름의 공통 한계는 평균 AUROC / Brier 한 수치로 신호와 fusion 의 우열을 보고한다는 점이다. 평균 한 수치는 (i) 어느 입력 조건에서 어느 신호가 안정적으로 작동하는지 가리고, (ii) fusion 결합기가 모든 영역에서 단일 신호보다 우수한지를 검증하지 않는다. 본 논문은 외부 corpus support 를 평가 단위로 채택해 단일 신호 (SE / Energy / logit-diagnostic) 와 fusion 의 영역별 행태를 10-decile 로 분해하여 평균 한 수치 뒤에 가려진 변동을 데이터로 드러낸다.

##### Corpus 기반 환각 경험 증거 (WildHallucinations).

Zhao 등(2024)은 “*Wikipedia 페이지가 없는 entity*에 대한 질문에서 LLM이 더 자주 환각한다”는 관찰을 보고하였다. 이는 본 연구가 corpus 빈도를 조건 축으로 쓰는 가설의 *선행 경험적 근거*이다. 차이점은 두 가지다. (i) WildHallucinations 는 Wikipedia 존재 여부 라는 *이진* 조건을 사용하는 반면, 본 논문은 entity frequency / pair co-occurrence 기반 *연속 corpus support 축*으로 일반화한다. (ii) WildHallucinations 는 환각률 자체를 corpus 조건으로 설명하는 *evaluation* 연구이고, 본 논문은 이미 환각이 일어난 후보 행에 대해 *어떤 탐지 신호가 corpus 조건에 따라 더 안정적으로 작동하는가*를 평가하는 *detection* 연구이다.

##### Conditional calibration framework (Valentin et al.).

Valentin 등 (2024) 은 환각 탐지 점수를 *입력 / 응답 attribute 에 conditional 하게 calibrate* 하는 multi-scoring framework 를 제안하였다. 본 논문과의 핵심 차이는 *조건의 출처* 이다. Valentin 은 모델 *내부 score attribute* 에 conditional 하게 calibrate 하는 반면, 본 논문은 모델 출력과 독립적인 *외부 corpus statistic* 으로 조건화한다. 두 방향은 직교적이며 결합 가능한 보완 구조로 자리매김할 수 있다.

# 제안 방법

<span id="ch:method" label="ch:method"></span>

본 장은 본 논문이 채택하는 *corpus-conditioned 분석 절차* 와 그 이론적 동기를 정리한다.

## 평균 metric 보고의 한계

기존 환각 탐지 연구는 단일 신호 (SE, Energy, logit-diagnostic) 와 fusion 결합기의 우열을 평균 AUROC 한 수치로 보고한다. 이 관행은 (i) 신호의 신뢰도가 입력 조건에 따라 어떻게 변하는지, (ii) fusion 결합기가 모든 입력 조건에서 단일 신호보다 우수한지를 가린다. 본 논문은 외부 입력 조건 변수로 corpus support 를 채택해 영역별 행태를 분해한다.

## Corpus support 를 외부 conditional 축으로 사용하는 근거

질문과 후보 답변에 등장하는 entity 의 corpus 내 frequency 와 entity-pair co-occurrence 는 모델이 “익숙한” entity 를 다루고 있는지를 외부 자료로 측정한 지표이다. Qiu 등 (2025) 의 QuCo-RAG 가 entity frequency 와 co-occurrence 를 retrieval grounding 신호로 활용하였고, Zhang 등 (2025) 은 pretraining corpus 의 n-gram coverage 가 환각 탐지 신호와 비단조적으로 연관됨을 보고하였다. Zhao 등 (2024) 의 WildHallucinations 는 Wikipedia 페이지가 없는 entity 에 대해 LLM 환각률이 증가한다고 관찰하였다. 세 결과의 공통 시사점은 corpus exposure 와 모델 출력의 신뢰도 분포가 경험적으로 연관되어 있다는 관찰이며 (인과 관계는 아님), 본 논문은 이 관찰을 차용해 corpus support 를 *탐지 신호의 영역별 행태를 분해하는 외부 평가 축* 으로 사용한다.

corpus support 를 *탐지기 입력 feature 가 아니라 평가 단위로만* 쓰는 이유는 평가 단위 (corpus support decile) 와 입력 feature (corpus aggregate) 가 같은 source 에서 나올 경우 self-conditioning artifact 가 발생하기 때문이다. 본 논문은 corpus 부족 영역에서 모델 자체 신호 (SE / Energy / logit-diagnostic) 와 fusion 의 AUROC 가 어떻게 변하는지를 직접 측정하는 것을 목표로 한다.

##### 왜 corpus 부족 영역에서 모델 신호가 약해질 수 있는가.

위 선행 연구를 잇는 가설 사슬이 가능하다 — corpus exposure 부족 → 모델 representation 불안정 → 환각률 증가 → free-sample 이 단일 cluster 로 수렴할 가능성 증가 → SE / Energy 의 sample-consistency 판별력 약화. 이 사슬은 Simhi 등 (2025) 의 CHOKE 패턴 (모델이 정답을 알면서도 일관된 오답을 확신 있게 생성) 과도 일관된다. 다만 본 논문은 representation 안정성이나 cluster 수렴 빈도를 직접 측정하지 않으므로 이 사슬은 결과 해석을 돕는 *가설 차원* 으로만 다루며, 영역별 AUROC 분해는 그 가설과 부합 / 불부합 관찰의 형태로 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a> 에서 보고한다.

## 본 논문의 분석 절차

본 논문의 절차는 다음 네 단계이다.

1.  **라벨링**. 각 질문의 free-sample N=10 정답 매칭 비율이 0.5 미만이면 *어려운 질문* 으로 binary 라벨을 부여한다 (Farquhar 2024 SE 의 평가 단위와 일치).

2.  **평가 단위 분할**. 5,815 개 질문을 corpus support 의 rank-quantile 10-decile 로 분할한다 (각 영역 약 580 질문).

3.  **단일 신호 AUROC**. SE / Energy / logit-diagnostic 각각의 score-based AUROC 를 각 decile 안에서 산출한다 (별도 학습 없음).

4.  **Fusion AUROC**. SE / Energy / NLL / confidence margin / logit variance 의 질문 단위 집계를 입력으로 하는 5-fold CV gradient boosting 의 out-of-fold 예측을 각 decile 안에서 평가한다.

corpus statistic 은 단계 (2) 의 평가 단위로만 사용하고 단계 (4) 의 fusion 입력으로는 사용하지 않는다 (self-conditioning artifact 회피).

# 실험

<span id="ch:experiment" label="ch:experiment"></span>

## 실험 설정

본 실험에서 후보 답변의 정답 / 환각 라벨은 데이터셋 annotation 을 그대로 사용하며, LLM-as-judge 로 후보의 정오를 새로 판정하지 않는다. 다만 질문 단위 *is_hard* proxy 라벨은 별도로 산출되며 (자세한 정의는 §평가 지표 참조) 모델의 free-sample N=10 답변과 데이터셋 정답의 token-overlap 매칭 비율로 정의된다.

### 데이터셋 구성

TruthfulQA 와 HaluEval-QA 두 데이터셋을 사용한다. TruthfulQA 는 정답 후보 list 와 오답 후보 list 에서 결정론적으로 한 쌍씩 선택하고, HaluEval-QA 는 제공되는 정답 답변과 환각 답변을 그대로 사용한다. 각 질문마다 정확히 두 후보 답 (정답 + 환각) 을 두며, 최종 데이터는 5,815 개 질문 + 11,630 개 후보 답이다 (TruthfulQA 815 질문 / 1,630 후보, HaluEval-QA 5,000 질문 / 10,000 후보).

### 모델 출력 수집

모델은 Qwen2.5-3B causal LM 을 사용한다. 두 종류의 출력을 함께 수집하여 단일 신호와 Fusion 입력 feature 를 산출한다.

- **Free sampling** (SE / Energy 산출용): 각 질문에 대해 $`N=10`$ 개의 짧은 답변을 자유 sampling 한다. 답변 길이는 최대 64 token 으로 제한하고 형식이 맞지 않은 표본은 일정 횟수까지 재샘플링한다. 짧은 entity-중심 답변에서 NLI cluster 판단이 안정적이며 후보 답변 단위와 일치한다.

- **Teacher-forced scoring** (logit-diagnostic 산출용): 정답 / 환각 후보 답을 모델에 입력으로 주입하면서 점수만 측정한다. 후보 답에 해당하는 token 위치 $`t`$ 마다 선택된 token 의 logit $`z_t(x_t)`$, $`\log p_\theta(x_t \mid \text{prompt}, x_{<t})`$, 분배 함수 $`\log Z_t`$, vocabulary logit 분산, top-1 과 top-2 logit 차 (confidence margin) 을 기록하고 후보 답 길이 $`T`$ 로 평균한다. 이 token-level 통계로부터 NLL, logit variance, confidence margin 의 질문 단위 집계가 산출되며 Fusion 입력 feature 로 사용된다.

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
로 계산한다. 부호 규약은 lower energy = higher reliability. teacher-forced scoring 의 NLL 평균, logit variance, confidence margin 은 logit-diagnostic 단일 신호 (§<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a> 종합 baseline) 와 Fusion 입력 feature 의 일부로 함께 사용된다 (정의 단위가 다르므로 Energy 와 같은 단일 신호 칸에 합치지 않는다).

##### Corpus-grounded feature.

질문과 두 후보 답변 (정답 후보 + 환각 후보) 에서 추출한 entity 를 합집합으로 모은 뒤 lowercase 정규화로 중복을 제거한 entity 집합 $`E`$ 를 사용해 질문 단위 corpus support score 를 산출한다. entity-pair 집합 $`P`$ 는 $`E`$ 안의 모든 unordered pair 로 정의한다. 즉 한 질문에 대해 정답 후보 entity 와 환각 후보 entity 가 모두 같은 $`E`$ 와 $`P`$ 에 포함된다. Entity 추출은 spaCy `en_core_web_lg` NER 로 수행하며, PERSON / ORG / GPE / LOC / DATE / EVENT / WORK_OF_ART / FAC / NORP / PRODUCT / LANGUAGE / LAW 12 개 label 만 corpus 조회 대상으로 유지한다 (CARDINAL / ORDINAL / MONEY / PERCENT / QUANTITY / TIME 은 factoid 답의 의미 단위로 부적합). NER 이 비면 spaCy noun-chunk 를 fallback 으로 사용하고, 그래도 비고 6 단어 이하 짧은 답이면 정규화한 텍스트 자체를 단일 entity 로 추가한다. 모든 entity 는 lowercase 정규화 + 중복 제거를 거치며 후보당 최대 8 개로 잘라낸다.

corpus count backend 는 Liu 등 (2024) 의 Infini-gram 의 `v4_dolmasample_olmo` 인덱스 (16B token, OLMo-7B-hf tokenizer) 를 사용한다. 한 질문의 entity 집합 $`E`$ 와 entity-pair 집합 $`P`$ 에 대해 다음을 산출한다.
``` math
\begin{align*}
f_{\min} &= \min_{e \in E} \mathrm{freq}(e), \qquad
f_{\text{axis}} = \frac{\log(1 + f_{\min})}{\log(1 + 10^6)} \\
p_{\text{mean}} &= \frac{1}{|P|} \sum_{(e_i, e_j) \in P} \mathrm{cooc}(e_i, e_j), \qquad
p_{\text{axis}} = \frac{\log(1 + p_{\text{mean}})}{\log(1 + 10^5)} \\
\text{coverage} &= \tfrac{1}{2}\,(f_{\text{axis}} + p_{\text{axis}})
\end{align*}
```
여기서 $`\mathrm{freq}(\cdot)`$ 와 $`\mathrm{cooc}(\cdot, \cdot)`$ 는 각각 Infini-gram 의 단일 entity count 와 `count_cnf` AND query 결과이다. coverage score 위에서 *rank-quantile 균등 분할* 로 corpus support 구간 label (3-bin / 5-bin / 10-bin) 을 산출한다. rank-quantile 분할을 사용한 이유는 spaCy NER 채택 후 entity_frequency 분포가 좌편향 (약 58% 가 0) 이라 fixed-cutoff (예: 0.1 절대 임계값) 분할 시 표본 쏠림 문제가 발생하기 때문이다.

본 논문에서 corpus support 신호는 *Fusion 입력 feature 로 사용하지 않고* 영역별 평가 단위 (rank-quantile decile 분할) 로만 사용된다. 이는 평가 단위와 입력 feature 가 같은 source 에서 나오면 self-conditioning artifact 가 발생할 수 있기 때문이다 (§<a href="#ch:method" data-reference-type="ref" data-reference="ch:method">[ch:method]</a>).

### 평가 지표

종합 표는 AUROC / AUPRC / Brier / ECE 를 보고한다. 단일 신호 (SE / Energy / logit-diagnostic) 의 AUROC 는 별도 학습 없이 score-based ranking 으로 산출하고, Fusion (gradient boosting) 의 AUROC 는 5-fold KFold 의 out-of-fold 예측에서 산출한다 (한 질문의 모든 정보가 같은 fold 에 들어가도록 강제). binary 라벨은 각 질문의 free-sample $`N=10`$ 정답 매칭 비율이 0.5 미만인 경우 *어려운 질문* 으로 정의한다 (SE 원논문 평가 단위와 일치).

## 종합 베이스라인

<div id="tab:current_thesis_evidence">

<table>
<caption>종합 baseline (n=5<span>,</span>815 질문). 라벨은 각 질문의 free-sample N=10 의 정답 매칭 비율이 0.5 미만인 경우 <em>어려운 질문</em> (is_hard=1). 단일 신호는 score-based AUROC (학습 없음). Fusion 은 5-fold CV gradient boosting (input: SE, Energy, NLL, confidence margin, logit variance 의 질문 단위 집계). 굵은 글씨 = 각 metric 별 1위 (AUROC / AUPRC / ECE 는 gradient boosting, Brier 는 random forest 가 근소 우위).</caption>
<thead>
<tr>
<th style="text-align: left;">신호</th>
<th style="text-align: center;">AUROC</th>
<th style="text-align: center;">AUPRC</th>
<th style="text-align: center;">Brier</th>
<th style="text-align: center;">ECE</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>단일 신호 (학습 없음)</em></td>
</tr>
<tr>
<td style="text-align: left;">logit-diagnostic-only</td>
<td style="text-align: center;">0.581</td>
<td style="text-align: center;">0.416</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">corpus-axis-only</td>
<td style="text-align: center;">0.634</td>
<td style="text-align: center;">0.546</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">Semantic Entropy (SE)</td>
<td style="text-align: center;">0.832</td>
<td style="text-align: center;">0.726</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">Semantic Energy</td>
<td style="text-align: center;">0.862</td>
<td style="text-align: center;">0.794</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Fusion (5-fold CV)</em></td>
</tr>
<tr>
<td style="text-align: left;">logistic regression</td>
<td style="text-align: center;">0.876</td>
<td style="text-align: center;">0.809</td>
<td style="text-align: center;">0.124</td>
<td style="text-align: center;">0.034</td>
</tr>
<tr>
<td style="text-align: left;">random forest</td>
<td style="text-align: center;">0.888</td>
<td style="text-align: center;">0.836</td>
<td style="text-align: center;"><strong>0.112</strong></td>
<td style="text-align: center;">0.016</td>
</tr>
<tr>
<td style="text-align: left;"><strong>gradient boosting</strong></td>
<td style="text-align: center;"><strong>0.889</strong></td>
<td style="text-align: center;"><strong>0.837</strong></td>
<td style="text-align: center;">0.113</td>
<td style="text-align: center;"><strong>0.014</strong></td>
</tr>
</tbody>
</table>

</div>

표 <a href="#tab:current_thesis_evidence" data-reference-type="ref" data-reference="tab:current_thesis_evidence">4.1</a> 의 결과에서 두 가지를 짚어 둔다. 첫째, SE 와 Energy 단독이 본 데이터에서도 강한 baseline 을 형성한다 (SE 0.832, Energy 0.862) — Farquhar 등 (2024) 의 TriviaQA / Llama-2 7B AUROC 0.79 와 비슷한 수준이다 (데이터셋 / 모델 / 라벨 정의가 다르므로 직접 재현이라기보다 같은 범위의 참고 비교). 둘째, Fusion (5-fold CV gradient boosting) 의 종합 AUROC 는 0.889으로 Energy 단독 대비 평균 +0.022우위에 그친다. 이 평균 한 수치 뒤에 영역별로 어떤 변동이 있는지가 다음 세 절의 분석 대상이다.

## 탐지 난이도와 corpus support 의 관계

##### 측정 방법.

5,815 개 질문을 corpus support coverage score 의 rank-quantile 로 10-decile 로 분할한다 (각 decile $`\approx`$ 580 질문). 각 decile 안에서 4 신호 (SE / Energy / logit-diagnostic / Fusion) 의 AUROC 를 산출한다. 단일 신호는 score-based ranking 으로 직접 계산 (학습 없음), Fusion 은 5-fold CV out-of-fold 예측을 사용한다.

<figure id="fig:per_decile_auroc" data-latex-placement="htbp">

<figcaption>Corpus support decile 별 AUROC (n=5<span>,</span>815, 라벨 = 어려운 질문). sample-consistency 계열 (SE, Energy, Fusion) 은 corpus 부족 영역 (왼쪽) 에서 가장 낮고 풍부 영역 (오른쪽) 에서 가장 높은 전반적 상승 경향을 보인다. logit-diagnostic 은 모든 영역에서 0.508–0.628 로 약하다 (“환각 답이 정답보다 덜 자신있을 것” 이라는 직관 위반이 corpus 영역에 무관하게 일관됨).</figcaption>
</figure>

그림 <a href="#fig:per_decile_auroc" data-reference-type="ref" data-reference="fig:per_decile_auroc">4.1</a> 는 sample-consistency 계열 (SE / Energy / Fusion) 이 corpus 부족 영역에서 동시에 약해지는 전반적 상승 패턴을 보여준다. 구체적으로 SE 는 가장 낮은 decile 에서 0.80, 가장 높은 decile 에서 0.92 로 변동하며, Energy 는 0.821에서 0.954로, Fusion 은 0.851에서 0.978로 동일한 상승 패턴을 보인다. 반면 logit-diagnostic 은 모든 영역에서 0.508–0.628 의 약한 판별력에 머무르며 전반적 상승 패턴을 따르지 않는다 — “환각 답이 정답보다 덜 자신있을 것” 이라는 직관과 부합하지 않는 패턴이 corpus 영역과 무관하게 반복적으로 관찰된다.

sample-consistency 계열의 상승 경향은 본 논문의 가설 — corpus 부족 영역에서 모델 representation 이 불안정하고 free-sample 이 단일 cluster 로 수렴해 sample-consistency 신호의 판별력이 약화된다 (§<a href="#ch:method" data-reference-type="ref" data-reference="ch:method">[ch:method]</a>) — 과 부합하는 관찰이다. 다만 본 논문은 cluster 수렴 빈도나 representation 안정성을 직접 측정하지 않았으므로 이 메커니즘은 가설 차원의 해석으로 다룬다. 또한 본 논문의 *is_hard* 라벨이 free-sample 매칭률 기반 proxy 이므로, “corpus 부족 영역에서 환각 탐지가 어렵다” 보다는 “corpus 부족 영역에서 sample-consistency 계열 신호의 hard question proxy 판별력이 낮다” 로 해석을 한정한다.

## 단일 신호 영역별 비교

##### 측정 방법.

§4.3 와 동일한 10-decile 분해 위에서 단일 신호 (SE / Energy / logit-diagnostic) 의 AUROC 를 직접 비교한다.

<div id="tab:bin10_method_matrix">

| decile |  n  |  SE   | **Energy** | logit-diag | Fusion |     |
|:-------|:---:|:-----:|:----------:|:----------:|:------:|:---:|
| 00–10  | 582 | 0.802 | **0.821**  |   0.628    | 0.851  |     |
| 10–20  | 570 | 0.718 | **0.746**  |   0.550    | 0.777  |     |
| 20–30  | 578 | 0.760 | **0.771**  |   0.579    | 0.787  |     |
| 30–40  | 585 | 0.788 | **0.812**  |   0.576    | 0.836  |     |
| 40–50  | 582 | 0.845 | **0.864**  |   0.592    | 0.848  |     |
| 50–60  | 568 | 0.821 | **0.850**  |   0.608    | 0.868  |     |
| 60–70  | 594 | 0.825 | **0.848**  |   0.615    | 0.851  |     |
| 70–80  | 584 | 0.832 | **0.868**  |   0.589    | 0.909  |     |
| 80–90  | 592 | 0.836 | **0.883**  |   0.540    | 0.930  |     |
| 90–100 | 580 | 0.924 | **0.954**  |   0.508    | 0.981  |     |

Method $`\times`$ corpus support decile AUROC (n=5,815, 각 decile $`\approx`$ 580 질문). 굵은 글씨 = 단일 신호 중 1위.

</div>

표 <a href="#tab:bin10_method_matrix" data-reference-type="ref" data-reference="tab:bin10_method_matrix">4.2</a> 가 보이는 패턴은 본 표본에서 단일 신호 비교의 결론이 corpus 영역에 따라 바뀌지 않았다는 점이다. Energy 가 모든 10 decile 에서 SE 를 0.011 에서 0.047 사이의 일관된 폭으로 능가하며, logit-diagnostic 은 모든 영역에서 0.508–0.628 의 약한 판별력에 머문다 (환각 답이 정답보다 더 자신있게 평가되는 CHOKE 패턴이 corpus 영역에 무관하게 유지되는 결과).

## Fusion 의 영역별 변동

##### 측정 방법.

§4.3 와 동일한 10-decile 분해 위에서 Fusion (5-fold CV GBM, 입력 = SE / Energy / NLL / margin / logit variance 의 질문 단위 집계) 의 AUROC 와 단일 신호 AUROC 의 영역별 차이를 측정한다.

<figure id="fig:fusion_delta" data-latex-placement="htbp">

<figcaption>Fusion AUROC 와 단일 신호 AUROC 의 decile 별 차이. Panel A 는 Fusion 과 SE / Energy 의 작은 폭의 차이를, Panel B 는 Fusion 과 logit-diagnostic 의 큰 폭의 차이를 별도 scale 에 보여준다. Panel A 에서 Fusion 은 SE 대비 모든 decile 에서 우위, Energy 대비는 영역별로 변동 (양 끝 영역에서 우위, 중간 영역에서 동률 또는 Energy 우위 점추정치). 정확한 수치와 95% bootstrap CI 는 표 <a href="#tab:fusion_energy_ci" data-reference-type="ref" data-reference="tab:fusion_energy_ci">4.3</a> 참조.</figcaption>
</figure>

그림 <a href="#fig:fusion_delta" data-reference-type="ref" data-reference="fig:fusion_delta">4.2</a> 가 보이는 패턴은 두 가지이다. 첫째, Fusion 은 SE 와 logit-diagnostic 단독에 대해 모든 영역에서 일관된 우위를 보인다 (vs SE 0.018–0.082, vs logit-diagnostic 0.30 이상). 단순 결합만으로도 약한 단일 신호를 보완하는 양상이 관찰된다. 둘째, Energy 단독과의 비교는 corpus 영역에 따라 변동한다.

이 변동의 통계적 유의성을 prompt 단위 bootstrap (n=1,000) 으로 측정한 결과를 표 <a href="#tab:fusion_energy_ci" data-reference-type="ref" data-reference="tab:fusion_energy_ci">4.3</a> 에 보고한다.

<div id="tab:fusion_energy_ci">

| decile |  n  |  점추정치  |        95% CI        | 통계 유의 |
|:-------|:---:|:----------:|:--------------------:|:---------:|
| 00–10  | 582 | $`+0.030`$ | $`[+0.005, +0.054]`$ |           |
| 10–20  | 570 | $`+0.031`$ | $`[+0.003, +0.061]`$ |           |
| 20–30  | 578 | $`+0.017`$ | $`[-0.007, +0.041]`$ |   null    |
| 30–40  | 585 | $`+0.024`$ | $`[-0.002, +0.052]`$ |   null    |
| 40–50  | 582 | $`-0.016`$ | $`[-0.036, +0.003]`$ |   null    |
| 50–60  | 568 | $`+0.018`$ | $`[-0.009, +0.046]`$ |   null    |
| 60–70  | 594 | $`+0.003`$ | $`[-0.017, +0.023]`$ |   null    |
| 70–80  | 584 | $`+0.041`$ | $`[+0.020, +0.063]`$ |           |
| 80–90  | 592 | $`+0.047`$ | $`[+0.029, +0.067]`$ |           |
| 90–100 | 580 | $`+0.027`$ | $`[+0.016, +0.040]`$ |           |

Decile 별 Fusion AUROC $`-`$ Energy AUROC 의 점추정치와 95% prompt-단위 bootstrap CI (n=1,000). 양수 = Fusion 우위. CI 가 0 을 포함하지 않으면 . 표시는 다중비교 보정 전의 탐색적 불확실성 진단이며, decile 10 개에 대한 동시 검정의 family-wise error 는 보정하지 않았다. 본 표와 표 <a href="#tab:bin10_method_matrix" data-reference-type="ref" data-reference="tab:bin10_method_matrix">4.2</a> 는 같은 5-fold OOF 예측에서 산출된 값이며, 표 4.4 의 반올림된 AUROC 단순 차이와 본 표의 점추정치는 일치한다.

</div>

표 <a href="#tab:fusion_energy_ci" data-reference-type="ref" data-reference="tab:fusion_energy_ci">4.3</a> 는 두 가지 새 사실을 보여준다. 첫째, Fusion 우위가 통계적으로 유의한 영역은 양 끝 (decile 00–20, 70–100) 5 개 영역이며, 평균 우위 +0.022의 대부분이 이 영역에서 발생한다. 둘째, decile 40–50 의 Energy 우위 점추정치 $`-0.016`$ 은 95% CI 가 0 을 포함하므로 *통계적으로 유의한 역전이라고 단정할 수 없다*. 본 절의 결론은 “Fusion 평균 우위 폭이 corpus 영역에 따라 균일하지 않으며, 중간 영역 (decile 20–70) 에서는 Energy 단독과 통계적으로 구별되지 않는 영역도 있다” 는 사실에 한정한다.

이 결과는 fusion 의 우월성이 단일 신호 선택과 corpus 영역에 동시에 의존한다는 것을 보인다. 평균 우위 한 수치 (Fusion 평균 vs Energy +0.022) 만 보면 “Fusion 이 항상 더 좋다” 는 인상이지만, 영역별 분해는 fusion 이 mid corpus 영역에서 단일 신호보다 약해질 수 있음을 보여준다. 기존 fusion 연구 (Valentin 2024, SelfCheckGPT) 가 평균 metric 한 수치로 fusion 우월성을 보고하는 관행이 가리는 정보가 있음을 시사한다.

## 데이터셋별 결과와 Calibration

<div id="tab:lodo">

| Method              |    AGG    | HaluEval-QA | TruthfulQA |   Brier   | ECE (비율) |
|:--------------------|:---------:|:-----------:|:----------:|:---------:|:----------:|
| **Fusion (GBM)**    | **0.889** |    0.815    |   0.951    |   0.113   | **0.014**  |
| Random Forest       |   0.888   |    0.813    | **0.953**  | **0.112** |   0.016    |
| Logistic Regression |   0.876   |    0.797    |   0.916    |   0.124   |   0.034    |
| Energy-only         |   0.862   |      —      |     —      |     —     |     —      |
| SE-only             |   0.832   |      —      |     —      |     —     |     —      |
| logit-diagnostic    |   0.581   |      —      |     —      |     —     |     —      |

데이터셋별 AUROC (5-fold CV pooled) 와 calibration. AGG 는 5,815 개 질문 전체를 5-fold CV 로 평가한 종합 AUROC 이며 단순 평균이 아니라 가중 평균에 가깝다 (HaluEval-QA 5,000 + TruthfulQA 815). Brier 와 ECE 는 종합 5,815 개 질문 위에서 산출. ECE 는 \[0,1\] 비율 단위.

</div>

AUROC 는 AGG (0.889) 와 HaluEval-QA (Fusion 0.815) 에서 Fusion 이 가장 높고, TruthfulQA 에서는 Random Forest (0.953) 가 Fusion (0.951) 을 0.002 차로 근소하게 앞선다. Calibration 은 ECE 기준 Fusion (1.4%) 이 가장 낮고, Brier 는 Random Forest (0.112) 가 Fusion (0.113) 보다 근소하게 낮으나 차이가 작다. 다만 TruthfulQA 단독 AUROC 0.951 은 라벨 노이즈에 의해 부풀려진 수치로 해석해야 한다. 본 논문의 token-overlap 라벨은 의미적으로 동등한 paraphrase 정답 (예: “Cardiff”, “the Welsh capital”) 을 충분히 잡지 못해 TruthfulQA 의 다양한 정답 표현을 *어려운 질문* 으로 잘못 분류한다 (TruthfulQA is_hard 비율 0.97). 본 논문 주요 결과 해석은 HaluEval-QA 단독 (Fusion 0.815) 기준이며, AGG (0.889) 와 §<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a> 의 영역별 분해 결과는 TruthfulQA 라벨 노이즈가 가중 평균 형태로 섞여 있다는 점을 감안해야 한다 (NLI 기반 라벨링으로의 교체는 §결론 한계 / 향후 연구 항목에서 다룬다).

본 절의 평가는 진정한 leave-one-dataset-out (한 데이터셋만으로 학습 후 다른 데이터셋에서 평가) 이 아니라 5,815 개 질문 전체에 대한 5-fold CV 결과를 데이터셋별로 분해한 것이다. 데이터셋 간 도메인 전이 검증은 본 논문 범위를 넘는다.

##### HaluEval-QA only 영역별 분해 (보조 검증).

같은 decile 분해를 HaluEval-QA 5,000 개 질문에 한정해 재산출하면 (TruthfulQA 의 라벨 노이즈 영향 제거), 영역별 *is_hard* prevalence 는 16% 에서 25% 사이로 변동하며 corpus 풍부 영역일수록 약간 낮은 경향 (decile 90–100 약 22%, decile 20–30 약 25%) 을 보인다. AUROC 의 영역별 상승 경향은 종합 분해보다 약하다 — Fusion AUROC 가 가장 낮은 decile 에서 약 0.78, 가장 높은 decile (90–100) 에서도 약 0.88 에 그쳐, 종합 분해의 0.85–0.98 범위와 차이가 있다. 종합 분해의 decile 90–100 의 0.98 신호는 TruthfulQA 의 비대칭 라벨이 가중 평균 형태로 섞인 결과로 해석된다. 그럼에도 Fusion – Energy 의 영역별 변동 패턴 (양 끝에서 우위 유의, 중간에서 동률 또는 Energy 우위 점추정치) 자체는 종합 분해와 일관된다.

# 결론

<span id="ch:conclusion" label="ch:conclusion"></span>

## 연구 요약

본 논문은 외부 corpus statistic 을 평가 단위로 사용하여 환각 탐지 신호 (SE, Semantic Energy, logit-diagnostic, fusion) 의 영역별 행태를 분해하였다. 세 연구 질문에 대한 답을 정리한다.

##### 첫째 (난이도 ↔ corpus support).

sample-consistency 계열 (SE, Energy, Fusion) 의 AUROC 가 corpus support 영역과 전반적 상승 경향을 보인다 — 가장 부족한 영역에서 낮고 풍부한 영역에서 높다 (SE 0.80 에서 0.92, Energy 0.821에서 0.954, Fusion 0.851에서 0.978). 반면 logit-diagnostic 은 모든 영역에서 0.508–0.628 의 약한 판별력에 머물러 이 경향을 따르지 않는다. sample-consistency 계열에 한정하면 *is_hard* proxy 의 판별이 corpus 부족 영역에서 상대적으로 어렵다는 점이 데이터로 관찰된다.

##### 둘째 (영역별 단일 신호 우열).

본 표본에서는 단일 신호 비교의 결론이 corpus 영역에 따라 바뀌지 않았다 — Energy 가 모든 10 decile 에서 SE 를 0.011 에서 0.047 사이로 능가하고 logit-diagnostic 은 모든 영역에서 약하다 (CHOKE 패턴).

##### 셋째 (Fusion 영역별 변동).

Fusion 은 SE 와 logit-diagnostic 단독에 대해 모든 영역에서 일관된 우위를 보인다. Energy 단독과는 영역별로 변동하며 prompt 단위 bootstrap (n=1,000) 으로 검정한 결과, Fusion 우위가 통계적으로 유의한 (95% CI 가 0 을 포함하지 않음) 영역은 양 끝 5 개 (decile 00–20, 70–100) 이고 중간 영역 (20–70) 에서는 CI 가 0 을 포함한다. 특히 decile 40–50 에서 Energy 가 Fusion 을 점추정치 0.016 능가하나 CI 가 \[-0.036, +0.003\] 으로 0 을 포함하므로 통계적으로 유의한 역전은 아니다. 평균 우위 한 수치 (+0.022) 만 보면 가려지는 영역별 변동 폭 자체는 데이터로 드러난다.

## 학술적 기여

1.  **영역별 분해를 통한 평균 metric 의 한계 노출**. sample-consistency 계열 신호의 탐지 난이도 (영역별 AUROC 가 낮은 정도) 가 corpus 부족 영역에 집중됨을 정량화. 평균 AUROC 한 수치가 가리는 입력 조건별 변동을 데이터로 보였다.

2.  **Fusion 우위 폭의 corpus 영역별 변동**. Fusion 이 Energy 단독을 평균 +0.022능가하나, decile 80–90 에서 +0.047까지 벌어지고 decile 40–50 에서는 오히려 -0.016로 줄어드는 변동을 보고하였다. 기존 fusion 연구 (Valentin 2024, SelfCheckGPT) 가 평균 metric 만 보고하는 관행이 가리는 변동이 있음을 시사한다.

3.  **외부 corpus 를 평가 단위로 사용하는 분석 절차**. Valentin 등 (2024) 이 모델 *내부* score attribute 로 conditional calibration 을 시도한 방향과 직교적으로, 본 논문은 모델 출력과 독립적인 *외부* corpus statistic 을 평가 단위로 사용해 영역별 행태를 분해한다. 두 방향은 결합 가능한 보완 구조이다.

## 한계

1.  **Token-overlap 라벨 proxy 한계**. is_hard 라벨은 free-sample N=10 의 정답 token-overlap ($`\ge 50\%`$) 매칭 결과로 정의된다. HaluEval-QA 의 짧은 factoid 답에서는 잘 작동하나 (is_hard 0.20), TruthfulQA 의 다양한 paraphrase 정답을 충분히 잡지 못해 라벨이 hard 쪽으로 치우친다 (0.97). NLI-기반 매칭으로의 교체는 향후 연구에서 검토한다.

2.  **is_hard 라벨과 SE / Energy 신호의 free-sampling 순환성**. is_hard 라벨과 SE / Energy 신호가 모두 동일한 free-sampling 절차 (N=10 답변 생성) 에서 파생된다. 따라서 본 결과는 독립 환각 라벨에 대한 일반 환각 탐지 성능이라기보다 sampling 기반 difficulty proxy 와 sample-consistency 신호 사이의 *조건부 관계 측정* 으로 해석해야 한다.

3.  **단일 corpus snapshot 및 단일 모델**. corpus support 축은 `v4_dolmasample_olmo` (16B Dolma sample) 단일 index 위에서 산출되며, 본 실험 corpus 와 Qwen2.5-3B 의 실제 pretraining corpus 가 다르므로 corpus support 와 환각의 연결은 *상관* 이지 *인과* 가 아니다. 단일 모델 평가이므로 모델 규모/계열 일반화는 본 논문 범위를 넘는다.

4.  **Self-conditioning artifact 회피**. 본 논문은 corpus statistic 을 평가 단위로만 사용하고 fusion 입력 feature 로는 사용하지 않았다. 따라서 “외부 corpus signal 의 직접 한계 효용” 은 본 논문의 주장 범위 밖이며, 본 논문의 결론은 단일 신호와 fusion 의 영역별 행태 자체에 한정된다.

## 향후 연구

1.  **NLI-기반 정답 매칭 라벨**. 현재 token-overlap proxy 를 NLI 매칭 (Farquhar 2024 SE 의 cluster 매칭과 일관) 으로 교체해 TruthfulQA 의 다양한 정답 표현을 잡는다.

2.  **모델 일반화**. 동일 평가 절차를 Llama, Mistral 계열 등 다른 causal LM 에 적용해 영역별 패턴이 모델에 의존하는지 본다.

3.  **여러 corpus snapshot 비교**. 도메인 특화 corpus 에서 영역별 분포가 어떻게 달라지는지 비교한다.

4.  **외부 SOTA 환각 탐지 시스템과의 비교**. SelfCheckGPT, INSIDE, FactScore 등과 본 논문 Fusion 의 절대 성능 비교.

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
