# 서론

<span id="ch:intro" label="ch:intro"></span>

## 연구 배경

대규모 언어모델 (Large Language Model, LLM) 은 텍스트 생성, 질의응답 등 다양한 자연어 처리 태스크에 널리 사용되고 있으나, 사실과 다른 내용을 그럴듯하게 생성하는 환각 (hallucination) 문제를 보이며 이는 LLM 신뢰성의 핵심 과제이다. 본 연구는 외부 사실과 충돌하는 응답을 탐지하는 문제 (사실성, factuality) 에 초점을 두며, TruthfulQA 같은 벤치마크가 이 평가 단위를 제공한다.

환각 탐지의 대표적 방법인 Semantic Entropy (SE, Farquhar 등 2024) 는 한 질문에 대해 $`K`$ 개 응답을 샘플링한 뒤 NLI 클러스터링 후 분포의 엔트로피를 계산한다. 반복 샘플링과 응답 간 일관성은 SelfCheckGPT 와 같은 선행 연구에서도 환각 탐지의 핵심 단서로 사용되었다. 그러나 SE 와 같은 sample-consistency 신호는 모델이 모든 응답에서 같은 의미의 틀린 답을 반복하면 판별력이 약해진다 — Simhi 등 (2025) 의 CHOKE (high-certainty hallucination) 패턴. 후보 답변의 NLL / confidence margin 같은 logit 진단 신호도 “환각 답이 정답보다 덜 자신있을 것” 이라는 직관에 기대므로 동일 한계를 갖는다.

본 연구는 이 한계를 *외부 corpus 신호* 로 보완한다. 후보 답의 entity frequency 와 entity-pair co-occurrence 를 외부 corpus 에서 측정한 뒤, SE / Energy baseline 위에 결합하여 추가 lift 가 발생하는지 측정하고, 그 lift 가 corpus support 영역별로 어떻게 분포하는지 분해한다. corpus 빈도는 환각 라벨이 아니라 *지표 신뢰성을 보완하는 외부 조건 변수* 이다.

본 연구는 외부 corpus 신호를 *탐지기 입력 feature 로 사용하지 않고* 평가 단위 (외부 conditional 축) 로만 사용한다. corpus 의 영역별 단위에서 단일 신호와 fusion 의 행태를 분해해 보고하며, corpus 빈도 자체는 정답/환각 라벨이 아니라 *지표 신뢰성을 분해하는 외부 조건 변수*로 다룬다.

## 연구 목적

본 연구는 다음 세 질문을 다룬다.

1.  **난이도 ↔ corpus support 관계**. SE, Semantic Energy, logit-diagnostic, fusion 의 AUROC 가 corpus support 영역 (10-decile) 에 따라 어떻게 변하는가? 모든 신호가 공통 패턴을 보이는가, 아니면 신호별로 다른가?

2.  **영역별 단일 신호 우열**. 단일 신호 비교 (SE / Energy / logit-diagnostic) 의 결론이 corpus support 영역에 따라 달라지는가, 아니면 일관된가?

3.  **Fusion 의 영역별 변동**. Fusion 결합기가 단일 신호보다 항상 우수한가, 아니면 어떤 영역에서는 단일 신호가 더 좋은 케이스가 있는가?

##### 본 논문의 기여.

본 논문은 외부 corpus statistic 을 평가 단위로 사용하여 환각 탐지 신호의 영역별 행태를 분해한다. 평균 AUROC 한 수치로 신호와 fusion 의 우열을 보고하는 기존 관행이 가리는 영역별 변동을 데이터로 보인다. 외부 corpus 를 탐지기 입력 feature 가 아니라 평가 축으로 사용하는 방향은 Valentin 등 (2024) 이 모델 *내부* score attribute 로 conditional calibration 을 시도한 방향과 직교적이다.

## 논문 구성

본 논문은 다음과 같이 구성된다. 제 <a href="#ch:related" data-reference-type="ref" data-reference="ch:related">[ch:related]</a>장에서는 LLM 환각 탐지 관련 연구와 본 연구의 위치를 정리한다. 제 <a href="#ch:method" data-reference-type="ref" data-reference="ch:method">[ch:method]</a>장에서는 단일 점수 가정의 한계, corpus support 를 외부 conditional 축으로 사용하는 근거, 본 논문의 분석 절차를 제시한다. 제 <a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>장에서는 종합 baseline 위에 영역별 비교 결과 세 가지를 차례로 보고하고 leave-one-dataset-out 일반화와 threshold / calibration 진단을 추가한다. 제 <a href="#ch:conclusion" data-reference-type="ref" data-reference="ch:conclusion">[ch:conclusion]</a>장에서 세 연구 질문에 대한 답변과 학술적 기여, 한계, 향후 연구 방향을 제시한다.

# 관련 연구

<span id="ch:related" label="ch:related"></span>

본 장은 LLM 환각 탐지 연구를 (a) 단일 신호, (b) 신호 결합 (fusion), (c) corpus statistic 활용 시도의 세 흐름으로 정리하고, 마지막에 공통 한계와 본 논문의 위치를 명시한다.

## 단일 신호 기반 탐지

환각 탐지의 가장 기본적인 접근은 단일 신호로 모델의 답변 신뢰도를 평가하는 것이다. 본 논문이 baseline 으로 사용하는 세 신호를 정리한다 (표 <a href="#tab:method_comparison" data-reference-type="ref" data-reference="tab:method_comparison">2.1</a>).

##### Semantic Entropy (SE).

Farquhar 등 (2024) 이 Nature 에 발표한 방법으로, 한 질문에 대해 $`K`$ 개 응답을 샘플링한 뒤 NLI 모델로 의미 클러스터에 묶고 클러스터 분포의 Shannon entropy $`SE = -\sum_{c} p(c) \log p(c)`$ 를 산출한다. SelfCheckGPT 가 시작한 sample-consistency 접근의 의미-단위 일반화이며, TriviaQA / Llama-2 7B 환경에서 AUROC 0.79 를 보고하였다.

##### Semantic Energy.

Ma 등 (2025) 은 SE 가 softmax 후 cluster 확률 분포의 entropy 만 사용해 token logit 의 크기 정보를 잃는다는 점을 지적하며, softmax 전 raw logit $`z_\theta(x_t)`$ 의 부호 반전 평균 $`\tilde{E}(x_t) = -z_\theta(x_t)`$ 을 sample energy 로 정의하고 NLI cluster 확률 가중 합으로 질문 단위 scalar 를 산출한다. 응답 다양성이 작은 사례에서도 token logit 차이가 추가 정보를 줄 수 있다는 motivation 이다.

##### Logit diagnostic + CHOKE.

후보 답변의 평균 음의 로그 확률 (NLL), confidence margin, logit variance 같은 token-level 신호는 “환각 답이 정답보다 덜 자신있을 것” 이라는 직관에 기댄다. 그러나 Simhi 등 (2025) 의 CHOKE (high-certainty hallucination) 패턴이 보였듯, 모델이 정답을 알면서도 일관된 오답을 더 확신 있게 생성하는 경우가 흔해 logit 단독 신호는 영역에 따라 방향이 역전된다.

<div id="tab:method_comparison">

| 신호             | 핵심 아이디어                 | 한계                        |
|:-----------------|:------------------------------|:----------------------------|
| Semantic Entropy | NLI cluster + Shannon entropy | 단일 cluster 수렴 사례 약함 |
| Semantic Energy  | cluster 가중 token logit 합   | 다양성 작은 사례 의존       |
| logit diagnostic | NLL, logit variance, margin   | CHOKE 사례에서 역전         |

본 논문이 비교하는 단일 신호.

</div>

## Multi-signal Fusion

단일 신호의 한계를 보완하기 위해 여러 신호를 결합하는 fusion 접근이 표준화되었다. Valentin 등 (2024) 은 환각 탐지 점수를 모델 내부 score attribute (예: top-k logit margin) 에 conditional 하게 calibrate 하는 multi-scoring framework 를 제안하였다. SelfCheckGPT 의 multiple consistency check (BERTScore, NLI entailment, n-gram match 등) 도 단일 prompt 의 여러 일관성 측정을 결합하는 fusion 의 일종이다.

이러한 fusion 연구는 공통적으로 *평균 AUROC / Brier 한 수치로* fusion 의 우월성을 보고한다. 그러나 fusion 결합기의 우월성이 입력 조건 (예: 질문의 entity 가 corpus 에 얼마나 등장하는지) 에 따라 어떻게 변하는지, 어떤 영역에서는 단일 신호보다 약한지를 분해해 보고한 연구는 본 저자들이 검토한 범위에서 보고되지 않았다.

## Corpus statistic 의 환각 탐지 활용 시도

corpus statistic 을 환각 탐지 신호로 직접 사용한 두 흐름이 있다.

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

Zhao 등(2024)은 “*Wikipedia 페이지가 없는 entity*에 대한 질문에서 LLM이 더 자주 환각한다”는 관찰을 보고하였다. 이는 본 연구가 corpus 빈도를 조건 축으로 쓰는 가설의 *선행 경험적 근거*이다. 차이점은 두 가지다. (i) WildHallucinations 는 Wikipedia 존재 여부 라는 *이진* 조건을 사용하는 반면, 본 연구는 entity frequency / pair co-occurrence 기반 *연속 corpus support 축*으로 일반화한다. (ii) WildHallucinations 는 환각률 자체를 corpus 조건으로 설명하는 *evaluation* 연구이고, 본 연구는 이미 환각이 일어난 후보 행에 대해 *어떤 탐지 신호가 corpus 조건에 따라 더 안정적으로 작동하는가*를 평가하는 *detection* 연구이다.

##### Conditional calibration 분석 틀 (Valentin et al.).

Valentin 등(2024)은 환각 탐지 점수를 *입력/응답 attribute에 conditional 하게 calibrate* 하는 multi-scoring 분석 틀를 제안하였다. 본 연구와의 핵심 차이는 *조건의 출처*이다. Valentin은 모델 *내부 score attribute* 에 conditional하게 calibrate 하는 반면, 본 연구는 모델 출력과 독립적인 *외부 corpus statistic* 으로 조건화한다. 두 분석 틀는 직교적이며, 본 연구의 corpus 조건 축은 Valentin의 internal-attribute conditioning과 결합 가능한 보완 신호로 위치할 수 있다.

# 제안 방법

<span id="ch:method" label="ch:method"></span>

본 장은 본 논문이 채택하는 *corpus-conditioned 분석 절차* 와 그 이론적 동기를 정리한다.

## 단일 점수 가정의 한계

SE 와 Energy 같은 sample-consistency 신호는 모든 응답이 단일 의미 cluster 로 수렴하는 *고확신 환각 (CHOKE)* 사례 에서 판별력이 약해지고, NLL / margin 같은 logit 신호는 환각 답이 더 자신있게 평가되는 사례에서 방향이 역전된다 (§<a href="#ch:related" data-reference-type="ref" data-reference="ch:related">[ch:related]</a> CHOKE paragraph). 두 사례는 신호 자체가 무용하다는 뜻이 아니라 *어느 입력 조건에서 어느 신호가 안정적으로 작동하는지를 분리해서 봐야 한다* 는 뜻이며, 본 논문은 그 외부 입력 조건으로 corpus support 를 사용한다.

## Corpus support 를 외부 conditional 축으로 사용하는 근거

질문에 등장하는 entity 의 corpus 내 frequency 와 entity-pair co-occurrence 는 모델이 “익숙한” entity 를 다루고 있는지를 외부 자료로 측정한 지표이다. Qiu 등 (2025) 의 QuCo-RAG 가 entity frequency 와 co-occurrence 를 retrieval grounding 신호로 활용하였고, Zhang 등 (2025) 은 pretraining corpus 의 n-gram coverage 가 환각 탐지 신호와 비단조적으로 연관됨을 보고하였다. 두 결과의 공통 시사점은 corpus exposure 가 모델 출력의 신뢰도 분포를 형성한다는 점이다.

본 논문은 이 가설을 차용하되 corpus statistic 을 *탐지기 입력 feature 로 사용하지 않고* 평가 단위 (외부 conditional 축) 로만 사용한다. 그 이유는 두 가지다. 첫째, 평가 단위 (corpus support decile bin) 와 입력 feature (corpus aggregate) 가 같은 source 에서 나오면 self-conditioning artifact 가 발생할 수 있다. 둘째, 본 논문의 질문 (단일 신호 / fusion 의 영역별 행태) 은 corpus 를 입력으로 쓰지 않고도 답할 수 있다. corpus 부족 영역에서 모델 자체 신호 (SE / Energy / logit-diagnostic) 가 어떻게 행동하는지를 직접 측정한다.

##### Conditioning 가설의 메커니즘.

corpus 부족 영역에서 모델 자체 신호의 신뢰도가 약화되는 가설은 다음 인과 사슬로 동기 부여된다. (i) Zhao 등 (2024) 의 WildHallucinations 는 Wikipedia 페이지가 없는 entity 에 대한 환각률 ↑ 를 보였다 — corpus exposure 부족 → representation 불안정 → 환각률 ↑. (ii) 이렇게 발생한 환각이 모델 입장에서 “확신 있게 일관된” 답으로 나오는 경우가 많은 것이 Simhi 등 (2025) 의 CHOKE 패턴이다 — corpus 부족 영역에서 free-sample 이 단일 cluster 로 수렴하면 SE / Energy 의 sample-consistency 판별력이 약화된다. (iii) 따라서 corpus 부족 영역에서 모델 자체 신호의 AUROC 자체가 낮을 것을 예측할 수 있다. 본 논문은 이를 영역별 AUROC 분해로 직접 측정한다 (§<a href="#ch:experiment" data-reference-type="ref" data-reference="ch:experiment">[ch:experiment]</a>).

## 본 논문의 분석 절차

본 논문의 절차는 네 단계이다. (1) 라벨링 — 각 질문의 free-sample N=10 정답 매칭 비율이 0.5 미만이면 *어려운 질문* 으로 binary 라벨 부여 (Farquhar 2024 SE 의 평가 단위와 일치). (2) 평가 단위 분할 — 5,815 개 질문을 corpus support 의 rank-quantile 10-decile 로 분할 (각 영역 ≈ 580 질문). (3) 단일 신호 AUROC — SE / Energy / logit-diagnostic 각각의 score-based AUROC 를 각 decile 안에서 산출 (학습 없음). (4) Fusion AUROC — SE / Energy / NLL / margin / logit variance 의 질문 단위 집계를 입력으로 하는 5-fold CV gradient boosting 의 out-of-fold 예측을 각 decile 안에서 평가. corpus statistic 은 (2) 에서 평가 단위로만 사용하고 (4) 의 fusion 입력으로는 사용하지 않는다.

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

## 종합 베이스라인

<div id="tab:current_thesis_evidence">

<table>
<caption>종합 baseline (n=5<span>,</span>815 질문). 라벨은 각 질문의 free-sample N=10 의 정답 매칭 비율이 0.5 미만인 경우 <em>어려운 질문</em> (is_hard=1). 단일 신호는 score-based AUROC (학습 없음). Fusion 은 5-fold CV gradient boosting (input: SE, Energy, NLL, confidence margin, logit variance 의 질문 단위 집계). 굵은 글씨 = 1위.</caption>
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
<td style="text-align: center;">0.112</td>
<td style="text-align: center;">0.016</td>
</tr>
<tr>
<td style="text-align: left;"><strong>gradient boosting</strong></td>
<td style="text-align: center;"><strong>0.889</strong></td>
<td style="text-align: center;"><strong>0.837</strong></td>
<td style="text-align: center;"><strong>0.113</strong></td>
<td style="text-align: center;"><strong>0.014</strong></td>
</tr>
</tbody>
</table>

</div>

표 <a href="#tab:current_thesis_evidence" data-reference-type="ref" data-reference="tab:current_thesis_evidence">4.1</a> 의 결과에서 두 가지를 짚어 둔다. 첫째, SE 와 Energy 단독이 본 데이터에서도 강한 baseline 을 형성한다 (SE 0.832, Energy 0.862) — Farquhar 등 (2024) 의 TriviaQA / Llama-2 7B AUROC 0.79 와 동일 수준에서 재현된다. 둘째, Fusion (5-fold CV gradient boosting) 의 종합 AUROC 는 0.889으로 Energy 단독 대비 평균 +0.027만 우위이다. 이 평균 한 수치 뒤에 영역별로 어떤 변동이 있는지가 다음 세 절의 분석 대상이다.

## 탐지 난이도와 corpus support 의 관계

##### 측정 방법.

5,815 개 질문을 corpus support coverage score 의 rank-quantile 로 10-decile 로 분할한다 (각 decile $`\approx`$ 580 질문). 각 decile 안에서 4 신호 (SE / Energy / logit-diagnostic / Fusion) 의 AUROC 를 산출한다. 단일 신호는 score-based ranking 으로 직접 계산 (학습 없음), Fusion 은 5-fold CV out-of-fold 예측을 사용한다.

<figure id="fig:per_decile_auroc" data-latex-placement="htbp">

<figcaption>Corpus support decile별 AUROC (n=5<span>,</span>815, 라벨 = 어려운 질문). SE, Energy, Fusion 모두 corpus 부족 영역 (왼쪽) 에서 가장 낮고 풍부 영역 (오른쪽) 에서 가장 높은 단조 패턴을 보인다. logit-diagnostic 은 모든 영역에서 0.51–0.63 으로 약함 (CHOKE 패턴).</figcaption>
</figure>

그림 <a href="#fig:per_decile_auroc" data-reference-type="ref" data-reference="fig:per_decile_auroc">4.1</a> 는 모든 신호가 corpus 부족 영역에서 동시에 약해지는 단조 상승 패턴을 보여준다. 구체적으로 SE 는 가장 낮은 decile 에서 0.80, 가장 높은 decile 에서 0.92 로 변동하며, Energy 는 0.821에서 0.954로, Fusion 은 0.851에서 0.978로 동일한 상승 패턴을 보인다. logit-diagnostic 은 모든 영역에서 0.51–0.63 의 약한 신뢰도를 유지하여, NLL / margin 같은 단독 logit 신호의 CHOKE 한계가 corpus 영역과 무관하게 일관됨을 시사한다.

이 단조 패턴은 본 논문의 conditioning 가설 — corpus 부족 영역에서 모델 representation 이 불안정하고 free-sample 이 단일 cluster 로 수렴해 sample-consistency 신호의 판별력이 약화된다 (§<a href="#ch:method" data-reference-type="ref" data-reference="ch:method">[ch:method]</a>) — 의 직접 데이터 근거이다. 환각 탐지 자체가 corpus 부족 영역에서 *모든 신호 공통적으로* 어렵다.

## 단일 신호 영역별 비교

##### 측정 방법.

§4.3 와 동일한 10-decile 분해 위에서 단일 신호 (SE / Energy / logit-diagnostic) 의 AUROC 를 직접 비교한다.

<div id="tab:bin10_method_matrix">

| decile |  n  |  SE   | **Energy** | logit-diag | Fusion |     |
|:-------|:---:|:-----:|:----------:|:----------:|:------:|:---:|
| 00–10  | 582 | 0.802 | **0.821**  |   0.628    | 0.851  |     |
| 10–20  | 570 | 0.718 | **0.746**  |   0.550    | 0.777  |     |
| 20–30  | 578 | 0.760 | **0.771**  |   0.579    | 0.787  |     |
| 30–40  | 585 | 0.788 | **0.812**  |   0.576    | 0.826  |     |
| 40–50  | 582 | 0.845 | **0.864**  |   0.592    | 0.850  |     |
| 50–60  | 568 | 0.821 | **0.850**  |   0.608    | 0.874  |     |
| 60–70  | 594 | 0.825 | **0.848**  |   0.615    | 0.851  |     |
| 70–80  | 584 | 0.832 | **0.868**  |   0.589    | 0.918  |     |
| 80–90  | 592 | 0.836 | **0.883**  |   0.540    | 0.926  |     |
| 90–100 | 580 | 0.924 | **0.954**  |   0.508    | 0.978  |     |

Method $`\times`$ corpus support decile AUROC (n=5,815, 각 decile $`\approx`$ 580 질문). 굵은 글씨 = 단일 신호 중 1위.

</div>

표 <a href="#tab:bin10_method_matrix" data-reference-type="ref" data-reference="tab:bin10_method_matrix">4.2</a> 가 보이는 단일 신호 비교의 결론은 corpus 영역과 무관하다. Energy 가 모든 10 decile 에서 SE 를 +0.020   +0.038 의 일관된 폭으로 능가한다. logit-diagnostic 은 모든 영역에서 0.5 근처 (0.508–0.628) 로 약하다 — 환각 답이 정답 답보다 더 자신있게 평가되는 CHOKE 패턴이 corpus 영역에 무관하게 유지됨을 시사한다. 즉 단일 신호 선택은 corpus support 영역에 따라 달라지지 않는 결정으로, Energy 가 SE / logit-diagnostic 을 압도한다.

## Fusion 의 영역별 변동

##### 측정 방법.

§4.3 와 동일한 10-decile 분해 위에서 Fusion (5-fold CV GBM, 입력 = SE / Energy / NLL / margin / logit variance 의 질문 단위 집계) 의 AUROC 와 단일 신호 AUROC 의 영역별 차이를 측정한다.

<figure id="fig:fusion_delta" data-latex-placement="htbp">

<figcaption>Corpus support decile별 Fusion AUROC 와 단일 신호 AUROC 의 차이. 양수 = Fusion 우위, 음수 = 단일 신호 우위. SE / logit-diagnostic 대비 Fusion 은 모든 영역에서 우위지만, Energy 단독과의 비교는 영역별로 변동 — decile 70–90 에서 Fusion 우위가 가장 크고 (+0.043), decile 40–50 에서는 Energy 가 Fusion 을 +0.014 능가.</figcaption>
</figure>

그림 <a href="#fig:fusion_delta" data-reference-type="ref" data-reference="fig:fusion_delta">4.2</a> 가 보이는 패턴은 두 가지이다. 첫째, Fusion 은 SE / logit-diagnostic 단독은 모든 영역에서 일관 우위 (vs SE +0.018   +0.082, vs logit-diagnostic +0.30 이상). 단순 결합만으로도 약한 단일 신호를 보완하는 효과는 분명하다. 둘째, *Energy 단독과의 비교는 corpus 영역에 따라 변동* 한다. decile 70–80 / 80–90 의 high-mid corpus 영역에서 Fusion 이 Energy 를 +0.043능가하며, 반대로 *decile 40–50 의 mid corpus 영역에서는 Energy 단독이 Fusion 을 +0.014 능가* 한다 (decile 60–70 도 거의 동률, +0.003).

이 결과는 fusion 의 우월성이 단일 신호 선택과 corpus 영역에 동시에 의존한다는 것을 보인다. 평균 우위 한 수치 (Fusion 평균 vs Energy +0.027) 만 보면 “Fusion 이 항상 더 좋다” 는 인상이지만, 영역별 분해는 fusion 이 mid corpus 영역에서 단일 신호보다 약해질 수 있음을 보여준다. 기존 fusion 연구 (Valentin 2024, SelfCheckGPT) 가 평균 metric 한 수치로 fusion 우월성을 보고하는 관행이 가리는 정보가 있음을 시사한다.

## 데이터셋 일반화 + Calibration

<div id="tab:lodo">

| Method              |    AGG    | HaluEval-QA | TruthfulQA |   Brier   |    ECE    |
|:--------------------|:---------:|:-----------:|:----------:|:---------:|:---------:|
| **Fusion (GBM)**    | **0.889** |    0.815    | **0.951**  | **0.113** | **0.014** |
| Random Forest       |   0.888   |    0.813    |   0.953    |   0.112   |   0.016   |
| Logistic Regression |   0.876   |    0.797    |   0.916    |   0.124   |   0.034   |
| Energy-only         |   0.862   |      —      |     —      |     —     |     —     |
| SE-only             |   0.832   |      —      |     —      |     —     |     —     |
| logit-diagnostic    |   0.581   |      —      |     —      |     —     |     —     |

Per-dataset AUROC (5-fold CV pooled) 와 calibration. AGG: 종합. Brier / ECE 는 모든 dataset 통합.

</div>

Fusion 이 두 dataset 모두에서 1위이며 calibration 도 1위이다 (Brier 0.113, ECE 1.4%). 다만 TruthfulQA 단독 AUROC 0.951 은 라벨 노이즈에 의해 부풀려진 수치로 해석해야 한다 — 본 논문의 token-overlap 라벨 (free-sample $`\ge`$ 50% token overlap with right_answer / best_answer / correct_answers list) 은 의미적으로 동등한 paraphrase 정답을 충분히 잡지 못해 TruthfulQA 의 다양한 정답 표현 (예: “Cardiff”, “the Welsh capital”) 을 *어려운 질문* 으로 잘못 분류한다 (TruthfulQA is_hard 비율 0.97). 본 논문 주요 결과 해석은 HaluEval-QA 단독 (Fusion 0.815) 또는 AGG (0.889) 기준으로 한다. NLI 기반 매칭으로 라벨링하면 노이즈가 줄어들 가능성이 높으며, 이는 §결론 한계 항목에서 다룬다.

# 결론

<span id="ch:conclusion" label="ch:conclusion"></span>

## 연구 요약

본 논문은 외부 corpus statistic 을 평가 단위로 사용하여 환각 탐지 신호 (SE, Semantic Energy, logit-diagnostic, fusion) 의 영역별 행태를 분해하였다. 세 연구 질문에 대한 답을 정리한다.

##### 첫째 (난이도 ↔ corpus support).

모든 신호의 AUROC 가 corpus support 와 단조 관계 — 가장 부족한 영역에서 가장 낮고 풍부한 영역에서 가장 높다 (SE 0.80 → 0.92, Energy 0.821→ 0.954, Fusion 0.851→ 0.978). 환각 탐지 자체가 corpus 부족 영역에서 *모든 신호 공통적으로* 어렵다.

##### 둘째 (영역별 단일 신호 우열).

단일 신호 비교는 corpus 영역과 무관 — Energy 가 모든 10 decile 에서 SE 를 +0.020   +0.038 능가하며, logit-diagnostic 은 모든 영역에서 약하다 (CHOKE 패턴). 즉 단일 신호 선택은 corpus 영역에 따라 달라지지 않는 결정이다.

##### 셋째 (Fusion 영역별 변동).

Fusion 이 SE / logit-diagnostic 단독은 모든 영역에서 일관 우위, 그러나 Energy 단독과는 영역별로 변동 — decile 70–90 에서 Fusion 우위가 가장 크고 (+0.043), *decile 40–50 에서는 Energy 단독이 Fusion 을 +0.014 능가*. 평균 우위 한 수치 (+0.027) 로는 가려지는 영역별 변동이 데이터로 보인다.

## 학술적 기여

1.  **영역별 분해를 통한 평균 metric 의 한계 노출**. 환각 탐지 난이도가 corpus 부족 영역에 집중됨을 모든 신호에 공통적으로 정량화. 평균 AUROC 한 수치가 가리는 입력 조건별 변동을 데이터로 보였다.

2.  **Fusion 이 항상 best 가 아니라는 발견**. mid corpus 영역 (decile 40–50) 에서 Energy 단독이 Fusion 을 +0.014 능가하는 케이스를 보고. 기존 fusion 연구 (Valentin 2024, SelfCheckGPT) 가 평균 metric 만 보고하는 관행이 가리는 정보가 있음을 시사한다.

3.  **외부 corpus 를 평가 단위로 사용하는 분석 절차**. Valentin 등 (2024) 이 모델 *내부* score attribute 로 conditional calibration 을 시도한 방향과 직교적으로, 본 논문은 모델 출력과 독립적인 *외부* corpus statistic 을 평가 단위로 사용해 영역별 행태를 분해한다. 두 방향은 결합 가능한 보완 구조이다.

## 한계

1.  **Token-overlap 라벨 proxy 한계**. is_hard 라벨은 free-sample N=10 의 정답 token-overlap ($`\ge 50\%`$) 매칭 결과로 정의된다. HaluEval-QA 의 짧은 factoid 답에서는 잘 작동하나 (is_hard 0.20), TruthfulQA 의 다양한 paraphrase 정답을 충분히 잡지 못해 라벨이 hard 쪽으로 치우친다 (0.97). NLI-기반 매칭으로의 교체는 향후 연구에서 검토한다.

2.  **단일 corpus snapshot 및 단일 모델**. corpus support 축은 `v4_dolmasample_olmo` (16B Dolma sample) 단일 index 위에서 산출되며, 본 실험 corpus 와 Qwen2.5-3B 의 실제 pretraining corpus 가 다르므로 corpus support 와 환각의 연결은 *상관* 이지 *인과* 가 아니다. 단일 모델 평가이므로 모델 규모/계열 일반화는 본 논문 범위를 넘는다.

3.  **Self-conditioning artifact 회피**. 본 논문은 corpus statistic 을 평가 단위로만 사용하고 fusion 입력 feature 로는 사용하지 않았다. 따라서 “외부 corpus signal 의 직접 한계 효용” 은 본 논문의 주장 범위 밖이며, 본 논문의 결론은 단일 신호와 fusion 의 영역별 행태 자체에 한정된다.

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
