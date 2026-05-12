# 서론

<span id="ch:intro" label="ch:intro"></span>

## 연구 배경

ChatGPT, Gemini 등으로 대표되는 LLM 은 인상적인 추론 및 질의응답 능력을
보이지만, 사실적으로 부정확하거나 지지되지 않는 답변 — 이른바
환각(hallucination) — 을 빈번하게 생성한다. 의료, 법률, 과학 등 고위험
도메인에서 환각은 중대한 오류로 이어질 수 있으며, 이에 따라 LLM 출력의
신뢰성을 평가하는 일반적인 방법론 개발이 중요한 연구 과제로 자리 잡았다.

환각 탐지의 핵심 접근법 중 하나는 불확실성 정량화(uncertainty
quantification)이다. 특히 Farquhar 등 (2024) 이 제안한 Semantic
Entropy(SE)는 동일 prompt 에 대해 N개의 sample 을 생성하고, NLI 기반
의미 cluster 의 확률 분포 엔트로피를 계산함으로써 모델의 의미적
불확실성을 포착한다. SE 는 모델, 데이터셋, 태스크에 걸쳐 강건하게
일반화되며 외부 지식 없이도 동작한다는 장점이 있다. 이후 Ma 등 (2025) 의
Semantic Energy 는 확률 대신 로짓(logit) 을 직접 활용해 SE 가 포착하지
못하는 에피스테믹 불확실성까지 추정한다.

그러나 이 두 방법 모두 모델 내부 신호에만 집중하며, 개별 prompt 의
*corpus support 조건* — 즉, 해당 사실(fact) 이 사전학습 corpus 에 얼마나
많이 등장했는지 — 에 따라 탐지 능력이 어떻게 달라지는지를 체계적으로
분석하지 않았다. 단순히 모델 전체의 평균 AUROC 를 보고하는 것은, 탐지
신호가 어떤 corpus support 영역에서 강하고 어느 영역에서 약한지를 가리는
한계가 있다. corpus 통계량을 환각 탐지에 활용한 선행 연구 가 있으나,
이들은 corpus signal 자체를 *탐지 신호* 로 사용하는 데 그쳤으며, corpus
signal 의 *단위(granularity)* 가 SE / Energy 의 decomposition 에 어떤
영향을 미치는지에 대한 체계적 분석은 존재하지 않는다.

## 연구 목적

본 논문의 핵심 질문은 다음과 같다: *환각 탐지 신호(SE, Semantic Energy,
sample-level token diagnostics) 의 corpus support 조건 의존성은 corpus
signal 의 단위(entity / fact-pair / phrase) 에 따라 어떻게 다르게
분해되는가?*

이를 위해 본 연구는 세 가지 세부 목표를 설정한다. 첫째, Farquhar (2024)
와 동일한 5개 데이터셋 및 평가 프레임워크(per-generation NLI
correctness) 위에서 Qwen2.5-3B 모델을 이용해 SE, Semantic Energy,
sample-level diagnostics 의 재현성을 검증한다. 둘째, entity,
entity-pair, QA-bridge, n-gram 등 단위가 다른 corpus signal 을 설계하고,
각 signal 이 단독 hallucination 탐지기로서 갖는 한계 (negative result)
를 정량화한다. 셋째, 동일 corpus signal 을 평가 axis 로 사용해 SE /
Energy 의 per-decile AUROC 변동을 분석하고, corpus signal 의 단위가
분해력에 어떤 영향을 미치는지를 규명한다.

##### 본 논문의 기여.

본 논문의 기여는 다음 세 가지이다. (1) 환각 탐지 신호의 corpus 조건부
행태를 단위 (granularity) 별로 multi-axis 분해하는 분석 프레임워크를
제안한다. (2) 본 표본에서 entity-pair co-occurrence axis 가 단순 entity
빈도 axis 보다 약 1.88배 큰 분해 진폭 ($`\Delta`$<!-- -->0.150 와
$`\Delta`$<!-- -->0.080) 을 보임을 single-condition point estimate 으로
관찰하였다 (CI 와 모델/corpus 일반화는
§<a href="#sec:limitations" data-reference-type="ref"
data-reference="sec:limitations">5.2</a> 참조). (3) corpus signal 의
설계 목표를 탐지 feature 에서 조건부 평가 도구 (conditional benchmarking
axis) 로 재정의하는 관점을 제안한다.

## 논문 구성

본 논문은 다음과 같이 구성된다.
제 <a href="#ch:related" data-reference-type="ref"
data-reference="ch:related">[ch:related]</a>장에서는 SE, Semantic
Energy, corpus 기반 환각 탐지, multi-signal fusion 관련 선행 연구를
정리한다. 제 <a href="#ch:method" data-reference-type="ref"
data-reference="ch:method">[ch:method]</a>장에서는 데이터셋, 실험 설정,
신호 정의, multi-axis 분해 프레임워크 등 제안 방법을 기술한다.
제 <a href="#ch:experiment" data-reference-type="ref"
data-reference="ch:experiment">[ch:experiment]</a>장에서는 단일 신호
평가, fusion 결과, multi-axis 분해 결과를 보고한다.
제 <a href="#ch:conclusion" data-reference-type="ref"
data-reference="ch:conclusion">[ch:conclusion]</a>장에서는 결과의 함의,
한계, 향후 연구 방향을 논의하고 결론을 맺는다.

# 관련 연구

<span id="ch:related" label="ch:related"></span>

본 장에서는 LLM 환각 탐지 연구를 sample-consistency 기반 탐지, logit
기반 불확실성 추정, corpus 통계 활용, multi-signal fusion 의 네 흐름으로
정리하고, 본 논문의 위치를 제시한다.

## Sample-Consistency 기반 환각 탐지

Farquhar 등 (2024) 의 Semantic Entropy(SE) 는 동일 prompt 에 대한 N개의
free-sample 을 NLI 양방향 entailment 로 의미 cluster 로 묶고, cluster
질량의 엔트로피를 환각 지표로 사용한다. 이 방법은 하나의 의미가 다양한
표현으로 나타날 수 있다는 언어적 특성을 해결하며, 태스크별 사전 지식
없이도 TriviaQA, SQuAD, BioASQ, NQ-Open, SVAMP 등 다양한 데이터셋에서
강건하게 작동한다. SE 는 naive entropy, $`p(\text{True})`$, embedding
회귀 기준선보다 일관되게 높은 AUROC 를 달성한다. SelfCheckGPT 가 시작한
sample-consistency 접근의 의미-단위 일반화이다.

Kossen 등 (2024) 의 Semantic Entropy Probes (SEPs) 는 SE 계산의 높은
연산 비용 문제를 해결하기 위해, 단일 생성의 hidden state 에서 직접 SE 를
근사하는 probe 를 학습한다. SEPs 는 다중 샘플링 없이도 높은 환각 탐지
성능을 유지하며, 분포 외 데이터에서도 강건한 일반화를 보인다.

Nikitin 등 (2024) 의 Kernel Language Entropy (KLE) 는 SE 를 일반화하여
hard clustering 대신 pairwise 의미 유사도 커널과 von Neumann 엔트로피를
사용한다. KLE 는 cluster 간 의존성을 더 세밀하게 포착하여 여러
데이터셋과 모델 아키텍처에서 SE 보다 높은 불확실성 정량화 성능을 보인다.
Ciosek 등 (2025) 은 Bayesian 접근법으로 SE 추정의 샘플 효율을 높여,
Farquhar 대비 53% 적은 샘플로 동등한 AUROC 를 달성함을 보인다.

## Logit 기반 불확실성 추정

Ma 등 (2025) 의 Semantic Energy 는 SE 가 확률 기반으로 포착하지 못하는
에피스테믹 불확실성을 보완하기 위해, 모델의 penultimate layer 로짓을
직접 활용하는 에너지 기반 불확실성 추정 프레임워크이다. 볼츠만 분포에서
영감을 받아 토큰 에너지 $`\tilde{E}(x_t) = -z_\theta(x_t)`$ 를 정의하고,
SE 와 동일한 semantic cluster 구조 위에서 cluster 에너지를 계산한다.
특히 SE 가 모든 sample 이 단일 cluster 에 집중될 때 분리력을 잃는
사례에서, Semantic Energy 는 평균 AUROC 13% 이상의 개선을 달성한다.
확률이 로짓 정규화 과정에서 강도 정보를 잃는다는 점이, LLM 의 불확실성
표현 능력을 제한하는 한 요인으로 지적된다.

Token-level 불확실성에 대한 다른 접근으로, Raghuvanshi 등 (2025) 은
token-level log-likelihood, 양방향 NLI contradiction 신호, Semantic
Entropy 를 결합한 하이브리드 탐지 파이프라인을 제안하고 SQuAD2.0 에서
AUC 0.818 을 달성함을 보고한다.

## Corpus 통계를 이용한 환각 탐지

LLM 의 환각은 종종 사전학습 corpus 에서 드물게 등장하는 사실과 연관된다.
HALoGEN (Ravichander 등 2025) 은 9개 도메인에 걸친 10,923 개의 포괄적
환각 벤치마크를 구축하고, LLM 생성물의 원자적 사실 단위를 고품질 지식
소스와 자동 검증하는 프레임워크를 제시한다. 이 연구는 최고 성능
모델에서도 생성된 원자적 사실의 최대 86% 가 환각일 수 있음을 발견하며,
환각의 원인을 학습 데이터 오류 회상(Type A), 학습 데이터 오류 지식(Type
B), 순수 조작(Type C) 으로 분류한다. WildHallucinations 는 *Wikipedia
페이지가 없는 entity* 에 대한 질문에서 LLM 환각률이 유의미하게 증가함을
관찰하여, corpus exposure 부족과 환각 발생률의 경험적 연관을 보고하였다.
Zhang 등 (2025) 은 RedPajama 1.3조 토큰 pretraining corpus 위에 suffix
array 를 구축하고 prompt / 답변의 n-gram 빈도를 환각 탐지 신호로
평가하였다.

Infini-gram 은 n-gram 카운팅을 통한 corpus 통계 질의를 지원하는
엔진으로, 특정 phrase 나 entity 가 사전학습 corpus 에 얼마나
등장하는지를 효율적으로 산출할 수 있다. 그러나 이를 hallucination 탐지의
corpus support 신호로 활용하는 연구는 아직 초기 단계이며, 어떤 단위
(entity / fact-pair / phrase) 의 corpus signal 이 탐지 신호와 어떻게
연계되는지에 대한 체계적 분석은 존재하지 않는다.

## Multi-Signal Fusion

단일 신호의 한계를 극복하기 위해 여러 신호를 결합하는 fusion 접근법이
제안되어 왔다. Raghuvanshi 등 (2025) 은 token-level 불확실성, NLI 신호,
SE 의 동적 가중 결합을 통해 calibration 한계를 완화하고 강건한 탐지를
달성한다. ECLIPSE (Singha 2025) 는 semantic entropy 와 novel perplexity
분해를 결합해 증거 활용 신호를 추가함으로써 금융 QA 에서 AUC 0.89 를
달성한다. SEReDeEP (Wang 2025) 는 RAG 환경에서 semantic entropy 와
attention 메커니즘 신호를 결합해 컨텍스트 의존성이 강한 환각 탐지를
강화한다. Valentin 등 (2024) 은 환각 탐지 점수를 모델 *내부* score
attribute 에 conditional 하게 calibrate 하는 multi-scoring framework 를
제안하였다.

이들 연구는 다양한 신호의 fusion 이 단일 신호 대비 성능 향상을 가져올 수
있음을 보이지만, corpus support signal 의 *역할* 에 대한 분석 — 입력
feature 로서의 한계 vs. 평가 axis 로서의 효용 — 은 부재하다. 본 논문은
corpus signal 을 fusion 입력에 추가했을 때의 lift 를 정량화하는 동시에,
같은 signal 을 *평가 axis* 로 전환했을 때의 분해력을 비교함으로써 이
공백을 메운다.

# 제안 방법

<span id="ch:method" label="ch:method"></span>

본 장에서는 데이터셋, 라벨, 신호 정의, multi-axis 분해 프레임워크,
fusion 모델 구성을 기술한다. 본 논문은 새 탐지기를 제안하기보다, 기존
신호 (SE, Energy, sample-level token diagnostics) 와 fusion 결합기가
generation-level correctness 를 corpus support *단위* 별로 얼마나 잘
분해하는지를 비교한다.

## 데이터셋 및 실험 설정

본 연구는 Farquhar 등 (2024)과 동일한 5개 QA 데이터셋을 사용한다:
TriviaQA 800개, SQuAD-1.1 800개, BioASQ 800개, NQ-Open 800개, SVAMP
300개, 총 3,500 prompts. 각 데이터셋에서 dataset 별 seed 를 고정한
결정론적 random shuffle 후 target 수만큼 선택하며, prompt_id 는
(dataset, split_id, source_index, source_row_id, prompt_hash) 의 stable
hash 로 생성하여 재실행 시 재현성을 확보한다.

모델은 Qwen/Qwen2.5-3B (base, instruction tuning 없음) 를 사용하며,
NVIDIA RTX 5090 32 GB CUDA 환경에서 float16 추론을 수행한다. Prompt
template 은 Farquhar (2024) 의 sentence-length 설정과 동일하게 context
passage 없이 “Answer the following question in a single brief but
complete sentence.\nQuestion: {q}\nAnswer:” 형식을 사용해 confabulation
을 유도한다. 각 prompt 당 N=10 free-sample 을 temperature=1.0,
top_p=0.9, top_k=50, max_new_tokens=64 설정으로 생성하며, 총 35,000
generation 이 평가 단위를 구성한다. 초기 max_new_tokens=64 에서 전체
35,000 generation 의 약 12.6% 가 truncation 되었고, 이로 인해 영향을
받은 2,426개 prompt 의 모든 sample 을 max_new_tokens=128 로 재생성하였다
(AUROC 변화는 $`\pm`$<!-- -->0.001 범위에 한정됨을 확인).

평가 지표로 AUROC, AURAC (Area Under Rejection-Accuracy Curve, Farquhar
2024 main metric), Brier Score, ECE 를 사용한다. Fusion 실험의
cross-validation 은 5-fold GroupKFold(prompt_id) 를 사용하여 같은 prompt
의 sample 이 같은 fold 에 배치되도록 강제한다(prompt 단위 leakage 차단).

## 라벨: Per-Generation `is_correct`

라벨 `is_correct` 는 NLI 양방향 entailment 기반으로 정의된다. 각 prompt
의 정답 후보 집합 $`C`$ 는 데이터셋 annotation 의 `best_answer` /
`correct_answers` / `alias_list` 를 lowercase 정규화 후 dedup 하여
구성한다. 생성 sample $`s_i`$ 에 대한 매칭 점수는 다음과 같이 정의된다:
``` math
\begin{align*}
m(s_i, c) &= \max\bigl(p_\text{entail}(c \rightarrow s_i),\ p_\text{entail}(s_i \rightarrow c)\bigr), \\
M(s_i)   &= \max_{c \in C} m(s_i, c).
\end{align*}
```
$`M(s_i) \geq 0.5`$ 이면 $`s_i`$ 를 정답 (`is_correct`=1), 그렇지 않으면
환각 (0) 으로 라벨링한다. NLI 모델은 `microsoft/deberta-large-mnli` 를
사용하며, 이는 Farquhar (2024) 의 DeBERTa 기반 접근법과 일치한다. 총
35,000 binary 라벨이 산출된다. 본 평가 단위는 Farquhar (Nature 2024)
Semantic Entropy 와 Ma (2025) Semantic Energy 의 generation-level
correctness 평가와 일치하므로 직접 비교가 가능하다.

## Semantic Entropy (SE)

Semantic Entropy 는 Farquhar 등 (2024) 의 정의를 그대로 따른다. 동일
prompt 에 대한 N=10 sample 을 NLI 양방향 entailment 로 의미 cluster
$`\mathbb{C}_k`$ 로 묶은 뒤, sequence log-likelihood 를 이용한 cluster
질량을 계산한다:
``` math
\begin{align*}
\log \tilde{p}(\mathbb{C}_k) &= \log\!\sum_{x^{(i)} \in \mathbb{C}_k} \exp\!\Bigl(\sum_t \log p_\theta(x_t^{(i)})\Bigr), \\
p(\mathbb{C}_k) &= \frac{\tilde{p}(\mathbb{C}_k)}{\sum_{k'} \tilde{p}(\mathbb{C}_{k'})}.
\end{align*}
```
최종 SE 값은
``` math
\mathrm{SE} = -\sum_{k} p(\mathbb{C}_k)\, \log p(\mathbb{C}_k)
```
로 정의되며, prompt 단위로 계산되어 같은 prompt 의 모든 sample 에
broadcast 된다.

## Semantic Energy

Semantic Energy 는 Ma 등 (2025) Eq. (11)–(14) 를 paper-faithful 하게
구현한다. SE 와 동일한 semantic cluster 구조를 공유하며, 확률 대신
로짓을 직접 활용해 모델의 에피스테믹 불확실성을 포착한다.
``` math
\begin{align}
\tilde{E}(x_t^{(i)}) &= -z_\theta(x_t^{(i)}) & \text{(token energy)} \\
E(x^{(i)}) &= \tfrac{1}{T_i}\sum_{t} \tilde{E}(x_t^{(i)}) & \text{(sample energy)} \\
E_{\text{Bolt}}(\mathbb{C}_k) &= \sum_{x^{(i)} \in \mathbb{C}_k} E(x^{(i)}) & \text{(cluster total)} \\
U(\mathbf{x}) &= \sum_{k} p(\mathbb{C}_k)\, E_{\text{Bolt}}(\mathbb{C}_k) & \text{(prompt 단위 uncertainty)}
\end{align}
```
값이 낮을수록 신뢰도가 높음을 의미한다.

## Per-Sample Token Logit 통계

Sample-level token logit 통계는 각 generation row
$`(\text{prompt\_id}, \text{sample\_index})`$ 에 대해 다음 네 가지
신호를 산출한다.

- `sample_nll`:
  $`\displaystyle -\tfrac{1}{T_i}\sum_t \log p(x_t^{(i)})`$ —
  token-level mean negative log-prob.

- `sample_sequence_log_prob`: $`\displaystyle \sum_t \log p(x_t^{(i)})`$
  — 전체 sequence likelihood.

- `sample_logit_variance`:
  $`\displaystyle \mathrm{Var}_t\!\bigl(z_\theta(x_t^{(i)})\bigr)`$ —
  sample 내 token 자신감 분산.

- `sample_logsumexp_mean`:
  $`\displaystyle \tfrac{1}{T_i}\sum_t \log Z_t^{(i)}`$ — 분포 flatness
  (높을수록 uncertain).

이 신호들은 sample-level 변동 정보를 담아 fusion 모델의 주요 입력으로
사용된다.

## Corpus Signal 설계

Corpus signal 은 Infini-gram local engine (index: `v4_dolmasample_olmo`,
16B token, OLMo-7B-hf tokenizer) 을 backend 로 사용한다. 본 논문은
단위가 다른 4가지 corpus signal 을 설계한다.

##### Entity 수준 신호.

spaCy `en_core_web_lg` NER 로 12개 label (PERSON, ORG, GPE, LOC, DATE,
EVENT, WORK_OF_ART, FAC, NORP, PRODUCT, LANGUAGE, LAW) entity 를
추출한다.
``` math
\begin{align*}
\mathrm{entity\_frequency\_axis} &= \frac{\log(1 + \min_{e \in E} \mathrm{freq}(e))}{\log(1 + 10^6)}, \\
\mathrm{entity\_pair\_cooccurrence\_axis} &= \frac{\log\!\bigl(1 + \tfrac{1}{|P|}\sum_{(e_i,e_j) \in P}\mathrm{cooc}(e_i,e_j)\bigr)}{\log(1 + 10^5)}.
\end{align*}
```
여기서 $`\mathrm{freq}(e)`$ 는 Infini-gram 단일 entity count,
$`\mathrm{cooc}(e_i,e_j)`$ 는 `count_cnf` AND query 결과이다. 본 논문의
*기존* corpus axis 는 두 값의 평균 (`corpus_axis_bin_10`) 이며, 본
논문은 두 단위를 분리해 보고하는 것이 핵심이다.

##### QA-bridge 동시 등장 신호.

$`E_q`$ = 질문 entity, $`E_a`$ = 답변 entity $`- E_q`$ (질문 entity
제거로 paraphrase 노이즈 차단). 모든 pair
$`(e_q, e_a) \in E_q \times E_a`$ 에 대해 Infini-gram `count_cnf` 를
산출한다. $`\mathrm{qa\_bridge\_axis}`$ 는 이 값들의 정규화 평균이다.
이는 (질문 entity, 답변 entity) 쌍이 corpus 에 함께 등장한 빈도, 즉 LLM
이 해당 fact 를 학습할 기회를 근사한다.

##### N-gram 등장 빈도 신호.

답변 token sequence 의 모든 n-gram (n=3, 5) 에 대해 Infini-gram `count`
를 산출한다. $`\mathrm{ans\_ngram\_n\_axis}`$ 는 해당 count 의 정규화
평균, $`\mathrm{ans\_ngram\_n\_zero\_count}`$ 는 corpus 에 0회 등장한
n-gram 수이다. 이는 답변 phrase 가 LLM 사전학습에 등장한 횟수를 직접
측정한다.

## Multi-Axis 분해 프레임워크

본 논문의 핵심 분석 도구는 corpus signal 을 환각 탐지 feature 가 아닌
*평가 axis* 로 전환하는 것이다. 각 corpus signal 을 prompt 단위
rank-quantile 10-decile bin 으로 이산화하여 다음 axis 를 구성한다:

- `corpus_axis_bin_10` (entity_freq + pair_cooc 평균)

- `entity_frequency_axis_bin_10`, `entity_pair_cooccurrence_axis_bin_10`

- `qa_bridge_axis_bin_10`

- `ans_ngram_3_axis_bin_10`, `ans_ngram_5_axis_bin_10`,
  `ans_ngram_3_zero_count_bin_10`

같은 prompt 의 모든 sample 이 같은 bin 값을 공유한다 (prompt-level
broadcast).

각 axis 에서 10개 decile 구간별로 SE, Energy, sample_nll 의 AUROC 를
계산하고, $`\Delta = \mathrm{AUROC}_{\max} - \mathrm{AUROC}_{\min}`$ 의
변동 폭을 axis 의 *분해력 (decomposition power)* 지표로 정의한다.
$`\Delta`$ 가 클수록 해당 corpus signal axis 가 환각 탐지 신호의
corpus-conditional 행태를 잘 분리함을 의미한다.

## Fusion 모델

Fusion 모델의 입력은 두 가지 구성으로 실험한다.

- **CORE_INPUTS**: prompt-level broadcast 신호 {SE, Semantic Energy} 와
  sample-level 신호 {`sample_nll`, `sample_logit_variance`,
  `sample_logsumexp_mean`, `sample_sequence_log_prob`}.

- **CORE_INPUTS + CORPUS_INPUTS**: 위 신호에 모든 corpus signal (entity
  / qa_bridge / n-gram) 을 prompt 단위 broadcast 로 추가.

분류 모델로는 Logistic Regression, Random Forest, Gradient Boosting 을
비교한다. 모든 실험은 5-fold GroupKFold(prompt_id) 로 평가해 data
leakage 를 방지한다.

# 실험

<span id="ch:experiment" label="ch:experiment"></span>

## 종합 baseline

본 절에서는 단일 신호와 fusion 의 종합 AUROC, AURAC, Brier 를 보고한다.

<div id="tab:current_thesis_evidence">

<table>
<caption>종합 baseline. n=35<span>,</span>000 generation (Qwen2.5-3B, 5
datasets, N=10 free-sample). 라벨 정의 / 평가 단위 / fusion 입력 / AURAC
정의는 §<a href="#sec:label" data-reference-type="ref"
data-reference="sec:label">3.2</a>–§<a href="#sec:fusion_def"
data-reference-type="ref" data-reference="sec:fusion_def">3.8</a> 참조.
<em>corpus 추가 lift</em> = 같은 분류기에서 CORE_INPUTS 만
vs. CORE+CORPUS_INPUTS 의 AUROC 차이. 굵은 글씨 = AUROC 1위.</caption>
<thead>
<tr>
<th style="text-align: left;">신호</th>
<th style="text-align: center;">AUROC</th>
<th style="text-align: center;">AURAC</th>
<th style="text-align: center;">Brier</th>
<th style="text-align: center;">corpus 추가 lift</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>단일 신호 (학습 없음,
sample-consistency &amp; token logit)</em></td>
</tr>
<tr>
<td style="text-align: left;">logit-variance</td>
<td style="text-align: center;">0.620</td>
<td style="text-align: center;">0.400</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">sequence-log-prob</td>
<td style="text-align: center;">0.656</td>
<td style="text-align: center;">0.466</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">sample_nll (logit-diagnostic)</td>
<td style="text-align: center;">0.670</td>
<td style="text-align: center;">0.468</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">Semantic Entropy (SE)</td>
<td style="text-align: center;">0.759</td>
<td style="text-align: center;">0.526</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">Semantic Energy</td>
<td style="text-align: center;">0.774</td>
<td style="text-align: center;">0.533</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>단일 corpus signal (학습
없음) — Negative result</em></td>
</tr>
<tr>
<td style="text-align: left;">entity_pair_cooccurrence_axis</td>
<td style="text-align: center;">0.546</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">entity_frequency_axis</td>
<td style="text-align: center;">0.565</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">ans_ngram_3_axis</td>
<td style="text-align: center;">0.563</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">qa_bridge_mean</td>
<td style="text-align: center;">0.583</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">ans_ngram_3_zero_count</td>
<td style="text-align: center;">0.583</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Fusion
(GroupKFold(prompt_id) 5-fold OOF)</em></td>
</tr>
<tr>
<td style="text-align: left;">logistic regression (no corpus)</td>
<td style="text-align: center;">0.793</td>
<td style="text-align: center;">0.548</td>
<td style="text-align: center;">0.164</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">logistic regression (with corpus)</td>
<td style="text-align: center;">0.795</td>
<td style="text-align: center;">0.550</td>
<td style="text-align: center;">0.162</td>
<td style="text-align: center;"><span
class="math inline">+0.002</span></td>
</tr>
<tr>
<td style="text-align: left;">random forest (no corpus)</td>
<td style="text-align: center;">0.790</td>
<td style="text-align: center;">0.546</td>
<td style="text-align: center;">0.166</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">random forest (with corpus)</td>
<td style="text-align: center;">0.800</td>
<td style="text-align: center;">0.555</td>
<td style="text-align: center;">0.161</td>
<td style="text-align: center;"><span
class="math inline">+0.010</span></td>
</tr>
<tr>
<td style="text-align: left;">gradient boosting (no corpus)</td>
<td style="text-align: center;">0.800</td>
<td style="text-align: center;">0.553</td>
<td style="text-align: center;">0.161</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;"><strong>gradient boosting (with
corpus)</strong></td>
<td style="text-align: center;"><strong>0.808</strong></td>
<td style="text-align: center;"><strong>0.559</strong></td>
<td style="text-align: center;"><strong>0.158</strong></td>
<td style="text-align: center;"><span
class="math inline">+0.008</span></td>
</tr>
</tbody>
</table>

</div>

표 <a href="#tab:current_thesis_evidence" data-reference-type="ref"
data-reference="tab:current_thesis_evidence">4.1</a> 에서 세 가지를
정리한다. 첫째, SE 와 Energy 단독 신호는 본 데이터에서 AUROC 0.759 /
0.774 로, Farquhar (2024) 의 TriviaQA / Llama-2 7B AUROC 0.79 및 Ma
(2025) 의 Qwen3-8B 보고치 범위 (0.75–0.85) 와 비슷한 범위에서 관찰된다.
이는 본 표본이 SE / Energy 의 일반적인 작동 범위에 속함을 시사하나
cross-model 일반화의 충분 조건은 아니다 (한계
§<a href="#sec:limitations" data-reference-type="ref"
data-reference="sec:limitations">5.2</a>). 둘째, 모든 단일 corpus signal
의 AUROC 는 0.50–0.58 범위에 한정되어 단독 환각 탐지기로서의 분리력이
제한적이다 (§<a href="#sec:negative" data-reference-type="ref"
data-reference="sec:negative">4.3</a>). 셋째, fusion 의 corpus 추가 lift
는 최대 +0.010 (RF), GBM 기준 +0.008 으로, corpus signal 이 fusion
feature 로 기여하는 폭 또한 제한적이다.

## 단일 신호 평가: Cross-Paper 검증

Farquhar (2024) 와 동일한 5개 데이터셋, 동일한 평가 단위(per-generation
NLI correctness) 위에서 Qwen2.5-3B 의 SE 및 Semantic Energy AUROC 를
측정한다.
표 <a href="#tab:single_signal_aurac" data-reference-type="ref"
data-reference="tab:single_signal_aurac">4.2</a> 가 그 결과이다.

<div id="tab:single_signal_aurac">

| 신호 | AUROC | AURAC | 비교 (선행 연구) |
|:---|:--:|:--:|:---|
| Semantic Entropy | 0.759 | 0.526 | Farquhar (2024) TriviaQA 0.79, SQuAD 0.83 (Llama-2 7B) |
| Semantic Energy | 0.774 | 0.533 | Ma (2025) TriviaQA $`\sim`$<!-- -->0.85 (Qwen3-8B) |
| sample_nll | 0.670 | 0.468 | — |
| sample_sequence_log_prob | 0.656 | 0.466 | — |
| sample_logit_variance | 0.620 | 0.400 | — |

단일 신호 AUROC / AURAC. 모든 신호는 score-based 평가 (학습 없음). 비교
컬럼은 SE / Semantic Energy 원 논문의 다른 모델에서 보고된 AUROC
범위이다.

</div>

Qwen2.5-3B 는 Llama-2 7B 보다 작은 모델임에도 SE / Energy AUROC 가
Farquhar 보고치 (0.75–0.85) 와 비슷한 범위에서 관찰되며, 본 표본이 SE /
Semantic Energy 의 일반적인 작동 범위에 속함을 시사한다 (직접
cross-model 재현이 아니므로 reference comparison 으로 한정한다).
`sample_nll` (0.670) 은 단독으로도 의미 있는 신호이나 SE / Energy 보다
0.09–0.10 낮아, 의미 cluster 정보의 부가 효과를 확인할 수 있다.

## 단일 corpus signal 평가

Corpus signal 각각의 단독 AUROC 를 측정한다
(표 <a href="#tab:corpus_only" data-reference-type="ref"
data-reference="tab:corpus_only">4.3</a>).

<div id="tab:corpus_only">

| Signal                        | AUROC |
|:------------------------------|:-----:|
| qa_bridge_mean                | 0.583 |
| ans_ngram_3_zero_count        | 0.583 |
| qa_bridge_zero_flag           | 0.574 |
| entity_frequency_axis         | 0.565 |
| ans_ngram_3_axis              | 0.563 |
| ans_ngram_5_axis              | 0.557 |
| entity_pair_cooccurrence_axis | 0.546 |
| qa_bridge_axis                | 0.511 |
| entity_frequency_mean         | 0.500 |

단일 corpus signal 의 환각 탐지 AUROC. 모든 신호는 prompt 단위 broadcast
후 score-based 로 평가하였다. 모든 corpus signal AUROC 가 0.50–0.58
범위에 한정된다.

</div>

모든 corpus signal 의 단독 AUROC 는 0.50–0.58 범위에 한정되어 환각
탐지기로서의 단독 분리력은 제한적이다. 단, 이 결과는 자명하지만 정량화가
필요한 진술로 해석한다. corpus signal 은 prompt 단위로만 broadcast 된
정적 통계량이므로, 같은 prompt 의 N=10 sample 사이의 정오 변동을 분리할
수 없는 것은 측정 단위 불일치에 따른 자연스러운 결과이다. 본 절은 corpus
signal 이 환각 탐지에 무용하다는 주장이 아니라, prompt 단위 broadcast 된
정적 corpus signal 은 prompt 내 sample 변동 (per-generation label) 을
분리하는 데 적합하지 않다는 점을 정량화하여 후속 분석의 동기를 제시하는
데 그 목적이 있다. 주목할 점은 후속 절
(§<a href="#sec:axis" data-reference-type="ref"
data-reference="sec:axis">4.5</a>) 에서 가장 큰 axis 분해 진폭을 보이는
`entity_pair_cooccurrence_axis` 가 단독 AUROC 0.546 으로
표 <a href="#tab:corpus_only" data-reference-type="ref"
data-reference="tab:corpus_only">4.3</a> 하위권에 속한다는 것이다. 단독
분리력 (per-generation label 위) 과 axis 분해 진폭 (decile별 신호 변동
위) 은 서로 다른 지표이며, 이 비대칭성이 본 논문의 핵심 결과로 이어진다.

## Fusion 결과

Fusion 모델의 AUROC 및 corpus signal 추가 효과를 측정한다
(표 <a href="#tab:fusion" data-reference-type="ref"
data-reference="tab:fusion">4.4</a>).

<div id="tab:fusion">

| Method                |   AUROC   |   AURAC   | corpus 추가 lift |
|:----------------------|:---------:|:---------:|:----------------:|
| Energy-only           |   0.774   |   0.533   |     — (기준)     |
| SE-only               |   0.759   |   0.526   |        —         |
| LR (no corpus)        |   0.793   |   0.548   |        —         |
| LR (with corpus)      |   0.795   |   0.550   |    $`+0.002`$    |
| RF (no corpus)        |   0.790   |   0.546   |        —         |
| RF (with corpus)      |   0.800   |   0.555   |    $`+0.010`$    |
| GBM (no corpus)       |   0.800   |   0.553   |        —         |
| **GBM (with corpus)** | **0.808** | **0.559** |    $`+0.008`$    |

Fusion AUROC. 모든 모델은 5-fold GroupKFold(prompt_id) OOF 로 평가.
*corpus 추가 lift* 는 같은 분류기에서 CORE_INPUTS 만 vs. CORE +
CORPUS_INPUTS 의 AUROC 차이.

</div>

세 가지 관찰이 도출된다. 첫째, Energy 단독 (0.774) 대비 GBM (no corpus,
0.800) 의 +0.026 lift 는 sample-level token logit 통계 (NLL, logit
variance 등) 와 prompt-level cluster 신호의 결합에서 비롯된다. 서로 다른
단위 (granularity) 의 불확실성 신호 결합이 단일 신호와 비교해 성능
향상에 기여함을 확인한다. 둘째, corpus signal 추가 시 lift 는 최대
+0.010 (RF), GBM 기준 +0.008 (95% prompt-grouped bootstrap CI \[+0.005,
+0.011\], B=500, 모든 iteration 에서 양수) 으로, 통계적으로 0 과
구분되는 양의 효과이나 절대 크기는 제한적이다. corpus signal 의 fusion
feature 로서의 기여 폭은 sample-level / prompt-level 신호 결합 효과
(+0.026) 의 약 1/3 정도에 해당한다. 셋째, fusion feature 로서의 기여는
제한적인 반면 평가 axis 로 사용했을 때의 분해 진폭은 큰 비대칭은 다음
절의 결과와 연결된다.

## Multi-Axis 분해 결과

각 corpus signal axis 의 10-decile 위에서 SE, Semantic Energy,
`sample_nll` 의 AUROC 변동 폭
($`\Delta = \mathrm{AUROC}_{\max} - \mathrm{AUROC}_{\min}`$) 을
측정한다. 이것이 본 논문의 핵심 표
(표 <a href="#tab:axis_decomp" data-reference-type="ref"
data-reference="tab:axis_decomp">[tab:axis_decomp]</a>) 이다.

두 axis 의 분해 패턴을 시각적으로 비교한 결과는
그림 <a href="#fig:axis_decomp" data-reference-type="ref"
data-reference="fig:axis_decomp">4.1</a> 에 제시한다.

<figure id="fig:axis_decomp" data-latex-placement="htbp">

<figcaption>Entity-pair co-occurrence axis 와 entity-frequency axis 의
per-decile SE AUROC. 같은 데이터, 같은 신호 (SE) 위에서 corpus signal
단위만 바꿔 분해한 결과이다. entity-pair axis 는 corpus 부족 영역
(decile 00–10, AUROC 0.643) 에서 가장 풍부한 영역 (max 는 decile 70–80,
0.793) 까지 <span class="math inline"><em>Δ</em></span>0.150 의 진폭을
보이며, SE 의 corpus 조건부 행태를 분리하는 정도가 단순 entity-frequency
axis (<span class="math inline"><em>Δ</em></span>0.080) 보다 약 1.88배
크다. 두 axis 모두 decile-by-decile 단조성은 엄밀히 성립하지 않는다 (예:
entity_pair 는 decile 20–30 에서 0.671 로 일시 하락한 뒤 50–80 에서
회복). <span class="math inline"><em>Δ</em></span> 는 진폭 (max<span
class="math inline">−</span>min) 이며 단조 분리력이 아니다.</figcaption>
</figure>

표 <a href="#tab:axis_decomp" data-reference-type="ref"
data-reference="tab:axis_decomp">[tab:axis_decomp]</a> 와
그림 <a href="#fig:axis_decomp" data-reference-type="ref"
data-reference="fig:axis_decomp">4.1</a> 에서 네 가지 핵심 발견이
도출된다.

##### Entity-pair co-occurrence axis 의 분해 진폭.

Entity-pair co-occurrence axis 는 SE 진폭 $`\Delta`$<!-- -->0.150,
Energy 진폭 $`\Delta`$<!-- -->0.144 로, 본 표본에서 비교한 7개 axis 중
가장 큰 분해 진폭을 보인다. prompt-grouped bootstrap (B=500) 으로 산출한
95% CI 는 SE Δ 0.121–0.211, entity_freq Δ 0.058–0.149 이다. 두 axis 의 Δ
차이의 95% CI 는 \[+0.002, +0.117\] 로 0 을 포함하지 않으며, 97.6%
iteration 에서 양수다 (Energy 도 \[+0.004, +0.109\], 98.0% positive). 즉
entity_pair Δ \> entity_freq Δ 라는 관찰은 sampling variability 를
통제한 뒤에도 통계적으로 유의하다. corpus 부족 영역 (decile 00–10)
에서는 SE 가 0.643, Energy 가 0.667 이며, corpus 가 가장 풍부한 영역
(decile 70–80) 에서는 SE 가 0.793, Energy 가 0.811 이다.
decile-by-decile 단조성은 엄밀하게 성립하지 않으며 중간 구간에 비단조
변동이 존재한다
(그림 <a href="#fig:axis_decomp" data-reference-type="ref"
data-reference="fig:axis_decomp">4.1</a>). 본 논문의 “체계적 ordering”
은 양 끝단 사이의 평균적 ordering 과 max$`-`$min 진폭을 의미하며,
decile-by-decile 단조성이 아니다. 이는 답변 내 entity-pair 가 corpus 에
함께 등장하는 정도가 LLM 의 해당 fact 학습 정도를 근사하며,
sample-consistency 신호의 행태와 관련된 한 요인임을 시사한다.

##### Entity-frequency axis 의 분해 진폭.

단순 entity 빈도 axis 는 SE $`\Delta`$<!-- -->0.080, Energy
$`\Delta`$<!-- -->0.077 로, 본 표에서 가장 작은 분해 진폭을 보인다. 본
연구의 baseline corpus axis (entity_freq + pair_cooc 평균) 도
$`\Delta`$<!-- -->0.082–0.086 으로 거의 동일하다. 이는 entity_pair 의 큰
진폭 ($`\Delta`$<!-- -->0.144) 이 entity_freq 의 작은 진폭
($`\Delta`$<!-- -->0.077) 과 평균되어 희석되었음을 의미하며, corpus axis
설계에서 단위를 분리해 보고할 필요가 있음을 시사한다. 또한 평균화는
진폭뿐 아니라 단조성도 희석한다. entity_pair (SE $`\rho`$=+0.648,
$`p`$=0.043) 와 entity_freq (SE $`\rho`$=+0.636, $`p`$=0.048) 는 각각
양의 단조성을 보이지만, 두 값의 평균인 `corpus_axis_bin_10` 의 SE
$`\rho`$=+0.418 ($`p`$=0.229) 는 유의 임계 아래로 떨어진다
(표 <a href="#tab:axis_decomp" data-reference-type="ref"
data-reference="tab:axis_decomp">[tab:axis_decomp]</a>). 즉 단순
평균화는 단위 정보뿐 아니라 ordering 일관성도 잃는 신호 가공이다. 본
논문의 정량적 비교 — entity-pair 단위가 entity 단위보다 SE 분해 진폭을
약 1.88배 크게 보인다는 점 — 가 본 결과에서 직접 도출된다.

##### QA-bridge axis 와 sample_nll 변동의 관계.

`qa_bridge_axis` 는 `sample_nll` 진폭 $`\Delta`$<!-- -->0.176 로, SE /
Energy 진폭 ($`\Delta`$<!-- -->0.089, $`\Delta`$<!-- -->0.087) 보다 큰
폭으로 분해한다. (질문 entity, 답변 entity) 쌍의 corpus 동시 등장은
모델의 token-level 자신감 (NLL) 과 가장 직접 연결되며, 이는 SE / Energy
와 다른 정보를 포착함을 시사한다.

##### ans_ngram_3_zero_count axis: 진폭과 단조성의 분리.

답변 phrase 중 corpus 에 한 번도 등장하지 않은 비율 (novel phrase count)
은 SE 진폭 측면에서 세 번째로 큰 axis ($`\Delta`$<!-- -->0.122) 이다.
그러나 단조성 ($`\rho`$=+0.261, $`p`$=0.467,
표 <a href="#tab:axis_decomp" data-reference-type="ref"
data-reference="tab:axis_decomp">[tab:axis_decomp]</a>) 은 다른 axis
(entity_pair $`\rho`$=+0.648, qa_bridge $`\rho`$=+0.648, ans_ngram_5
$`\rho`$=+0.685) 와 달리 유의수준 0.05 를 넘지 못한다. 본 axis 의 진폭이
단조 증가가 아닌 비단조 패턴 (U-shape, 부분 구간 spike 등) 에서 비롯됨을
시사한다. 따라서 “novel phrase count 가 클수록 SE 가 낮아진다” 라는 단조
해석은 본 데이터에서 성립하지 않으며, “novel phrase 비율이 높은 특정
영역에서 SE 분리력의 변동이 관찰된다” 라는 비단조 진술로 한정한다.

##### 강건성 검증: 질문 entity 한정 axis.

표 <a href="#tab:axis_decomp" data-reference-type="ref"
data-reference="tab:axis_decomp">[tab:axis_decomp]</a> 의
`entity_pair_cooccurrence_axis` 는 질문과 dataset gold answer 의 entity
합집합에서 산출되므로, 답변 entity 가 axis 값에 영향을 줄 가능성을
강건성 검증으로 점검하였다. 질문 텍스트에서만 spaCy NER 로 entity 를
추출한 question-entity-only axis (`q_entity_pair_axis_bin_10`, n=2,095
prompt; entity 가 2개 이상인 질문) 위에서 동일 분석을 재수행한 결과, SE
Δ는 0.178, Energy Δ는 0.162 로 원본 axis (SE 0.150 / Energy 0.144) 와
비교해 더 큰 진폭이 관찰되었다. 단순 entity 빈도 axis (질문 한정) 의 SE
Δ는 0.081 로 원본 0.080 과 차이가 거의 없다. 따라서 entity-pair 단위가
entity 단위보다 분해 진폭이 크다는 본 논문의 패턴은 답변 entity 를 axis
입력에서 제거해도 유지되며, ratio 는 약 1.88배에서 약 2.20배 (SE) 와 약
2.10배 (Energy) 로 증가한다. 이는 (a) corpus axis 의 분해 진폭이 모델
출력에 의해 인공적으로 증폭된 결과가 아님, (b) 질문 측 entity-pair 가
LLM 이 학습한 fact-pair 의 더 직접적인 proxy 임을 시사한다. 다만 본
ablation 의 부분표본 (n=2,095) 은 entity 가 2개 이상인 질문으로
한정된다. 이러한 질문은 단일 entity-fact 보다는 두 entity 사이의 관계
(relational fact) 를 묻는 factual QA — 예: “Cambodia 의 통화는 무엇인가”
(entity 1개) 가 아니라 “Cambodia 의 Khmer Rouge 정권은 누가 이끌었는가”
(entity 2개) 유형 — 에 편향된다. relational fact 는 entity-pair
co-occurrence 와 직접 대응하므로, 본 부분표본은 entity-pair axis 의
진폭이 본래 크게 나타나는 조건에 해당한다 (선택 편향). 따라서 “Q-only
ratio 2.20” 의 절대 크기는 이 부분표본의 특성을 반영하므로, 전체 표본
ratio (1.88) 와 직접 산술 비교하기보다는 “답변 entity 를 제거해도 ratio
의 부호가 유지된다” 라는 정성적 강건성 결과로 해석한다.

##### 강건성 검증: SVAMP 제외 민감도.

SVAMP 는 다른 4개 데이터셋과 달리 수학 word problem 이므로 entity-pair
co-occurrence 의 의미가 factual QA 와 다를 수 있다. SVAMP 를 제외
(n=3,200 prompt) 하고 동일 분석을 재수행하면 SE Δ ratio (entity_pair /
entity_freq) 는 1.68 로, 전체 표본 ratio (1.88) 보다 약 0.20 줄어든다.
Energy ratio 도 1.54 로 비슷한 폭으로 감소한다. 패턴 (entity_pair \>
entity_freq) 자체는 유지되지만, ratio 의 절대 크기는 SVAMP 포함 여부에
의존한다. 본 결과를 토대로 ratio 는 1.5–1.9 범위로 보고하는 편이
안정적이다.

##### 데이터셋별 분해 결과.

표 <a href="#tab:per_dataset_delta" data-reference-type="ref"
data-reference="tab:per_dataset_delta">4.5</a> 는 데이터셋별 SE Δ
(entity_pair 와 entity_freq) 를 보고한다. 4개 데이터셋 (BioASQ, SQuAD,
SVAMP, TriviaQA) 에서는 entity_pair Δ \> entity_freq Δ 패턴이 유지되나
비율은 1.0배 (TriviaQA) 에서 2.92배 (BioASQ) 까지 데이터셋에 따라
변동한다. NQ-Open 에서는 패턴이 역전되어 entity_freq Δ (0.372) 가
entity_pair Δ (0.250) 보다 크다. 즉 전체 표본 ratio 1.88배는 데이터셋별
ratio 의 평균이 아니라, BioASQ 의 큰 ratio (2.92) 가 NQ-Open 의 역전
(0.67) 을 보상해 형성된 표본 종합 통계이다. 따라서 entity-pair axis 가
모든 데이터셋에서 entity_freq axis 보다 큰 진폭을 보이지는 않으며,
데이터셋 도메인에 따라 corpus support signal 의 적절한 단위가 달라질 수
있다는 가설을 제시한다.

<div id="tab:per_dataset_delta">

| Dataset              | SE Δ pair | SE Δ freq | SE ratio | Energy Δ pair |
|:---------------------|:---------:|:---------:|:--------:|:-------------:|
| BioASQ               |   0.272   |   0.093   |   2.92   |     0.184     |
| SQuAD-1.1            |   0.166   |   0.125   |   1.33   |     0.271     |
| SVAMP                |   0.126   |   0.134   |   0.94   |     0.184     |
| TriviaQA             |   0.088   |   0.089   |   0.99   |     0.113     |
| NQ-Open ★            |   0.250   | **0.372** | **0.67** |     0.514     |
| **AGG (5 datasets)** | **0.150** | **0.080** | **1.88** |   **0.144**   |

데이터셋별 SE / Energy Δ (`entity_pair_cooccurrence_axis` 와
`entity_frequency_axis` 비교). NQ-Open 에서 entity_freq Δ 가 entity_pair
Δ 보다 크다 (★). 주의: NQ-Open Energy Δ 0.514 와 entity_freq Δ 0.372 는
부분적으로 class-imbalance 에 의한 산출 결과로 보인다 (NQ-Open
is_correct 비율 0.240, decile 20–30 에서 n=120 중 정답 7개로 AUROC 0.266
의 outlier 발생, decile 30–40 은 단일 class 로 산출 제외). 해당 outlier
decile 을 제외하면 NQ-Open Δ 는 약 0.10–0.15 수준으로 다른 데이터셋과
비슷해진다. 데이터셋별 Δ 의 절대 크기는 데이터셋별 정답률 분포에
민감하므로, 본 표는 표본 종합 ratio 가 데이터셋 평균과 일치하지 않는다는
점을 보이는 데 한정 사용한다.

</div>

## Per-Dataset 변동

5개 데이터셋별 단일 신호 / fusion AUROC 를
표 <a href="#tab:per_dataset" data-reference-type="ref"
data-reference="tab:per_dataset">4.6</a> 에 보고한다.

<div id="tab:per_dataset">

| Dataset | n | is_correct | SE | Energy | sample_nll | GBM (with corpus) |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| TriviaQA | 8,000 | 0.482 | 0.778 | 0.782 | 0.633 | 0.803 |
| SVAMP | 3,000 | 0.431 | 0.786 | 0.787 | 0.713 | 0.824 |
| BioASQ | 8,000 | 0.339 | 0.746 | 0.759 | 0.678 | 0.811 |
| NQ-Open | 8,000 | 0.240 | 0.709 | 0.731 | 0.635 | 0.796 |
| SQuAD-1.1 | 8,000 | 0.189 | 0.700 | 0.716 | 0.593 | 0.748 |
| **AGG** | **35,000** | **0.322** | **0.759** | **0.774** | **0.670** | **0.808** |

데이터셋별 AUROC (GroupKFold(prompt_id) 5-fold OOF). `is_correct rate`
는 dataset 별 평균 정답률.

</div>

`corpus_axis_bin_10` (entity_freq + pair 평균) 위에서 GBM (with corpus)
의 per-decile AUROC 는 최저 0.762 (decile 30–40), 최고 0.859 (decile
60–70) 로 $`\Delta`$<!-- -->0.097 의 변동을 보인다.

`entity_pair_cooccurrence_axis` 에서는 SE 기준 더 큰
$`\Delta`$<!-- -->0.150 로 이 패턴이 한층 명확하다
(표 <a href="#tab:axis_decomp" data-reference-type="ref"
data-reference="tab:axis_decomp">[tab:axis_decomp]</a>). `is_correct`
비율도 corpus 풍부 구간 (decile 60–70, 0.382) 이 corpus 부족 구간
(decile 10–20, 0.209) 보다 높아, corpus support 와 정답률의 양의 상관이
확인된다. 데이터셋 간 단조 패턴 — 정답률 (TriviaQA 0.482 $`\to`$ SQuAD
0.189) 과 모든 신호 AUROC (TriviaQA SE 0.778 $`\to`$ SQuAD SE 0.700) 가
같은 방향으로 감소 — 도 같은 corpus-support 효과의 dataset 간 발현으로
해석된다.

# 결론

<span id="ch:conclusion" label="ch:conclusion"></span>

## 논의: Corpus Signal의 역할 재정의

본 논문의 결과는 corpus signal 의 역할에 대한 재정의를 제안한다. corpus
signal 은 (a) 환각 탐지기의 입력 feature 로서의 기여가 제한적이나
(fusion lift $`\leq`$ +0.010), (b) 탐지 신호의 corpus 조건부 행태를
분해하는 평가 axis 로 사용했을 때는 분해 진폭이
$`\Delta`$<!-- -->0.080–0.150 범위에서 관찰된다. 이 비대칭은 corpus
signal 의 설계 목표가 달라야 함을 시사한다. 즉 “corpus signal 을 어떻게
더 효과적인 탐지 feature 로 가공할 것인가” 가 아니라 “corpus signal 의
어떤 단위가 탐지 신호의 어떤 측면을 분해하는가” 가 본 연구가 제기하는
질문이다.

Entity-pair co-occurrence 가 단순 entity 빈도보다 약 1.88배 큰 진폭 (SE
$`\Delta`$<!-- -->0.150 와 $`\Delta`$<!-- -->0.080) 을 보이는 결과는,
fact-level co-occurrence 가 entity-level unigram 빈도보다 LLM 의
in-context knowledge 를 더 직접 반영한다는 해석으로 설명할 수 있다. 단일
entity 가 corpus 에 자주 등장하더라도, 해당 entity-pair 관계가 corpus 에
함께 등장하지 않으면 LLM 이 해당 fact 를 학습할 가능성이 낮다. 이 단위
(granularity) 차이가 분해 진폭의 차이로 나타난다.

`qa_bridge_axis` 와 `sample_nll` 의 진폭 ($`\Delta`$<!-- -->0.176) 은
다른 관점을 제공한다. (질문 entity, 답변 entity) 쌍의 corpus 동시 등장은
LLM 이 “질문 $`\to`$ 답변” 경로를 사전학습에서 직접 학습했을 가능성을
근사하며, 이것이 token-level 생성 자신감 (NLL) 과 가장 직접 연결된다. SE
/ Energy 는 sample 간 일관성을 보는 다른 차원의 신호이므로, qa_bridge 와
sample_nll 의 연결이 SE / Energy 와 다른 패턴을 보이는 것은 두 신호군이
보완적인 정보를 포착함을 시사한다.

##### 효과 크기의 정밀도.

§<a href="#sec:axis" data-reference-type="ref"
data-reference="sec:axis">4.5</a> 의 SE Δ 차이의 95% CI \[+0.002,
+0.117\] 는 0 을 포함하지 않으나 하한이 +0.002 로 경계에 가까운 값이며,
CI 의 폭 (0.115) 은 점추정치 (+0.063) 의 약 1.8배에 해당한다. 즉
“entity_pair Δ \> entity_freq Δ” 라는 방향성은 본 표본에서 통계적으로
유의하나, 효과 크기 자체의 정밀도는 낮다. 본 논문은 “ratio 약 1.88배”
라는 점추정치가 모집단 ratio 와 일치한다고 보지 않으며, 다음 세 가지
점을 함께 보고하였다. (a) SVAMP 포함 여부에 따라 ratio 는 1.68–1.88
사이에서 변동한다 (§<a href="#sec:axis" data-reference-type="ref"
data-reference="sec:axis">4.5</a> SVAMP 민감도). (b) Q-only ablation
에서는 약 2.20배까지 증가한다. (c) 데이터셋별로는 NQ-Open 처럼 역전되는
경우가 관찰된다
(표 <a href="#tab:per_dataset_delta" data-reference-type="ref"
data-reference="tab:per_dataset_delta">4.5</a>). 따라서 본 논문의 정량적
주장은 “방향성과 약 1.5–1.9배 범위의 점추정치 ratio” 로 한정하며, 정밀한
effect size 추정은 다중 모델 / 다중 corpus 위에서의 추가 표본을 통해
가능하다.

## 한계

본 연구에는 다음과 같은 한계가 있다.

1.  **단일 모델 평가.** 본 연구는 Qwen2.5-3B 단일 모델 평가이다. corpus
    axis 분해 패턴이 Llama, Mistral, Gemma 계열 모델에서도 동일한
    방향으로 나타나는지 직접 검증되지 않았다. Farquhar (2024) 에서 서로
    다른 LLM 계열 간에도 SE 의 패턴이 유사했다는 점에서 일반화 가능성을
    시사하지만, 이는 향후 연구로 남긴다.

2.  **단일 corpus snapshot.** corpus support 축은 `v4_dolmasample_olmo`
    (16B Dolma sample) 단일 index 에서 산출되며, Infini-gram index 가
    Qwen2.5-3B 의 실제 사전학습 corpus 와 다르므로 corpus support
    $`\leftrightarrow`$ 환각의 연결은 *상관* 이며 *인과* 관계로 해석할
    수 없다. 도메인 특화 corpus (예: BioASQ 용 PubMed) 를 사용하면
    분해력이 달라질 가능성이 있다.

3.  **NLI 라벨링 한계.** `is_correct` 가 `microsoft/deberta-large-mnli`
    기반 양방향 entailment $`\geq`$ 0.5 기준에 의존하며, LLM-as-judge
    (GPT-4 등) 와 같은 다른 라벨링 방식과의 비교 검증은 향후 연구
    과제이다.

4.  **Free-sampling 부분 결합.** `is_correct` 라벨과 SE / Energy 신호는
    모두 동일한 N=10 free-sample 자료에서 산출된다. 라벨은 sample
    $`s_i`$ 와 정답 후보의 NLI 매칭으로 결정되고 SE / Energy 는 같은
    sample 들의 cluster 분포 / token logit 에서 산출되므로, 두 양이 동일
    sample 의 분산을 부분적으로 공유한다. 결과적으로 본 논문의 AUROC 는
    모델 출력의 정답 여부와 모델 출력 자신의 일관성 사이의 동일-source
    결합도 측정 (within-source consistency-to-correctness) 으로
    해석되며, 모델 출력과 외부 ground truth 의 일치도에 대한 독립 측정이
    아니다. 평가 단위 (per-generation) 가 Farquhar / Ma 와 일치하므로
    cross-paper 비교의 전제는 유지되며, 본 논문의 결과는 동일 라벨링
    protocol 위에서의 상대적 비교로 해석한다.

5.  **Self-conditioning artifact 회피.** 본 논문은 corpus 통계량을 (a)
    fusion 입력 feature 로서, (b) 평가 axis 로서 두 용도로 모두
    사용한다. (b) 의 axis 분석은 모델 출력과 무관한 외부 corpus 통계
    위에서 수행되므로 self-conditioning artifact 는 발생하지 않는다.

6.  **$`\Delta`$ 통계의 표본 변동성.** 본 논문의 ratio “약 1.88배” 는
    single-condition point estimate 이며, 두 $`\Delta`$ 의 차이에 대한
    prompt-grouped bootstrap 95% CI 를
    §<a href="#sec:axis" data-reference-type="ref"
    data-reference="sec:axis">4.5</a> 에 함께 보고하였다 (SE \[+0.002,
    +0.117\], Energy \[+0.004, +0.109\]). 두 CI 모두 0 을 포함하지
    않으므로 entity_pair Δ \> entity_freq Δ 는 본 표본에서 통계적으로
    유의하다. 다만 본 CI 는 단일 모델, 단일 corpus index 위에서의 표본
    변동성만을 통제하며, 모델 / corpus 일반화에는 별도 검증이 필요하다.
    max$`-`$min 통계량 자체가 순서통계량이므로 decile 수가 다른 binning
    에서는 ratio 의 절대 크기가 변할 수 있다.

7.  **Fusion lift +0.008 의 통계적 유의성.** GBM (with corpus) 과 GBM
    (no corpus) 의 lift 에 대한 prompt-grouped bootstrap 95% CI
    \[+0.005, +0.011\] 는 0 을 포함하지 않으며 모든 iteration 에서
    양수이다. 따라서 corpus signal 의 fusion 기여는 통계적으로 0 과
    구분되는 양의 효과이지만, 절대 크기 (+0.008) 는 sample-level /
    prompt-level 신호 결합 효과 (+0.026) 의 약 1/3 정도로 한정된다.

8.  **데이터셋별 이질성.**
    표 <a href="#tab:per_dataset_delta" data-reference-type="ref"
    data-reference="tab:per_dataset_delta">4.5</a> 에서 보이듯
    entity_pair Δ \> entity_freq Δ 패턴은 5개 데이터셋 중 NQ-Open 에서
    역전된다. 따라서 표본 종합 ratio 1.88배는 모든 데이터셋에서
    entity-pair axis 가 entity_freq axis 보다 큰 진폭을 보인다는 뜻이
    아니라, 데이터셋 종합 통계이다. 도메인별로 corpus support signal 의
    적절한 단위가 다를 수 있다는 가설은 본 논문 범위 밖이다.

9.  **Corpus axis 의 entity 추출 범위.** 본 논문의 corpus axis 는 질문
    텍스트와 데이터셋 annotation 의 정답 후보 텍스트 (gold answer) 에서
    추출한 entity 합집합에서 산출되며, 모델 free-sample 출력 entity 는
    axis 계산에 사용되지 않는다 (axis 값은 모델 출력과 독립이며, 따라서
    한계 4 의 free-sampling 결합 우려와는 다른 차원의 문제이다). 답변
    entity 의 영향을 분리한 question-entity-only axis 의 결과는
    §<a href="#sec:axis" data-reference-type="ref"
    data-reference="sec:axis">4.5</a> 에 함께 보고하였고, SE ratio 는
    1.88 에서 2.20 으로 증가한다. 다만 Q-only 부분표본 (n=2,095) 의 선택
    편향은 §<a href="#sec:axis" data-reference-type="ref"
    data-reference="sec:axis">4.5</a> 에 명시한다.

10. **$`\Delta`$ 와 Spearman $`\rho`$ 의 분리 해석.** $`\Delta`$ 는 진폭
    (max$`-`$min) 이며 단조 분리력과 비단조 진폭 (U-shape, CHOKE 등) 을
    구분하지 못한다. 본 논문은 $`\rho`$ 를
    표 <a href="#tab:axis_decomp" data-reference-type="ref"
    data-reference="tab:axis_decomp">[tab:axis_decomp]</a> 에 SE /
    Energy / sample_nll 세 신호 모두에 대해 함께 보고함으로써 진폭과
    단조성을 분리해 진단한다. 다만 (a) 21개 동시 검정에 대한 다중비교
    보정 (Bonferroni $`\alpha`$=0.05/21$`\approx`$<!-- -->0.0024) 을
    적용하면 본 표의 모든 $`\rho`$ 가 유의 기준을 충족하지 않으므로 본
    $`\rho`$ 결과는 예비 분석으로 한정한다. (b) n=10 (decile 수) 의
    Spearman 검정력이 낮아 $`|\rho|`$=0.65 부근의 결과는 binning 폭과
    tie 처리에 민감하다. 보다 robust 한 단조성 검정 (Mann-Kendall,
    partial $`\rho`$) 과 다중비교 보정은 향후 연구 과제로 남긴다.

## 향후 연구

본 연구의 결과는 세 가지 방향의 후속 연구를 제안한다.

1.  **모델 다양화.** Llama, Gemma, instruction-tuned 모델 등 다양한
    모델에서 entity-pair axis 의 분해력 패턴이 같은 방향으로 나타나는지
    검증한다.

2.  **LLM-as-judge 라벨 비교.** NLI 매칭 vs. GPT-4 judge 라벨 하에서
    multi-axis 분해 결과가 어떻게 달라지는지 분석한다.

3.  **도메인 특화 corpus.** BioASQ 의 경우 PubMed 기반 corpus, 수학
    문제의 경우 ArXiv 기반 corpus 등 도메인 특화 index 를 사용했을 때
    동일 axis 의 분해력이 더 강해지는지 검증한다.

## 결론

본 논문은 환각 탐지 신호 (Semantic Entropy, Semantic Energy,
sample-level token diagnostics) 의 corpus support 조건 의존성을 corpus
signal 의 단위 (entity, fact-pair, phrase) 에 따라 분해하는 multi-axis
분석 프레임워크를 제안하고, Qwen2.5-3B 와 5개 QA 데이터셋 (35,000
generation) 에서 초기 검증을 수행하였다. 핵심 결과는 entity-pair
co-occurrence axis 가 SE 의 decile 별 AUROC 진폭 $`\Delta`$<!-- -->0.150
를 분리하며, 이 진폭이 단순 entity 빈도 axis ($`\Delta`$<!-- -->0.080)
의 약 1.5–1.9배 (SVAMP 포함 시 1.88, 제외 시 1.68) 에 해당한다는 것이다.
두 Δ 차이는 prompt-grouped bootstrap 95% CI \[+0.002, +0.117\] 로 본
표본에서 방향성이 통계적으로 유의하나, CI 폭 대비 점추정치 비율로 보면
효과 크기의 정밀도는 낮다 (모집단 ratio 의 안정성은 추가 모델과 corpus
위에서의 검증이 필요하다). 동일한 corpus signal 이 fusion feature 로는
+0.008 의 작은 lift 만 보이는 반면, 평가 axis 로 사용했을 때 분해 진폭이
큰 비대칭은, corpus signal 의 설계 목표를 hallucination 탐지 feature 가
아닌 탐지 신호의 조건부 평가 도구 (conditional benchmarking) 로 재정의할
필요를 제기한다. 이러한 재정의는 향후 환각 탐지 연구의 corpus 조건부
평가에서 참조 가능한 분석 형식으로 활용될 수 있다.

<div class="thebibliography">

99

L. Huang et al., “A Survey on Hallucination in Large Language Models:
Principles, Taxonomy, Challenges, and Open Questions,” *ACM Transactions
on Information Systems*, vol. 43, no. 2, pp. 1–55, 2025.

J. Maynez et al., “On Faithfulness and Factuality in Abstractive
Summarization,” *Proceedings of ACL*, pp. 1906–1919, 2020.

S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting hallucinations
in large language models using semantic entropy,” *Nature*, vol. 630,
pp. 625–630, 2024.

P. Manakul, A. Liusie, and M. J. F. Gales, “SelfCheckGPT: Zero-Resource
Black-Box Hallucination Detection for Generative Large Language Models,”
*Proceedings of EMNLP*, pp. 9004–9017, 2023.

J. Kossen et al., “Semantic Entropy Probes: Robust and Cheap
Hallucination Detection in LLMs,” *arXiv preprint arXiv:2406.15927*,
2024.

A. Nikitin, J. Kossen, Y. Gal, and P. Marttinen, “Kernel Language
Entropy: Fine-grained Uncertainty Quantification for LLMs,” *arXiv
preprint arXiv:2405.20003*, 2024.

K. Ciosek et al., “Bayesian Estimation of Semantic Entropy for
Sample-Efficient Hallucination Detection,” *arXiv preprint
arXiv:2503.xxxxx*, 2025.

Z. Ma et al., “Semantic Energy: A novel approach for detecting
confabulation in language models,” *arXiv preprint arXiv:2412.07965*,
2025.

A. Raghuvanshi et al., “Hybrid Token-Level Uncertainty and Semantic
Entailment for Hallucination Detection,” *arXiv preprint
arXiv:2502.xxxxx*, 2025.

A. Ravichander et al., “HALoGEN: Fantastic LLM Hallucinations and Where
to Find Them,” *arXiv preprint arXiv:2501.08292*, 2025.

S. Singha, “ECLIPSE: Evidence-Conditioned Hallucination Detection via
Perplexity Decomposition for Financial QA,” *arXiv preprint
arXiv:2510.xxxxx*, 2025.

H. Wang, “SEReDeEP: Semantic Entropy + Retrieval-Aware Detection for
Enhanced Evaluation Pipelines,” *arXiv preprint arXiv:2509.xxxxx*, 2025.

S. Valentin et al., “Cost-Effective Hallucination Detection for LLMs,”
*arXiv preprint arXiv:2407.21424*, 2024.

W. Zhao et al., “WildHallucinations: Evaluating Long-form Factuality in
LLMs with Real-World Entity Queries,” *arXiv preprint arXiv:2407.17468*,
2024.

Y. Zhang et al., “Measuring the Impact of Lexical Training Data Coverage
on Hallucination Detection in Large Language Models,” *arXiv preprint
arXiv:2511.17946*, 2025.

J. Liu et al., “Infini-gram: Scaling Unbounded n-gram Language Models to
a Trillion Tokens,” *arXiv preprint arXiv:2401.17377*, 2024.

A. Yang et al., “Qwen2.5 Technical Report,” *arXiv preprint
arXiv:2412.15115*, 2024.

M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “TriviaQA: A Large
Scale Distantly Supervised Challenge Dataset for Reading Comprehension,”
*Proceedings of ACL*, pp. 1601–1611, 2017.

P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+
Questions for Machine Comprehension of Text,” *Proceedings of EMNLP*,
pp. 2383–2392, 2016.

G. Tsatsaronis et al., “An overview of the BIOASQ large-scale biomedical
semantic indexing and question answering competition,” *BMC
Bioinformatics*, vol. 16, no. 138, 2015.

T. Kwiatkowski et al., “Natural Questions: A Benchmark for Question
Answering Research,” *Transactions of the ACL*, vol. 7, pp. 453–466,
2019.

A. Patel, S. Bhattamishra, and N. Goyal, “Are NLP Models really able to
Solve Simple Math Word Problems?,” *Proceedings of NAACL*, pp.
2080–2094, 2021.

</div>
