# 서론

[]{#ch:intro label="ch:intro"}

## 연구 배경

ChatGPT 와 Gemini 로 대표되는 대규모 언어 모델 (LLM) 은 추론과
질의응답에서 뛰어난 성능을 보이지만, 사실과 다른 그럴듯한 답변, 이른바
환각 (hallucination) 을 자주
생성한다[@huang2025survey; @maynez2020faithfulness]. 의료, 법률, 과학 등
고위험 도메인에서 환각은 심각한 오류로 이어질 수 있다. 따라서 모델
출력의 신뢰성을 자동으로 판별하는 환각 탐지 연구가 중요한 과제로 자리
잡았다.

대표적인 접근법은 모델의 불확실성을 정량화하는 것이다. Farquhar 등
(2024)[@farquhar2024] 이 제안한 Semantic Entropy 는 동일 질문에 대해
생성된 여러 답변이 의미적으로 얼마나 흩어져 있는지를 측정한다. 답변들이
하나의 의미로 모이면 모델이 확신하는 경우로, 의미가 갈리면 확신하지
못하는 경우로 해석된다. Ma 등 (2025)[@ma2025] 의 Semantic Energy 는 확률
대신 logit 을 직접 활용하여, Semantic Entropy 가 놓치는 모델 지식 자체의
한계까지 반영한다. 이 두 신호는 외부 지식 없이도 LLM 출력의 신뢰도를
가늠할 수 있는 실질적인 도구로 자리매김했다.

그러나 이러한 탐지 신호의 성능은 대부분 평균 AUROC 라는 하나의 숫자로만
보고된다. 이 평균은 모델이 잘 아는 사실과 잘 모르는 사실을 한꺼번에 섞은
결과물이다. 모델은 사전학습 데이터에 자주 등장한 사실은 잘 기억하지만,
거의 등장하지 않은 사실은 잘 모를 가능성이 높다. 따라서 탐지 신호의
신뢰도도 두 영역에서 크게 다를 수밖에 없는데, 평균 숫자 하나만으로는 이
차이를 전혀 알 수 없다. "평균 AUROC 0.79" 라는 보고만으로는 어떤 질문에
대해 탐지 결과를 신뢰할 수 있고, 어떤 질문에 대해 신뢰하기 어려운지
판단하기 어렵다.

사전학습 corpus 통계량을 환각 탐지에 활용한 선행
연구[@wildhallucinations2024; @zhang2025corpus; @ravichander2025halogen]
도 있지만, 이들은 대부분 corpus 신호를 탐지기 자체의 입력 변수로
사용하는 데 그쳤다. corpus 통계가 탐지 신호의 영역별 신뢰도를 어떻게
분해하는지, 그리고 단일 entity 빈도, entity 쌍 동시 등장, n-gram 등 어느
단위가 분해를 가장 잘 설명하는지에 대한 체계적인 비교 분석은 아직
이루어지지 않았다.

본 연구의 핵심 결과는 다음과 같다. 동일 corpus 신호를 탐지기의 입력
변수로 추가했을 때는 성능 향상이 미미한 반면, 조건부 평가 척도로
사용했을 때는 AUROC 에 큰 폭의 변동이 나타난다. 이 비대칭 현상이 본
연구가 집중적으로 분석하는 대상이며, 이하의 내용은 이 비대칭을
정량적으로 규명하는 데 초점을 맞춘다. 본 연구에서 다루는 환각은 주로
사전학습 데이터의 사실 회상 (HALoGEN[@ravichander2025halogen] 의 Type A)
영역이며, 모델 내부의 잘못된 지식 (Type B) 이나 순수 조작 (Type C) 과의
차이는 후속 연구로 남긴다.

## 연구 목적

본 연구는 다음과 같은 직관에서 출발한다. "Paris" 라는 단어가 corpus 에
자주 등장한다고 해서 모델이 "Paris 가 France 의 수도" 라는 사실을
학습했다고 단정할 수는 없다. 이 사실을 학습할 가능성은 "Paris" 단일
단어의 빈도보다 "Paris" 와 "France" 가 corpus 에서 함께 등장한 빈도와
훨씬 더 직접적으로 연결될 것이다. 그렇다면 환각 탐지 신호의 신뢰도 역시
단일 entity 빈도가 아니라 entity 쌍의 동시 등장 빈도에 따라 더 뚜렷하게
갈릴 것이다.

이 직관을 바탕으로 본 연구의 핵심 질문은 다음과 같이 정리된다. 환각 탐지
신호 (Semantic Entropy, Semantic Energy, 답변 단위 logit 통계) 의
신뢰도가 corpus 의 사실 등장 패턴에 따라 어떻게 달라지는가? 그리고 단일
entity 빈도, entity co-occurrence, 답변 어구의 n-gram 빈도 가운데 어느
단위가 이 변동을 가장 잘 설명하는가?

이를 위해 세 가지 세부 목표를 설정하였다.

- 첫째, 기존 환각 탐지 신호 (Semantic Entropy, Semantic Energy, 답변
  단위 logit 통계) 의 성능을 동일한 평가 절차 아래에서 재현한다.

- 둘째, 단위가 서로 다른 corpus 신호 (entity 빈도, entity co-occurrence,
  question-answer bridge, n-gram) 를 설계하고, 각 신호가 단독 환각
  탐지기로서 어느 정도의 성능을 보이는지 평가한다.

- 셋째, 동일 corpus 신호로 sample 을 그룹별로 나누었을 때 환각 탐지
  신호의 구간별 AUROC 가 어떻게 변동하는지 비교·분석한다. 이를 통해
  corpus 신호의 단위가 AUROC range 에 미치는 영향을 정량적으로 규명한다.

본 연구의 주요 기여는 다음과 같다.

- \(1\) 환각 탐지 신호의 신뢰도가 corpus 조건에 따라 어떻게 달라지는지를
  corpus 신호의 단위별로 비교하는 분석 프레임워크를 제시한다.

- \(2\) 본 표본에서 entity co-occurrence 가 단일 entity 빈도보다 환각
  탐지 신호의 신뢰도 변동을 약 1.5\~1.9배 더 명확하게 구분한다는 점을
  정량적으로 확인한다 (단일 조건 점추정치 1.88배, 95% 신뢰구간 하한
  +0.002, SVAMP 제외 시 1.68배).

- \(3\) corpus 신호의 설계 목표를 "탐지기의 입력 변수" 에서 "탐지 신호의
  조건부 평가 척도" 로 재설정하는 새로운 관점을 제안한다. 이 관점은 향후
  환각 탐지 연구의 평가 방식과 실제 LLM 서비스에서의 조건부 신뢰 전략
  모두에 시사점을 준다.

## 논문 구성

본 논문은 다음과 같이 구성된다.
제 [\[ch:related\]](#ch:related){reference-type="ref"
reference="ch:related"}장에서는 Semantic Entropy, Semantic Energy,
corpus 기반 환각 탐지, 다중 신호 결합 (Multi-Signal Fusion) 관련 선행
연구를 정리한다. 제 [\[ch:method\]](#ch:method){reference-type="ref"
reference="ch:method"}장에서는 데이터셋, 실험 설정, 신호 정의, corpus
신호 단위별 분해 분석 절차를 기술한다.
제 [\[ch:experiment\]](#ch:experiment){reference-type="ref"
reference="ch:experiment"}장에서는 단일 신호 평가, fusion 결과, 단위별
구간 분석 결과를 보고한다.
제 [\[ch:conclusion\]](#ch:conclusion){reference-type="ref"
reference="ch:conclusion"}장에서는 연구 결과의 함의, 한계, 향후 연구
방향을 논의하고 결론을 맺는다.

# 관련 연구

[]{#ch:related label="ch:related"}

본 장에서는 LLM 환각 탐지 연구를 네 가지 주요 흐름으로 정리한다. 첫째,
생성된 답변 간 의미 일관성에 기반한 탐지 (sample-consistency). 둘째,
logit 을 활용한 불확실성 추정. 셋째, 사전학습 corpus 통계의 활용. 넷째,
여러 신호를 결합하는 multi-signal fusion 이다. 각 흐름의 대표 연구를
간략히 살펴본 뒤, 본 연구의 위치를 명확히 한다.

## Sample-Consistency 기반 환각 탐지

Farquhar 등 (2024)[@farquhar2024] 이 제안한 Semantic Entropy 는 동일
prompt 에 대해 생성된 N 개의 답변을 NLI 양방향 함의 (entailment) 관계로
의미 cluster 로 묶은 뒤, cluster 질량의 엔트로피를 환각 지표로 사용한다.
이 방법은 하나의 의미가 다양한 표현으로 나타날 수 있는 언어적 특성을 잘
반영하며, 태스크별 사전 지식 없이도 TriviaQA, SQuAD, BioASQ, NQ-Open,
SVAMP 등 여러 데이터셋에서 일관된 성능을 보인다. Semantic Entropy 는
naive entropy, $p(\text{True})$, embedding 회귀 기준선 대비 안정적으로
높은 AUROC 를 달성한다. 이는 SelfCheckGPT[@selfcheckgpt2023] 가 제시한
sample-consistency 접근을 의미 단위로 일반화한 결과로 평가된다.

Kossen 등 (2024)[@kossen2024seprobes] 의 Semantic Entropy Probes (SEPs)
는 Semantic Entropy 계산의 높은 연산 비용을 줄이기 위해, 단일 생성의
hidden state 로부터 Semantic Entropy 를 직접 근사하는 probe 를 학습한다.
SEPs 는 다중 샘플링 없이도 환각 탐지 성능을 유지하며, 분포 외
데이터에서도 안정적인 일반화를 보인다.

Nikitin 등 (2024)[@nikitin2024kle] 의 Kernel Language Entropy (KLE) 는
Semantic Entropy 를 일반화하여 hard clustering 대신 pairwise 의미 유사도
커널과 von Neumann 엔트로피를 사용한다. KLE 는 cluster 간 의존성을 더
세밀하게 포착하여, 여러 데이터셋과 모델 구조에서 Semantic Entropy 보다
우수한 불확실성 정량화 성능을 보고한다. Ciosek 등
(2025)[@ciosek2025bayes] 은 Bayesian 접근으로 Semantic Entropy 추정의
표본 효율을 높여, Farquhar 등 (2024) 대비 53% 적은 샘플로 동등한 AUROC
를 달성했다.

## Logit 기반 불확실성 추정

Ma 등 (2025)[@ma2025] 의 Semantic Energy 는 Semantic Entropy 의 한계를
보완하기 위해 logit 기반 에너지 모델을 도입한다. Semantic Entropy 는
모든 답변이 단일 cluster 에 집중될 때 분리력을 잃지만, Semantic Energy
는 모델의 penultimate layer logit 을 직접 활용하여 이러한 경우까지
포착한다. 토큰 에너지 $\tilde{E}(x_t) = -z_\theta(x_t)$ 를 정의하고,
Semantic Entropy 와 동일한 의미 cluster 구조 위에서 cluster 에너지를
산출한다. 보고된 개선 폭은 평균 AUROC 13% 이상이다. 확률이 logit 정규화
과정에서 강도 정보를 잃는다는 점은 LLM 의 불확실성 표현 능력을 제한하는
주요 요인으로 지적된다.

토큰 단위 불확실성에 대한 또 다른 접근으로, Raghuvanshi 등
(2025)[@raghuvanshi2025] 은 토큰 단위 log-likelihood, 양방향 NLI
contradiction 신호, Semantic Entropy 를 결합한 하이브리드 탐지
파이프라인을 제안하여 SQuAD2.0 에서 AUC 0.818 을 달성했다.

## Corpus 통계를 이용한 환각 탐지

LLM 의 환각은 사전학습 corpus 에서 드물게 등장하는 사실과 강하게
연관된다. HALoGEN (Ravichander 등, 2025)[@ravichander2025halogen] 은 9
개 도메인에 걸친 10,923 개 항목의 환각 벤치마크를 구축하고, LLM 생성물의
원자적 사실 단위를 고품질 지식 소스와 자동 검증하는 프레임워크를
제시했다. 이 연구는 최고 성능 모델에서도 생성된 원자적 사실의 최대 86%
가 환각으로 분류될 수 있음을 보여주며, 환각 원인을 학습 데이터 오류 회상
(Type A), 학습 데이터의 잘못된 지식 (Type B), 순수 조작 (Type C) 으로
분류했다.

WildHallucinations[@wildhallucinations2024] 는 Wikipedia 페이지가 없는
entity 에 대한 질문에서 LLM 의 환각률이 유의미하게 증가한다는 점을
관찰하여, corpus 노출 부족과 환각 발생률 사이의 경험적 연관성을
보고했다. Zhang 등 (2025)[@zhang2025corpus] 은 RedPajama 1.3 조 토큰
corpus 위에 suffix array 를 구축하고, prompt 와 답변의 n-gram 빈도를
환각 탐지 신호로 평가했다.

Infini-gram[@infinigram2024] 은 n-gram 카운팅을 통해 corpus 통계 질의를
지원하는 엔진으로, 특정 phrase 나 entity 의 등장 빈도를 효율적으로
산출한다. 그러나 이를 환각 탐지의 corpus 지지 신호로 활용하는 연구는
아직 초기 단계이며, 단위 (entity 빈도, entity co-occurrence, n-gram)
별로 corpus 통계 신호가 탐지 신호와 어떻게 연관되는지를 비교한 연구는
충분하지 않다.

## 다중 신호 결합 (Multi-Signal Fusion)

단일 신호의 한계를 극복하기 위해 여러 신호를 결합하는 fusion 접근이
활발히 연구되고 있다. Raghuvanshi 등 (2025)[@raghuvanshi2025] 은 토큰
단위 불확실성, NLI 신호, Semantic Entropy 를 동적 가중 결합하여
calibration 한계를 완화했다. ECLIPSE (Singha, 2025)[@singha2025eclipse]
는 semantic entropy 와 새로운 perplexity 분해를 결합해 증거 활용 신호를
추가하여 금융 QA 에서 AUC 0.89 를 달성했다. SEReDeEP (Wang,
2025)[@wang2025seredeep] 는 RAG 환경에서 semantic entropy 와 attention
메커니즘 신호를 결합하여 컨텍스트 의존성이 큰 환각 탐지를 강화했다.
Valentin 등 (2024)[@valentin2024] 은 환각 탐지 점수를 모델 내부 score
attribute 에 조건부로 calibrate 하는 다중 점수 프레임워크를 제안했다.

이들 연구는 여러 신호의 결합이 단일 신호 대비 성능 향상으로 이어질 수
있음을 보여준다. 그러나 선행 연구들은 공통적으로 corpus 통계를 탐지 신호
자체 또는 탐지기의 입력 변수로만 사용했을 뿐, 동일 신호가 탐지 신호의
영역별 신뢰도를 분해하는 평가 척도로서 어떤 효용을 가지는지는 비교하지
않았다.

본 연구는 이 두 역할을 명확히 분리하여 정량화한다. corpus 통계 신호를
fusion 입력에 추가했을 때의 성능 변화를 측정하는 한편, 동일 신호로
샘플을 나누었을 때 환각 탐지 신호의 AUROC 가 어떻게 달라지는지를
비교한다. 이 두 결과의 비대칭이 corpus 신호의 설계 목표 자체를 재정립할
필요성을 제기하는 것이 본 연구가 선행 연구와 구별되는 지점이다.

# 제안 방법

[]{#ch:method label="ch:method"}

본 장에서는 연구에 사용된 데이터셋과 실험 설정, 정답 라벨링 방식, 환각
탐지 신호 정의, corpus 신호 설계, 단위별 분해 분석 절차, 그리고 fusion
모델 구성까지 차례로 설명한다. 본 연구는 새로운 환각 탐지기를 제안하는
것이 아니라, 기존 탐지 신호의 성능이 사전학습 corpus 의 뒷받침 정도에
따라 어떻게 달라지는지를 *corpus 신호의 단위별로* 체계적으로 비교하는 데
초점을 맞춘다.

## 데이터셋 및 실험 설정 {#sec:exp_setup}

본 연구는 Farquhar 등 (2024)[@farquhar2024] 과 동일한 다섯 개의 질의응답
데이터셋을 사용한다. TriviaQA[@triviaqa] 800개, SQuAD-1.1[@squad] 800개,
BioASQ[@bioasq] 800개, NQ-Open[@nqopen] 800개, SVAMP[@svamp] 300개로 총
3,500개 sample 을 선정하였다. 각 데이터셋마다 무작위 추출 시드를
고정하여 동일한 sample 집합을 재현하였으며, 각 sample 에는 데이터셋
이름, split 식별자, 원본 색인, 본문 해시값을 결합한 고유 식별자를
부여하였다.

사용한 모델은 Qwen2.5-3B (base)[@qwen25] 이며, 별도의 instruction tuning
은 적용하지 않았다. 추론은 NVIDIA RTX 5090 32 GB 환경에서 16-bit
부동소수점 (float16) 으로 수행하였다. 입력 prompt 는 Farquhar 등 (2024)
와 동일하게 "Answer the following question in a single brief but
complete sentence." 라는 한 문장 응답 유도 형식을 따랐다. 각 sample 당
10개의 자유 생성 답변을 산출하였다. 샘플링 매개변수는 온도 1.0, top-p
0.9, top-k 50, 최대 생성 토큰 수 64로 설정하였다. 총 평가 단위는
35,000개의 생성 답변이다.

최대 토큰 수 64로 생성했을 때 전체 답변의 약 12.6% 가 중간에 절단되었다.
절단의 영향을 배제하기 위해 절단된 답변이 발생한 2,426개 sample 의 모든
답변을 최대 토큰 수 128로 다시 생성하였다. 재생성 전후 AUROC 변화는
$\pm$`<!-- -->`{=html}0.001 범위에 머물렀다.

평가 지표로는 AUROC, AURAC (Area Under Rejection-Accuracy Curve,
Farquhar 등 (2024) 의 주요 지표), Brier score, ECE 를 사용하였다. fusion
실험의 교차검증은 prompt 단위 5-fold GroupKFold 로 수행하였다. 같은
sample 의 모든 답변이 동일 fold 에 배치되도록 하여 학습-평가 간 sample
단위 정보 누출을 방지하였다.

## 답변 단위 정답 라벨 {#sec:label}

본 논문의 정답 라벨은 모델이 생성한 답변 $s_i$ 가 정답 후보 표현과
의미적으로 일치하는지를 NLI (자연어 추론) 양방향 함의 확률로 판정하여
정의한다. 각 sample 의 정답 후보 집합 $C$ 는 데이터셋이 제공한 정답
표현, 동의 표현, 별칭 목록을 모두 모은 뒤 소문자 정규화와 중복 제거를
거쳐 구성한다. 생성된 답변 $s_i$ 와 후보 $c$ 사이의 매칭 점수는 다음과
같이 정의한다. $$\begin{align*}
m(s_i, c) &= \max\bigl(p_\text{entail}(c \rightarrow s_i),\ p_\text{entail}(s_i \rightarrow c)\bigr), \\
M(s_i)   &= \max_{c \in C} m(s_i, c).
\end{align*}$$ $M(s_i) \geq 0.5$ 이면 $s_i$ 를 정답 (라벨 1), 그렇지
않으면 환각 (라벨 0) 으로 판정한다. NLI 모델로는
`microsoft/deberta-large-mnli` 를 사용하였으며, 절차는 Farquhar 등
(2024) 의 라벨링과 동일하다. 이 절차로 총 35,000개의 이진 라벨이
산출되었다. 본 평가 단위는 Farquhar 등 (2024) 의 Semantic Entropy 와 Ma
등 (2025) 의 Semantic Energy 가 사용한 답변 단위 정답성 평가와
일치하므로, 두 선행 연구와 직접 비교가 가능하다.

## Semantic Entropy {#sec:se_def}

Semantic Entropy 는 Farquhar 등 (2024) 의 정의를 그대로 따른다. 한
sample 에 대해 생성한 10개 답변을 NLI 양방향 함의 관계로 의미 cluster
$\mathbb{C}_k$ 로 묶은 뒤, 각 cluster 의 확률 질량을 답변의 시퀀스
로그우도로부터 계산한다. $$\begin{align*}
\log \tilde{p}(\mathbb{C}_k) &= \log\!\sum_{x^{(i)} \in \mathbb{C}_k} \exp\!\Bigl(\sum_t \log p_\theta(x_t^{(i)})\Bigr), \\
p(\mathbb{C}_k) &= \frac{\tilde{p}(\mathbb{C}_k)}{\sum_{k'} \tilde{p}(\mathbb{C}_{k'})}.
\end{align*}$$ Semantic Entropy 는 cluster 분포의 Shannon 엔트로피로
정의된다.
$$H(\mathbf{x}) = -\sum_{k} p(\mathbb{C}_k)\, \log p(\mathbb{C}_k).$$
Semantic Entropy 는 sample 단위로 산출되며, 같은 sample 의 모든 답변에
동일한 값이 적용된다.

## Semantic Energy {#sec:energy_def}

Semantic Energy 는 Ma 등 (2025) 의 정의를 그대로 구현한다. Semantic
Entropy 와 동일한 의미 cluster 구조 위에서, 확률 대신 모델의 token logit
을 직접 활용하여 모델 자체의 지식 한계에서 비롯되는 불확실성을 추정한다.
$$\begin{align}
\tilde{E}(x_t^{(i)}) &= -z_\theta(x_t^{(i)}) & \text{(token energy)} \\
E(x^{(i)}) &= \tfrac{1}{T_i}\sum_{t} \tilde{E}(x_t^{(i)}) & \text{(sample energy)} \\
E_{\text{Bolt}}(\mathbb{C}_k) &= \sum_{x^{(i)} \in \mathbb{C}_k} E(x^{(i)}) & \text{(cluster total)} \\
U(\mathbf{x}) &= \sum_{k} p(\mathbb{C}_k)\, E_{\text{Bolt}}(\mathbb{C}_k) & \text{(sample 단위 uncertainty)}
\end{align}$$ 값이 낮을수록 모델 신뢰도가 높음을 의미한다.

## 각 답변의 token logit 통계 {#sec:diag_def}

각 답변에 대해 다음 네 가지 token logit 통계를 산출한다.

- 평균 음의 로그우도:
  $\displaystyle -\tfrac{1}{T_i}\sum_t \log p(x_t^{(i)})$.

- 시퀀스 로그우도: $\displaystyle \sum_t \log p(x_t^{(i)})$.

- logit 분산:
  $\displaystyle \mathrm{Var}_t\!\bigl(z_\theta(x_t^{(i)})\bigr)$.

- 평균 로그 분배함수:
  $\displaystyle \tfrac{1}{T_i}\sum_t \log Z_t^{(i)}$ (분포의 평탄도
  지표).

이 신호들은 답변마다 산출된 변동 정보를 담아 fusion 모델의 입력으로
사용된다.

## Corpus 신호 설계 {#sec:corpus_def}

Corpus 신호는 Infini-gram[@infinigram2024] 의 local engine 을 backend 로
사용한다. 사용한 색인은 OLMo Dolma sample (16B token, OLMo-7B-hf
토크나이저) 이다. 본 연구는 단위가 서로 다른 네 가지 corpus 신호를
다음과 같이 정의한다.

##### Entity 수준 신호.

spaCy 의 `en_core_web_lg` 개체명 인식기로 12개 범주 (인물, 조직, 지명,
위치, 날짜, 사건, 작품, 시설, 국적/종교, 제품, 언어, 법) 의 entity 를
추출한다. 한 sample 에 대해 다음 두 가지 정규화 점수를 계산한다.
$$\begin{align*}
\text{entity 빈도} &= \frac{\log(1 + \min_{e \in E} \mathrm{freq}(e))}{\log(1 + 10^6)}, \\
\text{entity co-occurrence} &= \frac{\log\!\bigl(1 + \tfrac{1}{|P|}\sum_{(e_i,e_j) \in P}\mathrm{cooc}(e_i,e_j)\bigr)}{\log(1 + 10^5)}.
\end{align*}$$ $\mathrm{freq}(e)$ 는 entity $e$ 의 단일 등장 횟수이고,
$\mathrm{cooc}(e_i,e_j)$ 는 두 entity 가 corpus 안에서 함께 등장한
횟수이다. 본 연구의 baseline corpus 신호는 두 점수의 평균이며, 두 단위를
분리하여 비교하는 데 초점을 맞춘다.

##### Question-answer bridge 신호.

질문에서 추출한 entity 집합 $E_q$ 와, 답변에서 추출한 entity 중 질문과
겹치지 않는 entity 집합 $E_a$ 를 구성한다. 질문 entity 를 답변 entity
에서 제외하는 이유는 paraphrase 로 인한 자기 중복을 줄이기 위함이다.
모든 (질문 entity, 답변 entity) 쌍에 대해 corpus 동시 등장 횟수를
산출하고, 그 정규화 평균을 question-answer bridge 신호로 정의한다.

##### N-gram 등장 빈도 신호.

답변 토큰 시퀀스의 모든 3-gram 과 5-gram 에 대해 corpus 등장 횟수를
산출한다. 평균 등장 횟수를 정규화한 점수와, corpus 에 한 번도 등장하지
않은 n-gram 의 수를 함께 보고한다.

## Corpus 신호 단위별 분해 분석 {#sec:axis_framework}

본 논문 분석의 핵심은 §[3.6](#sec:corpus_def){reference-type="ref"
reference="sec:corpus_def"} 에서 정의한 corpus 신호를 환각 탐지의 입력
변수가 아니라 *sample 을 분할하는 신호* 로 사용하는 데 있다. 각 corpus
신호로 sample 을 순위에 따라 10분위 구간으로 나누어, 한 sample 의 corpus
뒷받침 정도가 어느 구간에 속하는지를 판별한다. sample 분할에 사용한
신호는 entity 수준 신호 세 가지 (entity 빈도, entity co-occurrence, 두
점수의 평균), question-answer bridge 한 가지, n-gram 수준 신호 세 가지
(3-gram 평균 등장, 5-gram 평균 등장, 3-gram 미등장 개수) 로 총 일곱
가지이다. 같은 sample 에 속한 모든 답변은 동일한 구간값을 공유한다.

각 신호의 10개 구간 안에서 환각 탐지 신호의 AUROC 를 따로 계산한 뒤, 그
최댓값과 최솟값의 차이
$\Delta = \mathrm{AUROC}_{\max} - \mathrm{AUROC}_{\min}$ 를 해당 신호의
*구간별 AUROC range* 으로 정의한다. AUROC range 가 클수록 해당 corpus
신호로 sample 을 나누었을 때 환각 탐지 신호의 성능이 corpus 조건에 따라
더 큰 차이를 보인다는 의미이다. 본 논문의 핵심 결과는 이 AUROC range 가
corpus 신호의 단위에 따라 어떻게 달라지는지를 비교한 것이다.

## Fusion 모델 {#sec:fusion_def}

Fusion 모델의 입력은 두 가지 구성으로 실험한다.

- **CORE 입력**: sample 단위 신호인 Semantic Entropy, Semantic Energy
  와, 답변마다 산출한 token logit 통계 4종 (평균 음의 로그우도, 시퀀스
  로그우도, logit 분산, 평균 로그 분배함수).

- **CORE + CORPUS 입력**: 위 신호에 entity, question-answer bridge,
  n-gram 의 모든 corpus 신호를 sample 단위로 함께 추가한 구성.

분류 모델로는 logistic regression, random forest, gradient boosting 세
가지를 비교한다. 모든 실험은 prompt 단위 5-fold GroupKFold 로 평가하여
학습-평가 사이의 정보 누출을 방지한다.

# 실험

[]{#ch:experiment label="ch:experiment"}

본 장에서는 주요 환각 탐지 신호와 corpus 신호의 종합 성능을 평가하고,
단일 신호 성능을 선행 연구와 비교하며, fusion 모델의 효과와 corpus
신호의 단위별 분해 결과를 보고한다. 마지막으로 데이터셋별 변동성을
분석한다.

## 주요 신호의 종합 성능

본 절에서는 단일 신호와 fusion 모델의 종합 AUROC, AURAC, Brier score 를
제시한다.

::: {#tab:current_thesis_evidence}
+-----------------+-------------+-------------+-------------+-------------+
| 신호            | AUROC       | AURAC       | Brier       | corpus 추가 |
|                 |             |             |             | 효과        |
+:================+:===========:+:===========:+:===========:+:===========:+
| *단일 신호 (학습 없음, 답변 일관성 및 토큰 logit 통계)*                 |
+-----------------+-------------+-------------+-------------+-------------+
| 답변 내 logit   | 0.620       | 0.400       | ---         | ---         |
| 분산            |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| 답변 전체       | 0.656       | 0.466       | ---         | ---         |
| 로그우도        |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| 답변 평균 음의  | 0.670       | 0.468       | ---         | ---         |
| 로그우도 (NLL)  |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| Semantic        | 0.759       | 0.526       | ---         | ---         |
| Entropy         |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| Semantic Energy | 0.774       | 0.533       | ---         | ---         |
+-----------------+-------------+-------------+-------------+-------------+
| *단일 corpus 신호 (학습 없음)*                                          |
+-----------------+-------------+-------------+-------------+-------------+
| entity          | 0.546       | ---         | ---         | ---         |
| co-occurrence   |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| entity 빈도     | 0.565       | ---         | ---         | ---         |
+-----------------+-------------+-------------+-------------+-------------+
| 답변 3-gram     | 0.563       | ---         | ---         | ---         |
| 등장 빈도       |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| question-answer | 0.583       | ---         | ---         | ---         |
| bridge (평균)   |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| 답변 3-gram     | 0.583       | ---         | ---         | ---         |
| 미등장 개수     |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| *Fusion 모델 (sample 단위 5-fold 교차검증, 검증 fold 예측 기준)*        |
+-----------------+-------------+-------------+-------------+-------------+
| logistic        | 0.793       | 0.548       | 0.164       | ---         |
| regression      |             |             |             |             |
| (corpus 미포함) |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| logistic        | 0.795       | 0.550       | 0.162       | $+0.002$    |
| regression      |             |             |             |             |
| (corpus 포함)   |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| random forest   | 0.790       | 0.546       | 0.166       | ---         |
| (corpus 미포함) |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| random forest   | 0.800       | 0.555       | 0.161       | $+0.010$    |
| (corpus 포함)   |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| gradient        | 0.800       | 0.553       | 0.161       | ---         |
| boosting        |             |             |             |             |
| (corpus 미포함) |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+
| **gradient      | **0.808**   | **0.559**   | **0.158**   | $+0.008$    |
| boosting        |             |             |             |             |
| (corpus 포함)** |             |             |             |             |
+-----------------+-------------+-------------+-------------+-------------+

: 주요 신호의 종합 성능. 생성 답변 35,000건 (Qwen2.5-3B, 5개 데이터셋,
sample 당 10개 자유 생성 답변) 을 대상으로 측정하였다. 라벨 정의, 평가
단위, fusion 입력, AURAC 정의는 §[3.2](#sec:label){reference-type="ref"
reference="sec:label"}--§[3.8](#sec:fusion_def){reference-type="ref"
reference="sec:fusion_def"} 에서 기술한다. corpus 추가 효과는 같은
분류기에서 핵심 입력만 사용한 경우와 핵심 입력에 corpus 신호를 함께 쓴
경우의 AUROC 차이로 정의한다. 굵은 글씨는 AUROC 가 가장 높은 행을
가리킨다.
:::

표 [4.1](#tab:current_thesis_evidence){reference-type="ref"
reference="tab:current_thesis_evidence"} 의 결과는 세 가지로 요약된다.
첫째, Semantic Entropy (0.759) 와 Semantic Energy (0.774) 단독 신호의
AUROC 는 Farquhar 등 (2024) 와 Ma 등 (2025) 의 보고 범위와 비슷하다
(다만 모델 간 일반화의 충분 조건은 아니다 ---
§[5.2](#sec:limitations){reference-type="ref"
reference="sec:limitations"}). 둘째, 모든 단일 corpus 신호의 AUROC 는
0.50--0.58 범위에 머물러 단독 탐지기로서의 분리력은 제한적이다
(§[4.3](#sec:negative){reference-type="ref" reference="sec:negative"}).
셋째, fusion 모델에 corpus 신호를 추가했을 때 AUROC 향상 폭은 random
forest 에서 +0.010, gradient boosting 에서 +0.008 으로 매우 작다. corpus
신호의 fusion 입력 변수로서의 기여는 제한적이다.

## 단일 신호 성능과 선행 연구 비교 {#sec:single_signal}

Farquhar 등 (2024) 과 동일한 다섯 개 데이터셋과 동일한 평가 단위 (생성
답변마다 산출한 NLI 정답 라벨) 에서 Qwen2.5-3B 의 Semantic Entropy 와
Semantic Energy AUROC 를 측정하였다. 결과는
표 [\[tab:single_signal_aurac\]](#tab:single_signal_aurac){reference-type="ref"
reference="tab:single_signal_aurac"} 에 정리되어 있다.

Qwen2.5-3B 는 Llama-2 7B 보다 작은 모델임에도 두 신호의 AUROC 가 선행
연구 보고 범위 (0.75--0.85) 안에 들어간다. 이는 본 표본이 Semantic
Entropy 와 Semantic Energy 의 일반적인 작동 범위에 속한다는 점을
보여준다. 다만 이는 직접 재현이 아닌 참고용 비교이며, 답변 평균 NLL
(0.670) 은 단독 신호로도 일정한 분리력을 보이지만 의미 cluster 정보를
추가했을 때 부가 효과가 크다.

## 단일 corpus 신호의 환각 탐지 성능 {#sec:negative}

각 corpus 신호를 단독으로 환각 탐지 신호로 사용했을 때의 AUROC 를
측정하였다. 결과는 표 [4.2](#tab:corpus_only){reference-type="ref"
reference="tab:corpus_only"} 에 정리되어 있다.

::: {#tab:corpus_only}
  신호                                    AUROC
  -------------------------------------- -------
  question-answer bridge (평균)           0.583
  답변 3-gram 미등장 개수                 0.583
  question-answer bridge (미등장 표시)    0.574
  entity 빈도                             0.565
  답변 3-gram 등장 빈도                   0.563
  답변 5-gram 등장 빈도                   0.557
  entity co-occurrence                    0.546
  question-answer bridge (정규화)         0.511
  entity 빈도 (평균)                      0.500

  : 단일 corpus 신호를 환각 탐지 신호로 사용했을 때의 AUROC. 각 신호는
  한 sample 의 모든 답변에 같은 값으로 적용되며, 신호 점수의 순위만으로
  평가한다. 모든 결과가 0.50--0.58 범위에 한정된다.
:::

모든 corpus 신호의 단독 AUROC 는 0.50--0.58 범위에 머물러 단독
탐지기로서의 분리력은 제한적이다. 이는 corpus 신호가 sample 단위의 정적
신호라는 특성상, 같은 sample 안 10개 답변 사이의 정답 여부 변동을
구분하기 어렵기 때문이다. 따라서 본 절은 corpus 신호가 환각 탐지에
무용하다는 주장이 아니라, sample 단위 정적 신호만으로는 답변 단위 정답
여부 변동을 분리하기 어렵다는 점을 정량화하여 후속 분석의 동기를
제시한다.

주목할 점은 후속 절 (§[4.5](#sec:axis){reference-type="ref"
reference="sec:axis"}) 에서 가장 큰 구간별 AUROC range 를 보이는 entity
co-occurrence 신호가, 표 [4.2](#tab:corpus_only){reference-type="ref"
reference="tab:corpus_only"} 의 단독 AUROC 비교에서는 0.546 으로
하위권에 머문다는 사실이다. 단독 신호로서의 분리력과 같은 신호로 sample
을 나누어 분석했을 때의 구간별 AUROC range 는 서로 다른 지표이며, 이
비대칭이 본 연구 핵심 결과의 출발점이다.

## 신호 결합 (fusion) 결과 {#sec:fusion_results}

Fusion 모델의 AUROC 와 corpus 신호 추가 효과를
표 [4.3](#tab:fusion){reference-type="ref" reference="tab:fusion"} 에
정리하였다.

::: {#tab:fusion}
  Method                                   AUROC       AURAC     corpus 추가 lift
  ------------------------------------- ----------- ----------- ------------------
  Semantic Energy 단독                     0.774       0.533        --- (기준)
  Semantic Entropy 단독                    0.759       0.526           ---
  logistic regression (corpus 미포함)      0.793       0.548           ---
  logistic regression (corpus 포함)        0.795       0.550         $+0.002$
  random forest (corpus 미포함)            0.790       0.546           ---
  random forest (corpus 포함)              0.800       0.555         $+0.010$
  gradient boosting (corpus 미포함)        0.800       0.553           ---
  **gradient boosting (corpus 포함)**    **0.808**   **0.559**       $+0.008$

  : Fusion 모델의 AUROC. 모든 모델은 sample 단위 5-fold GroupKFold 의
  검증 fold 예측으로 평가하였다. *corpus 추가 효과* 는 같은 분류기에서
  핵심 입력만 사용한 경우와 핵심 입력에 corpus 신호를 함께 쓴 경우의
  AUROC 차이이다.
:::

Semantic Energy 단독 (0.774) 대비 gradient boosting (corpus 미포함,
0.800) 의 +0.026 향상은 답변 단위 logit 통계 (NLL, logit 분산 등) 와
sample 단위 cluster 신호의 결합에서 비롯된다. 측정 단위가 서로 다른
불확실성 신호를 결합하는 방식이 단일 신호보다 분리력을 끌어올리는 데
기여한다. corpus 신호 추가 효과는 random forest 에서 +0.010, gradient
boosting 에서 +0.008 이다 (95% 신뢰구간 \[+0.005, +0.011\], sample 단위
부트스트랩 반복 500회, 모든 반복에서 양수). 통계적으로 유의하지만 절대
크기는 작으며, 답변 단위와 sample 단위 신호 결합 효과 (+0.026) 의 약
3분의 1 수준이다. fusion 입력 변수로서의 기여는 제한적인 반면, 같은
신호로 sample 을 분할했을 때의 구간별 AUROC range 는 큰 비대칭을 보인다
(다음 절 참조).

## Corpus 신호 단위별 분해 결과 {#sec:axis}

각 corpus 신호로 sample 을 10분위 구간으로 나눈 뒤, 구간 안에서 Semantic
Entropy, Semantic Energy, 답변 평균 NLL 의 AUROC 를 계산하고 최댓값과
최솟값의 차이 $\Delta = \mathrm{AUROC}_{\max} - \mathrm{AUROC}_{\min}$
를 AUROC range 로 측정하였다. 본 논문의 핵심 결과는
표 [\[tab:axis_decomp\]](#tab:axis_decomp){reference-type="ref"
reference="tab:axis_decomp"} 와
그림 [4.1](#fig:axis_decomp){reference-type="ref"
reference="fig:axis_decomp"} 이다.

두 corpus 신호의 분해 패턴을 세 환각 탐지 신호 (Semantic Entropy,
Semantic Energy, 답변 평균 NLL) 에 대해 비교한 결과를
그림 [4.1](#fig:axis_decomp){reference-type="ref"
reference="fig:axis_decomp"} 에 제시한다.

<figure id="fig:axis_decomp" data-latex-placement="htbp">

<figcaption>세 환각 탐지 신호 (Semantic Entropy, Semantic Energy, 답변
평균 NLL) 의 구간별 AUROC 를 entity co-occurrence 신호 (실선) 와 entity
빈도 신호 (점선) 로 sample 을 분할하여 비교한 결과. 세 환각 탐지 신호
모두 entity co-occurrence 신호로 분할했을 때의 AUROC range 가 entity
빈도 신호로 분할했을 때보다 크며, Semantic Entropy 와 Semantic Energy
에서 그 차이가 뚜렷하게 나타난다. Semantic Entropy 의 AUROC range 는
entity co-occurrence 신호에서 <span
class="math inline"><em>Δ</em></span>0.150, entity 빈도 신호에서 <span
class="math inline"><em>Δ</em></span>0.080 이다. Semantic Energy 도
비슷한 패턴을 따른다 (<span class="math inline"><em>Δ</em></span>0.144
와 <span class="math inline"><em>Δ</em></span>0.077). 답변 평균 NLL 은
두 신호 모두에서 AUROC range 가 작다 (<span
class="math inline"><em>Δ</em></span>0.092 와 <span
class="math inline"><em>Δ</em></span>0.082). 두 신호 모두 구간이
올라갈수록 AUROC 가 일관되게 증가하지는 않는다. 예를 들어 entity
co-occurrence 신호로 분할했을 때 Semantic Entropy 는 구간 20–30 에서
0.671 로 일시 하락한 뒤 회복한다.</figcaption>
</figure>

표 [\[tab:axis_decomp\]](#tab:axis_decomp){reference-type="ref"
reference="tab:axis_decomp"} 와
그림 [4.1](#fig:axis_decomp){reference-type="ref"
reference="fig:axis_decomp"} 에서 다음 결과가 도출된다.

##### Entity co-occurrence 신호의 구간별 AUROC range.

entity co-occurrence 신호로 sample 을 10분위 구간으로 나눌 때, Semantic
Entropy 의 AUROC range ($\Delta$) 는 0.150, Semantic Energy 는 0.144 로
나타났다. 이는 본 연구에서 비교한 일곱 가지 corpus 신호 가운데 가장 큰
범위이다.

sample 단위 부트스트랩 (500회 반복) 으로 계산한 95% 신뢰구간은 Semantic
Entropy AUROC range 가 0.121--0.211, entity 빈도 신호는 0.058--0.149
였다. 두 값의 차이는 95% 신뢰구간 \[+0.002, +0.117\] 으로 0 을 포함하지
않으며, 부트스트랩 반복의 97.6% 에서 양의 값으로 관찰되었다. Semantic
Energy 에서도 차이가 \[+0.004, +0.109\] 로 98.0% 가 양수였다. 따라서
entity co-occurrence 신호로 분할했을 때의 AUROC range 가 entity 빈도
신호보다 크다는 결과는, 표본 변동성을 통제한 뒤에도 통계적으로 유지된다.

corpus 가 가장 부족한 구간 (00--10) 에서는 Semantic Entropy 가 0.643,
Semantic Energy 가 0.667 로 관찰되었으며, 가장 풍부한 구간 (70--80)
에서는 각각 0.793, 0.811 로 측정되었다. 다만 구간이 한 단계씩 올라갈
때마다 AUROC 가 단조 증가하지는 않으며, 중간 구간에서는 일시적인 하락과
회복이 관찰된다 (그림 [4.1](#fig:axis_decomp){reference-type="ref"
reference="fig:axis_decomp"}). "체계적 정렬" 이라는 표현은 양 끝단
사이의 평균적인 증가 경향과 최댓값과 최솟값의 폭을 가리키며, 인접 구간
사이의 엄밀한 단조 증가를 뜻하지 않는다.

이러한 결과는 답변에 등장하는 entity 쌍이 corpus 에 함께 등장하는 정도가
LLM 의 해당 사실 학습 가능성을 반영하는 지표로 작용하며, 답변 간
일관성을 측정하는 신호의 변동과 연결되는 한 요인임을 시사한다.

##### Entity 빈도 신호의 구간별 AUROC range.

단일 entity 빈도 (entity frequency) 신호로 sample 을 분할했을 때의 AUROC
range 는 Semantic Entropy 기준 $\Delta$`<!-- -->`{=html}0.080, Semantic
Energy 기준 $\Delta$`<!-- -->`{=html}0.077 로 본 표에서 가장 작다. 본
연구의 baseline corpus 신호 (entity 빈도와 entity co-occurrence의 평균)
도 AUROC range 가 $\Delta$`<!-- -->`{=html}0.082--0.086 으로 entity 빈도
신호와 거의 같은 수준이다. entity co-occurrence 신호의 큰 AUROC range 가
entity 빈도 신호의 작은 AUROC range 와 평균되며 희석된 결과로 해석되며,
corpus 신호를 설계할 때 두 단위를 따로 보고해야 한다는 점을 시사한다.
평균화의 영향은 AUROC range 뿐 아니라 단조성에서도 나타난다. entity
co-occurrence 신호 (Semantic Entropy $\rho$=+0.648, $p$=0.043) 와 entity
빈도 신호 (Semantic Entropy $\rho$=+0.636, $p$=0.048) 모두 양의 단조성을
보이지만, 두 점수의 평균인 baseline corpus 신호는 Semantic Entropy
$\rho$=+0.418 ($p$=0.229) 로 유의 임계 아래에 머무른다
(표 [\[tab:axis_decomp\]](#tab:axis_decomp){reference-type="ref"
reference="tab:axis_decomp"}). 단위가 다른 두 신호를 평균하면 각 신호가
가지고 있던 단조 증가 경향까지 무뎌진다. 본 논문의 정량 비교 --- entity
co-occurrence 단위가 entity 빈도 단위보다 Semantic Entropy AUROC range
를 약 1.88배 크게 보인다는 점 --- 은 본 결과에서 곧바로 도출된다.

##### Question-answer bridge 신호와 NLL 변동의 관계.

Question-answer bridge 신호 ((질문 entity, 답변 entity) 쌍의 corpus 동시
등장 빈도) 로 sample 을 분할하면, 답변 평균 NLL 의 AUROC range 가
$\Delta$`<!-- -->`{=html}0.176 로 측정된다. Semantic Entropy 와 Semantic
Energy 의 AUROC range ($\Delta$`<!-- -->`{=html}0.089,
$\Delta$`<!-- -->`{=html}0.087) 보다 큰 폭이다. (질문, 답변) 의 사실
관계가 corpus 에 함께 등장하는 정도가 모델의 토큰 단위 자신감 (NLL) 과
가장 직접 연결되며, Semantic Entropy 나 Semantic Energy 가 포착하는 답변
사이 일관성과는 다른 정보를 측정함을 시사한다.

##### 답변 3-gram 미등장 개수 신호: AUROC range 와 단조성의 분리.

답변에서 corpus 에 한 번도 등장하지 않은 3-gram 의 개수로 sample 을
분할하면, Semantic Entropy AUROC range 는 $\Delta$`<!-- -->`{=html}0.122
로 비교적 큰 편에 속한다. 그러나 단조성은 $\rho$=+0.261 ($p$=0.467,
표 [\[tab:axis_decomp\]](#tab:axis_decomp){reference-type="ref"
reference="tab:axis_decomp"}) 로, entity co-occurrence ($\rho$=+0.648),
question-answer bridge ($\rho$=+0.648), 답변 5-gram ($\rho$=+0.685) 등
다른 신호와 달리 유의수준 0.05 를 넘지 않는다. 이 신호의 AUROC range 는
구간이 올라갈수록 일관되게 증가하는 패턴이 아니라, U자형이나 일부 구간의
일시적 변동에서 비롯된 것으로 보인다. 따라서 "corpus 미등장 3-gram 이
많을수록 Semantic Entropy 가 낮아진다" 라고 단정하기는 어렵고, 미등장
3-gram 비율이 높은 일부 구간에서 Semantic Entropy AUROC 의 변동이
관찰된다는 정도로 해석을 한정한다.

##### 강건성 검증: 질문 entity 한정 신호.

앞서 사용한 entity co-occurrence 신호는 질문과 데이터셋이 제공한 정답
텍스트의 entity 를 모두 합쳐 산출하므로, 답변 측 entity 가 신호 값에
영향을 줄 가능성이 있다. 본 절은 이 가능성을 강건성 검증으로 점검한다.
질문 텍스트에서만 entity 를 추출하여 같은 분석을 재수행하였으며, 질문이
두 개 이상의 entity 를 포함하는 2,095 sample 이 분석 대상이 된다. 그
결과 Semantic Entropy 의 AUROC range 는 0.178, Semantic Energy 의 AUROC
range 는 0.162 로, 원래 신호 (Semantic Entropy 0.150, Semantic Energy
0.144) 보다 큰 AUROC range 가 관찰되었다. 질문에 한정한 entity 빈도
신호의 Semantic Entropy AUROC range 는 0.081 로 원래 신호의 0.080 과
거의 같다. 따라서 entity co-occurrence 단위가 entity 빈도 단위보다 AUROC
range 가 크다는 본 논문의 결과는 답변 측 entity 를 신호 산출에서
제거해도 유지된다. 비율은 약 1.88배에서 Semantic Entropy 기준 2.20배,
Semantic Energy 기준 2.10배로 오히려 증가한다. 이는 두 가지를 시사한다.
첫째, 본 신호의 AUROC range 는 모델 출력으로 인해 인공적으로 부풀려진
결과가 아니다. 둘째, 질문 측 entity co-occurrence이 LLM 의 사전학습된
지식을 더 직접 반영하는 지표일 가능성이 있다. 다만 본 분석의 부분표본은
entity 가 두 개 이상인 질문에 한정된다. 이러한 질문은 단일 entity 의
사실을 묻기보다 entity pair 의 관계를 묻는 지식 질문에 치우친다. entity
pair 관계는 정의상 entity co-occurrence 와 직접 대응하므로, 본
부분표본은 entity co-occurrence 신호의 AUROC range 가 본래 크게 나타나는
조건이며, 이는 *선택 편향* (selection bias) 에 해당한다. 따라서 비율
2.20 의 절대 크기는 이 부분표본의 특성을 반영한다. 전체 표본의 비율 1.88
과 산술적으로 직접 비교하기보다는, 답변 측 entity 를 제거해도 비율의
부호가 유지된다는 정성적 강건성 결과로 해석한다.

##### 강건성 검증: SVAMP 제외 민감도.

SVAMP 는 다른 네 데이터셋과 달리 수학 단어 문제이므로, entity
co-occurrence의 의미가 지식 질문과 다를 수 있다. SVAMP 를 제외한 3,200
sample 으로 같은 분석을 재수행하면, Semantic Entropy 기준 비율 (entity
co-occurrence AUROC range / entity 빈도 AUROC range) 은 1.68 로 전체
표본의 비율 1.88 보다 약 0.20 줄어든다. Semantic Energy 기준 비율도 1.54
로 비슷한 폭으로 감소한다. 두 단위 사이의 부호 (entity co-occurrence \>
entity 빈도) 는 유지되지만, 절대 크기는 SVAMP 포함 여부에 따라 달라진다.
본 결과를 토대로 비율은 1.5--1.9 범위로 보고하는 것이 안정적이다.

##### 데이터셋별 분해 결과.

표 [\[tab:per_dataset_delta\]](#tab:per_dataset_delta){reference-type="ref"
reference="tab:per_dataset_delta"} 는 데이터셋별 Semantic Entropy AUROC
range 를 entity co-occurrence 신호와 entity 빈도 신호 모두에 대해
보고한다. 다섯 데이터셋 중 네 데이터셋 (BioASQ, SQuAD-1.1, SVAMP,
TriviaQA) 에서는 entity co-occurrence AUROC range 가 entity 빈도 AUROC
range 보다 크거나 같다. 다만 비율은 TriviaQA 의 1.0배에서 BioASQ 의
2.92배까지 큰 폭으로 변동한다. NQ-Open 에서는 부호가 역전되어 entity
빈도 AUROC range (0.372) 이 entity co-occurrence AUROC range (0.250)
보다 크다. 따라서 전체 표본의 비율 1.88배는 데이터셋별 비율의 평균이
아니라, BioASQ 의 큰 비율 (2.92) 이 NQ-Open 의 역전 (0.67) 을 보상하여
형성된 표본 종합 통계이다. 즉 entity co-occurrence 신호가 모든
데이터셋에서 entity 빈도 신호보다 큰 AUROC range 를 보이지는 않는다. 본
결과는 데이터셋 도메인에 따라 적절한 corpus 신호의 단위가 달라질 수
있다는 가설을 제기한다.

## 데이터셋별 변동 {#sec:per_dataset}

다섯 데이터셋별 단일 신호와 fusion 모델의 AUROC 를
표 [\[tab:per_dataset\]](#tab:per_dataset){reference-type="ref"
reference="tab:per_dataset"} 에 정리하였다.

baseline corpus 신호 (entity 빈도와 entity co-occurrence 의 평균) 로
sample 을 분할하면, gradient boosting (corpus 포함) 의 AUROC 는 구간
30--40 에서 최저 0.762, 구간 60--70 에서 최고 0.859 까지
$\Delta$`<!-- -->`{=html}0.097 의 변동을 보인다. 같은 분석을 entity
co-occurrence 신호로 수행하면 Semantic Entropy 기준 AUROC range 가
$\Delta$`<!-- -->`{=html}0.150 로 더 커져, 패턴이 한층 분명하다
(표 [\[tab:axis_decomp\]](#tab:axis_decomp){reference-type="ref"
reference="tab:axis_decomp"}). 정답률도 corpus 가 풍부한 구간 (60--70,
0.382) 이 부족한 구간 (10--20, 0.209) 보다 높아, corpus 뒷받침 정도와
정답률 사이의 양의 상관이 확인된다. 데이터셋 사이의 패턴도 같은 방향으로
나타난다. 정답률은 TriviaQA (0.482) 에서 SQuAD (0.189) 로 감소하며, 환각
탐지 신호 AUROC 도 같은 순서로 감소한다 (Semantic Entropy 기준 TriviaQA
0.778, SQuAD 0.700). 이는 corpus 뒷받침 효과가 데이터셋 사이에서도
동일한 방향으로 나타남을 시사한다.

# 결론

[]{#ch:conclusion label="ch:conclusion"}

## 논의

본 연구는 환각 탐지 신호의 성능이 사전학습 corpus 의 뒷받침 정도에 따라
크게 달라진다는 사실을 체계적으로 보여주었다. 특히 entity co-occurrence
신호로 sample 을 분할했을 때 Semantic Entropy 와 Semantic Energy 의
AUROC range 가 entity 빈도 신호보다 약 1.5--1.9배 컸다 (Semantic Entropy
기준 $\Delta$`<!-- -->`{=html}0.150 와 $\Delta$`<!-- -->`{=html}0.080).
이는 "Paris--France 가 함께 등장하는지" 와 같은 entity pair 정보가 단일
entity 빈도보다 모델의 사실 학습 정도를 더 직접 반영한다는 직관과
일치한다.

가장 중요한 발견은 비대칭이다. 동일 corpus 신호를 fusion 모델 입력
변수로 추가했을 때 AUROC 향상 폭은 +0.008--+0.010 에 머물렀으나, 조건부
평가 척도로 사용했을 때 AUROC range 는 $\Delta$`<!-- -->`{=html}0.150 에
달했다. 이는 corpus 신호의 설계 목표를 "탐지기의 입력 변수" 에서 "탐지
신호의 영역별 신뢰도 분해 도구" 로 재설정할 필요를 시사한다. 향후 환각
탐지 연구에서는 평균 AUROC 뿐 아니라 corpus 조건에 따른 AUROC range 를
함께 보고하는 형식이 자리 잡는 것이 바람직하다.

실제 LLM 서비스 관점에서도 함의가 있다. corpus 뒷받침이 약한 질문 (예:
entity co-occurrence 가 낮은 구간) 은 Semantic Entropy 와 Semantic
Energy 의 신뢰도가 큰 폭으로 떨어지므로, 별도 검증 절차 (RAG 호출, 인간
검토, 더 강력한 fusion 모델 적용 등) 를 거치는 조건부 신뢰 전략을 도입할
수 있다. Question-answer bridge 신호와 답변 평균 NLL 사이의 AUROC range
($\Delta$`<!-- -->`{=html}0.176) 은 같은 맥락의 보완 정보를 제공한다.
질문 entity 와 답변 entity 의 corpus 동시 등장은 LLM 이 "질문에서
답변으로 이어지는 경로" 를 사전학습에서 직접 학습했을 가능성을 근사하며,
이것이 토큰 단위 자신감을 반영하는 NLL 과 가장 직접 연결된다. Semantic
Entropy 와 Semantic Energy 는 답변 사이 일관성을 보는 다른 차원의
신호이므로, 두 신호군이 서로 보완적인 정보를 포착한다고 해석할 수 있다.

##### 효과 크기의 정밀도.

§[4.5](#sec:axis){reference-type="ref" reference="sec:axis"} 에서 보고한
Semantic Entropy AUROC range 차이의 95% 신뢰구간 \[+0.002, +0.117\] 은 0
을 포함하지 않으나, 하한이 +0.002 로 경계에 가까우며 신뢰구간 폭 (0.115)
이 점추정치 (+0.063) 의 약 1.8배에 해당한다. 즉 "entity co-occurrence
AUROC range 가 entity 빈도 AUROC range 보다 크다" 는 방향성은 본
표본에서 통계적으로 유의하지만, 효과 크기 자체의 정밀도는 낮다. 약
1.88배라는 점추정치가 모집단 비율과 일치한다고 보지 않으며, 다음 세 가지
점을 함께 보고하였다. 첫째, SVAMP 포함 여부에 따라 비율은 1.68 에서 1.88
사이에서 변동한다. 둘째, 질문 entity 만으로 산출한 강건성 검증에서는
비율이 약 2.20배까지 증가한다. 셋째, 데이터셋별로는 NQ-Open 처럼 부호가
역전되는 경우가 관찰된다
(표 [\[tab:per_dataset_delta\]](#tab:per_dataset_delta){reference-type="ref"
reference="tab:per_dataset_delta"}). 따라서 정량적 주장은 "방향성이 본
표본에서 유의하며, 비율의 점추정치는 약 1.5--1.9배 범위에 있다" 로
한정한다. 보다 정밀한 효과 크기 추정은 추가 모델과 corpus 위에서의 추가
표본을 통해 가능하다.

## 한계 {#sec:limitations}

##### Proxy corpus index 의 사용.

본 논문은 corpus 신호 산출에 Infini-gram 색인 `v4_dolmasample_olmo`
(OLMo 의 16B Dolma sample) 을 사용하였다. 이는 Qwen2.5-3B 의 실제
사전학습 corpus 가 아니며, Qwen 의 사전학습 데이터는 공개되지 않았다.
따라서 본 논문이 측정한 "corpus 뒷받침" 은 모델이 실제로 학습한 데이터의
양이 아니라 투명하고 포괄적인 web-scale 색인 위에서 측정한 사실 등장
빈도이다.

이러한 proxy 사용에는 정당화 근거가 있다. Web-scale 사전학습 corpus
(Dolma, RedPajama, C4 등) 는 Common Crawl 기반의 공통 source 를 상당
부분 공유하므로, 한 transparent corpus 에서 산출한 통계가 다른 LLM 의
학습 노출에 대한 합리적 proxy 역할을 할 수 있다는 가정은 선행 연구에서도
채택되어 왔다 (예: Qiu 등 2025[@qucorag] 의 QuCo-RAG). 그러나 이 proxy
대체는 형식적으로 보장되지 않으며, 본 논문 결과의 해석에 두 가지 제약을
가한다. 첫째, corpus 뒷받침이 큰 sample 에서 Semantic Entropy AUROC 가
높게 관찰되는 결과는 "모델이 학습해서 잘 안다" 라는 인과 진술이 아니라,
"Dolma 와 같은 web 분포에서 자주 등장하는 사실은 Qwen 학습에서도 자주
등장했을 가능성이 높다" 라는 간접 추론을 통해서만 해석된다. 둘째, entity
co-occurrence 가 entity 빈도보다 약 1.88배 큰 분해를 보인다는 비율 또한
Dolma 색인에 한정된 관찰이며, Qwen 의 실제 corpus 나 다른 모델의 corpus
에서는 두 단위의 비율이 달라질 수 있다. 따라서 본 결과는 corpus 빈도와
환각 사이의 상관 관찰에 한정되며, 모델 학습 가능성에 대한 인과 진술로
해석해서는 안 된다. 이 한계는 모델 제공자가 사전학습 corpus 를 공개하기
전까지 본 연구 설계에 내재하며, 후속 연구가 corpus 색인을 모델별 실제
학습 데이터로 교체하기 전에는 수치 자체의 외부 타당성에 분명한 제약이
있다.

이외에도 다음과 같은 한계가 있다.

1.  **단일 모델 평가.** 본 연구는 Qwen2.5-3B 단일 모델에서 수행되었다.
    corpus 신호의 구간별 변동 패턴이 Llama, Mistral, Gemma 계열
    모델에서도 같은 방향으로 나타나는지는 직접 검증되지 않았다. Farquhar
    등 (2024)[@farquhar2024] 에서 서로 다른 LLM 계열 사이에서도 Semantic
    Entropy 의 패턴이 유사하게 관찰되었다는 점은 일반화 가능성을
    시사하나, 이는 향후 연구로 남긴다.

2.  **NLI 라벨링 한계.** 정답 라벨은 `microsoft/deberta-large-mnli` 기반
    양방향 entailment $\geq$ 0.5 기준에 의존한다. LLM-as-judge 와 같은
    다른 라벨링 방식과의 비교 검증은 향후 과제로 남는다.

3.  **라벨과 신호의 부분 결합.** 정답 라벨과 Semantic Entropy / Semantic
    Energy 신호는 모두 동일한 N=10 개 생성 답변에서 산출된다. 라벨은 각
    답변과 정답 후보의 NLI 매칭으로 결정되고, Semantic Entropy /
    Semantic Energy 는 같은 답변들의 의미 cluster 분포와 토큰 logit 에서
    산출되므로, 두 양은 동일 답변 집합의 분산을 부분적으로 공유한다.
    결과적으로 본 논문의 AUROC 는 모델 출력의 정답 여부와 모델 출력 사이
    일관성이라는 동일 source 상의 결합도를 측정하며, 모델 출력과 외부
    정답 텍스트 사이 일치도에 대한 독립 측정은 아니다. 평가 단위 (생성
    답변 단위) 가 Farquhar 와 Ma 의 연구와 일치하므로 선행 연구 비교의
    전제는 유지되며, 본 결과는 동일 라벨링 절차 위에서의 상대적 비교로
    해석한다.

4.  **Self-conditioning 회피.** corpus 통계량은 fusion 입력 변수와 분해
    분석 신호 두 용도로 사용된다. 분해 분석은 모델 출력과 무관한 외부
    corpus 통계 위에서 수행되므로 self-conditioning 문제는 발생하지
    않는다.

5.  **AUROC range 차이의 표본 변동성.** 약 1.88배라는 비율은 단일
    조건에서 산출한 점추정치이며, 두 AUROC range 의 차이에 대한 sample
    단위 bootstrap 95% 신뢰구간을 §[4.5](#sec:axis){reference-type="ref"
    reference="sec:axis"} 에 함께 보고하였다 (Semantic Entropy \[+0.002,
    +0.117\], Energy \[+0.004, +0.109\]). 두 신뢰구간 모두 0 을 포함하지
    않으므로, entity co-occurrence AUROC range 가 entity 빈도 AUROC
    range 보다 크다는 결과는 본 표본에서 통계적으로 유의하다. 다만 이
    신뢰구간은 단일 모델, 단일 corpus 색인 위에서의 표본 변동성만을
    통제하며, 모델과 corpus 일반화에는 별도 검증이 필요하다. 또한
    max$-$min 통계량 자체가 순서통계량이므로 구간 수가 다른 분할에서는
    비율의 절대 크기가 달라질 수 있다.

6.  **Fusion lift 의 통계적 유의성.** corpus 신호 포함 여부에 따른
    gradient boosting 의 lift 에 대한 sample 단위 bootstrap 95% 신뢰구간
    \[+0.005, +0.011\] 은 0 을 포함하지 않으며 모든 반복에서 양수이다.
    따라서 corpus 신호의 fusion 기여는 통계적으로 0 과 구분되는 양의
    효과이지만, 절대 크기 (+0.008) 는 답변 단위·sample 단위 신호 결합
    효과 (+0.026) 의 약 3분의 1 정도에 한정된다.

7.  **데이터셋별 이질성.**
    표 [\[tab:per_dataset_delta\]](#tab:per_dataset_delta){reference-type="ref"
    reference="tab:per_dataset_delta"} 에서 보이듯, entity co-occurrence
    AUROC range 가 entity 빈도 AUROC range 보다 크다는 패턴은 다섯
    데이터셋 중 NQ-Open 에서 역전된다. 따라서 표본 종합 비율 1.88배는
    모든 데이터셋에서 entity co-occurrence 신호가 entity 빈도 신호보다
    큰 AUROC range 를 보인다는 뜻이 아니라 데이터셋 종합 통계이다.
    도메인별로 corpus 신호의 적절한 단위가 달라질 수 있다는 가설은 본
    논문의 범위 밖이다.

8.  **Corpus 신호의 entity 추출 범위.** 본 논문의 corpus 신호는 질문
    텍스트와 데이터셋이 제공한 정답 텍스트의 entity 를 합쳐 산출하며,
    모델이 생성한 답변의 entity 는 사용되지 않는다. 따라서 신호 값은
    모델 출력과 독립이며, 한계 4 에서 다룬 답변 결합 문제와는 차원이
    다르다. 답변 측 entity 의 영향을 분리하여 질문 entity 만으로 신호를
    다시 산출한 결과는 §[4.5](#sec:axis){reference-type="ref"
    reference="sec:axis"} 에 함께 보고하였으며, Semantic Entropy 기준
    비율은 1.88 에서 2.20 으로 증가한다. 다만 부분표본 (n=2,095) 의 선택
    편향은 §[4.5](#sec:axis){reference-type="ref" reference="sec:axis"}
    에 명시하였다.

9.  **AUROC range 와 Spearman $\rho$ 의 분리 해석.** AUROC range 는
    max$-$min 통계량이므로, 구간이 올라갈수록 AUROC 가 단조 증가하는
    양상과 비단조 변동 (U자형 등) 을 구분하지 못한다. 본 논문은
    표 [\[tab:axis_decomp\]](#tab:axis_decomp){reference-type="ref"
    reference="tab:axis_decomp"} 에 Semantic Entropy / Energy / 답변
    평균 NLL 세 신호 모두에 대해 Spearman $\rho$ 를 함께 보고함으로써
    AUROC range 와 단조성을 분리해 진단한다. 다만 21개 동시 검정에 대한
    다중비교 보정 (Bonferroni
    $\alpha$=0.05/21$\approx$`<!-- -->`{=html}0.0024) 을 적용하면 표의
    모든 $\rho$ 가 유의 기준을 충족하지 않으므로, $\rho$ 결과는 예비
    분석으로 한정한다. 또한 n=10 (구간 수) 의 Spearman 검정력이 낮아
    $|\rho|$=0.65 부근의 결과는 구간 폭과 동순위 처리에 민감하다. 보다
    강건한 단조성 검정 (Mann-Kendall, partial $\rho$) 과 다중비교 보정은
    향후 과제로 남긴다.

10. **NLI 임계값 0.5 의 민감도 미검증.** 본 논문은 정답 라벨을 양방향
    entailment 확률 0.5 임계로 정의하였다. 임계값을 0.4 또는 0.6 으로
    변경했을 때 AUROC range 와 비율이 어떻게 달라지는지에 대한 민감도
    분석은 수행하지 않았으며, 본 결과가 임계 0.5 에 어느 정도
    의존하는지는 후속 검증이 필요하다.

11. **Fusion 입력 변수의 기여도 분해 미수행.** fusion 모델의 +0.026
    향상이 SE / Energy / 답변 단위 logit 통계 가운데 어느 신호의
    결합에서 비롯되었는지를 분해하는 분석 (예: permutation importance,
    SHAP) 은 수행하지 않았다. 본 논문은 fusion 결과를 종합 AUROC
    수준에서만 보고하며, 입력 변수별 기여도 분리는 후속 과제로 남긴다.

12. **NQ-Open 의 부분 구간 outlier 원인 미규명.** NQ-Open 의 두 번째
    구간 (20--30) 에서 AUROC 가 0.266 까지 떨어지는 outlier 는 정답률
    0.058 (표본 120개 중 정답 7개) 의 클래스 불균형에서 일부 비롯된
    것으로 보이지만, 이러한 분포 skew 의 근본 원인 (질문 유형, entity
    분포 등) 에 대한 추가 분석은 수행하지 않았다.

13. **단순 prompt 설정.** 본 연구는 추가 context 없는 한 문장 응답
    prompt 위에서 수행되었다. 실무 LLM 서비스에서 표준이 된 RAG 환경이나
    multi-turn agentic 설정에서의 일반화는 검증 대상이다. 본 결과를
    retrieval-augmented 환경에 그대로 적용하기 전에 별도 검증이
    필요하다.

## 향후 연구

본 연구의 결과를 토대로 다음 방향의 후속 연구를 제안한다.

1.  **모델 다양화.** Llama, Gemma, Mistral, instruction-tuned 모델 등
    다양한 모델에서 entity co-occurrence 신호의 AUROC range 패턴이 같은
    방향으로 나타나는지 검증한다.

2.  **실제 사전학습 corpus 사용.** 본 연구가 사용한 OLMo Dolma sample
    proxy 색인을 Qwen2.5 의 실제 사전학습 corpus 또는 모델별 학습
    데이터에 가까운 색인으로 교체하여 proxy corpus 의 한계를 극복한다.

3.  **도메인 특화 corpus.** BioASQ 의 PubMed, 수학 문제의 ArXiv 등
    도메인 특화 색인을 사용했을 때 동일 신호의 AUROC range 가 더
    커지는지 확인한다.

4.  **LLM-as-judge 라벨과의 비교.** NLI 매칭 라벨을 LLM-as-judge 라벨로
    대체했을 때 corpus 신호 단위별 AUROC range 가 어떻게 달라지는지
    분석한다.

5.  **RAG 및 agentic 환경 확장.** retrieval 없는 단순 prompt 설정을 RAG
    환경이나 multi-turn agentic 설정으로 확장하여, corpus 뒷받침 정도와
    retrieval 결과가 환각 탐지 신호의 신뢰도를 어떻게 함께 결정하는지를
    검증한다.

6.  **환각 원인 유형별 분해.** 본 연구는 사실 회상 (Type A) 영역을 주로
    다루었다. 모델 내부 잘못된 지식 (Type B) 과 순수 조작 (Type C) 에서
    corpus 신호 단위별 AUROC range 가 어떻게 달라지는지에 대한 별도
    분석이 필요하다.

7.  **정교한 corpus 신호 설계와 표준 벤치마크.** entity co-occurrence
    외에 syntactic pattern, semantic similarity 기반 co-occurrence,
    temporal decay 를 고려한 최근성 가중치 등 정교한 corpus 신호를
    설계하고, 데이터셋과 도메인별 최적 단위를 자동 선택하는 meta-learner
    를 연구할 가치가 있다. 동시에 AUROC range 를 표준 평가 지표로
    포함하는 새로운 환각 탐지 벤치마크의 제안도 자연스러운 확장이다.

## 결론

본 논문은 LLM 환각 탐지 신호 (Semantic Entropy, Semantic Energy, 답변
단위 token logit 통계) 의 성능이 사전학습 corpus 의 사실 등장 패턴에
따라 크게 달라진다는 점을 정량적으로 입증하였다. Qwen2.5-3B 와 다섯 개
질의응답 데이터셋 (총 35,000 개 생성 답변) 위에서 분석한 결과, entity
co-occurrence 신호로 sample 을 나누었을 때 Semantic Entropy 의 구간별
AUROC range 는 $\Delta$`<!-- -->`{=html}0.150 로, 단일 entity 빈도
신호의 AUROC range $\Delta$`<!-- -->`{=html}0.080 의 약 1.5--1.9배에
해당한다 (SVAMP 포함 시 1.88배, 제외 시 1.68배). 두 AUROC range 차이에
대한 95% 신뢰구간 \[+0.002, +0.117\] 은 본 표본에서 방향성이 통계적으로
유의함을 보여주지만, 효과 크기의 정밀도는 낮으며 본 결과는 corpus 빈도와
환각 사이의 상관 관계 관찰로 해석한다 (현 색인이 모델의 실제 사전학습
corpus 와 다르므로 인과 해석은 불가).

가장 중요한 발견은 비대칭이다. 동일 corpus 신호가 fusion 모델 입력
변수로는 +0.008 의 작은 향상에 한정되는 반면, 조건부 분해 신호로
사용했을 때는 큰 AUROC range 를 보인다. 이 비대칭은 corpus 신호의 설계
목표를 환각 탐지기의 입력 변수가 아니라 탐지 신호의 조건부 평가 도구로
재정의할 필요를 제기하며, 평균 AUROC 하나만 보고하는 기존 관행을 넘어
corpus 조건에 따른 AUROC range 를 함께 보고하는 평가 형식을 제안한다.
동시에 실제 LLM 서비스에서 corpus 뒷받침이 약한 질문을 별도로 검증하는
조건부 신뢰 전략의 이론적 근거를 제공한다.

본 연구는 학사 논문으로서 단일 모델, proxy corpus, 표본 규모 등의 분명한
한계를 갖는다. 그럼에도 환각 탐지 연구에 corpus 중심의 조건부 분석이라는
새로운 관점을 제시했다는 점에서 의의가 있다. 더 다양한 모델과 실제
corpus 위에서의 후속 연구를 통해 본 관점이 한층 견고해지기를 기대한다.

::: thebibliography
99

L. Huang et al., "A Survey on Hallucination in Large Language Models:
Principles, Taxonomy, Challenges, and Open Questions," *ACM Transactions
on Information Systems*, vol. 43, no. 2, pp. 1--55, 2025.

J. Maynez et al., "On Faithfulness and Factuality in Abstractive
Summarization," *Proceedings of ACL*, pp. 1906--1919, 2020.

S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, "Detecting hallucinations
in large language models using semantic entropy," *Nature*, vol. 630,
pp. 625--630, 2024.

P. Manakul, A. Liusie, and M. J. F. Gales, "SelfCheckGPT: Zero-Resource
Black-Box Hallucination Detection for Generative Large Language Models,"
*Proceedings of EMNLP*, pp. 9004--9017, 2023.

J. Kossen et al., "Semantic Entropy Probes: Robust and Cheap
Hallucination Detection in LLMs," *arXiv preprint arXiv:2406.15927*,
2024.

A. Nikitin, J. Kossen, Y. Gal, and P. Marttinen, "Kernel Language
Entropy: Fine-grained Uncertainty Quantification for LLMs," *arXiv
preprint arXiv:2405.20003*, 2024.

K. Ciosek et al., "Bayesian Estimation of Semantic Entropy for
Sample-Efficient Hallucination Detection," *arXiv preprint
arXiv:2503.xxxxx*, 2025.

Z. Ma et al., "Semantic Energy: A novel approach for detecting
confabulation in language models," *arXiv preprint arXiv:2412.07965*,
2025.

A. Raghuvanshi et al., "Hybrid Token-Level Uncertainty and Semantic
Entailment for Hallucination Detection," *arXiv preprint
arXiv:2502.xxxxx*, 2025.

A. Ravichander et al., "HALoGEN: Fantastic LLM Hallucinations and Where
to Find Them," *arXiv preprint arXiv:2501.08292*, 2025.

S. Singha, "ECLIPSE: Evidence-Conditioned Hallucination Detection via
Perplexity Decomposition for Financial QA," *arXiv preprint
arXiv:2510.xxxxx*, 2025.

H. Wang, "SEReDeEP: Semantic Entropy + Retrieval-Aware Detection for
Enhanced Evaluation Pipelines," *arXiv preprint arXiv:2509.xxxxx*, 2025.

S. Valentin et al., "Cost-Effective Hallucination Detection for LLMs,"
*arXiv preprint arXiv:2407.21424*, 2024.

W. Zhao et al., "WildHallucinations: Evaluating Long-form Factuality in
LLMs with Real-World Entity Queries," *arXiv preprint arXiv:2407.17468*,
2024.

Y. Zhang et al., "Measuring the Impact of Lexical Training Data Coverage
on Hallucination Detection in Large Language Models," *arXiv preprint
arXiv:2511.17946*, 2025.

Z. Qiu et al., "QuCo-RAG: Query-aware Corpus Grounding for
Retrieval-Augmented Generation," *arXiv preprint arXiv:2512.19134*,
2025.

J. Liu et al., "Infini-gram: Scaling Unbounded n-gram Language Models to
a Trillion Tokens," *arXiv preprint arXiv:2401.17377*, 2024.

A. Yang et al., "Qwen2.5 Technical Report," *arXiv preprint
arXiv:2412.15115*, 2024.

M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, "TriviaQA: A Large
Scale Distantly Supervised Challenge Dataset for Reading Comprehension,"
*Proceedings of ACL*, pp. 1601--1611, 2017.

P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, "SQuAD: 100,000+
Questions for Machine Comprehension of Text," *Proceedings of EMNLP*,
pp. 2383--2392, 2016.

G. Tsatsaronis et al., "An overview of the BIOASQ large-scale biomedical
semantic indexing and question answering competition," *BMC
Bioinformatics*, vol. 16, no. 138, 2015.

T. Kwiatkowski et al., "Natural Questions: A Benchmark for Question
Answering Research," *Transactions of the ACL*, vol. 7, pp. 453--466,
2019.

A. Patel, S. Bhattamishra, and N. Goyal, "Are NLP Models really able to
Solve Simple Math Word Problems?," *Proceedings of NAACL*, pp.
2080--2094, 2021.
:::
