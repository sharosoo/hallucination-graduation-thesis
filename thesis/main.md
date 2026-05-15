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
훨씬 더 직접적으로 연결될 것이다. 이러한 직관은 Kandpal 등
(2023)[@kandpal2023longtail] 이 사전학습 corpus 에서 (질문 entity, 정답
후보 entity) 쌍의 동시 등장 횟수가 단일 entity 빈도보다 LLM 의 사실
정답률을 더 잘 예측한다는 것을 정량적으로 보인 결과와 직접 맞닿는다.
그렇다면 환각 탐지 신호의 신뢰도 역시 단일 entity 빈도가 아니라 entity
쌍의 동시 등장 빈도에 따라 더 뚜렷하게 갈릴 것이다.

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

- \(2\) Kandpal 등 (2023)[@kandpal2023longtail] 이 모델 정답률에 대해
  보고한 entity co-occurrence 의 우월성이 환각 탐지 신호의 신뢰도 영역
  분해에서도 같은 방향으로 나타남을 본 표본에서 확인한다 (entity
  co-occurrence 가 단일 entity 빈도보다 환각 탐지 신호의 신뢰도 변동을
  약 1.5\~1.9배 더 명확하게 구분; 단일 조건 점추정치 1.88배, 95%
  신뢰구간 하한 +0.002, SVAMP 제외 시 1.68배). 핵심은 측정 대상을 *모델
  정답률* 에서 *환각 탐지 신호의 AUROC* 로 한 단계 더 높은 수준으로
  옮겼음에도 같은 방향성이 유지된다는 점이다.

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

Semantic Entropy 의 개념적 기원은 Kuhn 등 (2023)[@kuhn2023semantic] 의
*Semantic Uncertainty* 논문에 있다 (의미적 등가의 linguistic invariance
를 명시적으로 다룬 entropy 정의). Farquhar 등 (2024)[@farquhar2024] 은
이를 hallucination detection 으로 구체화한 후속 연구이다.

Farquhar 등 (2024)[@farquhar2024] 이 제안한 Semantic Entropy 는 동일
prompt 에 대해 생성된 N 개의 답변을 NLI 양방향 함의 관계로 의미 cluster
로 묶은 뒤, cluster 질량의 엔트로피를 환각 지표로 사용한다. 이 방법은
태스크별 사전 지식 없이도 TriviaQA, SQuAD, BioASQ, NQ-Open, SVAMP 등에서
일관된 성능을 보이며, naive entropy, $p(\text{True})$, embedding 회귀
기준선 대비 안정적으로 높은 AUROC 를 달성한다.
SelfCheckGPT[@selfcheckgpt2023] 가 제시한 sample-consistency 접근을 의미
단위로 일반화한 결과로 평가된다.

Semantic Entropy 의 확장 연구로는 SEPs[@kossen2024seprobes],
KLE[@nikitin2024kle], Bayesian SE[@ciosek2025bayes] 등이 있으며, 각각
추론 비용 감소 (단일 hidden state 로부터의 probe 근사), NLI hard
clustering 의존성 완화 (pairwise 커널 기반 von Neumann 엔트로피), 표본
효율 개선 (Bayesian 추정) 을 시도한다.

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

## Corpus 통계를 이용한 환각 탐지

LLM 의 환각은 사전학습 corpus 에서 드물게 등장하는 사실과 강하게
연관된다. HALoGEN (Ravichander 등, 2025)[@ravichander2025halogen] 은 9
개 도메인에 걸친 10,923 개 항목의 환각 벤치마크를 구축하고, LLM 생성물의
원자적 사실 단위를 고품질 지식 소스와 자동 검증하는 프레임워크를
제시했다. 이 연구는 최고 성능 모델에서도 생성된 원자적 사실의 최대 86%
가 환각으로 분류될 수 있음을 보여주며, 환각 원인을 학습 데이터 오류 회상
(Type A), 학습 데이터의 잘못된 지식 (Type B), 순수 조작 (Type C) 으로
분류했다.

corpus 노출의 어떤 측도가 모델의 사실 회상을 가장 잘 예측하는가에 대한
정량 연구로는 Kandpal 등 (2023)[@kandpal2023longtail] 과 Mallen 등
(2023)[@mallen2023whennot] 이 본 연구의 직접적인 선행 연구이다. Kandpal
등 (2023) 은 The Pile 과 ROOTS 사전학습 corpus 에서 (질문 entity, 정답
후보 entity) 쌍의 동시 등장 횟수를 산출하고, 이 횟수와 LLM 의 정답률
사이에 강한 양의 상관이 있음을 보였다. 특히 단일 entity 빈도보다 entity
쌍 동시 등장이 정답률을 더 잘 예측한다는 점을 명시적으로 보고했다.
Mallen 등 (2023) 은 정답 후보 entity 의 Wikipedia 페이지뷰 (popularity)
를 corpus 노출의 proxy 로 삼아 PopQA 데이터셋의 sample 을 popularity bin
으로 나누고, popularity 가 낮은 bin 에서 LLM 정답률이 급격히 떨어지는
패턴을 보였다. Razeghi 등 (2022)[@razeghi2022termfreq] 은 few-shot
numerical reasoning 에서 동일 패턴 (corpus 노출-성능 양의 관계) 이
reasoning 전반에 걸쳐 나타남을 뒷받침한다. 본 연구는 측정 대상을
정답률에서 환각 탐지 신호의 AUROC 로 한 단계 끌어올려 분석한다.

WildHallucinations[@wildhallucinations2024] 는 Wikipedia 페이지가 없는
entity 에 대한 질문에서 LLM 의 환각률이 유의미하게 증가한다는 점을
관찰하여, corpus 노출 부족과 환각 발생률 사이의 경험적 연관성을
보고했다. Zhang 등 (2025)[@zhang2025corpus] 은 RedPajama 1.3 조 토큰
corpus 위에 suffix array 를 구축하고, prompt 와 답변의 n-gram 빈도를
환각 탐지 신호로 평가했다. 그들은 occurrence 기반 feature 가 단독 사용
시 예측력이 제한적이며, log-probability 와 결합할 때 modest 한 성능
향상을 보인다고 보고했다. 본 연구는 이 결론에서 한 걸음 더 나아가,
동일한 corpus 신호를 fusion 입력 feature 로 사용할 때와 sample 을
분할하는 기준으로 사용할 때의 효과를 명시적으로 분리하여 측정한다. 또한
Zhang 등 (2025) 이 다루지 않은 corpus 신호의 단위 비교 (entity 빈도
vs. entity co-occurrence vs. n-gram) 를 동일한 AUROC range 척도 위에서
수행함으로써, corpus 신호 단위의 상대적 효용을 정량적으로 규명한다.

Min 등 (2025)[@qucorag] 의 QuCo-RAG 는 사전학습 corpus 의 low-frequency
entity 와 entity co-occurrence 를 Infini-gram 으로 검증하여 RAG
retrieval trigger 로 활용한다. 본 연구와 동일한 두 corpus 신호 (entity
빈도, entity co-occurrence) 와 동일한 도구 (Infini-gram) 를 사용하지만,
목적이 retrieval trigger 라는 점에서 다르다. QuCo-RAG 는 RAG 환경에서
uncertainty quantification 을 corpus-grounded 로 전환하는 데 초점을
맞추는 반면, 본 연구는 환각 탐지 신호의 조건부 신뢰도를 corpus 단위별로
분해하는 데 목적을 둔다. 또한 QuCo-RAG 는 단위 비교 없이 두 신호를 함께
사용하지만, 본 연구는 일곱 가지 단위를 같은 평가 척도 위에서 비교한다.

Infini-gram[@infinigram2024] 은 n-gram 카운팅을 통해 corpus 통계 질의를
지원하는 엔진으로, 특정 phrase 나 entity 의 등장 빈도를 효율적으로
산출한다. 그러나 이를 환각 탐지의 corpus 지지 신호로 활용하는 연구는
아직 초기 단계이며, 단위 (entity 빈도, entity co-occurrence, n-gram)
별로 corpus 통계 신호가 탐지 신호와 어떻게 연관되는지를 비교한 연구는
충분하지 않다.

## 다중 신호 결합 (Multi-Signal Fusion)

단일 신호의 한계를 극복하기 위해 여러 신호를 결합하는 fusion 접근이
연구되고 있다. ECLIPSE[@singha2025eclipse] 는 perplexity 분해 기반 증거
활용 신호로 금융 QA 에서, SEReDeEP[@wang2025seredeep] 는 semantic
entropy 와 context-parameter knowledge 균형 신호 결합으로 RAG 환경에서
각각 향상을 보고했다 (두 연구 모두 peer review 미완료 preprint).
Valentin 등 (2024)[@valentin2024] 은 환각 탐지 점수를 모델 내부 score
attribute 에 조건부로 calibrate 하는 다중 점수 프레임워크를 제안했다.

이들 연구는 여러 신호의 결합이 단일 신호 대비 성능 향상으로 이어질 수
있음을 보여주나, 공통적으로 corpus 통계를 탐지기의 입력 변수로만 사용할
뿐, 동일 신호가 탐지 신호의 영역별 신뢰도를 분해하는 평가 척도로서 어떤
효용을 가지는지는 비교하지 않았다. 본 연구는 이 두 역할을 명확히
분리하여 정량화하며, 두 결과의 비대칭이 corpus 신호의 설계 목표 자체를
재정립할 필요성을 제기한다.

선행 연구와의 차이는 다음과 같다. Kandpal 등
(2023)[@kandpal2023longtail] 과 Mallen 등 (2023)[@mallen2023whennot] 은
corpus 노출 / popularity 와 *모델 정답률* 의 관계를 보였다. 본 연구는
같은 corpus 신호를 사용하되, 측정 대상을 *환각 탐지 신호의 AUROC* 로 한
단계 끌어올렸다. 정답률과 AUROC 는 자동으로 비례하지 않는다 (정답률이
다른 영역에서도 AUROC 가 동일할 수 있고 그 역도 성립). 또한 동일 corpus
신호의 두 역할 --- 탐지기 입력 변수와 영역별 평가 척도 --- 사이의
비대칭은 두 선행 연구에 정의 자체가 없는 분석 축이며, 일곱 가지 corpus
신호 단위 (entity 빈도, entity co-occurrence, baseline, question-answer
bridge, 답변 3-gram / 5-gram, 3-gram 미등장) 의 비교도 본 연구의 범위에
속한다.

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
3,500개 sample 을 선정하였다. 각 데이터셋마다 시드를 고정하여 sample
집합을 재현하였으며, 각 sample 에는 데이터셋 이름·split·원본 색인·본문
해시를 결합한 고유 식별자를 부여하였다.

사용한 모델은 Qwen2.5-3B (base)[@qwen25] 이며, instruction tuning 은
적용하지 않았다. 추론은 NVIDIA RTX 5090 32 GB 환경, float16 으로
수행하였다. 입력 prompt 는 Farquhar 등 (2024) 와 동일하게 "Answer the
following question in a single brief but complete sentence." 형식을
따랐고, 각 sample 당 10개의 자유 생성 답변을 산출하였다. 샘플링
매개변수는 온도 1.0, top-p 0.9, top-k 50, 최대 생성 토큰 수 64이며, 총
평가 단위는 35,000개의 생성 답변이다. 최대 토큰 수 64에서 약 12.6% 가
중간에 절단되어, 절단이 발생한 2,426개 sample 의 답변을 최대 토큰 수
128로 재생성하였으며, 재생성 전후 AUROC 변화는
$\pm$`<!-- -->`{=html}0.001 범위에 머물렀다.

평가 지표로는 AUROC, AURAC (Area Under Rejection-Accuracy Curve,
Farquhar 등 (2024) 의 주요 지표), Brier score, ECE 를 사용한다. Fusion
교차검증은 prompt 단위 5-fold GroupKFold 로, 같은 sample 의 모든 답변을
동일 fold 에 배치하여 sample 단위 정보 누출을 방지한다.

## 답변 단위 정답 라벨 {#sec:label}

정답 라벨은 모델이 생성한 답변 $s_i$ 가 정답 후보 표현과 의미적으로
일치하는지를 NLI 양방향 함의 확률로 판정하여 정의한다. 정답 후보 집합
$C$ 는 데이터셋이 제공한 정답·동의·별칭 표현을 모은 뒤 소문자 정규화와
중복 제거로 구성한다. $$\begin{align*}
m(s_i, c) &= \max\bigl(p_\text{entail}(c \to s_i),\ p_\text{entail}(s_i \to c)\bigr), \quad M(s_i) = \max_{c \in C} m(s_i, c).
\end{align*}$$ $M(s_i) \geq 0.5$ 이면 $s_i$ 를 정답 (라벨 1), 그렇지
않으면 환각 (라벨 0) 으로 판정한다. NLI 모델로는
`microsoft/deberta-large-mnli` 를 사용하며, 절차는 Farquhar 등 (2024) 의
라벨링과 동일하다. 총 35,000개의 이진 라벨이 산출되어, Farquhar 등
(2024) 와 Ma 등 (2025) 의 답변 단위 정답성 평가와 직접 비교가 가능하다.

## Semantic Entropy {#sec:se_def}

Semantic Entropy 는 Farquhar 등 (2024) 의 정의를 그대로 따른다. 한
sample 의 10개 답변을 NLI 양방향 함의로 의미 cluster $\mathbb{C}_k$ 로
묶고, cluster 확률 질량을 시퀀스 로그우도로부터 계산한다.
$$\begin{align*}
\log \tilde{p}(\mathbb{C}_k) &= \log\!\sum_{x^{(i)} \in \mathbb{C}_k} \exp\!\Bigl(\sum_t \log p_\theta(x_t^{(i)})\Bigr), \quad p(\mathbb{C}_k) = \frac{\tilde{p}(\mathbb{C}_k)}{\sum_{k'} \tilde{p}(\mathbb{C}_{k'})}.
\end{align*}$$ Semantic Entropy 는 cluster 분포의 Shannon 엔트로피
$H(\mathbf{x}) = -\sum_{k} p(\mathbb{C}_k)\, \log p(\mathbb{C}_k)$ 로
정의되며, sample 단위로 산출되어 같은 sample 의 모든 답변에 동일하게
적용된다.

## Semantic Energy {#sec:energy_def}

Semantic Energy 는 Ma 등 (2025) 의 정의를 그대로 구현한다. Semantic
Entropy 와 동일한 cluster 구조 위에서 확률 대신 token logit 을 직접
활용하여 모델 지식 한계에서 비롯되는 불확실성을 추정한다.
$$\begin{align}
\tilde{E}(x_t^{(i)}) &= -z_\theta(x_t^{(i)}), \quad E(x^{(i)}) = \tfrac{1}{T_i}\sum_{t} \tilde{E}(x_t^{(i)}), \\
E_{\text{Bolt}}(\mathbb{C}_k) &= \sum_{x^{(i)} \in \mathbb{C}_k} E(x^{(i)}), \quad U(\mathbf{x}) = \sum_{k} p(\mathbb{C}_k)\, E_{\text{Bolt}}(\mathbb{C}_k).
\end{align}$$ 값이 낮을수록 모델 신뢰도가 높다.

## 각 답변의 token logit 통계 {#sec:diag_def}

각 답변에 대해 다음 네 가지 token logit 통계를 산출하여 fusion 입력으로
사용한다: 평균 음의 로그우도 $-\tfrac{1}{T_i}\sum_t \log p(x_t^{(i)})$,
시퀀스 로그우도 $\sum_t \log p(x_t^{(i)})$, logit 분산
$\mathrm{Var}_t(z_\theta(x_t^{(i)}))$, 평균 로그 분배함수
$\tfrac{1}{T_i}\sum_t \log Z_t^{(i)}$ (분포 평탄도 지표).

## Corpus 신호 설계 {#sec:corpus_def}

Corpus 신호는 Infini-gram[@infinigram2024] 의 local engine 을 backend 로
사용하며, 색인은 OLMo Dolma sample (16B token, OLMo-7B-hf 토크나이저)
이다.

##### Entity 수준 신호.

spaCy `en_core_web_lg` 의 NER 로 12개 범주 (인물, 조직, 지명, 위치,
날짜, 사건, 작품, 시설, 국적/종교, 제품, 언어, 법) 의 entity 를
추출하고, 소문자 정규화·정관사 (the/a/an 등) 제거·중복 제거를 거쳐
entity 집합을 구성한다. 한 sample 에 대해 다음 두 점수를 계산한다.
$$\begin{align*}
\text{entity 빈도} &= \frac{\log(1 + \min_{e \in E} \mathrm{freq}(e))}{\log(1 + 10^6)}, \quad \text{entity co-occurrence} = \frac{\log\!\bigl(1 + \tfrac{1}{|P|}\sum_{(e_i,e_j) \in P}\mathrm{cooc}(e_i,e_j)\bigr)}{\log(1 + 10^5)}.
\end{align*}$$ $\mathrm{freq}(e)$ 는 entity $e$ 의 단일 등장 횟수이며,
$\mathrm{cooc}(e_i,e_j)$ 는 Infini-gram 의 `count_cnf`($e_i$ AND $e_j$)
로 두 entity 가 *같은 문서 단위* 안에서 함께 등장한 횟수이다 (sentence /
window 단위가 아니다). 본 연구는 raw count 위에 $\log(1+\cdot)$ 변환과
max-clipping 으로 정규화하며, zero count 는 0으로 유지하여 별도
smoothing 을 적용하지 않는다. baseline corpus 신호는 두 점수의 평균이며,
entity 빈도와 entity co-occurrence 두 단위의 분리 비교가 본 연구의
초점이다.

##### Pair 구성과 question-answer bridge 신호.

entity co-occurrence 신호는 (질문 entity $\cup$ 정답 후보 entity) 의
모든 pairwise 쌍을 사용한다. 반면 question-answer bridge 신호는 질문
entity 집합 $E_q$ 와 정답 후보 entity 중 질문과 겹치지 않는 집합 $E_a$
의 cross-pair $(e_q, e_a)$ 만을 사용한다 (paraphrase 로 인한 자기 중복
제거 목적). 모든 cross-pair 의 corpus 동시 등장 횟수를 산출하고 그
정규화 평균을 question-answer bridge 신호로 정의한다.

##### N-gram 등장 빈도 신호.

답변 토큰 시퀀스의 모든 3-gram 과 5-gram 에 대해 corpus 등장 횟수를
산출한다. 평균 등장 횟수를 정규화한 점수와, corpus 에 한 번도 등장하지
않은 n-gram 의 수를 함께 보고한다.

## Corpus 신호 단위별 분해 분석 {#sec:axis_framework}

본 분석의 핵심은 §[3.6](#sec:corpus_def){reference-type="ref"
reference="sec:corpus_def"} 의 corpus 신호를 환각 탐지의 입력이 아니라
*sample 을 분할하는 신호* 로 사용하는 데 있다. 각 신호로 sample 을 순위
기반 10분위 구간으로 나누고, 분할에 사용한 신호는 entity 수준 세 가지
(entity 빈도, entity co-occurrence, 두 점수 평균), question-answer
bridge 한 가지, n-gram 수준 세 가지 (3-gram·5-gram 평균 등장, 3-gram
미등장 개수) 로 총 일곱 가지이다. 각 신호의 10개 구간 안에서 환각 탐지
신호의 AUROC 를 따로 계산하고, 최댓값과 최솟값의 차이
$\Delta = \mathrm{AUROC}_{\max} - \mathrm{AUROC}_{\min}$ 를 해당 신호의
*구간별 AUROC range* 로 정의한다. AUROC range 가 클수록 해당 corpus
신호로 분할했을 때 탐지 성능이 corpus 조건에 따라 큰 차이를 보인다.

## Fusion 모델 {#sec:fusion_def}

Fusion 모델의 입력은 두 가지 구성으로 실험한다. **CORE 입력** 은 sample
단위 Semantic Entropy / Semantic Energy 와 답변마다 산출한 token logit
통계 4종 (평균 NLL, 시퀀스 로그우도, logit 분산, 평균 로그 분배함수)
이며, **CORE + CORPUS 입력** 은 여기에 entity, question-answer bridge,
n-gram 의 모든 corpus 신호를 sample 단위로 추가한다. 분류 모델은
logistic regression, random forest, gradient boosting 세 가지를
비교하며, 모든 실험은 prompt 단위 5-fold GroupKFold 로 평가한다.

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

Qwen2.5-3B 는 Llama-2 7B 보다 작은 모델임에도 두 신호의 AUROC 가
Farquhar (2024) 의 보고 범위 (TriviaQA 0.79 / SQuAD 0.83 등 0.75--0.85
대) 안에 들어간다. 이는 본 표본이 Semantic Entropy 의 일반적인 작동
범위에 속한다는 점을 보여준다. Semantic Energy 의 경우 Ma (2025) 가
Semantic Entropy 대비 평균 AUROC 13% 이상 개선을 보고하였으나,
모델·데이터셋 구성이 본 실험과 동일하지 않아 직접 비교는 한정적이다.
답변 평균 NLL (0.670) 은 단독 신호로도 일정한 분리력을 보이지만 의미
cluster 정보를 추가했을 때 부가 효과가 크다.

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

주목할 점은 후속 절 (§[4.6](#sec:axis){reference-type="ref"
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

## 비대칭 메커니즘 부분 분석 {#sec:mechanism}

§[4.4](#sec:fusion_results){reference-type="ref"
reference="sec:fusion_results"} 의 fusion lift (+0.008) 와
§[4.6](#sec:axis){reference-type="ref" reference="sec:axis"} 의 구간별
AUROC range ($\Delta 0.150$, 전체 표본 기준이며 데이터셋별 변동 있음 ---
표 [\[tab:per_dataset_delta\]](#tab:per_dataset_delta){reference-type="ref"
reference="tab:per_dataset_delta"}) 사이의 큰 비대칭은 corpus 신호가
환각 탐지 신호의 *선형 예측 변수* 로서가 아니라 *영역별 신뢰도를
조절하는 요인* (effect modifier) 으로 작동할 가능성을 시사한다. 본 절은
세 가지 보조 분석으로 이 가설을 부분 검증한다.

##### 1) Semantic Entropy 와 corpus 신호 사이의 직접 상관.

표 [4.4](#tab:se_corpus_corr){reference-type="ref"
reference="tab:se_corpus_corr"} 는 SE 와 6개 corpus 신호 사이의 Spearman
$\rho$ 를 보고한다. 모두 $|\rho| \leq 0.16$ 으로 두 신호군은 거의
직교한다. 따라서 "corpus 신호의 fusion lift 가 작은 이유가 SE 와 상관이
높아서다" 라는 가설은 본 데이터에서 약하게 부정된다.

::: {#tab:se_corpus_corr}
  Corpus 신호                                             Semantic Entropy $\rho$   Semantic Energy $\rho$
  ------------------------------------------------------ ------------------------- ------------------------
  entity 빈도 (entity_frequency_axis)                            $-0.124$                  $-0.125$
  entity co-occurrence (entity_pair_cooccurrence_axis)           $-0.121$                  $-0.117$
  question-answer bridge (qa_bridge_axis)                        $-0.142$                  $-0.151$
  답변 3-gram 등장 (ans_ngram_3_axis)                            $+0.095$                  $+0.108$
  답변 5-gram 등장 (ans_ngram_5_axis)                            $+0.021$                  $+0.030$
  답변 3-gram 미등장 (ans_ngram_3_zero_count)                    $+0.147$                  $+0.164$

  : Semantic Entropy / Semantic Energy 와 6개 corpus 신호 사이의
  Spearman $\rho$ (n=35,000). $|\rho|$ 가 모두 0.16 이하로 두 신호군은
  거의 직교한다. $p$ 값은 양측 검정.
:::

##### 2) SE × corpus interaction term 의 회귀 계수.

표준화된 입력 ($-$SE, entity_pair_cooccurrence_axis, 두 변수의 곱)
위에서 logistic regression 을 prompt 단위 5-fold GroupKFold OOF 로
학습하여 main effect 모델과 interaction 포함 모델을 비교한다.
interaction 추가는 OOF AUROC 를 +0.0005 만 변동시킨다 (0.7585 $\to$
0.7590). interaction 표준화 계수는 $-0.37$ 로 0 이 아니지만 예측
분리력에는 기여하지 않는다.

##### 3) Corpus 영역별 calibration.

표 [\[tab:per_bin_calibration\]](#tab:per_bin_calibration){reference-type="ref"
reference="tab:per_bin_calibration"} 에서 SE 의 per-bin AUROC 가
0.643--0.793 ($\Delta 0.150$) 로, ECE 도 0.07--0.28 의 큰 변동을 보인다.
SE 의 ranking 분리력과 calibration 모두 corpus 영역에 따라 변동한다.

##### 종합 해석.

세 보조 분석을 종합하면 비대칭의 부분적 메커니즘은 다음과 같다. (i)
corpus 신호와 SE 는 거의 직교한다 ($|\rho| \leq 0.16$). (ii) corpus
신호는 SE 의 sample 단위 확률 예측 자체를 보강하지는 않는다 (interaction
term 이 OOF AUROC 에 +0.0005 만 기여). (iii) 그럼에도 corpus 영역별로 SE
의 AUROC 와 ECE 가 큰 폭으로 변동하며 ($\Delta 0.150$ / ECE 0.07--0.28),
이는 SE 의 *성능 자체* 가 corpus 영역에 의해 조절되는 effect
modification 패턴이다.

즉 corpus 신호는 "어느 영역에서 SE 를 신뢰해야 하는가" 라는 *메타 정보*
로 작용하며, "개별 답변이 환각인가" 를 직접 예측하는 정보로는 약하다. 본
비대칭은 이 두 역할의 차이를 정량화한 결과이며, 특정 분석 방식에서만
우연히 나타난 결과 (예: ranking metric 의 분포 의존성) 가 아니라 corpus
환경에 따른 모델 행동 변화를 반영한다고 해석된다. 다만 본 절은 부분
분석이며, 가능한 다른 메커니즘 (예: fusion 모델의 비선형성 한계, 10분위
이산화 효과) 의 완전한 배제는 향후 과제로 남긴다.

이 메타 정보를 실제 LLM 서비스에서 활용하는 방안 가운데 하나로, entity
co-occurrence 가 낮은 구간에 속하는 질문에 대해서는 SE 탐지 결과를
그대로 신뢰하지 않고 RAG 호출 또는 추가 검토를 트리거하는 임계 기반
전략을 제안할 수 있다. 이는 비대칭 발견의 직접적인 응용이며, 구체적 임계
설정과 실서비스 ROC curve 검증은
§[\[ch:conclusion\]](#ch:conclusion){reference-type="ref"
reference="ch:conclusion"} 의 향후 연구로 남긴다.

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

##### Entity 단위 신호: co-occurrence 와 빈도의 분해 폭 비교.

entity co-occurrence 신호로 sample 을 10분위 구간으로 나눌 때 Semantic
Entropy 의 AUROC range 는 $\Delta$`<!-- -->`{=html}0.150, Semantic
Energy 는 $\Delta$`<!-- -->`{=html}0.144 로, 일곱 corpus 신호 가운데
가장 크다. corpus 가 가장 부족한 구간 (00--10) 에서 Semantic Entropy 와
Semantic Energy 는 각각 0.643, 0.667 이고, 가장 풍부한 구간 (70--80)
에서는 0.793, 0.811 이다. 반면 단일 entity 빈도 신호로 분할했을 때의
AUROC range 는 Semantic Entropy 기준 $\Delta$`<!-- -->`{=html}0.080,
Semantic Energy 기준 $\Delta$`<!-- -->`{=html}0.077 로 본 표에서 가장
작다 (entity co-occurrence 의 약 1.88배 차이). baseline corpus 신호 (두
점수의 평균) 도 $\Delta$`<!-- -->`{=html}0.082--0.086 으로 entity 빈도
신호와 거의 같은 수준이며, 평균화가 entity co-occurrence 의 큰 분해 폭을
희석시킨다. 단조성도 같은 방향이다. entity co-occurrence ($\rho$=+0.648,
$p$=0.043) 와 entity 빈도 ($\rho$=+0.636, $p$=0.048) 모두 양의 단조성을
보이나, 두 점수의 평균은 $\rho$=+0.418 ($p$=0.229) 로 유의 임계 아래로
떨어진다.

sample 단위 부트스트랩 (500회) 으로 계산한 두 AUROC range 차이의 95%
신뢰구간은 Semantic Entropy 에서 \[+0.002, +0.117\] (97.6% 양수),
Semantic Energy 에서 \[+0.004, +0.109\] (98.0% 양수) 로 0 을 포함하지
않는다. 다만 약 1.88배라는 비율은 데이터셋별로 0.67--2.92 범위에서
변동하며 (BioASQ 2.92, TriviaQA 0.99, NQ-Open 0.67 로 부호 역전), 안정적
요약은 1.5\~1.9배이다. 인접 구간의 단조 증가가 보장되지 않으며
(그림 [4.1](#fig:axis_decomp){reference-type="ref"
reference="fig:axis_decomp"}), 양 끝단 평균 경향과 max$-$min 폭만을
의미한다.

##### Question-answer bridge 와 답변 3-gram 미등장 개수.

Question-answer bridge 신호 ((질문 entity, 정답 후보 entity 중 질문과
겹치지 않는 것) 의 corpus 동시 등장 빈도) 로 sample 을 분할하면 답변
평균 NLL 의 AUROC range 가 $\Delta$`<!-- -->`{=html}0.176 로 측정되어,
Semantic Entropy / Semantic Energy 의 AUROC range
($\Delta$`<!-- -->`{=html}0.089, $\Delta$`<!-- -->`{=html}0.087) 보다
크다. (질문, 정답 후보) 사실 관계의 corpus 동시 등장이 토큰 단위 자신감
(NLL) 과 가장 직접 연결되며, 답변 사이 일관성을 측정하는 SE / Energy
와는 다른 정보를 포착함을 시사한다. 답변에서 corpus 미등장 3-gram 의
개수로 분할하면 Semantic Entropy AUROC range 는
$\Delta$`<!-- -->`{=html}0.122 로 큰 편이지만 단조성은 $\rho$=+0.261
($p$=0.467) 로 유의 임계 아래이며, 일관된 증가가 아닌 일부 구간의 일시
변동에서 비롯된 것으로 해석된다.

##### 강건성 검증: 질문 entity 한정 및 SVAMP 제외.

질문 텍스트만으로 entity 를 추출하여 재산출하면 비율은 SE 기준
1.88배에서 2.20배로 증가하나 (n=2,095 부분표본), 부분표본이 entity pair
관계 질문에 치우친 *선택 편향* 이 있어 절대 크기 비교에는 한계가 있다.
정답 후보 측 entity 를 제거해도 부호가 유지된다는 정성적 강건성 결과로
해석한다. 수학 단어 문제인 SVAMP 를 제외한 3,200 sample 에서는 SE 기준
비율이 1.88 에서 1.68 로, Energy 기준도 1.54 로 비슷한 폭으로 감소한다.
부호는 유지되며 안정적 보고 범위는 1.5--1.9배이다.

##### 데이터셋별 분해 결과.

표 [\[tab:per_dataset_delta\]](#tab:per_dataset_delta){reference-type="ref"
reference="tab:per_dataset_delta"} 는 데이터셋별 Semantic Entropy AUROC
range 를 entity co-occurrence 신호와 entity 빈도 신호에 대해 보고한다.
다섯 데이터셋 중 네 (BioASQ, SQuAD-1.1, SVAMP, TriviaQA) 에서는 entity
co-occurrence AUROC range 가 entity 빈도 AUROC range 보다 크거나 같으며,
비율은 TriviaQA 의 1.0배에서 BioASQ 의 2.92배까지 변동한다. NQ-Open
에서는 부호가 역전되어 entity 빈도 (0.372) 가 entity co-occurrence
(0.250) 보다 크다. 전체 표본 비율 1.88배는 데이터셋 평균이 아니라 BioASQ
의 큰 비율이 NQ-Open 의 역전을 보상한 종합 통계이며, 모든 도메인에서
성립하지 않는다. 본 결과는 도메인에 따라 적절한 corpus 신호 단위가
달라질 수 있다는 가설을 제기한다.

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
AUROC range 가 entity 빈도 신호보다 전체 표본 기준 약 1.5--1.9배 컸다
(Semantic Entropy 기준 $\Delta$`<!-- -->`{=html}0.150 와
$\Delta$`<!-- -->`{=html}0.080). 이 비율은 BioASQ (2.92배) 에서 NQ-Open
(0.67배, 역전) 까지 데이터셋별 변동이 크므로, 도메인 무관한 일반화로
해석하지는 않는다. "Paris--France 가 함께 등장하는지" 와 같은 entity
pair 정보가 단일 entity 빈도보다 모델의 사실 학습 정도를 더 직접
반영한다는 직관과 일치한다.

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
질문 entity 와 정답 후보 entity 의 corpus 동시 등장은 LLM 이 "질문에서
정답으로 이어지는 경로" 를 사전학습에서 직접 학습했을 가능성을 근사하며,
이것이 토큰 단위 자신감을 반영하는 NLL 과 가장 직접 연결된다. Semantic
Entropy 와 Semantic Energy 는 답변 사이 일관성을 보는 다른 차원의
신호이므로, 두 신호군이 서로 보완적인 정보를 포착한다고 해석할 수 있다.

##### 효과 크기의 정밀도.

§[4.6](#sec:axis){reference-type="ref" reference="sec:axis"} 에서 보고한
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
표본에서 유의하며, 비율의 점추정치는 약 1.5--1.9배 범위에 있으나 NQ-Open
처럼 역전되는 도메인이 존재한다" 로 한정한다. 또한 AUROC range
(max$-$min) 통계량은 이상 구간에 민감하므로, 후속 연구에서는 IQR 기반
dispersion 지표 (예: 25분위 / 75분위 AUROC 차) 를 병행 보고하는 것을
권장한다. 보다 정밀한 효과 크기 추정은 추가 모델과 corpus 위에서의 추가
표본을 통해 가능하다.

## 한계 {#sec:limitations}

##### \[핵심 한계 1\] Proxy corpus index 사용.

본 논문은 corpus 신호 산출에 Infini-gram 색인 `v4_dolmasample_olmo`
(OLMo 의 16B Dolma sample) 을 사용하며, 이는 Qwen2.5-3B 의 실제 사전학습
corpus 가 아니다 (Qwen 학습 데이터 비공개). 따라서 본 논문이 측정한
"corpus 뒷받침" 은 모델이 실제로 학습한 데이터 양이 아니라 web-scale
transparent 색인 위의 등장 빈도이다. Web-scale 사전학습 corpus (Dolma,
RedPajama, C4 등) 는 Common Crawl 기반 source 를 공유하므로 합리적 proxy
가정은 선행 연구[@qucorag] 에서도 채택되었으나, 이 대체는 형식적으로
보장되지 않는다. 따라서 본 결과는 corpus 빈도와 환각 사이의 상관 관찰에
한정되며, "모델이 학습해서 잘 안다" 라는 인과 진술이 아니라 "Dolma
분포에서 자주 등장하는 사실은 Qwen 학습에서도 자주 등장했을 가능성이
높다" 라는 간접 추론으로 해석한다. 1.88배 비율 또한 Dolma 색인에 한정된
관찰이다.

##### \[핵심 한계 2\] 단일 모델 평가.

본 연구는 Qwen2.5-3B 단일 모델에서 수행되었다. corpus 신호의 구간별 변동
패턴이 Llama, Mistral, Gemma 계열에서도 같은 방향으로 나타나는지는 직접
검증되지 않았다. Farquhar 등 (2024)[@farquhar2024] 가 LLM 계열 사이에서
SE 패턴의 유사성을 보고한 점은 일반화 가능성을 시사하나, 본 연구 범위
밖이다.

1.  **라벨링 절차.** (i) 정답 라벨은 `microsoft/deberta-large-mnli`
    양방향 entailment $\geq 0.5$ 기준에 의존하며, 임계값 0.4 / 0.6
    민감도 분석과 LLM-as-judge 비교 검증은 수행하지 않았다. (ii) 라벨과
    SE / Energy 신호 모두 동일 N=10 답변에서 산출되어 동일 답변 집합의
    분산을 부분 공유하므로, 본 AUROC 는 모델 출력 사이 일관성과 정답
    여부의 결합도를 측정하며 외부 정답 텍스트와의 독립 일치도 측정은
    아니다. (iii) 다만 corpus 신호 자체는 질문 텍스트와 데이터셋 정답
    텍스트의 entity 만으로 산출되어 모델 출력과 독립이며, 분해 분석에
    self-conditioning 문제는 발생하지 않는다.

2.  **데이터셋 이질성.**
    표 [\[tab:per_dataset_delta\]](#tab:per_dataset_delta){reference-type="ref"
    reference="tab:per_dataset_delta"} 에서 NQ-Open 은 부호가 역전되며
    (entity 빈도 0.372 \> entity co-occurrence 0.250), 비율은 BioASQ
    2.92배에서 NQ-Open 0.67배까지 변동한다. NQ-Open 두 번째 구간
    (20--30) 의 AUROC 0.266 outlier 는 정답률 0.058 (120개 중 7개) 의
    클래스 불균형에서 일부 비롯되나 분포 skew 의 근본 원인 (질문 유형,
    entity 분포) 은 미규명이다. 따라서 표본 종합 비율은 데이터셋 종합
    통계이며, 도메인별 최적 단위가 달라질 가능성은 본 연구 범위 밖이다.

3.  **통계 정밀도.** (i) 두 AUROC range 차이의 95% 신뢰구간은 SE
    \[+0.002, +0.117\], Energy \[+0.004, +0.109\] 로 0 을 포함하지
    않으나 단일 모델·단일 corpus 색인 위의 표본 변동성만 통제한다.
    max$-$min 순서통계량 특성상 구간 수에 따라 비율 절대 크기가
    달라진다. (ii) Fusion lift 95% 신뢰구간 \[+0.005, +0.011\] 도 0 을
    포함하지 않으나 절대 크기 (+0.008) 는 답변·sample 단위 신호 결합
    효과 (+0.026) 의 약 3분의 1 수준이다. (iii)
    표 [\[tab:axis_decomp\]](#tab:axis_decomp){reference-type="ref"
    reference="tab:axis_decomp"} 의 21개 동시 검정에 다중비교 보정
    (Bonferroni $\alpha$=0.05/21$\approx$`<!-- -->`{=html}0.0024) 을
    적용하면 모든 $\rho$ 가 유의 기준을 충족하지 않으므로 $\rho$ 결과는
    예비 분석으로 한정한다. n=10 검정력 부족으로 $|\rho|$=0.65 부근의
    결과는 구간 폭과 동순위 처리에 민감하다. (iv) AUROC range 와 단조성
    ($\rho$) 은 분리 진단을 위해 함께 보고되었으나, 보다 강건한 검정
    (Mann-Kendall, partial $\rho$) 은 후속 과제이다.

4.  **분석 미수행.** (i) Fusion 모델 +0.026 향상이 SE / Energy / 답변
    단위 logit 통계 중 어느 신호 결합에서 비롯되었는지에 대한 입력
    변수별 기여도 분해 (permutation importance, SHAP) 는 수행하지
    않았다. (ii) 본 연구는 추가 context 없는 단순 prompt 위에서
    수행되었으며, RAG 또는 multi-turn agentic 환경에서의 일반화는
    미검증이다.

5.  **선행 연구 인용 검증 한계.** 본 논문이 인용한 2025년 arXiv preprint
    일부 (Singha 2025[@singha2025eclipse], Wang 2025[@wang2025seredeep],
    Zhang 2025[@zhang2025corpus], Min 2025[@qucorag] 등) 는 제출 시점
    (2026년 2월) 기준 peer review 를 거치지 않았으며, 보고된 수치의
    재현성은 독립 검증이 필요하다. 본 논문 §2 의 해당 인용은 갈래 정리
    목적에 한정한다. 메인 결과의 비교 기준 가운데 Kandpal 2023
    ICML[@kandpal2023longtail], Mallen 2023 ACL[@mallen2023whennot],
    Farquhar 2024 Nature[@farquhar2024] 는 peer-reviewed venue
    출판물이다. 한편 Ma 2025[@ma2025] Semantic Energy 는 OpenReview 기준
    ICLR 2026 제출본 (preprint) 으로 peer review 절차가 진행 중이며, 본
    논문은 이를 "최신 비교 기준" 으로만 사용한다.

## 향후 연구

본 연구의 결과를 토대로 다음 방향의 후속 연구를 제안한다.

1.  **모델 다양화.** Llama, Gemma, Mistral 등 다양한 모델에서 entity
    co-occurrence 패턴을 검증한다.

2.  **실제 사전학습 corpus 사용.** 본 연구가 사용한 OLMo Dolma sample
    proxy 색인을 Qwen 의 실제 학습 데이터 또는 모델별 색인으로 교체하여
    proxy corpus 의 한계를 극복한다.

3.  **RAG / agentic 환경 확장 + 평가 도구로의 발전.** retrieval
    환경에서의 일반화를 검증하는 한편, AUROC range 를 표준 평가 지표로
    포함하는 새 환각 탐지 벤치마크를 제안한다.

이외에 LLM-as-judge 라벨 비교, 도메인 특화 corpus, Type B / C 환각 분해,
syntactic / semantic 기반 corpus 신호 정교화, AUROC range 외 IQR 기반
dispersion 지표 병행 보고 등이 후속 과제다.

## 결론

본 논문은 LLM 환각 탐지 신호의 성능이 사전학습 corpus 의 사실 등장
패턴에 따라 달라진다는 점을 확인하였다. Qwen2.5-3B 와 다섯 개 QA
데이터셋 (총 35,000 개 생성 답변) 위에서, entity co-occurrence 신호로
sample 을 나누었을 때 Semantic Entropy 의 AUROC range 는
$\Delta$`<!-- -->`{=html}0.150 로, 단일 entity 빈도 신호의
$\Delta$`<!-- -->`{=html}0.080 의 약 1.5--1.9배이다. 다만 NQ-Open 에서
비율이 역전되며 전체 수치는 데이터셋 종합 통계로 해석한다. 현 색인이
모델의 실제 사전학습 corpus 와 다르므로 인과 해석은 불가하다.

가장 중요한 발견은 비대칭이다. 동일 corpus 신호가 fusion 입력 변수로는
+0.008 의 작은 향상에 그치는 반면, 조건부 분해 신호로 사용하면 큰 AUROC
range 를 보인다. 이 비대칭은 corpus 신호의 설계 목표를 "탐지기 입력
변수" 에서 "탐지 신호의 조건부 평가 도구" 로 재정의할 필요를 제기하며,
평균 AUROC 와 함께 corpus 조건별 AUROC range 를 보고하는 평가 형식과
실서비스의 조건부 신뢰 전략의 근거가 된다.

본 연구는 학사 논문으로서 단일 모델·proxy corpus 등의 한계를 가지나,
환각 탐지 연구에 corpus 중심의 조건부 분석 관점을 제시한 점에 의의가
있다.

::: thebibliography
99

N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, "Large
Language Models Struggle to Learn Long-Tail Knowledge," *Proceedings of
the 40th International Conference on Machine Learning (ICML)*, vol. 202,
pp. 15696--15707, 2023. arXiv:2211.08411.

L. Kuhn, Y. Gal, and S. Farquhar, "Semantic Uncertainty: Linguistic
Invariances for Uncertainty Estimation in Natural Language Generation,"
*Proceedings of ICLR*, 2023. arXiv:2302.09664.

Y. Razeghi, R. L. Logan IV, M. Gardner, and S. Singh, "Impact of
Pretraining Term Frequencies on Few-Shot Numerical Reasoning," *Findings
of the Association for Computational Linguistics: EMNLP*, pp. 840--854,
2022. arXiv:2202.07206.

A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, and H. Hajishirzi,
"When Not to Trust Language Models: Investigating Effectiveness of
Parametric and Non-Parametric Memories," *Proceedings of ACL*, pp.
9802--9822, 2023. arXiv:2212.10511.

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
Entropy: Fine-grained Uncertainty Quantification for LLMs," *Advances in
Neural Information Processing Systems (NeurIPS)*, 2024.
arXiv:2405.20003.

K. Ciosek et al., "Hallucination Detection on a Budget: Efficient
Bayesian Estimation of Semantic Entropy," *Transactions on Machine
Learning Research (TMLR)*, 2025. arXiv:2504.03579.

H. Ma et al., "Semantic Energy: Detecting LLM Hallucination Beyond
Entropy," *arXiv preprint arXiv:2508.14496*, 2025.

A. Ravichander et al., "HALoGEN: Fantastic LLM Hallucinations and Where
to Find Them," *Proceedings of ACL (Long Papers)*, 2025.
arXiv:2501.08292.

S. Singha, "Detecting AI Hallucinations in Finance: An
Information-Theoretic Method Cuts Hallucination Rate by 92%," *arXiv
preprint arXiv:2512.03107*, 2025.

L. Wang, "SEReDeEP: Hallucination Detection in Retrieval-Augmented
Models via Semantic Entropy and Context-Parameter Fusion," *arXiv
preprint arXiv:2505.07528*, 2025.

S. Valentin et al., "Cost-Effective Hallucination Detection for LLMs,"
*KDD 2024 GenAI Evaluation Workshop*, 2024. arXiv:2407.21424.

W. Zhao et al., "WildHallucinations: Evaluating Long-form Factuality in
LLMs with Real-World Entity Queries," *arXiv preprint arXiv:2407.17468*,
2024.

Y. Zhang et al., "Measuring the Impact of Lexical Training Data Coverage
on Hallucination Detection in Large Language Models," *arXiv preprint
arXiv:2511.17946*, 2025.

D. Min et al., "QuCo-RAG: Quantifying Uncertainty from the Pre-training
Corpus for Dynamic Retrieval-Augmented Generation," *Findings of ACL (to
appear)*, 2026. arXiv:2512.19134.

J. Liu et al., "Infini-gram: Scaling Unbounded n-gram Language Models to
a Trillion Tokens," *Conference on Language Modeling (COLM)*, 2024.
arXiv:2401.17377.

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
