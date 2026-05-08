# Simhi et al. 2025 (Trust Me, I'm Wrong / CHOKE) — Corpus axis 동기 강화

## Scope

Simhi et al. 2025는 LLM이 *정답 지식을 가지고 있음에도* high-certainty
hallucination을 만드는 현상 (CHOKE: Certain Hallucinations Overriding Known
Evidence) 을 보고했다. 이는 본 연구가 다루는 SE의 low-diversity wrong answer
한계와 직접 연결되며, corpus-axis conditioning의 동기를 강화하는 외부 evidence가
된다.

## Simhi et al. 2025 핵심

- **Source**: arXiv:2502.12964 (Feb 2025, revised Aug 2025).
- **CHOKE 현상**: 모델이 정답을 *알고 있을* 때도, 입력의 사소한 변화 (paraphrase,
  prompt 미세 조정)에 의해 confident한 잘못된 답을 생성한다. 이는 일반적인
  hallucination과 구별된다 — 지식 부재가 아닌 prompt-condition된 confident error.
- **Persistence**: prompt와 모델에 걸쳐 일관되게 관측됨. uncertainty 기반 mitigation
  (entropy threshold 등) 이 underperform.
- **제안**: probing-based approach로 부분 개선. high-stakes domain (의료, 법률) 에서
  중요.

## 본 연구와의 연결

### 1. SE의 low-diversity wrong answer 한계와 동형(同形)

본 연구는 Semantic Entropy (Farquhar 2024)의 한계로 "low-diversity wrong answer"
케이스를 명시한다 — 모델이 자신감 있게 똑같이 잘못된 답을 N=10번 반복하면 cluster=1,
entropy=0이 되어 신뢰함으로 판정 (= 틀린 답 통과). Simhi et al.의 CHOKE는 이
현상의 외부 검증 evidence이며, 단지 "uncertainty가 낮다"는 사실로 신뢰성을 추론하는
것이 위험함을 독립적으로 보여준다.

### 2. Corpus axis가 보완하는 영역

CHOKE 사례는 *모델 내부 uncertainty 신호로는 잡히지 않는* hallucination이다. 본
연구의 corpus axis는 모델 내부와 *분리된* 외부 지표 (entity가 corpus에 얼마나
자주 등장하는지) 이므로:

- 모델이 *prompt-induced confident error* 를 만들 때, 그 답에 등장하는 entity의
  corpus frequency가 보통 어떤 패턴인지 (예: low frequency entity + confident wrong
  answer) 를 corpus-axis bin별 분석으로 정량화 가능.
- "high-certainty + low-corpus-support" 가 high-risk bin이라는 가설을 실험적으로
  검증하는 자연스러운 도구가 본 연구의 framework이다.

### 3. 본 연구 본문에서 다룰 positioning

Ch.2 관련 연구 §"기존 연구의 한계"에서:

> "Simhi et al. 2025 (CHOKE) further demonstrate that LLMs hallucinate confidently
> even when the model has internal access to the correct answer, suggesting that
> *internal uncertainty signals alone are insufficient*. This motivates our
> framework's use of *external* corpus statistics as a complementary
> conditioning axis: when high model certainty co-occurs with low corpus support,
> the framework can flag a CHOKE-like risk profile that internal-only signals
> miss."

Ch.3 §이론적 분석 §"환각이 드러나는 양상"에서 confabulation pattern을 다룰 때
Simhi et al.을 인용해 강화.

## Guardrails

- Simhi et al.은 *probing* 으로 일부 개선을 제안하지만, 본 연구는 hidden-state probe를
  사용하지 않는다 (Phillips PC Probe와 같은 이유). Simhi의 *현상 보고* 만 인용하고
  *해결책 (probing)* 은 채택하지 않는다.
- CHOKE는 본 연구의 직접 측정 대상이 아니다. Simhi et al.의 발견을 본 연구
  framework이 *간접적으로 explain 가능* 하다는 정도로만 framing.
- "corpus axis가 CHOKE를 완전히 해결한다"고 claim하지 않는다. Corpus support는
  conditioning 변수이지 fix가 아니다.

## Citation status

- arXiv preprint (Feb 2025, revised Aug 2025). 인용 시 preprint 표기 유지.
- 인용 위치 추천: 서론 hook 보강, Ch.2 SE 한계 단락, Ch.3 confabulation 단락.
