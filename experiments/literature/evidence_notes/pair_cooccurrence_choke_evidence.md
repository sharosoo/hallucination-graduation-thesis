# Pair-cooccurrence CHOKE Evidence — corpus-side validation of confident hallucination

## Scope

본 노트는 thesis-valid run (pipeline-20260507T103225Z, Qwen2.5-3B, 11,630 candidate
rows / 5,815 prompts) 에서 발견한 핵심 finding 두 가지가 *같은 현상의 두 측면*
이라는 점을 기록한다. 이는 Simhi et al. 2025 (CHOKE: Certain Hallucinations
Overriding Known Evidence) 가 보고한 high-certainty hallucination 현상의 본
실험 데이터에서의 직접 재현이다.

## 두 측면의 evidence

### (a) candidate-level — 모델이 환각을 *더 자신있게* 만든다

paired discriminative protocol에서 같은 prompt의 (정답 후보, 환각 후보) 두 row를
비교한 paired win-rate (paired Δ≠0인 prompt만 집계, ties 제외):

| Signal | win-rate | n_pairs | 방향 |
|---|---|---|---|
| `mean_negative_log_probability` | 0.360 ~ 0.402 | ~5,200 | **역방향** — 환각이 NLL *더 작음* (= 더 자신있게 생성) |
| `confidence_margin` | 0.315 ~ 0.448 | ~5,200 | **역방향** — 환각의 top-1 margin *더 큼* (= 더 압도적) |
| `logit_variance` | 0.625 ~ 0.678 | ~5,200 | 정방향 (유일) |

→ 단일 candidate-level 신호 3개 중 2개가 *역방향*. 즉 **모델은 환각 답을 만들 때
정답 답을 만들 때보다 더 confident하다.**

### (b) corpus-level — 직접 marker 약화 (RETRACTED, 2026-05-08)

**스pacy entity extractor + rank-quantile binning 으로 갱신한 결과, corpus-level
직접 marker 증거는 retract 한다.**

| Corpus signal | win-rate (Δ≠0 only) | n_pairs (Δ≠0) | 방향 |
|---|---|---|---|
| `entity_frequency_axis` | ~0.5 | (분포 좌편향, 직접 비교 의미 약화) | 무작위 |
| `entity_pair_cooccurrence_axis` | **0.493** | **2,908 (50% of paired prompts)** | 무작위 |

→ regex era 의 0.551 (n=2,499) corpus-level CHOKE 우위는 **spaCy NER 채택 후 사라진다**.
spaCy 가 후보당 더 많은 entity (정답/환각 양쪽 모두) 를 추출하면서 두 paired
후보의 entity-pair 집합 차이가 작아지고, 차이가 남아 있는 prompt 에서도 우위
방향이 무작위에 가까워진 결과로 해석된다.

→ **corpus support 신호의 가치는 직접 marker 가 아니라 conditioning 변수에 있다.**
corpus-bin weighted fusion 이 rank-quantile 균등 분할 (각 decile n=1,163) 위에서
모든 10 decile 통계적 유의 우위를 보이는 것이 그 정량적 증거 (sweet spot
decile 20--60, win 0.635--0.650). 자세한 결과는 thesis §5 참조.

## 통합 그림 — 한 문장 요약 (갱신)

> **모델은 환각을 더 confident 하게 생성한다 (candidate-level 강한 evidence).
> corpus support 의 가치는 신호의 신뢰도가 어느 corpus 영역에서 살아나는지를
> conditioning 하는 데 있다 (직접 marker 가 아닌 conditioning 변수).**

- candidate-level: NLL/margin 역방향 — 환각이 더 자신있게 생성 (강한 evidence, 유지)
- corpus-level direct marker: 본 데이터에서 검출되지 않음 (spaCy 채택 후 retract)
- corpus-level conditioning: corpus-bin weighted fusion 의 모든 decile 우위로 입증

## 본 논문 본문 framing

### 서론/배경 (Simhi et al. 2025 인용; 갱신본)

> "We test whether the CHOKE phenomenon (Simhi et al. 2025), where models
> hallucinate confidently despite internal knowledge, is reproducible in our
> paired discriminative protocol. The candidate-level evidence is strong:
> hallucinated answers receive *lower* mean NLL and *higher* top-1 confidence
> margin than correct answers (paired win rate 0.36 / 0.34, n≈5,785, ties
> excluded). At the corpus level, the entity-pair co-occurrence signal does
> not act as a direct marker in our data after switching to spaCy NER (paired
> win rate 0.493, n=2,908). Instead, corpus support functions as a
> *conditioning axis*: a corpus-bin weighted fusion attains a statistically
> significant paired win-rate above 0.5 in *every one of ten* rank-quantile
> deciles, with peak win rates of 0.635--0.650 in the middle deciles."

### 결과 단락 (갱신본)

> "After switching the entity extractor to spaCy en_core_web_lg, the direct
> corpus-level CHOKE signal weakens (entity-pair co-occurrence paired win-rate
> 0.493, indistinguishable from chance). The value of corpus support in our
> data is therefore not as a stand-alone marker but as a conditioning variable
> for fusion: a corpus-bin weighted fusion that splits corpus support into
> ten rank-quantile-balanced bins and learns per-bin weights attains
> statistically significant paired uplift in every decile, with the strongest
> effect in the middle deciles (20--60) where corpus partially knows the
> entities involved."

### Limitations 보고

- 본 finding은 *paired pair 안* delta. dataset 전반의 인과관계 (예: corpus
  exposure가 hallucination을 *야기* 한다) 는 입증하지 않음.
- corpus axis는 v4_dolmasample_olmo (16B Dolma sample) 한 인덱스 의존. 다른
  corpus snapshot에서 패턴이 동일한지는 향후 검증.

## Money Figure (논문 visual identity)

본 논문 핵심 figure 권장:

```
y-axis (8 signals):
  prompt-level:
    - semantic_entropy_score
    - semantic_energy_score
    - semantic_energy_sample
    - semantic_energy_diagnostic
    - entity_frequency_axis
  candidate-level:
    - mean_negative_log_probability
    - confidence_margin
    - logit_variance
    - entity_pair_cooccurrence_axis

x-axis (5 corpus support bins): very_low, low, mid, high, very_high

cell:
  - prompt-level signal: bin 평균값 또는 distribution quantile (heatmap)
  - candidate-level signal: paired win-rate (paired Δ≠0 prompts only, ties excluded)

legend로 cell의 metric type 구분 (color scale 분리 또는 hatch).
```

이 figure 한 장에서 다음을 모두 시각화:
- prompt-level vs candidate-level 평가 단위 차이가 자연스럽게 드러남
- corpus support 조건에 따라 단일 신호의 방향 / 강도가 변함
- pair_cooccurrence 정방향 + NLL/margin 역방향이 같은 corpus support 구간에서
  나타나는 것을 직접 시각화 → CHOKE pattern의 corpus×candidate 결합 evidence

## Citation chain

- Farquhar 2024 (Semantic Entropy) — 본 논문이 측정하는 prompt-level 신호의 토대
- Ma 2025 (Semantic Energy) — paper-faithful Energy 구현
- QuCo-RAG 2025 — corpus statistics를 평가 도구로 재해석한 본 연구의 직접 inspiration
- **Simhi et al. 2025 (CHOKE)** — 본 finding이 직접 재현하는 현상
- Valentin et al. 2024 (Cost-Effective Detection) — conditional calibration framework로 직교적 비교

## Guardrails

- "환각이 *더 흔한* pair 사용" 이 *모든* 환각의 보편 패턴이라고 일반화하지 않는다.
  43% prompt에서 관측된 paired delta 안의 통계.
- pair_cooccurrence 정방향이 약 win 0.55라는 점에서 단독 detector로 약하다는 점
  명시 — fusion 입력일 때 가치.
- CHOKE 현상의 *해결책* 은 본 연구 범위 밖. 본 연구는 *측정 framework* 까지만.
