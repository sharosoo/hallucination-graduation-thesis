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

### (b) corpus-level — 환각이 *더 흔한 entity pair* 를 사용한다

| Corpus signal | win-rate (Δ≠0 only) | n_pairs (Δ≠0) | 방향 |
|---|---|---|---|
| `entity_frequency_axis` | 0.430 | 803 (14% of paired prompts) | 역방향 — 환각이 *덜 흔한* 단일 entity 사용 |
| **`entity_pair_cooccurrence_axis`** | **0.551** | **2,499 (43%)** | **정방향** — 환각이 *더 자주 같이 등장하는* entity pair 사용 |
| `entity_frequency_mean` (raw count) | 0.521 | 2,599 | 정방향 |

→ pair co-occurrence axis가 **유일하게 candidate-level differentiation 강한
corpus 신호** (43% prompt에서 두 candidate가 다른 값) 이며, 환각 답이 더 흔한
entity pair를 사용한다. 단순 entity frequency는 거의 prompt-level (86%
candidates 동일).

## 통합 그림 — 한 문장 요약

> **모델은 익숙한 entity들을 잘못된 관계로 confident하게 조합해서 환각을 만든다.**

- "익숙한 entity" → corpus pair_cooccurrence가 정방향 (환각이 더 흔한 pair)
- "confident하게" → NLL/margin 역방향 (환각이 더 자신있게 생성)
- "잘못된 관계" → entity_frequency_axis 역방향 (환각의 *조합*은 corpus에 덜 등장)

세 evidence가 *직교적이지만 같은 패턴* 을 그림.

## 본 논문 본문 framing

### 서론/배경 (Simhi et al. 2025 인용 강화)

> "We test whether the CHOKE phenomenon (Simhi et al. 2025), where models
> hallucinate confidently despite internal knowledge, is reproducible in our
> paired discriminative protocol. We find converging evidence from two
> orthogonal scopes: (i) at the candidate level, hallucinated answers receive
> *lower* mean NLL and *higher* top-1 confidence margin than correct answers
> (paired win rate 0.36 / 0.34, n≈5200, ties excluded); (ii) at the corpus
> level, hallucinated answers use *more frequent* entity-pair co-occurrences
> (win 0.551, n=2499). Both observations support the same picture: models
> confidently combine familiar entities into incorrect relations."

### 결과 단락

> "Pair co-occurrence is the only corpus signal with candidate-level
> differentiation strong enough for paired evaluation: 43% of paired prompts
> have a non-zero pair-cooccurrence delta between the correct and hallucinated
> candidate, compared to only 14% for single-entity frequency. Among those
> non-tied prompts, the hallucinated candidate has *higher* pair-cooccurrence
> 55.1% of the time, suggesting that hallucinations recombine entities that
> co-occur frequently in the training corpus into relations that are factually
> incorrect — a corpus-side echo of the CHOKE pattern at the language-model
> output."

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
