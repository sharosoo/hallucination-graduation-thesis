# Corpus-conditioned hallucination metric reliability plan

이 문서는 다음 실험 사이클의 연구 계획을 고정한다. 본 연구는 RAG 시스템 논문이 아니라 **환각 탐지 지표의 조건부 신뢰성**을 분석하는 논문이다. QuCo-RAG에서 영감을 받은 corpus entity frequency와 entity-pair co-occurrence는 환각의 직접 지표가 아니라, 기존 hallucination metric이 어느 조건에서 신뢰로운지 나누는 연속 조건 축으로 사용한다.

## 1. Revised thesis claim

본 연구의 중심 주장은 다음이다.

> 말뭉치 기반 entity 등장 빈도와 entity-pair co-occurrence가 만드는 연속 조건 축에 따라 hallucination detection feature의 신뢰도와 유효성이 달라진다. 따라서 Semantic Entropy, Semantic Energy, likelihood/logit diagnostics, corpus statistics를 하나의 전역 점수로만 비교하는 대신, corpus 조건 구간별 성능과 condition-aware fusion을 평가해야 한다.

이 주장은 다음을 말하지 않는다.

- RAG 시스템을 새로 제안한다는 뜻이 아니다.
- corpus entity frequency가 곧 hallucination probability라는 뜻이 아니다.
- global learned fusion이 항상 더 좋다는 뜻이 아니다.
- SE가 hallucination 유형 분류의 유일한 기본 축이라는 뜻이 아니다.

## 2. Feature taxonomy

| Family | Feature | Status | Thesis role |
| --- | --- | --- | --- |
| Semantic Entropy | N=10 NLI likelihood-based SE, cluster count | paper-faithful target | semantic diversity feature and optional secondary axis |
| Semantic Energy | sampled-response cluster-level selected-token-logit energy | paper-faithful target | semantic-cluster logit-energy feature |
| Candidate logit diagnostics | NLL, confidence margin, logit variance | thesis-derived diagnostics | candidate-level reliability probes |
| QuCo-style corpus statistics | entity frequency, entity-pair co-occurrence, low/zero support flags | paper-inspired corpus axis | continuous conditioning axis, not direct hallucination score |
| Fusion | global and condition-aware fusion | thesis proposal | compare global vs corpus-conditioned reliability |

Existing checked artifacts remain useful for diagnosis, but current N=5 exact-match SE, candidate-level Boltzmann diagnostic, and local candidate-corpus counts are not final paper-faithful implementations.

## 3. Research questions

### RQ1. Corpus entity frequency axis

Does entity frequency in a corpus condition which hallucination metrics are reliable?

- Build continuous entity support scores such as `log(1 + mean_entity_frequency)` and `log(1 + min_entity_frequency)`.
- Preserve raw counts and corpus provenance.
- Bin into predefined fixed or train-split quantile bins.
- Evaluate every metric globally and per bin.

### RQ2. Entity-pair co-occurrence axis

Does entity-pair co-occurrence condition relation-level hallucination risk and metric reliability?

- Compute `head AND tail` or entity-pair co-occurrence using an Infini-gram-compatible count backend or fixed corpus snapshot.
- Treat zero/low co-occurrence as evidence-sparsity conditions, not as deterministic hallucination labels.
- Evaluate whether features behave differently when entities are individually frequent but jointly rare.

### RQ3. Paper-faithful uncertainty features

Do paper-faithful SE and Semantic Energy behave differently from thesis-derived logit diagnostics across corpus bins?

- Recompute SE with N=10 answer-only samples, DeBERTa-family NLI clustering, and likelihood-based cluster probabilities.
- Recompute Semantic Energy over sampled response clusters using selected-token logits, not only teacher-forced candidate `-logZ` diagnostics.
- Keep NLL, confidence margin, and logit variance as diagnostic baselines, not source-paper features.

### RQ4. Condition-aware fusion

Does corpus-conditioned fusion improve reliability or interpretability over global fusion?

- Compare single features, global fusion, and condition-aware fusion.
- Report aggregate metrics, per-bin metrics, worst-bin behavior, calibration, and paired win-rate.
- Treat improvements as empirical observations, not assumed thesis claims.

## 4. Required experiments

### E1. Feature implementation alignment audit

Produce a table mapping each feature to source paper, current implementation, final target implementation, and known deviations.

### E2. Incremental N=10 generation

Extend existing N=5 answer-only samples to N=10 without destroying old artifacts.

- Preserve sample indexes 0--4.
- Generate sample indexes 5--9.
- Write a versioned artifact such as `free_sample_rows_n10.json` and matching sidecar.
- Validate exactly sample indexes `{0, ..., 9}` per prompt.

### E3. NLI likelihood-based SE

- Use DeBERTa-family NLI with explicit label mapping from `model.config.id2label`.
- Cluster answer samples via a documented bidirectional or relaxed entailment rule.
- Compute cluster probability with likelihood log-sum-exp.
- Export both likelihood-based SE and discrete cluster entropy as separate fields.

### E4. Paper-faithful Semantic Energy

- Use the same sampled responses and semantic clusters as SE.
- Compute selected-token-logit energy for each generated answer.
- Aggregate energy at the semantic cluster level.
- Keep current candidate-level `semantic_energy_boltzmann` only as a diagnostic unless realigned.

### E5. QuCo-style corpus axis

- Replace or clearly demote local candidate-corpus count proxies.
- Use Infini-gram-compatible counts or a fixed external corpus snapshot.
- Store raw entity frequency, raw pair co-occurrence, log-transformed support scores, bin ids, and corpus provenance.

### E6. Corpus-bin metric reliability

For each entity-frequency and co-occurrence bin, report:

- AUROC
- AUPRC
- F1 or fixed-threshold metrics
- paired win-rate where pairs exist
- mean/median hallucinated-minus-normal delta
- prompt-grouped bootstrap confidence intervals

### E7. Corpus × secondary-axis analysis

Run a coarse grid rather than overfitting many cells.

- corpus support low/mid/high × SE low/high
- corpus support low/mid/high × confidence margin low/high
- entity frequency high but co-occurrence low as a special analysis slice

### E8. Condition-aware fusion

Compare:

- best single feature per global setting
- global logistic fusion
- corpus-conditioned per-bin feature selection
- small interaction model with a frozen, limited feature set

## 5. Primary tables and figures

1. Feature alignment table: paper-faithful / adapted / diagnostic.
2. Corpus support distribution: entity frequency and co-occurrence histograms.
3. Corpus bin × feature metric table.
4. Corpus bin × feature heatmap for AUROC/AUPRC/win-rate.
5. Corpus frequency × co-occurrence grid showing best feature and sample count.
6. Global fusion vs condition-aware fusion table.
7. Failure-case table for high-confidence hallucination under low corpus support.

## 6. Interpretation rules

- Corpus-only performance may be weak; that does not invalidate corpus as a conditioning axis.
- SE is not assumed to be the base axis. It is one feature and optional secondary axis.
- NLL, confidence margin, and logit variance must be named as diagnostics unless tied to a cited paper.
- Any feature orientation or bin boundary chosen after seeing test results is exploratory.
- Final claims must distinguish current diagnostic artifacts from paper-faithful final artifacts.

## 7. Related work positioning

| Paper | Role in this thesis |
| --- | --- |
| Farquhar 2024 (Nature) — Semantic Entropy | S4 SE 구현 토대 + Eq.(8) cluster probability를 paper-faithful Energy에서 상속 |
| Ma 2025 — Semantic Energy | S6 paper-faithful Energy 식 (Eq.11–14) 그대로 구현. preprint 인용 caveat 유지 |
| QuCo-RAG 2025 | corpus statistics를 *retrieval trigger* 가 아닌 **conditioning axis** 로 재해석 (같은 통계, 다른 용도) |
| Phillips 2026 (PC Probe) | hidden-state probe — **사용 안 함**. black-box compatibility 유지를 위한 framing reference |
| **Valentin et al. 2024 (Cost-Effective Hallucination Detection)** | conditional calibration framework — *내부* score attribute 조건화. 본 연구의 *외부* corpus statistics 조건화와 직교적. positioning에서 명시 차별화 필요 |
| **Simhi et al. 2025 (CHOKE)** | 모델이 정답 지식을 가지고도 high-certainty hallucination 생성 — SE의 low-diversity wrong answer 한계의 외부 evidence. corpus-axis conditioning 동기 강화. probing 해결책은 채택 안 함 |

자세한 positioning은 `experiments/literature/evidence_notes/valentin_calibration_vs_corpus_axis.md`,
`experiments/literature/evidence_notes/simhi_choke_motivates_corpus_axis.md`,
`experiments/literature/evidence_notes/quco_vs_probe.md` 참조.

### 핵심 발견 (thesis-valid run, paper-faithful 결과 도착 후)

본 실험 데이터에서 Simhi et al. 2025의 CHOKE 현상이 **두 직교 scope에서 동시에 재현됨**:
- **candidate-level**: 환각 답이 정답 답보다 *더 작은 NLL* + *더 큰 confidence margin*
  (paired win-rate 0.36 / 0.34, n≈5,200, ties 제외) — 모델이 환각을 *더 자신있게* 생성
- **corpus-level**: `entity_pair_cooccurrence_axis` 의 paired win-rate 0.551
  (n=2,499) — 환각 답이 corpus에 *더 자주 같이 등장하는* entity pair 사용

두 evidence가 동일 그림을 그림: **"모델은 익숙한 entity를 잘못된 관계로
confident하게 조합해서 환각을 만든다."** 자세한 분석은
`experiments/literature/evidence_notes/pair_cooccurrence_choke_evidence.md`.

## 8. Entity 추출 backend 변경 (2026-05 pivot)

기존 S5 corpus features 는 `phrase_candidates` regex 휴리스틱으로 entity 를
추출하였으나, 다음 두 reviewer 지적으로 NER backend 를 pluggable 하게 만들고
QuCo-RAG 의 공개 distilled 모델로 교체할 수 있도록 했다.

- **Reviewer 지적**: 연도 (`1958`, `1941`) 와 같은 4 자 entity 누락, 문장 시작
  capitalized 일반 단어 (`The`, `What`) false positive, 5+ 자 일반 명사가
  stopword 누락 시 후보로 들어옴. 이 노이즈가 corpus-level CHOKE 효과 크기
  (paired win-rate 0.551) 를 약화시켰을 가능성.
- **결정**: `EntityExtractorPort` 를 신설하고 두 adapter 제공:
  - `RegexEntityExtractor` (legacy, default): 기존 `phrase_candidates` 그대로.
    재현 호환을 위해 보존.
  - `QucoEntityExtractor` (권장): `ZhishanQ/QuCo-extractor-0.5B`
    (Qwen2.5-0.5B-Instruct fine-tuned, GPT-4o-mini 에서 distill 된 QuCo-RAG
    공개 모델). knowledge triplet `(head, relation, tail)` 출력에서 head /
    tail 만 entity 로 사용.
- **CLI**: `compute_corpus_features.py --entity-extractor {regex,quco}`,
  `run_pipeline.py --entity-extractor {regex,quco}`. 기본은 `regex` 로 유지
  (기존 산출물 재현 가능). 신규 thesis-valid run 은 `quco` 권장.
- **재실행 scope**: S5 → S7 → S8 → S9 만 재실행 (전체 30%). S2 모델 sampling /
  scoring, S4 NLI cluster, S6 Semantic Energy 는 entity 와 무관하므로 재사용.
- **Provenance**: `corpus_features.parquet` 의 report 에 `entity_extractor`
  필드 추가 (`version`, `kind`, `model_ref`, `prompt_template`).
- **Caveat 보존**: Infini-gram 16B index ≠ 모델 (Qwen2.5-3B) 의 실제
  pretraining corpus. corpus support 는 여전히 *proxy* 이며 인과 관계가 아닌
  상관 관계로만 해석한다 (Zhang et al. 2025 의 동일 caveat 인용).

자세한 결정 근거는 `experiments/literature/evidence_notes/quco_extractor_adoption.md`.
