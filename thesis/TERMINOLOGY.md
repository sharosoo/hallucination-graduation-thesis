# 본문 용어 규칙 (학술 한국어 재작성용)

본 문서는 thesis 본문 재작성 시 모든 subagent 가 따라야 할 용어 규칙이다.
세 가지 범주로 분류한다.

## A. 영문 그대로 유지 (학계 표준 용어, 한국어 번역이 더 모호함)

| 영문 | 사용 예 |
|---|---|
| Semantic Entropy (SE) | "Semantic Entropy" 첫 등장 시 풀어 쓰고 이후 "SE" |
| Semantic Energy | 그대로 |
| AUROC, AURAC, Brier, ECE | 평가 지표는 영문 약어 |
| NLI, NER | 표준 약어 |
| Infini-gram, spaCy, OLMo | 도구·자원 고유명사 |
| Spearman $\rho$ | "Spearman 상관" 또는 "Spearman $\rho$" |
| LLM-as-judge | 사용자 명시: 그대로 유지 |
| logit | "로짓" 도 가능하나 본문에서 "logit" 그대로 |
| corpus | 한국어 "말뭉치" 보다 corpus 그대로 |
| cluster | "의미 cluster" 형태로 사용 |
| free-sample | 그대로 (또는 "자유 생성 sample") |
| paraphrase | 그대로 |
| fusion | "fusion 모델" / "신호 결합 (fusion)" 형태 |
| gradient boosting / random forest / logistic regression | 분류 모델 표준명, 그대로 |
| GroupKFold | 그대로 (필요 시 "prompt 단위 5-fold 교차검증") |
| top-k / top-p | 그대로 |
| Bonferroni | 그대로 |
| n-gram, 3-gram, 5-gram | 그대로 |
| (부트스트랩) bootstrap | 그대로 (또는 "부트스트랩") |

## B. 한국어로 풀어 쓴다 (영어 직삽 / 코드 식별자 제거)

| 영문 / 코드 식별자 | 한국어 |
|---|---|
### B-1. corpus signal 명칭 (분해 \emph{기준} 으로 사용 시 “기준” / 일반 언급 시 “신호” 중 문맥별 선택)

| 영문 / 코드 식별자 | 한국어 |
|---|---|
| `entity_pair_cooccurrence_axis` | entity 쌍 동시 등장 (entity pair co-occurrence) |
| `entity_frequency_axis` | entity 빈도 (entity frequency) |
| `corpus_axis_bin_10` | baseline corpus 기준 (entity 빈도와 entity 쌍 동시 등장의 평균) |
| `qa_bridge_axis` | question-answer bridge |
| `ans_ngram_3_axis` / `ans_ngram_5_axis` | 답변 3-gram 등장 빈도 / 답변 5-gram 등장 빈도 |
| `ans_ngram_3_zero_count` | 답변 3-gram 미등장 개수 |
| `q_entity_pair_axis_bin_10` | 질문 entity 쌍 (질문 텍스트에서만 추출) |

### B-2. sample 단위 / token 로짓 통계

> 원칙: `-level` 식 명사화 표현 (sample-level / prompt-level / token-level) 은 한국어로 풀어 쓴다.

| 영문 / 코드 식별자 | 한국어 |
|---|---|
| `free-sample` | 생성 답변 (또는 자유 생성 답변) |
| `sample-level (signal)` | "생성 답변마다 산출한 (신호)" / "답변 단위로 산출된 (신호)" |
| `prompt-level (signal)` | "문항 단위로 산출된 (신호)" / "한 문항에 하나의 값을 갖는 (신호)" |
| `token-level` | "토큰 단위로" / "각 토큰에 대해" |
| `sample_nll` | 답변 평균 음의 로그우도 (NLL) |
| `sample_logit_variance` | 답변 내 로짓 분산 |
| `sample_sequence_log_prob` | 답변 전체 로그우도 |
| `sample_logsumexp_mean` | 평균 로그 분배함수 |
| `prompt-grouped bootstrap` | 문항 단위 bootstrap |

### B-3. 라벨 / 데이터셋 / 실험 설정

| 영문 / 코드 식별자 | 한국어 |
|---|---|
| `is_correct` | 정답 라벨 (1) / 환각 라벨 (0) |
| `prompt_id`, `sample_index` | 본문에 노출 금지 (실험 노트 식별자) |
| `best_answer / correct_answers / alias_list` | "데이터셋이 제공한 정답 표현·동치 표현·별칭 목록" |
| `annotation` | "데이터셋이 제공한 (라벨)" |
| `gold answer` | 정답 텍스트 |
| `max_new_tokens` | 최대 생성 토큰 수 |
| `temperature` | 온도 |
| `truncation / truncated` | "잘림 / 중간에 절단됨" |
| `confabulation` | 그대로 (또는 "꾸며낸 답") |

### B-4. 측정 / 평가 / 통계 어휘 (\emph{분해 진폭} 어휘 포함)

| 영문 / 코드 식별자 | 한국어 |
|---|---|
| `decomposition power / amplitude` | **구간별 AUROC 변동 폭** (또는 단순히 “변동 폭”) — “분해 진폭” / “분해력” 사용 금지 |
| `axis` (분해 기준 의미) | **명사화하지 않는다.** 동사·소유격으로 풀어 쓴다. 예: “X 신호의 구간별 AUROC”, “X 신호로 문항을 10분위로 나누었을 때”, “X 신호를 따라 문항을 분할하면” 등. 추상명사 “기준 / 분해 기준 / 조건 기준 / 평가 축” 같은 표현은 모두 어색하므로 사용하지 않는다. |
| `axis` (좌표축 의미) | "축" (그래프 좌표 한정) |
| `evaluation axis 로 사용` | "문항을 분할하는 신호로 사용" / "문항 분할에 사용" |
| `analyze on axis X` | "X 신호로 문항을 나누어 분석" / "X 신호의 구간별로 분석" |
| `decile` | 10분위 구간 (또는 단순히 "구간") |
| `score-based` | 신호 점수 기반 |
| `broadcast` | "한 문항의 모든 답변에 같은 값으로 적용" |
| `leakage` | 정보 누출 |
| `out-of-fold` | "검증 fold 에서 예측한" (또는 그대로) |
| `granularity` | 단위 |
| `co-occurrence` | 동시 등장 |
| `ablation` | 강건성 검증 |
| `robustness` | 강건성 |
| `cross-paper` | 선행 연구 비교 |
| `single-condition point estimate` | "단일 조건에서 산출한 점추정치" |
| `effect size` | 효과 크기 |
| `selection bias` | 선택 편향 |
| `sampling variability` | 표본 변동성 |
| `cluster count` | cluster 수 |
| `feature` | 입력 변수 (또는 feature 그대로) |

### B-5. 의미·해석 어휘

| 영문 / 코드 식별자 | 한국어 |
|---|---|
| `fact-pair` | 두 entity 사이의 관계 (또는 “관계형 정보”) — “사실 쌍” 사용 금지 |
| `relational fact` | 두 entity 사이의 관계 |
| `factual QA` | 지식 질문 / QA — “사실 질의응답” 사용 금지 |
| `in-context knowledge` | 사전학습된 지식 |
| `conditional benchmarking` | 조건부 평가 도구 |
| `reference comparison` | 참고용 비교 |
| `negative result` | "분리력이 제한적이라는 결과" 등으로 풀어 씀 |
| `paper-faithful` | "원 논문 그대로" / "동일하게 구현" |
| `feature 가공` | 신호 가공 |

## C. 금지 패턴 (lab-노트체 / 번역투)

다음은 어떤 경우에도 본문에 등장해서는 안 된다.

1. **paragraph 헤딩에 괄호로 미니 결론**:
   `\paragraph{발견 1 (... 가 최강 분해력 ...).}` → 헤딩에는 짧은 명사구만.
2. **자평 라벨**: `(Baseline)`, `(신규)`, `(핵심 제안)`, `(개선됨)`, `(negative result)` 등 괄호 자평.
3. **메타 부연 괄호**: 본문 끝에 `(... 차단)`, `(... 방지)`, `(... 보고됨)` 같은 짧은 부연.
4. **최상급 / 주관적 형용사**: `최강`, `최약`, `매우 약/강함`, `현저히`, `훨씬`, `완전히`, `오히려`, `흥미로운`, `흥미롭게도`, `강력한`, `심각한`, `핵심 통찰`.
5. **번역투 동사**:
   - `~을 갖는다` → `~을 보인다`
   - `~할 수 있을 것으로 기대한다` → `~할 수 있다`
   - `~에 그치다` → `~에 한정되다`
   - `~에 직결되다` → `~과 연결된다`
   - `~을 깨지지 않는다` → `~이 유지된다`
   - `~함을 시사한다 / ~함을 의미한다` 남발 금지 (한 단락에 두 번 이상)
6. **영어 복합 표현 직삽**: `multi-axis 분해`, `paper-faithful`, `motivation 정당화`, `trivial-but-quantified`, `↔ 화살표`, `Robustness check —`, `Effect-size precision`.
7. **자기 강조 \emph{}**: `\emph{체계적으로 제안}`, `\emph{참조 형식}`, `\emph{단위}` 등 자평 강조.
8. **에피스테믹** → "모델 자체에 대한 / 모델 지식 한계의" 로 풀어 씀.
9. **"세 번째로 큰 ~ 이다"** 식 영어 번역체 서수 표현 → "비교적 크다 / 큰 편이다" 또는 수치 직접 비교.

## D. 인용 / 수식 / 표 인용 형식

- 표/그림 인용: `표~\ref{...}` / `그림~\ref{...}` (`표 4.6` 같은 hardcoded 금지)
- 인용: `Farquhar 등 (2024)` (저자 등) — 첫 등장 시 \cite{} 결합
- 수식 변수: `$\Delta$\HeadlineEntityPairSEDelta{}` 처럼 매크로로 통일
- 통계 인용: `Spearman $\rho$=+0.648 ($p$=0.043)` 형식

## E. 통일 어조

- 1인칭 ("우리는") 금지 → "본 논문/본 연구는"
- 미래형 ("~일 것이다") 가급적 단언형 ("~이다 / ~한다") 으로
- 객관적 사실 진술 우선, 단정형 대신 "~로 관찰된다 / ~로 확인된다"
