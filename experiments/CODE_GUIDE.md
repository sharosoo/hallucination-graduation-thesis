# Code guide — hexagonal layout + plug-in points

이 문서는 본 repo 의 핵심 코드 구조와 plug-in 변경 절차를 정리한다. 새 모델
이나 새 backend 를 도입할 때 어디를 손대야 하는지 한 곳에서 찾기 위한 reference
이다. 본 트랙은 **트랙 B — SE 5-dataset Single-candidate (Phase 3,
generation-level NLI correctness)** 를 따른다 (`experiments/PIPELINE.md` 참조).
삭제된 Phase 1/2 모듈에 대한 내용은 본 문서에 더 이상 등장하지 않는다 (사유는
`HISTORY.md` 참조).

## 1. Layered structure

```
experiments/
├── domain/      # 순수 도메인 dataclass (포트 의존성 없음)
├── ports/       # ABC / Protocol 정의 (interface)
├── adapters/    # 외부 의존성을 가진 구체 구현
├── application/ # use case orchestration (도메인 + ports 사용)
├── scripts/     # CLI 진입점 (argparse + adapter 선택)
└── configs/     # YAML configuration
```

규칙:

- `domain/` 은 다른 패키지 안 import 하지 않는다. 순수 dataclass 만 둔다.
- `ports/` 는 `domain/` 만 import 가능. 외부 라이브러리 사용 금지.
- `adapters/` 는 `ports/` 와 `domain/` import 가능. transformers / torch / pyarrow
  같은 외부 의존성은 모두 여기에 isolate 한다.
- `application/` 은 `ports/`, `domain/` import. adapter 인스턴스를 dependency
  injection 으로 받는다.
- `scripts/` 는 argparse 로 사용자 입력을 받고 adapter 선택 + DI 만 한다.
  비즈니스 로직 들어가지 않는다.

## 2. Plug-in points

### 2.1 Entity extractor

- Port: `experiments/ports/entity_extractor.py::EntityExtractorPort`
- Adapters:
  - `experiments/adapters/entity_extractor_spacy.py::SpacyEntityExtractor`
    (**default, recommended**) — spaCy `en_core_web_lg` NER, PERSON / ORG /
    GPE / LOC / DATE / EVENT / WORK\_OF\_ART / FAC / NORP / PRODUCT /
    LANGUAGE / LAW 필터, noun-chunk + 텍스트-자체 fallback. CPU-only,
    1.4 ms/text.
  - `experiments/adapters/entity_extractor_regex.py::RegexEntityExtractor`
    (legacy, archived) — regex heuristic. 짧은 entity 일부 누락, 일반
    명사 false positive. 새 run 에서는 사용하지 않음.
  - `experiments/adapters/entity_extractor_quco.py::QucoEntityExtractor`
    (실험적, archived) — `ZhishanQ/QuCo-extractor-0.5B` knowledge triplet
    모델. 짧은 factoid 답변 (`"Delhi"`, `"1941"`) 에서 100% empty triplet
    을 출력하여 우리 데이터셋에 부적합. 어댑터 보존, 실험 미사용.
- Consumer: `experiments/adapters/corpus_features.py::CorpusFeatureAdapter`
  (S8' stage). `combine_entities(row, extractor=...)` 또는
  `_cached_combine_entities(row)` 가 port 호출.
- CLI: `experiments/scripts/compute_corpus_features.py
  --entity-extractor {spacy,regex,quco}` (default `spacy`).
- Provenance: 새 extractor 추가 시 `describe()` 가 `entity_extractor_version`,
  `entity_extractor_kind`, `entity_extractor_model_ref`,
  `entity_extractor_prompt_template` 필드를 반환해야 함.

새 extractor 추가 절차:

1. `experiments/adapters/entity_extractor_<name>.py` 작성. `EntityExtractorPort`
   subclass. `extract(text, role)` 와 `describe()` 구현.
2. `experiments/scripts/compute_corpus_features.py` 의 `_build_entity_extractor`
   에 `<name>` 분기 추가.
3. `experiments/literature/evidence_notes/<name>_extractor_adoption.md`
   evidence note 작성 (모델 출처, 채택 이유, 선행 연구 caveats).
4. `experiments/PIPELINE.md` §S8', `experiments/OVERVIEW.md` §5,
   `experiments/README.md` Entity extractor backends 절 업데이트.

### 2.2 Corpus count backend

- Port: `experiments/ports/corpus_counts.py::CorpusCountBackendPort`
- Adapter: `experiments/adapters/corpus_counts.py` (Infini-gram local + REST)
- 새 backend 추가 시: 같은 절차 적용. 단 `count_entity` / `count_pair` /
  `describe` / `warmup` 4 메서드 모두 구현 필수.

### 2.3 Fusion + decile decomposition

- Port: `experiments/ports/fusion_strategy.py`,
  `experiments/ports/evaluator.py`
- Application: `experiments/application/generation_level_eval.py` 가 본
  파이프라인의 fusion + robustness 통합 entry point.
  - `run_generation_fusion(df)` — 5-fold GroupKFold(prompt_id) OOF, 단일
    신호 baselines (SE-only / Energy-only / 7 corpus single-signal) +
    learned fusion (LR / RF / GBM, no-corpus / with-corpus 6 변형).
  - `corpus_bin_reliability(df, preds, methods, bin_field=...)` — 7 corpus
    axis 별 per-decile AUROC + AURAC.
  - `bootstrap_ci_per_decile`, `per_dataset_breakdown`, `calibration`,
    `compute_aurac` — robustness 보조 함수.
- 새 fusion variant 추가 시 `generation_level_eval.run_generation_fusion` 의
  `fusion_specs` 리스트에 한 줄 추가.

### 2.4 Generation correctness labeling

- Application: `experiments/application/generation_correctness.py`
- 핵심 함수: `build_generation_correctness_frame(free_sample_rows, *, use_nli,
  nli_model_name, threshold)` — 자유 생성 답변 N=10 과 데이터셋 정답 후보
  간 NLI 양방향 함의 (≥ threshold) 매칭으로 row=(prompt_id, sample_index)
  단위 `is_correct` 라벨 산출.
- Audit: 같이 산출되는 `.audit.json` 에 NLI model / threshold / per-dataset
  is_correct rate / token-overlap fallback 비율 기록.

## 3. Stage / artifact 매핑

본 표는 트랙 B (SE 5-dataset, generation-level) 기준이다. 상세 명령은
`experiments/PIPELINE.md` 참조.

| Stage | Script | Adapter / Application | Output | Plug-in |
|---|---|---|---|---|
| S0 | `validate_architecture.py` | `application/architecture_validation.py` | pass/fail | n/a |
| S1' | `prepare_datasets_se.py` | `adapters/hf_datasets_single_candidate.py` | `prompt_groups.jsonl`, `candidate_rows.jsonl` | dataset adapter |
| S2' | `run_generation.py` | `adapters/model_generation.py` | `free_sample_rows.json` (+ checkpoint), `candidate_scores.json` | model backend |
| S3' | `consolidate_checkpoints_se.py` | `adapters/model_generation.py` | consolidated `free_sample_rows.json` | n/a |
| S4' | `compute_semantic_entropy.py` | `adapters/semantic_entropy_features.py` | `semantic_entropy_features.parquet` | NLI model |
| S5' | `compute_energy_se_minimal.py` | `adapters/energy_features.py` | `energy_features.parquet` | n/a |
| S6' | (어댑터 직접 호출) | `adapters/free_sample_diagnostics.py` | `free_sample_diagnostics.parquet` | n/a |
| S7' | (S11' 안에서 호출) | `application/generation_correctness.py` | `generation_correctness.parquet` + `.audit.json` | NLI model |
| **S8'** | `compute_corpus_features.py` | `adapters/corpus_features.py`, `adapters/entity_extractor_*.py`, `adapters/corpus_counts.py` | `corpus_features.parquet` | **entity extractor**, corpus count backend |
| S9' | `compute_qa_bridge_features.py` | `adapters/qa_bridge_features.py` | `qa_bridge_features.parquet` | (entity extractor 공유) |
| S10' | `compute_ngram_coverage_features.py` | `adapters/ngram_coverage_features.py` | `ngram_coverage_features.parquet` | n/a |
| S11' | `run_generation_se_analysis.py` | `application/generation_level_eval.py` | `fusion.generation_level/{summary, predictions}.json(l)` + `robustness.generation_level/*.json` | **fusion strategy**, **evaluator** |
| S12' | `review_ablations.py` (보조: `question_only_axis.py`) | (자체 구현, scipy / sklearn 직접) | `review_ablations.json` (Spearman ρ 21 cells, B=500 bootstrap CI, SVAMP-excl, fusion lift CI) | n/a |
| S13' | `build_results_macros.py` | (자체 구현, JSON 소비만) | `thesis/results_macros.tex` (30 `\providecommand`) | n/a |

## 4. Reusability scope on backend swap

Entity extractor 또는 corpus count backend 를 바꾸면 다음만 재실행:

- S8' (entity 추출 + count 조회 결과 변경)
- S9' (질문↔답변 entity 쌍 cooc 도 entity 결과에 의존)
- S11' (fusion 입력 변경, robustness 재계산)
- S12' / S13' (재현성 macro 재산출)

다음은 재사용:

- S2' (모델 sampling) — entity 와 무관. 가장 무거움.
- S3' (consolidate) — entity 와 무관.
- S4' (NLI cluster + Semantic Entropy) — entity 와 무관.
- S5' (Semantic Energy) — entity 와 무관.
- S6' (free-sample token 통계) — entity 와 무관.
- S7' (NLI is_correct 라벨) — entity 와 무관.
- S10' (n-gram coverage) — entity 가 아닌 token n-gram 사용.

NLI 모델, 모델 sampling, Semantic Energy 계산을 바꾸면 S4' / S5' 부터 재실행
필요 (그 위 stage 도 모두 invalidate).

## 5. Provenance 정책

모든 stage 산출물은 다음 provenance 필드를 함께 기록한다.

- `run_id` — UTC timestamp 기반.
- `code_version` — git commit hash.
- `model_ref` / `tokenizer_ref` — HF id 또는 local path.
- 외부 backend / extractor 의 `describe()` 결과를 그대로 첨부.

이는 향후 audit 와 재현을 위한 핵심 안전장치다. 신규 plug-in 추가 시 반드시
`describe()` 메서드에 위 4 필드를 포함시킬 것.

## 6. Configuration

YAML 설정은 `experiments/configs/` 에 두고, CLI 가 `--config` 또는 stage
별 yaml 을 받아 dataclass 로 deserialize 한다. 트랙 B (Phase 3) 의 기본 설정:

- `experiments/configs/datasets_se.yaml` — 5 SE datasets (TriviaQA / SQuAD-1.1 /
  BioASQ / NQ-Open / SVAMP) 선정 + sample count + seed.
- `experiments/configs/generation_se_qwen.yaml`,
  `experiments/configs/generation_se_gemma.yaml` — Qwen2.5-3B / Gemma 4 E4B
  모델별 sampling parameters (T=1.0, top-p 0.9, top-k 50, N=10, max_new_tokens 64).
- `experiments/configs/fusion.yaml` — fusion baseline 목록 (LR/RF/GBM 6 변형) +
  GroupKFold(prompt_id) split 규칙.
- `experiments/configs/formulas.yaml` — paper-faithful formula 정의 (SE
  Eq. 8, Energy Eq. 11–14, QuCo entity_freq / entity_pair_cooc).
- `experiments/configs/literature.yaml` — 참조 문헌 manifest.

새 plug-in (e.g. entity extractor) 가 config 항목을 가지면 위 yaml 중 하나에
sub-section 으로 추가하고 schema doc 을 본 파일에 갱신할 것.
