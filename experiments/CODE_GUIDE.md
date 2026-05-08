# Code guide — hexagonal layout + plug-in points

이 문서는 본 repo 의 핵심 코드 구조와 plug-in 변경 절차를 정리한다. 새 모델
이나 새 backend 를 도입할 때 어디를 손대야 하는지 한 곳에서 찾기 위한 reference
이다.

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
  (S5 stage). `combine_entities(row, extractor=...)` 또는
  `_cached_combine_entities(row)` 가 port 호출.
- CLI: `experiments/scripts/compute_corpus_features.py
  --entity-extractor {spacy,regex,quco}` (default `spacy`). 같은 옵션이
  `experiments/scripts/run_pipeline.py` 에도 있음.
- Provenance: 새 extractor 추가 시 `describe()` 가 `entity_extractor_version`,
  `entity_extractor_kind`, `entity_extractor_model_ref`,
  `entity_extractor_prompt_template` 필드를 반환해야 함.

새 extractor 추가 절차:

1. `experiments/adapters/entity_extractor_<name>.py` 작성. `EntityExtractorPort`
   subclass. `extract(text, role)` 와 `describe()` 구현.
2. `experiments/scripts/compute_corpus_features.py` 의 `_build_entity_extractor`
   에 `<name>` 분기 추가.
3. `experiments/scripts/run_pipeline.py` argparse `--entity-extractor` choices
   에 `<name>` 추가, `build_stages` 에 forward 인자 작성.
4. `experiments/literature/evidence_notes/<name>_extractor_adoption.md`
   evidence note 작성 (모델 출처, 채택 이유, 선행 연구 caveats).
5. `experiments/PIPELINE.md` §S5, `experiments/OVERVIEW.md` §5,
   `experiments/README.md` Entity extractor backends 절 업데이트.

### 2.2 Corpus count backend

- Port: `experiments/ports/corpus_counts.py::CorpusCountBackendPort`
- Adapter: `experiments/adapters/corpus_counts.py` (Infini-gram local + REST)
- 새 backend 추가 시: 같은 절차 적용. 단 `count_entity` / `count_pair` /
  `describe` / `warmup` 4 메서드 모두 구현 필수.

### 2.3 Fusion strategy

- Port: `experiments/ports/fusion_strategy.py`
- Adapter: `experiments/application/fusion.py` 가 사이킷런 LR / RF / GBM /
  SVM, corpus-bin weighted, axis-interaction 등을 함께 정의.
- 새 fusion variant 추가 시 `application/fusion.py` 의 `BASELINE_RECIPES`
  딕셔너리에 등록.

### 2.4 Evaluator

- Port: `experiments/ports/evaluator.py`
- Adapter: `experiments/application/robustness.py` 가 prompt-grouped
  bootstrap, LODO, calibration, decile reliability 통합.

## 3. Stage / artifact 매핑

| Stage | Script | Adapter / Application | Output | Plug-in |
|---|---|---|---|---|
| S0 | `validate_architecture.py` | `application/architecture_check.py` | pass/fail | n/a |
| S1 | `prepare_datasets.py` | `adapters/hf_datasets.py` | `prompt_groups.jsonl`, `candidate_rows.jsonl` | dataset adapter |
| S2 | `run_generation.py` | `adapters/model_generation.py` | `free_sample_rows.json`, `candidate_scores.json` | model backend |
| S3 | `build_correctness_dataset.py` | `adapters/correctness_dataset.py` | `correctness_judgments.jsonl` | label policy |
| S4 | `compute_semantic_entropy.py` | `adapters/semantic_entropy_features.py` | `semantic_entropy_features.parquet` | NLI model |
| **S5** | `compute_corpus_features.py` | `adapters/corpus_features.py`, `adapters/entity_extractor_*.py`, `adapters/corpus_counts.py` | `corpus_features.parquet` | **entity extractor**, corpus count backend |
| S6 | `compute_energy_features.py` | `adapters/energy_features.py` | `energy_features.parquet` | model backend |
| S7 | `build_feature_table.py` | `application/labeling.py` | `features.parquet` | n/a |
| S8 | `run_fusion.py` | `application/fusion.py` | `fusion/summary.json`, `predictions.jsonl` | **fusion strategy** |
| S9 | `run_robustness.py` | `application/robustness.py` | `robustness/summary.json`, `bootstrap_ci.json`, `corpus_bin_reliability.json`, `leave_one_dataset_out.json`, `threshold_calibration.json` | **evaluator** |

## 4. Reusability scope on backend swap

Entity extractor 또는 corpus count backend 를 바꾸면 다음만 재실행:

- S5 (entity 추출 + count 조회 결과 변경)
- S7 (feature table 결합)
- S8 (fusion 입력 변경)
- S9 (모든 robustness 지표 재계산)

다음은 재사용:

- S2 (모델 sampling + teacher-forced scoring) — 가장 무거움. entity 와 무관.
- S4 (NLI cluster + Semantic Entropy) — entity 와 무관.
- S6 (Semantic Energy) — entity 와 무관.

NLI 모델, 모델 sampling, Semantic Energy 계산을 바꾸면 S4 / S6 부터 재실행
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
별 yaml 을 받아 dataclass 로 deserialize 한다. 기본 설정:

- `experiments/configs/datasets.yaml` — TruthfulQA / HaluEval-QA 선택, paired
  candidate selection 규칙
- `experiments/configs/generation.yaml` — N=10 free sampling cap, teacher-forced
  scoring 옵션
- `experiments/configs/fusion.yaml` — fusion baseline 목록, train / eval split
  규칙

새 plug-in (e.g. entity extractor) 가 config 항목을 가지면 위 yaml 중 하나에
sub-section 으로 추가하고 schema doc 을 본 파일에 갱신할 것.
