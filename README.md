# 환각 탐지 졸업논문 아카이브

이 repo는 환각(hallucination) 탐지 주제 졸업논문과, 논문 수치를 재생성하기 위한 `uv` 기반 실험 파이프라인을 함께 관리합니다.

현재 실험 논지는 RAG 시스템 구축이 아니라 **corpus 조건 축에 따른 hallucination metric reliability 분석**입니다. QuCo-RAG에서 영감을 받은 entity frequency와 entity-pair co-occurrence를 continuous corpus-support axis로 만들고, 그 축의 bin마다 Semantic Entropy, Semantic Energy, likelihood/logit diagnostics, condition-aware fusion의 신뢰도가 어떻게 달라지는지 평가합니다.

포함 내용:

- 논문 LaTeX 원본 + 컴파일된 PDF
- 논문에 반영하거나 반영 예정인 실험 결과 JSON/Parquet 요약
- paper-faithful feature pipeline 계약 문서

제외 내용:

- 대용량 원본 데이터셋 캐시
- 대용량 full-logits 산출물
- 로컬 Elasticsearch/Infini-gram 호환 인덱스 데이터

## 실험 파이프라인

실험 실행 순서와 thesis-valid gate는 `experiments/PIPELINE.md`에 고정되어 있습니다.

```bash
uv sync --group generation
uv run python experiments/scripts/validate_architecture.py
uv run python experiments/scripts/run_pipeline.py --dry-run --out experiments/results/runs
```

실제 모델 실행은 `experiments/configs/generation.yaml`의 `model.model_name` / `model.tokenizer_name`에 고정된 Qwen2.5 계열 causal LM과 같은 계열 tokenizer를 사용합니다. CUDA GPU가 있는 환경에서는 `uv sync --group generation` 후 `uv run python experiments/scripts/run_pipeline.py --execute --out <large-nvme-run-root>`로 live run을 시작합니다. full-vocabulary logits는 JSON에 직접 저장하지 않고 같은 stem의 `.full_logits.parquet` sidecar에 저장합니다. full-logits 저장량은 모델 파라미터 수보다 tokenizer vocab size와 token position 수에 좌우됩니다.

Semantic Entropy용 prompt free sampling은 answer-only protocol로 고정합니다. redesigned thesis-valid artifact는 prompt당 N=10 valid answer samples, DeBERTa-family NLI semantic clustering, likelihood-based cluster probability fields를 포함해야 합니다.

Semantic Energy는 current candidate-level `-logsumexp` diagnostic만으로 paper-faithful하다고 쓰지 않습니다. 최종 구현은 multiple generated answers, semantic clusters, selected-token logit-derived energy, cluster-level aggregation을 포함해야 합니다.

Corpus feature는 hallucination label이 아니라 reliability conditioning axis입니다. Entity frequency와 entity-pair co-occurrence는 Infini-gram-compatible count backend 또는 고정 corpus snapshot provenance를 가져야 하며, Elasticsearch/BM25 retrieval score로 대체하지 않습니다.

## 현재 결과 해석 상태

현재 repo-local evidence는 full paired Qwen2.5-3B answer-only run의 S7 feature table, S8 fusion, S9 robustness까지 포함합니다. 이 결과는 baseline/preliminary evidence로 유지합니다. 다만 final thesis claim은 새 `experiments/PIPELINE.md` 계약에 따라 N=10 NLI likelihood SE, paper-faithful Semantic Energy, QuCo-style corpus continuous axis, corpus-bin reliability analysis, condition-aware fusion을 다시 산출한 뒤 작성해야 합니다.

기존 분석에서 중요한 관찰은 다음입니다.

- Semantic Entropy는 prompt-level 신호라 같은 prompt의 correct/hallucinated candidate row에 동일하게 broadcast된다.
- Candidate-row aggregate SE-only AUROC는 구조적으로 0.5가 될 수 있으므로 SE 실패의 직접 증거가 아니다.
- Candidate-level logit diagnostics는 corpus-axis bin slice 별로 paired win-rate와 paired delta를 보고하여 어떤 corpus support 조건에서 신호가 가장 안정적인지 본다.
- Corpus-only direct classifier 성능이 약하더라도 corpus axis가 metric reliability를 나누는 조건 축인지 여부는 별도 검증 대상이다.

## repo 구조

```text
hallucination-graduation-thesis/
├── experiments/
│   ├── PIPELINE.md
│   ├── README.md
│   ├── OVERVIEW.md
│   ├── RESEARCH_PLAN.md
│   ├── configs/
│   ├── domain/
│   ├── ports/
│   ├── adapters/
│   ├── application/
│   ├── scripts/
│   ├── literature/
│   ├── manifests/
│   └── results/
└── thesis/
    ├── main.tex
    ├── main.pdf
    ├── figures_tikz.tex
    ├── snuthesis.cls
    ├── snutocstyle.tex
    └── sections/experiment_method.tex
```

## 논문 PDF 다시 빌드

```bash
cd thesis
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
