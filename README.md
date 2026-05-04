# 환각 탐지 졸업논문 아카이브

이 repo는 환각(hallucination) 탐지 주제 졸업논문과, 논문 수치를 재생성하기 위한 `uv` 기반 실험 파이프라인을 함께 관리합니다.

포함 내용:

- 논문 LaTeX 원본 + 컴파일된 PDF
- 논문에 반영된 실험 결과 JSON

제외 내용:

- 대용량 원본 데이터셋 캐시
- 대용량 full-logits 산출물
- 로컬 Elasticsearch/Infini-gram 호환 인덱스 데이터

## 실험 파이프라인

실험 실행 순서와 thesis-valid gate는 `experiments/PIPELINE.md`에 고정되어 있습니다.

```bash
uv sync --group generation
uv run python experiments/scripts/validate_pipeline_contract.py experiments/PIPELINE.md
uv run python experiments/scripts/validate_paper_feature_alignment.py --formulas experiments/configs/formulas.yaml --notes experiments/literature/formula_notes.md --pipeline experiments/PIPELINE.md
uv run python experiments/scripts/run_pipeline.py --mode smoke --dry-run --out experiments/results/runs
```

현재 작은 upstream/cache/proxy 결과는 파일럿 진단용이며, 논문 최종 근거는 `full-core` 또는 명시적으로 승격된 `full-extended` 실행이 모든 gate를 통과한 뒤에만 사용할 수 있습니다.

## repo 구조

```text
hallucination-graduation-thesis/
├── experiments/
│   ├── PIPELINE.md
│   ├── configs/
│   ├── scripts/
│   └── results/
├── results/
│   ├── analysis.json
│   └── results.json
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
