# 환각 탐지 졸업논문 아카이브

이 repo는 환각(hallucination) 탐지 주제 졸업논문의 최종 산출물만 모아둔 아카이브입니다.

포함 내용:

- 논문 LaTeX 원본 + 컴파일된 PDF
- 논문에 반영된 실험 결과 JSON

제외 내용:

- 실험 실행 코드
- 실험 실행용 패키지 소스 코드
- 레퍼런스 조사 노트, 회의 메모 등 연구 보조 자료

## repo 구조

```text
hallucination-graduation-thesis/
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
