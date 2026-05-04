# QuCo-RAG vs PC Probe

## Scope

This note exists to keep corpus-stat signals and probe-style signals conceptually separate.

## Core distinction

- QuCo-RAG-style features in this repo are **corpus-grounded proxies** derived from external corpus statistics such as entity frequency and entity-pair co-occurrence (QuCo-RAG arXiv page 1 abstract; pages 3-4, Section 3.2 and Section 3.3).
- PC Probe is a **model-internal supervised signal** family and is reference-only here (Phillips et al. arXiv page 1 abstract; page 2, Section 2.2).
- Any claim that QuCo-style corpus statistics are objective or model-agnostic must be read with an **objective/model-agnostic caveat**: QuCo-RAG calls the approach corpus-grounded and practically model-agnostic in the context of shared web-scale corpora (QuCo-RAG arXiv page 1 abstract; page 6, “Why Proxy Corpus Works”), but this is still about the chosen external corpus snapshot, not proof of model pretraining exposure, latent knowledge, or hidden-state causality.

## Guardrails

- Do not claim QuCo-RAG proves model exposure to facts.
- Do not implement PC Probe or any hidden-state feature in this experiment layer.
- Do not describe corpus counts as interchangeable with internal confidence or probe outputs.

## Citation status

- QuCo-RAG should retain a preprint or status caveat unless archival publication is directly verified and recorded in `literature_manifest.json`.
- Phillips/PC Probe remains reference-only in this repo regardless of citation status.

## Thesis-safe interpretation

- Use QuCo-RAG to justify corpus-frequency and corpus co-occurrence signals as external evidence proxies, not as hidden-state or confidence substitutes.
- Use Phillips et al. only to justify reference-only framing around selective prediction, deployment-facing risk metrics, and the possibility of confidently wrong low-entropy errors.
