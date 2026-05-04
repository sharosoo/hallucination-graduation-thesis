# Semantic Energy single-cluster and Zero-SE claims

## Claim handling rule

Unsupported numeric claims about single-cluster or Zero-SE behavior must not be cited as facts unless they are tied to a saved PDF location.

## Current status

- The paper explicitly claims on arXiv page 2, Section 1, final bullet that Semantic Energy improves AUROC by **more than 13%** over Semantic Entropy in cases where Semantic Entropy is confident.
- Table 2 on arXiv page 7 reports **single-cluster** AUROC gains from **50.0% to 66.7%** for Qwen3-8B on CSQA, **50.0% to 62.1%** for Qwen3-8B on TriviaQA, **50.0% to 58.9%** for ERNIE-21B-A3B on CSQA, and **50.0% to 65.8%** for ERNIE-21B-A3B on TriviaQA.
- The paragraph immediately below Table 2 on arXiv page 7 states that Semantic Energy achieves an average AUROC improvement of more than 13% in the cases where Semantic Entropy is confident.

## Guardrail on what remains unsupported

- Any claim about **Zero-SE** specifically, rather than the paper’s **single-cluster** setup, is `UNVERIFIED_DO_NOT_CITE` unless a later note ties Zero-SE wording to an exact page/table/section reference.
- Any transfer of the Table 2 numbers to this thesis’s datasets, labels, or methods is `UNVERIFIED_DO_NOT_CITE` unless new experiments in this repo reproduce them.

## Allowed downstream use

- Semantic Energy may still be used as the motivating source for low-diversity hallucination handling with a preprint/submission caveat.
- Numeric thesis prose must wait for direct PDF verification.
