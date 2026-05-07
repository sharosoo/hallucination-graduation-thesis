# Paired/type-routed reanalysis summary

This note re-evaluates the current full paired Qwen2.5-3B answer-only feature table at the evaluation granularity required by the thesis claims. It uses `experiments/results/features.parquet` only; no S1--S7 generation or feature recomputation is required.

## Archived diagnostic claims

1. **Semantic Entropy was previously treated as a high-diversity routing signal.**
   This remains useful only as archived diagnostic framing. The redesigned thesis-valid path requires N=10 NLI likelihood SE and evaluates it across corpus-support bins.
2. **Semantic Energy/logit diagnostics were previously treated as low-diversity candidate-level signals.**
   Current Energy-family scores should be read as candidate-level diagnostics unless multi-generation semantic clustering and cluster-level energy aggregation are implemented.
3. **Corpus-grounded entity statistics were previously used in selective fusion.**
   The redesigned framing uses entity frequency and entity-pair co-occurrence as continuous corpus-support axes that condition metric reliability, not as direct hallucination labels.

## Reanalysis results

The feature table contains 5,815 valid prompt pairs and 11,630 candidate rows. Semantic Entropy is identical within all 5,815 prompt pairs, so SE-only candidate-row AUROC is structurally uninformative for within-pair candidate ranking.

LOW_DIVERSITY contains 1,078 prompt pairs. Paired win-rate means the hallucinated candidate receives a higher risk score than the correct candidate in the same prompt pair.

| Signal | Risk orientation | LOW_DIVERSITY win-rate | 95% bootstrap CI | LOW_DIVERSITY row AUROC |
| --- | --- | ---: | ---: | ---: |
| `semantic_energy_boltzmann` | higher is riskier | 0.5798 | [0.5501, 0.6095] | 0.5529 |
| `logit_variance` | higher is riskier | 0.7078 | [0.6800, 0.7347] | 0.6787 |
| `-confidence_margin` | lower margin is riskier | 0.8052 | [0.7811, 0.8284] | 0.7704 |
| `-mean_negative_log_probability` | lower NLL / overconfidence is riskier | 0.5872 | [0.5584, 0.6160] | 0.6024 |
| `corpus_risk_only` | higher is riskier | 0.4917 | [0.4610, 0.5204] | 0.4875 |

Dataset caveat: LOW_DIVERSITY is dominated by HaluEval-QA (1,066 pairs) and has only 12 TruthfulQA pairs. Therefore LOW_DIVERSITY Energy/logit claims should be stated as primarily HaluEval-QA-supported unless additional paired datasets are added.

## Claim assessment

Defensible:

- The paired dataset and label construction are thesis-valid: annotation-backed candidate pairs, no LLM judge, no heuristic answer matching.
- Semantic Entropy should be evaluated as a prompt-level type-routing signal, not as a candidate-row ranking signal.
- In LOW_DIVERSITY pairs, logit diagnostics show directional candidate-level signal, especially `-confidence_margin` and `logit_variance`.
- Corpus fusion did not improve the current aggregate learned-fusion result; this should be reported as a negative result.

Not defensible from the current artifacts:

- Claiming that the archived global fusion method improves overall hallucination detection.
- Claiming corpus-grounded features improve aggregate learned fusion.
- Claiming Semantic Energy is a strong universal hallucination detector.
- Claiming LOW_DIVERSITY results generalize equally across TruthfulQA and HaluEval-QA.

## Recommended thesis framing

The thesis should be framed as a diagnostic paired study and then extended into a corpus-axis reliability study: prompt-level Semantic Entropy, paper-faithful Semantic Energy, candidate-level logit diagnostics, and corpus-support axes operate at different granularities. The positive current result is not aggregate fusion performance; it is an archived directional diagnostic finding that motivates corpus-bin reliability analysis.
