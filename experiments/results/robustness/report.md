# Robustness report

Observed robustness summary for the current fusion outputs. This report uses neutral wording whenever a bootstrap interval crosses zero.

## Evaluation-granularity caveat

These bootstrap deltas apply to the current candidate-row fusion outputs. They are useful as robustness checks for the generated S8 table, but they do not settle the intended SE/Energy complementarity claim. SE must be interpreted as a prompt-level/type-level diagnostic, and Energy must be re-evaluated with LOW_DIVERSITY matched-pair metrics and prompt-grouped bootstrap CIs.

## Bootstrap deltas

| candidate | reference | metric | observed_delta | ci_95_lower | ci_95_upper | crosses_zero | statistically_significant |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| learned fusion with corpus | learned fusion without corpus | auroc | -0.029893 | -0.034938 | -0.024429 | False | True |
| learned fusion with corpus | learned fusion without corpus | auprc | -0.013207 | -0.016573 | -0.010180 | False | True |
| learned fusion with corpus | SE-only | auroc | -0.156463 | -0.165031 | -0.147889 | False | True |
| learned fusion with corpus | SE-only | auprc | -0.093516 | -0.098168 | -0.088723 | False | True |
| learned fusion without corpus | SE-only | auroc | -0.126570 | -0.135127 | -0.118072 | False | True |
| learned fusion without corpus | SE-only | auprc | -0.080309 | -0.084451 | -0.075479 | False | True |
| learned fusion with corpus | corpus-risk-only | auroc | -0.174137 | -0.190792 | -0.157707 | False | True |
| learned fusion with corpus | corpus-risk-only | auprc | -0.062272 | -0.071127 | -0.053147 | False | True |

## Threshold sensitivity

Threshold sensitivity uses alternative low/high Semantic Entropy cutoffs around 0.1 and 0.5, then recomputes type-specific subsets.

| low_cutoff | high_cutoff | HIGH_DIVERSITY | LOW_DIVERSITY | AMBIGUOUS_INCORRECT | NORMAL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.4 | 4737 | 1078 | 0 | 5815 |
| 0.05 | 0.5 | 4737 | 1078 | 0 | 5815 |
| 0.05 | 0.6 | 3477 | 1078 | 1260 | 5815 |
| 0.1 | 0.4 | 4737 | 1078 | 0 | 5815 |
| 0.1 | 0.5 | 4737 | 1078 | 0 | 5815 |
| 0.1 | 0.6 | 3477 | 1078 | 1260 | 5815 |
| 0.2 | 0.4 | 4737 | 1078 | 0 | 5815 |
| 0.2 | 0.5 | 4737 | 1078 | 0 | 5815 |
| 0.2 | 0.6 | 3477 | 1078 | 1260 | 5815 |

## Selective-risk framing

Phillips-inspired evaluation only. This is a selective-prediction framing check, not a probe-paper reproduction.

| method | AURC | overall_error_rate | low_confidence_error_rate | error_concentration_ratio |
| --- | ---: | ---: | ---: | ---: |
| SE-only | 0.449945 | 0.500000 | 0.500000 | 1.000000 |
| corpus-risk-only | 0.575875 | 0.502236 | 0.256664 | 0.511043 |
| learned fusion with corpus | 0.473483 | 0.494325 | 0.439381 | 0.888850 |
| learned fusion without corpus | 0.453768 | 0.481857 | 0.408426 | 0.847609 |

## Caveats

- Phillips is used only for selective-prediction framing. No probe-style correctness or hidden-representation feature is implemented here.
- Energy-dependent robustness uses feature-table candidate-level Energy/logit diagnostics; these are not paper-faithful Semantic Energy claims.
- The current feature table contains no AMBIGUOUS_INCORRECT rows, so gray-zone threshold sensitivity reports absence explicitly instead of fabricating that slice.
