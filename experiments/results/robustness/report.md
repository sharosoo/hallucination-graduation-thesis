# Robustness report

Observed robustness summary for the current fusion outputs. This report uses neutral wording whenever a bootstrap interval crosses zero.

## Bootstrap deltas

| candidate | reference | metric | observed_delta | ci_95_lower | ci_95_upper | crosses_zero | statistically_significant |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| learned fusion with corpus | learned fusion without corpus | auroc | -0.033087 | -0.053205 | -0.011724 | False | True |
| learned fusion with corpus | learned fusion without corpus | auprc | -0.010632 | -0.031273 | 0.009852 | True | False |
| learned fusion with corpus | SE-only | auroc | -0.231038 | -0.259493 | -0.200704 | False | True |
| learned fusion with corpus | SE-only | auprc | -0.122755 | -0.147472 | -0.095634 | False | True |
| learned fusion without corpus | SE-only | auroc | -0.197950 | -0.219158 | -0.177051 | False | True |
| learned fusion without corpus | SE-only | auprc | -0.112123 | -0.126082 | -0.096216 | False | True |
| learned fusion with corpus | corpus-risk-only | auroc | -0.054926 | -0.101081 | -0.008134 | False | True |
| learned fusion with corpus | corpus-risk-only | auprc | -0.009464 | -0.043211 | 0.027163 | True | False |

## Threshold sensitivity

Threshold sensitivity uses alternative low/high Semantic Entropy cutoffs around 0.1 and 0.5, then recomputes type-specific subsets.

| low_cutoff | high_cutoff | HIGH_DIVERSITY | LOW_DIVERSITY | AMBIGUOUS_INCORRECT | NORMAL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.4 | 481 | 117 | 0 | 402 |
| 0.05 | 0.5 | 481 | 117 | 0 | 402 |
| 0.05 | 0.6 | 410 | 117 | 71 | 402 |
| 0.1 | 0.4 | 481 | 117 | 0 | 402 |
| 0.1 | 0.5 | 481 | 117 | 0 | 402 |
| 0.1 | 0.6 | 410 | 117 | 71 | 402 |
| 0.2 | 0.4 | 481 | 117 | 0 | 402 |
| 0.2 | 0.5 | 481 | 117 | 0 | 402 |
| 0.2 | 0.6 | 410 | 117 | 71 | 402 |

## Selective-risk framing

Phillips-inspired evaluation only. This is a selective-prediction framing check, not a probe-paper reproduction.

| method | AURC | overall_error_rate | low_confidence_error_rate | error_concentration_ratio |
| --- | ---: | ---: | ---: | ---: |
| SE-only | 0.440037 | 0.462000 | 0.350000 | 0.757576 |
| corpus-risk-only | 0.494120 | 0.470000 | 0.375000 | 0.797872 |
| learned fusion with corpus | 0.498079 | 0.509000 | 0.520000 | 1.021611 |
| learned fusion without corpus | 0.444304 | 0.462000 | 0.585000 | 1.266234 |

## Caveats

- Phillips is used only for selective-prediction framing. No probe-style correctness or hidden-representation feature is implemented here.
- True Energy robustness remains unavailable because current rows do not carry row-level full logits. semantic_energy_proxy is not treated as thesis-valid Energy evidence.
- The current feature table contains no AMBIGUOUS_INCORRECT rows, so gray-zone threshold sensitivity reports absence explicitly instead of fabricating that slice.
- All current rows mark true Boltzmann Energy as unavailable, so Energy-dependent robustness baselines stay null or rerun-required in this report.
