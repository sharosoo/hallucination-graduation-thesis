# SE-binned feature reanalysis

- pairs: 5815
- unique SE values: 7
- binning: Fixed threshold-focused SE intervals: exactly zero, (0,0.1), [0.1,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0), [1.0,1.25), [1.25,1.5), [1.5,1.75), [1.75,2.0), [2.0,2.5), [2.5,inf). Empty bins omitted.

## semantic_energy_boltzmann

| SE bin | n pairs | hallucinated labels | AUROC | AUPRC | paired win-rate |
| --- | ---: | --- | ---: | ---: | ---: |
| SE_eq_0 | 1078 | {'LOW_DIVERSITY': 1078} | 0.5529 | 0.5108 | 0.5798 |
| SE_[0.5,0.75) | 1559 | {'HIGH_DIVERSITY': 1559} | 0.5187 | 0.4801 | 0.5420 |
| SE_[0.75,1.0) | 1026 | {'HIGH_DIVERSITY': 1026} | 0.4898 | 0.4588 | 0.4942 |
| SE_[1.0,1.25) | 284 | {'HIGH_DIVERSITY': 284} | 0.4854 | 0.4607 | 0.4507 |
| SE_[1.25,1.5) | 893 | {'HIGH_DIVERSITY': 893} | 0.4538 | 0.4362 | 0.4614 |
| SE_[1.5,1.75) | 975 | {'HIGH_DIVERSITY': 975} | 0.4467 | 0.4512 | 0.4636 |

## logit_variance

| SE bin | n pairs | hallucinated labels | AUROC | AUPRC | paired win-rate |
| --- | ---: | --- | ---: | ---: | ---: |
| SE_eq_0 | 1078 | {'LOW_DIVERSITY': 1078} | 0.6787 | 0.5867 | 0.7078 |
| SE_[0.5,0.75) | 1559 | {'HIGH_DIVERSITY': 1559} | 0.6512 | 0.5654 | 0.6684 |
| SE_[0.75,1.0) | 1026 | {'HIGH_DIVERSITY': 1026} | 0.6035 | 0.5273 | 0.6355 |
| SE_[1.0,1.25) | 284 | {'HIGH_DIVERSITY': 284} | 0.6646 | 0.5953 | 0.7042 |
| SE_[1.25,1.5) | 893 | {'HIGH_DIVERSITY': 893} | 0.5773 | 0.5076 | 0.5812 |
| SE_[1.5,1.75) | 975 | {'HIGH_DIVERSITY': 975} | 0.4845 | 0.4720 | 0.4913 |

## negative_confidence_margin

| SE bin | n pairs | hallucinated labels | AUROC | AUPRC | paired win-rate |
| --- | ---: | --- | ---: | ---: | ---: |
| SE_eq_0 | 1078 | {'LOW_DIVERSITY': 1078} | 0.7704 | 0.6902 | 0.8052 |
| SE_[0.5,0.75) | 1559 | {'HIGH_DIVERSITY': 1559} | 0.6898 | 0.5961 | 0.7197 |
| SE_[0.75,1.0) | 1026 | {'HIGH_DIVERSITY': 1026} | 0.6340 | 0.5421 | 0.6472 |
| SE_[1.0,1.25) | 284 | {'HIGH_DIVERSITY': 284} | 0.6148 | 0.5370 | 0.6338 |
| SE_[1.25,1.5) | 893 | {'HIGH_DIVERSITY': 893} | 0.5436 | 0.4782 | 0.5431 |
| SE_[1.5,1.75) | 975 | {'HIGH_DIVERSITY': 975} | 0.4445 | 0.4385 | 0.4554 |

## negative_mean_negative_log_probability

| SE bin | n pairs | hallucinated labels | AUROC | AUPRC | paired win-rate |
| --- | ---: | --- | ---: | ---: | ---: |
| SE_eq_0 | 1078 | {'LOW_DIVERSITY': 1078} | 0.6024 | 0.5873 | 0.5872 |
| SE_[0.5,0.75) | 1559 | {'HIGH_DIVERSITY': 1559} | 0.6258 | 0.5893 | 0.6023 |
| SE_[0.75,1.0) | 1026 | {'HIGH_DIVERSITY': 1026} | 0.6347 | 0.5765 | 0.6189 |
| SE_[1.0,1.25) | 284 | {'HIGH_DIVERSITY': 284} | 0.6406 | 0.6001 | 0.6268 |
| SE_[1.25,1.5) | 893 | {'HIGH_DIVERSITY': 893} | 0.6363 | 0.5670 | 0.6327 |
| SE_[1.5,1.75) | 975 | {'HIGH_DIVERSITY': 975} | 0.6391 | 0.5867 | 0.6379 |

## corpus_risk_only

| SE bin | n pairs | hallucinated labels | AUROC | AUPRC | paired win-rate |
| --- | ---: | --- | ---: | ---: | ---: |
| SE_eq_0 | 1078 | {'LOW_DIVERSITY': 1078} | 0.4875 | 0.4467 | 0.4917 |
| SE_[0.5,0.75) | 1559 | {'HIGH_DIVERSITY': 1559} | 0.5166 | 0.4655 | 0.5170 |
| SE_[0.75,1.0) | 1026 | {'HIGH_DIVERSITY': 1026} | 0.5678 | 0.5052 | 0.5624 |
| SE_[1.0,1.25) | 284 | {'HIGH_DIVERSITY': 284} | 0.5830 | 0.5148 | 0.5704 |
| SE_[1.25,1.5) | 893 | {'HIGH_DIVERSITY': 893} | 0.4988 | 0.4598 | 0.4815 |
| SE_[1.5,1.75) | 975 | {'HIGH_DIVERSITY': 975} | 0.4825 | 0.4631 | 0.4472 |
