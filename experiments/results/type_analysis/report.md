# Type-specific signal analysis

This report keeps unavailable Energy visible as unavailable. `semantic_energy_boltzmann` is never replaced with proxy selected-logit energy.

## HaluEval-QA

### overall

All rows. Positive class is any non-NORMAL hallucination label.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 200 | 21 | 179 | 0.505720 |  | 0.114961 |  |
| corpus_risk_only | 200 | 21 | 179 | 0.437350 |  | 0.099043 |  |
| semantic_energy_boltzmann | 200 | 21 | 179 | null | full_logits_required | null | full_logits_required |

### high_diversity_vs_normal

Rows labeled HIGH_DIVERSITY or NORMAL only. Positive class is HIGH_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 188 | 9 | 179 | 0.800124 |  | 0.128243 |  |
| corpus_risk_only | 188 | 9 | 179 | 0.415270 |  | 0.043949 |  |
| semantic_energy_boltzmann | 188 | 9 | 179 | null | full_logits_required | null | full_logits_required |

### low_diversity_vs_normal

Rows labeled LOW_DIVERSITY or NORMAL only. Positive class is LOW_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 191 | 12 | 179 | 0.284916 |  | 0.062827 |  |
| corpus_risk_only | 191 | 12 | 179 | 0.453911 |  | 0.061724 |  |
| semantic_energy_boltzmann | 191 | 12 | 179 | null | full_logits_required | null | full_logits_required |

### zero_se_vs_normal

Exact-zero Semantic Entropy bucket only. Positive class is any non-NORMAL row inside the zero-SE bucket.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 114 | 12 | 102 | 0.500000 |  | 0.105263 |  |
| corpus_risk_only | 114 | 12 | 102 | 0.450572 |  | 0.101588 |  |
| semantic_energy_boltzmann | 114 | 12 | 102 | null | full_logits_required | null | full_logits_required |


## HaluEval-dialogue

### overall

All rows. Positive class is any non-NORMAL hallucination label.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 200 | 188 | 12 | 0.598626 |  | 0.954389 |  |
| corpus_risk_only | 200 | 188 | 12 | 0.462101 |  | 0.935572 |  |
| semantic_energy_boltzmann | 200 | 188 | 12 | null | full_logits_required | null | full_logits_required |

### high_diversity_vs_normal

Rows labeled HIGH_DIVERSITY or NORMAL only. Positive class is HIGH_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 156 | 144 | 12 | 0.717882 |  | 0.958786 |  |
| corpus_risk_only | 156 | 144 | 12 | 0.465567 |  | 0.917960 |  |
| semantic_energy_boltzmann | 156 | 144 | 12 | null | full_logits_required | null | full_logits_required |

### low_diversity_vs_normal

Rows labeled LOW_DIVERSITY or NORMAL only. Positive class is LOW_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 56 | 44 | 12 | 0.208333 |  | 0.785714 |  |
| corpus_risk_only | 56 | 44 | 12 | 0.450758 |  | 0.769550 |  |
| semantic_energy_boltzmann | 56 | 44 | 12 | null | full_logits_required | null | full_logits_required |

### zero_se_vs_normal

Exact-zero Semantic Entropy bucket only. Positive class is any non-NORMAL row inside the zero-SE bucket.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 49 | 44 | 5 | 0.500000 |  | 0.897959 |  |
| corpus_risk_only | 49 | 44 | 5 | 0.509091 |  | 0.899629 |  |
| semantic_energy_boltzmann | 49 | 44 | 5 | null | full_logits_required | null | full_logits_required |


## Natural Questions

### overall

All rows. Positive class is any non-NORMAL hallucination label.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 200 | 114 | 86 | 0.635557 |  | 0.659113 |  |
| corpus_risk_only | 200 | 114 | 86 | 0.492809 |  | 0.566380 |  |
| semantic_energy_boltzmann | 200 | 114 | 86 | null | full_logits_required | null | full_logits_required |

### high_diversity_vs_normal

Rows labeled HIGH_DIVERSITY or NORMAL only. Positive class is HIGH_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 189 | 103 | 86 | 0.689151 |  | 0.668630 |  |
| corpus_risk_only | 189 | 103 | 86 | 0.499266 |  | 0.544577 |  |
| semantic_energy_boltzmann | 189 | 103 | 86 | null | full_logits_required | null | full_logits_required |

### low_diversity_vs_normal

Rows labeled LOW_DIVERSITY or NORMAL only. Positive class is LOW_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 97 | 11 | 86 | 0.133721 |  | 0.113402 |  |
| corpus_risk_only | 97 | 11 | 86 | 0.432347 |  | 0.101110 |  |
| semantic_energy_boltzmann | 97 | 11 | 86 | null | full_logits_required | null | full_logits_required |

### zero_se_vs_normal

Exact-zero Semantic Entropy bucket only. Positive class is any non-NORMAL row inside the zero-SE bucket.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 34 | 11 | 23 | 0.500000 |  | 0.323529 |  |
| corpus_risk_only | 34 | 11 | 23 | 0.409091 |  | 0.287074 |  |
| semantic_energy_boltzmann | 34 | 11 | 23 | null | full_logits_required | null | full_logits_required |


## TriviaQA

### overall

All rows. Positive class is any non-NORMAL hallucination label.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 200 | 110 | 90 | 0.668687 |  | 0.657430 |  |
| corpus_risk_only | 200 | 110 | 90 | 0.510253 |  | 0.555376 |  |
| semantic_energy_boltzmann | 200 | 110 | 90 | null | full_logits_required | null | full_logits_required |

### high_diversity_vs_normal

Rows labeled HIGH_DIVERSITY or NORMAL only. Positive class is HIGH_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 181 | 91 | 90 | 0.763065 |  | 0.679860 |  |
| corpus_risk_only | 181 | 91 | 90 | 0.505433 |  | 0.505746 |  |
| semantic_energy_boltzmann | 181 | 91 | 90 | null | full_logits_required | null | full_logits_required |

### low_diversity_vs_normal

Rows labeled LOW_DIVERSITY or NORMAL only. Positive class is LOW_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 109 | 19 | 90 | 0.216667 |  | 0.174312 |  |
| corpus_risk_only | 109 | 19 | 90 | 0.533333 |  | 0.184466 |  |
| semantic_energy_boltzmann | 109 | 19 | 90 | null | full_logits_required | null | full_logits_required |

### zero_se_vs_normal

Exact-zero Semantic Entropy bucket only. Positive class is any non-NORMAL row inside the zero-SE bucket.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 58 | 19 | 39 | 0.500000 |  | 0.327586 |  |
| corpus_risk_only | 58 | 19 | 39 | 0.538462 |  | 0.345455 |  |
| semantic_energy_boltzmann | 58 | 19 | 39 | null | full_logits_required | null | full_logits_required |


## TruthfulQA

### overall

All rows. Positive class is any non-NORMAL hallucination label.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 200 | 165 | 35 | 0.619048 |  | 0.872733 |  |
| corpus_risk_only | 200 | 165 | 35 | 0.443810 |  | 0.805987 |  |
| semantic_energy_boltzmann | 200 | 165 | 35 | null | full_logits_required | null | full_logits_required |

### high_diversity_vs_normal

Rows labeled HIGH_DIVERSITY or NORMAL only. Positive class is HIGH_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 169 | 134 | 35 | 0.729211 |  | 0.883776 |  |
| corpus_risk_only | 169 | 134 | 35 | 0.440192 |  | 0.768993 |  |
| semantic_energy_boltzmann | 169 | 134 | 35 | null | full_logits_required | null | full_logits_required |

### low_diversity_vs_normal

Rows labeled LOW_DIVERSITY or NORMAL only. Positive class is LOW_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 66 | 31 | 35 | 0.142857 |  | 0.469697 |  |
| corpus_risk_only | 66 | 31 | 35 | 0.459447 |  | 0.455504 |  |
| semantic_energy_boltzmann | 66 | 31 | 35 | null | full_logits_required | null | full_logits_required |

### zero_se_vs_normal

Exact-zero Semantic Entropy bucket only. Positive class is any non-NORMAL row inside the zero-SE bucket.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 41 | 31 | 10 | 0.500000 |  | 0.756098 |  |
| corpus_risk_only | 41 | 31 | 10 | 0.483871 |  | 0.744761 |  |
| semantic_energy_boltzmann | 41 | 31 | 10 | null | full_logits_required | null | full_logits_required |


## AGGREGATE

### overall

All rows. Positive class is any non-NORMAL hallucination label.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 1000 | 598 | 402 | 0.675244 |  | 0.716412 |  |
| corpus_risk_only | 1000 | 598 | 402 | 0.499133 |  | 0.603121 |  |
| semantic_energy_boltzmann | 1000 | 598 | 402 | null | full_logits_required | null | full_logits_required |

### high_diversity_vs_normal

Rows labeled HIGH_DIVERSITY or NORMAL only. Positive class is HIGH_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 883 | 481 | 402 | 0.785338 |  | 0.745215 |  |
| corpus_risk_only | 883 | 481 | 402 | 0.503155 |  | 0.552188 |  |
| semantic_energy_boltzmann | 883 | 481 | 402 | null | full_logits_required | null | full_logits_required |

### low_diversity_vs_normal

Rows labeled LOW_DIVERSITY or NORMAL only. Positive class is LOW_DIVERSITY.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 519 | 117 | 402 | 0.222637 |  | 0.225434 |  |
| corpus_risk_only | 519 | 117 | 402 | 0.482598 |  | 0.223311 |  |
| semantic_energy_boltzmann | 519 | 117 | 402 | null | full_logits_required | null | full_logits_required |

### zero_se_vs_normal

Exact-zero Semantic Entropy bucket only. Positive class is any non-NORMAL row inside the zero-SE bucket.

| signal | rows | positives | negatives | AUROC | AUROC reason | AUPRC | AUPRC reason |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| semantic_entropy | 296 | 117 | 179 | 0.500000 |  | 0.395270 |  |
| corpus_risk_only | 296 | 117 | 179 | 0.496514 |  | 0.400061 |  |
| semantic_energy_boltzmann | 296 | 117 | 179 | null | full_logits_required | null | full_logits_required |


## Signal notes

- `semantic_entropy`: Semantic Entropy score from the feature table.
- `corpus_risk_only`: Corpus-only risk baseline from cached or proxy corpus features.
- `semantic_energy_boltzmann`: True Boltzmann Semantic Energy. Remains unavailable until full logits are regenerated.
