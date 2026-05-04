# Fusion baseline evaluation

Primary split: deterministic leave-one-dataset-out. This keeps dataset-level visibility front and center and avoids a pooled random split as the headline.

True Boltzmann Semantic Energy stays explicit. If full logits are unavailable, Energy-dependent baselines remain listed with null metrics and rerun flags instead of quietly using `semantic_energy_proxy`.

## Aggregate baseline table

| baseline | status | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SE-only | ok | 0.675244 | 0.716412 | 0.660000 | 0.683239 | 0.804348 | 0.738863 |  |
| Energy-only | unavailable | null | null | null | null | null | null | full_logits_required |
| corpus-risk-only | ok | 0.499133 | 0.603121 | 0.591000 | 0.595745 | 0.983278 | 0.741956 |  |
| fixed linear 0.1/0.9 | unavailable | null | null | null | null | null | null | full_logits_required |
| fixed linear 0.5/0.5 | unavailable | null | null | null | null | null | null | full_logits_required |
| fixed linear 0.9/0.1 | unavailable | null | null | null | null | null | null | full_logits_required |
| hard cascade | unavailable | null | null | null | null | null | null | full_logits_required |
| old coverage-adaptive baseline | unavailable | null | null | null | null | null | null | full_logits_required |
| learned fusion without corpus | ok | 0.477294 | 0.604289 | 0.598000 | 0.598000 | 1.000000 | 0.748436 |  |
| learned fusion with corpus | ok | 0.444207 | 0.593657 | 0.521000 | 0.566332 | 0.849498 | 0.679599 |  |

## Learned fusion comparison

### aggregate

Observed delta, with corpus minus without corpus: AUROC -0.033087, AUPRC -0.010632, F1 -0.068837.

### per_dataset

| dataset | status | ΔAUROC | ΔAUPRC | ΔF1 | reason |
| --- | --- | ---: | ---: | ---: | --- |
| HaluEval-QA | ok | 0.063182 | 0.044797 | 0.000000 |  |
| HaluEval-dialogue | ok | 0.008865 | 0.001833 | 0.000000 |  |
| Natural Questions | ok | -0.000306 | 0.004184 | -0.004070 |  |
| TriviaQA | ok | -0.001616 | 0.006867 | 0.000000 |  |
| TruthfulQA | ok | -0.091169 | -0.031149 | -0.213524 |  |
