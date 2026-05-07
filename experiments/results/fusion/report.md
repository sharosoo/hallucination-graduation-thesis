# Fusion baseline evaluation

Primary split: deterministic leave-one-dataset-out. This keeps dataset-level visibility front and center and avoids a pooled random split as the headline.

True Boltzmann Semantic Energy stays explicit. If full logits are unavailable, Energy-dependent baselines remain listed with null metrics and rerun flags instead of quietly using `semantic_energy_proxy`.

## Interpretation caveat

The aggregate table below is a candidate-row headline comparison only. It is not sufficient for the intended Semantic Entropy / Semantic Energy thesis claim because `semantic_entropy` is prompt-level and broadcast to both paired candidate rows, while Energy diagnostics are candidate-level and intended for the LOW_DIVERSITY matched-pair slice. Do not interpret SE-only AUROC 0.5 as SE failure. Do not interpret Energy-only aggregate AUROC as the final Energy result. The next required analysis is paired/type-routed reassessment by `prompt_id`.

## Aggregate baseline table

| baseline | status | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SE-only | ok | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 1.000000 | 0.666667 |  |
| Energy-only | ok | 0.494768 | 0.464656 | 0.515305 | 0.508028 | 0.968530 | 0.666469 |  |
| corpus-risk-only | ok | 0.517674 | 0.468756 | 0.500000 | 0.500000 | 1.000000 | 0.666667 |  |
| fixed linear 0.1/0.9 | ok | 0.493906 | 0.471318 | 0.540843 | 0.525431 | 0.843852 | 0.647618 |  |
| fixed linear 0.5/0.5 | ok | 0.495696 | 0.485485 | 0.509544 | 0.505335 | 0.904041 | 0.648292 |  |
| fixed linear 0.9/0.1 | ok | 0.499486 | 0.492577 | 0.507911 | 0.504397 | 0.907481 | 0.648400 |  |
| hard cascade | ok | 0.501818 | 0.500486 | 0.511092 | 0.505842 | 0.960447 | 0.662672 |  |
| old coverage-adaptive baseline | ok | 0.672976 | 0.590870 | 0.587962 | 0.548460 | 0.995529 | 0.707269 |  |
| learned fusion without corpus | ok | 0.373430 | 0.419691 | 0.520808 | 0.511208 | 0.949097 | 0.664499 |  |
| learned fusion with corpus | ok | 0.343537 | 0.406484 | 0.491144 | 0.495025 | 0.881169 | 0.633923 |  |

## Learned fusion comparison

### aggregate

Observed delta, with corpus minus without corpus: AUROC -0.029893, AUPRC -0.013207, F1 -0.030576.

### per_dataset

| dataset | status | ΔAUROC | ΔAUPRC | ΔF1 | reason |
| --- | --- | ---: | ---: | ---: | --- |
| HaluEval-QA | ok | -0.035976 | -0.011923 | -0.009774 |  |
| TruthfulQA | ok | -0.017050 | 0.005063 | -0.012554 |  |
