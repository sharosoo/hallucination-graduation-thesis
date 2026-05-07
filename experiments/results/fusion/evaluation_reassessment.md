# Fusion evaluation reassessment

Generated after inspecting the May 7 full paired run artifacts and metric code.

## Current status

S7 feature table, S8 fusion, and S9 robustness are complete and validated for the current repo-local artifacts. However, the current headline S8/S9 candidate-row AUROC tables are not sufficient for the intended Semantic Entropy / Semantic Energy thesis claims.

## Why SE-only row-level AUROC is structurally uninformative

`semantic_entropy` is a prompt-level feature. It is joined into the feature table by `prompt_id`, so each paired prompt contributes one NORMAL candidate and one hallucinated candidate with the same SE value. In this paired-candidate setup, a candidate-row binary AUROC for SE-only is forced toward 0.5 because the positive and negative row in each prompt tie.

Therefore, SE-only aggregate candidate-row AUROC should not be interpreted as evidence that Semantic Entropy is useless. SE should be reported as a prompt-level/type-level signal: high-diversity assignment, SE-bin composition, threshold sensitivity, and high-vs-low diversity separation.

## Why Energy needs low-diversity paired evaluation

Semantic Energy is candidate-level and is intended to complement SE for low-diversity/confident hallucinations. The current aggregate Energy-only AUROC pools all NORMAL rows against all hallucinated rows, which can hide the intended low-diversity within-prompt effect.

The relevant question is matched-pair: within the same prompt, does the hallucinated low-diversity candidate receive higher risk than the correct candidate?

Initial read-only diagnostics on the current feature table found directional signal in the low-diversity paired slice:

- `semantic_energy_boltzmann`: hallucinated > normal in about 57.98% of LOW_DIVERSITY pairs.
- `logit_variance`: hallucinated > normal in about 70.78% of LOW_DIVERSITY pairs.
- `confidence_margin`: hallucinated < normal in about 80.61% of LOW_DIVERSITY pairs, so `-confidence_margin` is the risk-oriented form.
- `mean_negative_log_probability`: hallucinated < normal in about 58.72% of LOW_DIVERSITY pairs, suggesting confident-wrong behavior and requiring explicit orientation checks.

## Claim guidance

Do not claim from the current headline table that corpus learned fusion improves aggregate detection; it does not under the current candidate-row evaluation. Also do not claim that SE/Energy fail conceptually. The safer claim is that the current headline evaluation mixes feature granularities and must be supplemented by type-routed, prompt-grouped, matched-pair metrics.

## Required next analysis before final thesis performance claims

1. Add matched-pair metrics by `prompt_id`.
2. Report low-diversity Energy-family win rates and prompt-grouped bootstrap CIs.
3. Report risk-oriented transforms: `semantic_energy_boltzmann`, `logit_variance`, `-confidence_margin`, and `-mean_negative_log_probability`.
4. Separate prompt-level SE diagnostics from candidate-level Energy diagnostics.
5. Reframe fusion as type-routed evaluation rather than a single all-row classifier if the thesis claim is about SE/Energy complementarity.
