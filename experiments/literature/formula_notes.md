# Formula Notes

These notes are source-traceability scaffolding for the experiment contract. Any line marked `UNVERIFIED_DO_NOT_CITE` still requires direct PDF inspection before thesis prose cites it as settled evidence.

## Formula: semantic_entropy
Source ID: farquhar_semantic_entropy_2024
Paper: Detecting hallucinations in large language models using semantic entropy
Feature Family: semantic_entropy_nli_likelihood, semantic_entropy_cluster_count, semantic_entropy_discrete_cluster_entropy
Page Reference: Nature 2024 PDF pages 7-8 (Methods, “Semantic entropy” and “Computing the semantic entropy”).
Section Reference: Methods → “Principles of semantic uncertainty”; Methods → “Computing the semantic entropy”.
Equation Reference: Eq. (2) defines cluster probability mass P(c|x); Eq. (3) defines semantic entropy SE(x); Eq. (5) gives the sampled estimator used in practice.
Notes: The thesis-valid implementation must use N=10 answer-only samples, strict bidirectional NLI entailment equivalence clustering in deterministic sample-index order, and likelihood-based cluster probability from mean token log-likelihoods. Exact normalized-string clustering and N=5 count entropy are archived diagnostics, not paper-faithful final evidence.

## Formula: semantic_energy_cluster_uncertainty
Source ID: ma_semantic_energy_2025
Paper: Semantic Energy
Feature Family: semantic_energy_cluster_uncertainty, semantic_energy_sample_energy
Page Reference: arXiv PDF pages 4-6 (Sections 2.2, 3.2.1, and 3.2.2).
Section Reference: Section 2.2 “Semantic Entropy and Response Clustering”; Section 3.2.1 “Boltzmann Distribution”; Section 3.2.2 “Specific Implementation in LLMs”.
Equation Reference: Eq. (8) defines likelihood-based cluster probability p(C_k) = sum of normalized response likelihoods of cluster members; Eq. (11) defines per-response sequence-level energy E(x^(i)) = (1/T_i) * sum_t E_t^(i); Eq. (12) defines total cluster energy E_Bolt(C) = sum_{x in C} E(x); Eq. (13) approximates token energy as the negative logit of the selected token, tilde-E(x_t) = -z_theta(x_t); Eq. (14) defines the final uncertainty U(x^(i)).
Notes: This paper is a preprint and should stay citation-caveated. The thesis-valid implementation uses the same N=10 sampled responses and Task 4 semantic clusters as Semantic Entropy; clusters are joined by `(prompt_id, sample_index)` and are not recomputed in the Energy stage. Per Eq. (13), token energy is `-selected_token_logit`. Per Eq. (11), per-response sample energy is `sample_energy = (1/T_i) * sum_t -z_theta(x_t) = mean(-selected_token_logits)`. Per Eq. (12), cluster total energy is `cluster_energy = sum_{x in C_k} sample_energy(x)` (SUM over cluster members, not mean — the Boltzmann thermodynamic argument is that more states in a cluster contribute more total energy). The final uncertainty is `semantic_energy_cluster_uncertainty = sum_k cluster_probability(C_k) * cluster_energy(C_k)` where `cluster_probability(C_k)` is the Eq. (8) likelihood-based cluster mass inherited unchanged from the Task 4 Semantic Entropy stage; this preserves the Farquhar-style likelihood weighting that Ma 2025 cites in Section 2.2. The supporting prompt-level `semantic_energy_sample_energy` is the simple arithmetic mean over the ten per-response sample energies and matches the most literal collapsed reading of Eq. (14). Lower raw Energy means more reliable; higher raw Energy is treated as higher uncertainty. Candidate-level full-logit `-logsumexp` means remain `semantic_energy_boltzmann_diagnostic` only and are reported separately from the paper-faithful sampled-response Semantic Energy.

## Formula: candidate_logit_diagnostics
Source ID: local_diagnostic_features
Paper: Local experiment diagnostic specification
Feature Family: mean_negative_log_probability, logit_variance, confidence_margin, semantic_energy_boltzmann_diagnostic
Page Reference: Local implementation note, not an external source-paper formula.
Section Reference: experiments/PIPELINE.md §5 Paper-derived feature alignment.
Equation Reference: Candidate token mean NLL, token-logit variance, confidence margin, and candidate-window mean `-logsumexp` are defined by the repo formula manifest rather than an external named method.
Notes: These features are useful candidate-level diagnostics and may be compared against Semantic Entropy and Semantic Energy, but they must be labeled diagnostic/adapted instead of paper-faithful Semantic Energy.

## Formula: quco_entity_frequency
Source ID: quco_rag_2025
Paper: QuCo-RAG
Feature Family: entity_frequency, entity_frequency_axis, low_frequency_entity_flag
Page Reference: arXiv PDF page 3.
Section Reference: Section 3.2 “Pre-Generation Knowledge Assessment”.
Equation Reference: Eq. (2) triggers retrieval when the average entity frequency falls below a threshold; the surrounding text defines freq(e; P) over the pre-training corpus.
Notes: QuCo-RAG uses low entity frequency as an input-uncertainty proxy. This thesis uses the same direct-count idea as a continuous corpus-support axis for metric reliability analysis, not as a direct hallucination label or as a RAG retrieval policy.

## Formula: quco_entity_pair_cooccurrence
Source ID: quco_rag_2025
Paper: QuCo-RAG
Feature Family: entity_pair_cooccurrence, entity_pair_cooccurrence_axis, zero_cooccurrence_flag
Page Reference: arXiv PDF pages 3-4.
Section Reference: Section 3.3 “Runtime Claim Verification”.
Equation Reference: Eq. (3) defines entity co-occurrence count cooc(h, t; P); Eq. (4) defines the retrieval trigger when the minimum co-occurrence in extracted triplets falls below the threshold.
Notes: This is a corpus-grounded proxy over an indexed pre-training corpus. Zero co-occurrence is a risk/support-sparsity signal, while non-zero co-occurrence does not prove correctness. Corpus coverage and entity extraction errors must be reported as caveats.

## Formula: condition_aware_fusion
Source ID: local_reliability_analysis
Paper: Local thesis reliability-analysis protocol
Feature Family: corpus-bin feature selection, corpus-bin weighted fusion, axis-interaction logistic fusion
Page Reference: Local implementation note, not an external source-paper formula.
Section Reference: experiments/PIPELINE.md §S8 and §S9.
Equation Reference: Global logistic fusion and corpus-bin-aware comparisons are defined in the experiment protocol; report AUROC, AUPRC, paired win rate, paired delta, and prompt-grouped bootstrap CI per bin.
Notes: The claim is not that corpus-aware fusion universally improves hallucination detection. The claim to test is whether conditioning on corpus-support axes changes metric reliability or makes fusion more interpretable/stable than a single global fusion rule.
