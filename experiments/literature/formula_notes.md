# Formula Notes

These notes are source-traceability scaffolding for the experiment contract. Any line marked `UNVERIFIED_DO_NOT_CITE` still requires direct PDF inspection before thesis prose cites it as settled evidence.

## Formula: semantic_entropy
Source ID: farquhar_semantic_entropy_2024
Paper: Detecting hallucinations in large language models using semantic entropy
Feature Family: semantic_entropy, cluster_count
Page Reference: Nature 2024 PDF pages 7-8 (Methods, “Semantic entropy” and “Computing the semantic entropy”).
Section Reference: Methods → “Principles of semantic uncertainty”; Methods → “Computing the semantic entropy”.
Equation Reference: Eq. (2) defines cluster probability mass P(c|x); Eq. (3) defines semantic entropy SE(x); Eq. (5) gives the sampled estimator used in practice.
Notes: The paper explicitly defines semantic clustering by bidirectional entailment on page 8 and uses sampled generations plus cluster probability mass to compute semantic entropy. `cluster_count` is a derived implementation artifact from the semantic clustering stage and should be traced to the same Methods section rather than treated as an independent paper formula.

## Formula: semantic_energy_boltzmann_or_proxy
Source ID: ma_semantic_energy_2025
Paper: Semantic Energy
Feature Family: semantic_energy_boltzmann or semantic_energy_proxy, logit_variance, confidence_margin
Page Reference: arXiv PDF pages 4-6 (Sections 2.2, 3.2.1, and 3.2.2).
Section Reference: Section 2.2 “Semantic Entropy and Response Clustering”; Section 3.2.1 “Boltzmann Distribution”; Section 3.2.2 “Specific Implementation in LLMs”.
Equation Reference: Eq. (9) defines Semantic Entropy over semantic clusters; Eq. (10) gives the Boltzmann distribution; Eq. (12) defines total cluster energy; Eq. (13) maps token energy to negative logits; Eq. (14) defines the final uncertainty U(x(i)).
Notes: This paper is a preprint and should stay citation-caveated. It directly supports a Boltzmann/logit-based semantic energy formulation. `logit_variance` and `confidence_margin` are not explicitly introduced here as standalone features, so if they are implemented later they must be labeled as downstream proxies rather than quoted as named Ma et al. formulas.

## Formula: quco_entity_frequency
Source ID: quco_rag_2025
Paper: QuCo-RAG
Feature Family: entity_frequency, low_frequency_entity_flag
Page Reference: arXiv PDF page 3.
Section Reference: Section 3.2 “Pre-Generation Knowledge Assessment”.
Equation Reference: Eq. (2) triggers retrieval when the average entity frequency falls below a threshold; the surrounding text defines freq(e; P) over the pre-training corpus.
Notes: QuCo-RAG uses low entity frequency as an input-uncertainty proxy, not as proof of model exposure. `low_frequency_entity_flag` in this repo should therefore be documented as a thresholded derivative of corpus entity frequency, with the same corpus-grounded caveat.

## Formula: quco_entity_pair_cooccurrence
Source ID: quco_rag_2025
Paper: QuCo-RAG
Feature Family: entity_pair_cooccurrence
Page Reference: arXiv PDF pages 3-4.
Section Reference: Section 3.3 “Runtime Claim Verification”.
Equation Reference: Eq. (3) defines entity co-occurrence count cooc(h, t; P); Eq. (4) defines the retrieval trigger when the minimum co-occurrence in extracted triplets falls below the threshold.
Notes: This is a corpus-grounded proxy over an indexed pre-training corpus. The paper explicitly states that zero co-occurrence strongly indicates hallucination risk but does not guarantee correctness when co-occurrence is non-zero, so downstream prose must not overstate what this signal proves.

## Formula: selective_risk_metrics
Source ID: phillips_pc_probe_2026
Paper: Entropy Alone is Insufficient for Safe Selective Prediction in LLMs
Feature Family: selective/risk framing only; not an implemented feature source
Page Reference: arXiv PDF pages 1-2.
Section Reference: Section 2.1 “Selective Prediction Setup”; Section 2.3 “Evaluation Metrics”; Section 3.2 “Model-Specific Failure Modes”.
Equation Reference: Eq. (1) defines the abstention policy Aτ(x) = I{r(x) ≤ τ}; E-AURC and TCE are defined in Section 2.3 rather than as numbered equations.
Notes: Phillips et al. is reference-only here. It supports the framing that entropy-only methods can fail in a confidently wrong regime and that selective prediction should be judged with deployment-facing risk/coverage metrics. It must not be used as a source for implementing hidden-state correctness probes in this repo.
