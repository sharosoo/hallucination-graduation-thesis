---
dataset_info:
  features:
    - name: prompt_id
      dtype: string
    - name: candidate_id
      dtype: string
    - name: pair_id
      dtype: string
    - name: dataset
      dtype: string
    - name: candidate_role
      dtype: string
    - name: is_correct
      dtype: bool
    - name: label_source
      dtype: string
---

# Annotation-driven correctness labels

This correctness label manifest is derived only from `candidate_rows.jsonl` dataset annotations.
It does not call an LLM judge, does not read generated answers, and does not use heuristic text matching.

## Construction

- Source candidate rows: `experiments/results/datasets/candidate_rows.jsonl`
- Label rows: `data/correctness_judgments.jsonl`
- Row count: 11630
- Heuristic matching used: `False`
- LLM-as-judge used: `False`
