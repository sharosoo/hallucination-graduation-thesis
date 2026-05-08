#!/usr/bin/env bash
# Run S2b validate -> S3 -> S4 -> S6 -> S7 -> S8 -> S9 against an existing
# pipeline run dir. Each stage checks its inputs before launching, fails fast
# on validator non-zero exits, and writes per-stage stdout/stderr logs.
#
# Required env: RUN_DIR points at an existing pipeline-<utc> run dir whose S2
# outputs (free_sample_rows.json + candidate_scores.json) have just landed.
#
# Usage:
#   RUN_DIR=/mnt/data/hallucination-graduation-thesis-runs/pipeline-20260507T103225Z \
#       bash experiments/scripts/run_downstream_chain.sh
set -euo pipefail

RUN_DIR=${RUN_DIR:?"set RUN_DIR to the pipeline run directory"}
RESULTS=$RUN_DIR/results
LOGS=$RUN_DIR/logs
mkdir -p "$LOGS"

FREE_SAMPLES=$RESULTS/generation/free_sample_rows.json
CANDIDATE_SCORES=$RESULTS/generation/candidate_scores.json
CANDIDATE_ROWS=$RESULTS/datasets/candidate_rows.jsonl
PROMPT_GROUPS=$RESULTS/datasets/prompt_groups.jsonl
CORRECTNESS_DIR=$RESULTS/correctness
SE_OUT=$RESULTS/semantic_entropy_features.parquet
ENERGY_OUT=$RESULTS/energy_features.parquet
CORPUS_OUT=$RESULTS/corpus_features.parquet
FEATURES_OUT=$RESULTS/features.parquet
FUSION_DIR=$RESULTS/fusion
ROBUSTNESS_DIR=$RESULTS/robustness
FUSION_CONFIG=experiments/configs/fusion.yaml

stage_header () {
  printf '\n=========================================================\n'
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1"
  printf '=========================================================\n'
}

require_file () {
  local label=$1 path=$2
  if [ ! -e "$path" ]; then
    echo "[FATAL] missing $label: $path" >&2
    exit 2
  fi
}

run_stage () {
  local stage_id=$1; shift
  local label=$1; shift
  stage_header "$stage_id $label"
  printf '+ %s\n' "$*"
  if ! "$@" > "$LOGS/$stage_id.stdout.log" 2> "$LOGS/$stage_id.stderr.log"; then
    echo "[FATAL] $stage_id failed; tail of stderr:" >&2
    tail -40 "$LOGS/$stage_id.stderr.log" >&2 || true
    exit 2
  fi
  echo "[ok] $stage_id"
}

# ---------- S2b: validate generation outputs (pre-flight gate for S4/S6) ----------
require_file 'free_sample_rows.json' "$FREE_SAMPLES"
require_file 'candidate_scores.json' "$CANDIDATE_SCORES"
run_stage S2b.free_samples 'validate free_sample_rows logits' \
  uv run python experiments/scripts/validate_generation_logits.py "$FREE_SAMPLES"
run_stage S2b.candidate_scores 'validate candidate_scores logits' \
  uv run python experiments/scripts/validate_generation_logits.py "$CANDIDATE_SCORES"

# ---------- S3: annotation-driven correctness labels ----------
# Out is a directory (build_correctness_dataset.py creates data/ + manifest + README inside).
require_file 'candidate_rows.jsonl' "$CANDIDATE_ROWS"
mkdir -p "$CORRECTNESS_DIR"
run_stage S3 'build correctness labels' \
  uv run python experiments/scripts/build_correctness_dataset.py \
    --candidates "$CANDIDATE_ROWS" \
    --out "$CORRECTNESS_DIR" \
    --progress "$LOGS/S3.correctness.progress.json"

# ---------- S4: NLI likelihood Semantic Entropy (batched DeBERTa-large MNLI on GPU) ----------
run_stage S4 'compute semantic entropy (batched NLI)' \
  uv run python experiments/scripts/compute_semantic_entropy.py \
    --free-samples "$FREE_SAMPLES" \
    --out "$SE_OUT" \
    --progress "$LOGS/S4.semantic_entropy.progress.json" \
    --resume

# ---------- S5: corpus features (skip if existing valid prebuild, else rebuild) ----------
# 일반 흐름: prebuild parquet (corpus_axis_bin_10 포함, schema_version=corpus_axis_counts_v1)
# 이 이미 존재 → --resume이 즉시 skip. 없으면 thread cap을 걸고 새로 산출.
# Infini-gram local engine은 OpenMP/Rust thread pool fan-out (이전 stuck 시 256 threads +
# 31 GB RSS) 이 발생하므로 OMP/MKL/OPENBLAS/RAYON num_threads를 4로 캡. cap 적용 시 RSS
# ~17 GB 안에서 안정적으로 종료 확인됨.
if [ -f "$CORPUS_OUT" ]; then
  echo "[$(date -Iseconds)] S5: existing $CORPUS_OUT detected; --resume will skip if schema matches"
else
  echo "[$(date -Iseconds)] S5: $CORPUS_OUT missing; will rebuild with thread cap"
  S5_MIN_AVAIL_GB=15
  for attempt in 1 2 3 4 5; do
    AVAIL_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
    echo "[$(date -Iseconds)] memory check attempt=$attempt available=${AVAIL_GB} GiB threshold=${S5_MIN_AVAIL_GB} GiB"
    [ "$AVAIL_GB" -ge "$S5_MIN_AVAIL_GB" ] && break
    if [ "$attempt" -eq 5 ]; then
      echo "[$(date -Iseconds)] FATAL: host MEM tight after 5 retries; aborting before S5 rebuild."
      exit 3
    fi
    sleep 60
  done
fi
run_stage S5 'compute corpus features (resume-skip if existing)' \
  env \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    RAYON_NUM_THREADS=4 \
    INFINIGRAM_DEFAULT_PARALLELISM=4 \
  uv run python experiments/scripts/compute_corpus_features.py \
    --candidates "$CANDIDATE_ROWS" \
    --out "$CORPUS_OUT" \
    --progress "$LOGS/S5.corpus.progress.json" \
    --resume

# ---------- S6: paper-faithful Semantic Energy + diagnostics ----------
run_stage S6 'compute energy features' \
  uv run python experiments/scripts/compute_energy_features.py \
    --candidate-scores "$CANDIDATE_SCORES" \
    --free-samples "$FREE_SAMPLES" \
    --semantic-entropy "$SE_OUT" \
    --out "$ENERGY_OUT" \
    --progress "$LOGS/S6.energy.progress.json" \
    --resume

# ---------- S7: feature table ----------
run_stage S7 'build feature table' \
  uv run python experiments/scripts/build_feature_table.py \
    --inputs "$RESULTS" \
    --out "$FEATURES_OUT" \
    --progress "$LOGS/S7.features.progress.json" \
    --resume

# ---------- S8: condition-aware fusion ----------
mkdir -p "$FUSION_DIR"
run_stage S8 'run fusion' \
  uv run python experiments/scripts/run_fusion.py \
    --features "$FEATURES_OUT" \
    --config "$FUSION_CONFIG" \
    --out "$FUSION_DIR" \
    --progress "$LOGS/S8.fusion.progress.json"

# ---------- S9: robustness ----------
mkdir -p "$ROBUSTNESS_DIR"
run_stage S9 'run robustness' \
  uv run python experiments/scripts/run_robustness.py \
    --features "$FEATURES_OUT" \
    --fusion "$FUSION_DIR" \
    --out "$ROBUSTNESS_DIR" \
    --progress "$LOGS/S9.robustness.progress.json"

stage_header 'ALL DOWNSTREAM STAGES COMPLETE'
ls -lh "$RESULTS"
