#!/usr/bin/env bash
# Watch the running S2 generation process and, when it exits cleanly with
# both final artifacts produced, kick off run_downstream_chain.sh against the
# same run dir. If S2 dies without producing both final artifacts, log and exit
# without launching downstream (so we don't waste GPU on an incomplete S2).
set -u

RUN_DIR=${RUN_DIR:?"set RUN_DIR to the pipeline run directory"}
S2_PID=${S2_PID:?"set S2_PID to the running run_generation.py pid"}

FREE_SAMPLES=$RUN_DIR/results/generation/free_sample_rows.json
CANDIDATE_SCORES=$RUN_DIR/results/generation/candidate_scores.json
LOG_DIR=$RUN_DIR/logs
mkdir -p "$LOG_DIR"
DOWNSTREAM_STDOUT=$LOG_DIR/downstream.stdout.log
DOWNSTREAM_STDERR=$LOG_DIR/downstream.stderr.log
WATCHER_LOG=$LOG_DIR/await_s2.log

log () {
  printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$*" | tee -a "$WATCHER_LOG"
}

log "watcher started; S2_PID=$S2_PID RUN_DIR=$RUN_DIR"

# Step 1: wait for the S2 process to exit (succeed or fail).
while kill -0 "$S2_PID" 2>/dev/null; do
  sleep 60
done
log "S2 process $S2_PID has exited"

# Step 2: verify both final artifacts landed. run_generation.py writes them
# atomically only on full success, so presence == success.
if [ ! -f "$FREE_SAMPLES" ]; then
  log "FAIL: $FREE_SAMPLES is missing; S2 did not finalize free samples. Skipping downstream."
  exit 2
fi
if [ ! -f "$CANDIDATE_SCORES" ]; then
  log "FAIL: $CANDIDATE_SCORES is missing; S2 did not finalize candidate scores. Skipping downstream."
  exit 2
fi
log "both S2 final artifacts present; proceeding to downstream chain"

# Step 3: kick off the chain script. It runs S2b validate -> S3 -> S4 -> S5
# (resume-skip) -> S6 -> S7 -> S8 -> S9 with per-stage stdout/stderr logs.
log "launching: bash experiments/scripts/run_downstream_chain.sh"
RUN_DIR="$RUN_DIR" bash experiments/scripts/run_downstream_chain.sh \
  > "$DOWNSTREAM_STDOUT" 2> "$DOWNSTREAM_STDERR"
RC=$?
if [ $RC -eq 0 ]; then
  log "downstream chain SUCCEEDED (rc=0)"
else
  log "downstream chain FAILED (rc=$RC); see $DOWNSTREAM_STDERR"
fi
exit $RC
