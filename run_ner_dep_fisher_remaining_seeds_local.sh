#!/usr/bin/env bash
# Complete NER/DEP Fisher and probe+Fisher runs for remaining seeds.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
SINGLE_RUNNER="$PROJECT_ROOT/run_ner_dep_fisher_single_seed_pilot_local.sh"
SEEDS="${SEEDS:-20260807 20260808 20260809}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/ner_dep_fisher_remaining_seeds_$RUN_ID"
STATUS="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tstatus\trun_id\tlog\n' > "$STATUS"

for seed in $SEEDS; do
  seed_run_id="${RUN_ID}_seed${seed}"
  log="$LOG_DIR/seed_${seed}.launcher.log"
  echo "[$(date --iso-8601=seconds)] START seed=$seed run_id=$seed_run_id"
  if env SEED="$seed" RUN_ID="$seed_run_id" bash "$SINGLE_RUNNER" > "$log" 2>&1; then
    status=completed
  else
    status=failed
    failures=$((failures + 1))
  fi
  printf '%s\t%s\t%s\t%s\n' "$seed" "$status" "$seed_run_id" "$log" >> "$STATUS"
  echo "[$(date --iso-8601=seconds)] ${status^^} seed=$seed"
done

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS"
exit "$failures"
