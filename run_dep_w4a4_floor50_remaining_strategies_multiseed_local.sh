#!/usr/bin/env bash
# Complete DEP floor50 early/inverse/late/random pilot and BOP-fix runs.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PILOT_SCRIPT="$PROJECT_ROOT/run_dep_w4a4_floor50_low_boundary_six_strategy_single_seed_local.sh"
BOPFIX_SCRIPT="$PROJECT_ROOT/run_dep_w4a4_floor50_low_boundary_bopfix2_single_seed_local.sh"
SEEDS="${SEEDS:-20260807 20260808 20260809}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_floor50_remaining_strategies_multiseed_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tphase\tstatus\tchild_run_id\tlog\n' > "$STATUS_TSV"

run_phase() {
  local seed="$1"
  local phase="$2"
  local script="$3"
  local child_run_id="${RUN_ID}_seed${seed}_${phase}"
  local log_path="$LOG_DIR/${seed}_${phase}_runner.log"
  local status

  echo "[$(date --iso-8601=seconds)] START seed=$seed phase=$phase"
  if env SEED="$seed" RUN_ID="$child_run_id" bash "$script" >"$log_path" 2>&1; then
    status=completed
  else
    status=failed
    failures=$((failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$seed" "$phase" "$status" "$child_run_id" "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} seed=$seed phase=$phase"
  [[ "$status" == "completed" ]]
}

for seed in $SEEDS; do
  if run_phase "$seed" pilot "$PILOT_SCRIPT"; then
    run_phase "$seed" bopfix2 "$BOPFIX_SCRIPT" || true
  else
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$seed" bopfix2 not_run "${RUN_ID}_seed${seed}_bopfix2" \
      "$LOG_DIR/${seed}_bopfix2_runner.log" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] NOT_RUN seed=$seed phase=bopfix2 due_to_pilot_failure"
  fi
done

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
