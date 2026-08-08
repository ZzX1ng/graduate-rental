#!/usr/bin/env bash
# Queue the DEP low-budget probing workflow for four calibration seeds.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
SINGLE_RUNNER="$PROJECT_ROOT/run_dep_w4a4_probe_guided_outlier8_low_single_seed_local.sh"
SEEDS="${SEEDS:-20260806 20260807 20260808 20260809}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_probe_guided_out8_low_four_seeds_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tstatus\tseed_run_id\tlauncher_log\n' > "$STATUS_TSV"

for seed in $SEEDS; do
  seed_run_id="${RUN_ID}_seed${seed}"
  launcher_log="$LOG_DIR/seed_${seed}.launcher.log"
  echo "[$(date --iso-8601=seconds)] START seed=$seed run_id=$seed_run_id"
  if env SEED="$seed" RUN_ID="$seed_run_id" \
      bash "$SINGLE_RUNNER" >"$launcher_log" 2>&1; then
    status=completed
  else
    status=failed
    failures=$((failures + 1))
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "$seed" "$status" "$seed_run_id" "$launcher_log" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} seed=$seed"
done

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
