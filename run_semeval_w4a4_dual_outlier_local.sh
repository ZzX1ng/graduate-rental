#!/usr/bin/env bash
# Run the SemEval W4A4 dual-sided outlier16 anchor configurations sequentially.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_local.sh"
BASE_RUN="${BASE_RUN:-formal_e10_ms256}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_w4a4_dual_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR"
printf 'weight_ratio\tactivation_ratio\tlabel\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

run_one() {
  local weight_ratio="$1"
  local activation_ratio="$2"
  local label="$3"
  local run_name="ptq_oa_w4a4_dualout16_${label}_ms256_semeval_e10"
  local log_path="$LOG_DIR/${label}.log"

  echo "[$(date --iso-8601=seconds)] START W4A4 W-outlier=$weight_ratio A-outlier=$activation_ratio run=$run_name"
  if env \
      BASE_RUN="$BASE_RUN" \
      PTQ_WEIGHT_BITS=4 \
      PTQ_ACT_BITS=4 \
      PTQ_WEIGHT_OUTLIER_RATIO="$weight_ratio" \
      PTQ_ACT_OUTLIER_RATIO="$activation_ratio" \
      PTQ_OUTLIER_BITS=16 \
      PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    printf '%s\t%s\t%s\t%s\tcompleted\t%s\n' \
      "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] DONE  $run_name"
  else
    printf '%s\t%s\t%s\t%s\tfailed\t%s\n' \
      "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] FAIL  $run_name"
    failures=$((failures + 1))
  fi
}

run_one 0.001 0.02 w0p1_a2
run_one 0.001 0.04 w0p1_a4
run_one 0.02 0.02 w2_a2
run_one 0.02 0.04 w2_a4

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
