#!/usr/bin/env bash
# Screen selected SemEval W2A4/W4A2 dual-outlier8 configurations.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_local.sh"
BASE_RUN="${BASE_RUN:-formal_e10_ms256}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_w2a4_w4a2_out8_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'weight_bits\tactivation_bits\tweight_ratio\tactivation_ratio\tlabel\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

run_one() {
  local weight_bits="$1"
  local activation_bits="$2"
  local weight_ratio="$3"
  local activation_ratio="$4"
  local label="$5"
  local run_name="ptq_oa_${label}_ms256_semeval_e10"
  local run_dir="$RUN_ROOT/$run_name"
  local log_path="$LOG_DIR/${label}.log"

  if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tskipped_existing\t%s\n' \
      "$weight_bits" "$activation_bits" "$weight_ratio" "$activation_ratio" \
      "$label" "$run_name" "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] SKIP  $run_name"
    skipped=$((skipped + 1))
    return
  fi

  echo "[$(date --iso-8601=seconds)] START W${weight_bits}A${activation_bits} W-outlier8=$weight_ratio A-outlier8=$activation_ratio run=$run_name"
  if env \
      BASE_RUN="$BASE_RUN" \
      PTQ_WEIGHT_BITS="$weight_bits" \
      PTQ_ACT_BITS="$activation_bits" \
      PTQ_WEIGHT_OUTLIER_RATIO="$weight_ratio" \
      PTQ_ACT_OUTLIER_RATIO="$activation_ratio" \
      PTQ_OUTLIER_BITS=8 \
      PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tcompleted\t%s\n' \
      "$weight_bits" "$activation_bits" "$weight_ratio" "$activation_ratio" \
      "$label" "$run_name" "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] DONE  $run_name"
    completed=$((completed + 1))
  else
    printf '%s\t%s\t%s\t%s\t%s\t%s\tfailed\t%s\n' \
      "$weight_bits" "$activation_bits" "$weight_ratio" "$activation_ratio" \
      "$label" "$run_name" "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] FAIL  $run_name"
    failures=$((failures + 1))
  fi
}

run_one 2 4 0 0 w2a4_uniform
run_one 2 4 0.005 0.01 w2a4_dualout8_w0p5_a1
run_one 2 4 0.01 0.02 w2a4_dualout8_w1_a2
run_one 2 4 0.02 0.04 w2a4_dualout8_w2_a4

run_one 4 2 0 0 w4a2_uniform
run_one 4 2 0.0025 0.02 w4a2_dualout8_w0p25_a2
run_one 4 2 0.005 0.04 w4a2_dualout8_w0p5_a4
run_one 4 2 0.01 0.06 w4a2_dualout8_w1_a6

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
