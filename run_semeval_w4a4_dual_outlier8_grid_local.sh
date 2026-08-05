#!/usr/bin/env bash
# Run the full SemEval W4A4 W/A-outlier8 ratio grid sequentially.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_local.sh"
BASE_RUN="${BASE_RUN:-formal_e10_ms256}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_w4a4_dual_out8_grid_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'weight_ratio\tactivation_ratio\tlabel\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

run_one() {
  local weight_ratio="$1"
  local weight_label="$2"
  local activation_ratio="$3"
  local activation_label="$4"
  local label="w${weight_label}_a${activation_label}"
  local run_name="ptq_oa_w4a4_dualout8_${label}_ms256_semeval_e10"
  local run_dir="$RUN_ROOT/$run_name"
  local log_path="$LOG_DIR/${label}.log"

  if [[ -s "$run_dir/val_metrics.json" && \
        -s "$run_dir/quant_summary.json" && \
        -s "$run_dir/outlier_runtime_stats.json" ]]; then
    printf '%s\t%s\t%s\t%s\tskipped_existing\t%s\n' \
      "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] SKIP  $run_name"
    skipped=$((skipped + 1))
    return
  fi

  echo "[$(date --iso-8601=seconds)] START W4A4 W-outlier8=$weight_ratio A-outlier8=$activation_ratio run=$run_name"
  if env \
      BASE_RUN="$BASE_RUN" \
      PTQ_WEIGHT_BITS=4 \
      PTQ_ACT_BITS=4 \
      PTQ_WEIGHT_OUTLIER_RATIO="$weight_ratio" \
      PTQ_ACT_OUTLIER_RATIO="$activation_ratio" \
      PTQ_OUTLIER_BITS=8 \
      PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    printf '%s\t%s\t%s\t%s\tcompleted\t%s\n' \
      "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] DONE  $run_name"
    completed=$((completed + 1))
  else
    printf '%s\t%s\t%s\t%s\tfailed\t%s\n' \
      "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] FAIL  $run_name"
    failures=$((failures + 1))
  fi
}

weight_grid=(
  '0.001 0p1'
  '0.0025 0p25'
  '0.005 0p5'
  '0.01 1'
  '0.02 2'
  '0.04 4'
)

activation_grid=(
  '0.005 0p5'
  '0.01 1'
  '0.02 2'
  '0.04 4'
  '0.06 6'
)

for weight in "${weight_grid[@]}"; do
  read -r weight_ratio weight_label <<< "$weight"
  for activation in "${activation_grid[@]}"; do
    read -r activation_ratio activation_label <<< "$activation"
    run_one "$weight_ratio" "$weight_label" "$activation_ratio" "$activation_label"
  done
done

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
