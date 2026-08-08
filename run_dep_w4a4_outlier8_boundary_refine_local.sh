#!/usr/bin/env bash
# Refine DEP W4A4 dual-outlier8 low/boundary budgets with fixed calibration.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
BASE_RUN="${BASE_RUN:-formal_e3_ms256_retry1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_out8_boundary_refine_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/dep"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'weight_ratio\tactivation_ratio\tlabel\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

run_one() {
  local weight_ratio="$1"
  local activation_ratio="$2"
  local label="$3"
  local run_name="ptq_oa_w4a4_dualout8_boundaryrefine_${label}_fixedseed20260806_cal16_ms256_dep_e3_retry1"
  local run_dir="$RUN_ROOT/$run_name"
  local log_path="$LOG_DIR/${label}.log"

  if [[ -s "$run_dir/val_metrics.json" && \
        -s "$run_dir/quant_summary.json" && \
        -s "$run_dir/outlier_runtime_stats.json" && \
        -s "$run_dir/activation_calibration_stats.json" ]]; then
    printf '%s\t%s\t%s\t%s\tskipped_existing\t%s\n' \
      "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] SKIP  $run_name"
    skipped=$((skipped + 1))
    return
  fi

  echo "[$(date --iso-8601=seconds)] START W=$weight_ratio A=$activation_ratio run=$run_name"
  if env \
      TASK=dep \
      BASE_RUN="$BASE_RUN" \
      PTQ_WEIGHT_BITS=4 \
      PTQ_ACT_BITS=4 \
      PTQ_WEIGHT_OUTLIER_RATIO="$weight_ratio" \
      PTQ_ACT_OUTLIER_RATIO="$activation_ratio" \
      PTQ_OUTLIER_BITS=8 \
      PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
      PTQ_ACT_CALIBRATION_BATCHES=16 \
      PTQ_ACT_CALIBRATION_MAX_VALUES=65536 \
      PTQ_ALLOCATION_STRATEGY=uniform \
      PTQ_RUN_SEED=20260806 \
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

run_one 0.00055 0.00275 w0p055_a0p275
run_one 0.00060 0.00300 w0p060_a0p300
run_one 0.00065 0.00325 w0p065_a0p325
run_one 0.00070 0.00350 w0p070_a0p350
run_one 0.00080 0.00400 w0p080_a0p400
run_one 0.00090 0.00450 w0p090_a0p450

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
