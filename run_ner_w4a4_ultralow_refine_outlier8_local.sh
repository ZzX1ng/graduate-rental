#!/usr/bin/env bash
# Refine the NER W4A4 dual-outlier8 transition between the 1/4 and 1/2 budgets.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
BASE_RUN="${BASE_RUN:-formal_e3_ms256_tner}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/ner_w4a4_ultralow_refine_out8_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/ner"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'scale\tweight_ratio\tactivation_ratio\tlabel\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

run_one() {
  local scale="$1"
  local weight_ratio="$2"
  local activation_ratio="$3"
  local label="$4"
  local run_name="ptq_oa_w4a4_dualout8_ultralow_refine_${label}_fixedseed20260806_cal16_ms256_ner_e3"
  local run_dir="$RUN_ROOT/$run_name"
  local log_path="$LOG_DIR/${label}.log"

  if [[ -s "$run_dir/val_metrics.json" && \
        -s "$run_dir/quant_summary.json" && \
        -s "$run_dir/outlier_runtime_stats.json" && \
        -s "$run_dir/activation_calibration_stats.json" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\tskipped_existing\t%s\n' \
      "$scale" "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    skipped=$((skipped + 1))
    return
  fi

  echo "[$(date --iso-8601=seconds)] START scale=$scale W=$weight_ratio A=$activation_ratio run=$run_name"
  if env \
      BASE_RUN="$BASE_RUN" \
      PTQ_WEIGHT_BITS=4 \
      PTQ_ACT_BITS=4 \
      PTQ_WEIGHT_OUTLIER_RATIO="$weight_ratio" \
      PTQ_ACT_OUTLIER_RATIO="$activation_ratio" \
      PTQ_OUTLIER_BITS=8 \
      PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
      PTQ_ACT_CALIBRATION_BATCHES=16 \
      PTQ_ACT_CALIBRATION_MAX_VALUES=65536 \
      PTQ_RUN_SEED=20260806 \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\tcompleted\t%s\n' \
      "$scale" "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] DONE  $run_name"
    completed=$((completed + 1))
  else
    printf '%s\t%s\t%s\t%s\t%s\tfailed\t%s\n' \
      "$scale" "$weight_ratio" "$activation_ratio" "$label" "$run_name" \
      "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] FAIL  $run_name"
    failures=$((failures + 1))
  fi
}

run_one 5/16 0.0003125 0.0015625 s5of16_w0p03125_a0p15625
run_one 3/8 0.000375 0.001875 s3of8_w0p0375_a0p1875
run_one 7/16 0.0004375 0.0021875 s7of16_w0p04375_a0p21875

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
