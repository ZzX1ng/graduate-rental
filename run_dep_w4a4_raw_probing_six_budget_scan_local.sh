#!/usr/bin/env bash
# Scan six DEP budgets with raw probing allocation and no layer floor.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/dep"
SEED="${SEED:-20260806}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_raw_probing_six_budget_scan_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'weight_ratio\tactivation_ratio\tlabel\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

run_one() {
  local weight_ratio="$1"
  local activation_ratio="$2"
  local label="$3"
  local allocation_json="$ANALYSIS_DIR/dep_w4a4_raw_probing_${label}_allocations.json"
  local run_name="ptq_pg_w4a4_out8_${label}_probing_raw_fixedseed${SEED}_cal16_dep_e3_retry1"
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

  "$PYTHON" "$GENERATOR" \
    --probing-csv "$PROBING_CSV" \
    --task dep \
    --low-weight-ratio "$weight_ratio" \
    --low-activation-ratio "$activation_ratio" \
    --boundary-weight-ratio "$weight_ratio" \
    --boundary-activation-ratio "$activation_ratio" \
    --shape-floor 0 \
    --output "$allocation_json"

  readarray -t allocation < <("$PYTHON" - "$allocation_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
item = next(
    item
    for item in payload["allocations"]
    if item["budget"] == "low" and item["strategy"] == "probing"
)
print(f'{item["nominal_bop_overhead"]:.12g}')
print(",".join(f"{value:.12g}" for value in item["weight_layer_ratios"]))
print(",".join(f"{value:.12g}" for value in item["activation_layer_ratios"]))
PY
)

  local nominal_bop="${allocation[0]}"
  local weight_ratios="${allocation[1]}"
  local act_ratios="${allocation[2]}"

  echo "[$(date --iso-8601=seconds)] START W=$weight_ratio A=$activation_ratio run=$run_name"
  if env \
      TASK=dep \
      BASE_RUN=formal_e3_ms256_retry1 \
      PTQ_WEIGHT_BITS=4 \
      PTQ_ACT_BITS=4 \
      PTQ_WEIGHT_OUTLIER_RATIO=0 \
      PTQ_ACT_OUTLIER_RATIO=0 \
      PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$weight_ratios" \
      PTQ_ACT_OUTLIER_LAYER_RATIOS="$act_ratios" \
      PTQ_OUTLIER_BITS=8 \
      PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
      PTQ_ACT_CALIBRATION_BATCHES=16 \
      PTQ_ACT_CALIBRATION_MAX_VALUES=65536 \
      PTQ_RUN_SEED="$SEED" \
      PTQ_ALLOCATION_STRATEGY=probing_raw_no_floor \
      PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    status=completed
    completed=$((completed + 1))
  else
    status=failed
    failures=$((failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$weight_ratio" "$activation_ratio" "$label" "$run_name" "$status" \
    "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} run=$run_name"
}

run_one 0.00055 0.00275 w0p055_a0p275
run_one 0.00060 0.00300 w0p060_a0p300
run_one 0.00065 0.00325 w0p065_a0p325
run_one 0.00070 0.00350 w0p070_a0p350
run_one 0.00080 0.00400 w0p080_a0p400
run_one 0.00090 0.00450 w0p090_a0p450

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
