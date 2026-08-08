#!/usr/bin/env bash
# Test raw DEP probing allocation at W0.055%/A0.275%, without a layer floor.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
ALLOCATION_JSON="$ANALYSIS_DIR/dep_w4a4_raw_probing_w0p055_a0p275_allocations.json"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/dep"
SEED="${SEED:-20260806}"
RUN_NAME="ptq_pg_w4a4_out8_w0p055_a0p275_probing_raw_fixedseed${SEED}_cal16_dep_e3_retry1"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_raw_probing_w0p055_a0p275_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
RUN_LOG="$LOG_DIR/probing.log"

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tstrategy\tshape_floor\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task dep \
  --low-weight-ratio 0.00055 \
  --low-activation-ratio 0.00275 \
  --boundary-weight-ratio 0.00055 \
  --boundary-activation-ratio 0.00275 \
  --shape-floor 0 \
  --output "$ALLOCATION_JSON"

readarray -t allocation < <("$PYTHON" - "$ALLOCATION_JSON" <<'PY'
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

nominal_bop="${allocation[0]}"
weight_ratios="${allocation[1]}"
act_ratios="${allocation[2]}"

if [[ -s "$RUN_DIR/val_metrics.json" && \
      -s "$RUN_DIR/quant_summary.json" && \
      -s "$RUN_DIR/outlier_runtime_stats.json" && \
      -s "$RUN_DIR/activation_calibration_stats.json" ]]; then
  printf '%s\tprobing\t0\t%s\tskipped_existing\t%s\n' \
    "$SEED" "$RUN_NAME" "$RUN_LOG" >> "$STATUS_TSV"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] START strategy=probing shape_floor=0 run=$RUN_NAME"
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
    RUN_NAME="$RUN_NAME" \
    bash "$WORKER" >"$RUN_LOG" 2>&1; then
  status=completed
else
  status=failed
fi

printf '%s\tprobing\t0\t%s\t%s\t%s\n' \
  "$SEED" "$RUN_NAME" "$status" "$RUN_LOG" >> "$STATUS_TSV"
echo "[$(date --iso-8601=seconds)] ${status^^} strategy=probing shape_floor=0"
[[ "$status" == completed ]]
