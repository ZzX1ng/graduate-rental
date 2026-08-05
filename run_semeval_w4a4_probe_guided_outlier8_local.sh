#!/usr/bin/env bash
# Run BOP-matched fixed-probing and control allocations sequentially on one rental GPU.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ALLOCATION_JSON="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq/semeval_w4a4_probe_guided_outlier8_allocations.json"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_w4a4_probe_guided_out8_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
failures=0

mkdir -p "$LOG_DIR" "$(dirname "$ALLOCATION_JSON")"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'budget\tstrategy\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task semeval \
  --output "$ALLOCATION_JSON"

while IFS=$'\t' read -r budget strategy nominal_bop weight_ratios act_ratios; do
  run_name="ptq_pg_w4a4_out8_${budget}_${strategy}_cal16_semeval_e10"
  run_dir="$RUN_ROOT/$run_name"
  log_path="$LOG_DIR/${budget}_${strategy}.log"
  if [[ -s "$run_dir/val_metrics.json" && \
        -s "$run_dir/quant_summary.json" && \
        -s "$run_dir/outlier_runtime_stats.json" && \
        -s "$run_dir/activation_calibration_stats.json" ]]; then
    printf '%s\t%s\t%s\tskipped_existing\t%s\n' \
      "$budget" "$strategy" "$run_name" "$log_path" >> "$STATUS_TSV"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] START budget=$budget strategy=$strategy run=$run_name"
  if env \
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
      PTQ_ALLOCATION_STRATEGY="$strategy" \
      PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    status=completed
  else
    status=failed
    failures=$((failures + 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$budget" "$strategy" "$run_name" "$status" "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} run=$run_name"
done < <("$PYTHON" - "$ALLOCATION_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    activations = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(
        item["budget"],
        item["strategy"],
        f'{item["nominal_bop_overhead"]:.12g}',
        weights,
        activations,
        sep="\t",
    )
PY
)

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
