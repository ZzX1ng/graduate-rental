#!/usr/bin/env bash
# Correct measured BOP after the deterministic fixed-seed pilot.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_fixedseed_local.sh"
INITIAL_STATUS="${INITIAL_STATUS:?Set INITIAL_STATUS to the 16-run pilot status.tsv}"
INITIAL_PID="${INITIAL_PID:-0}"
ALLOCATION_JSON="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq/semeval_w4a4_probe_guided_outlier8_allocations.json"
CORRECTED_JSON="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq/semeval_w4a4_probe_guided_outlier8_bopfix2_fixedseed_allocations.json"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_w4a4_probe_guided_out8_bopfix2_fixedseed_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'budget\tstrategy\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

while true; do
  if [[ ! -f "$INITIAL_STATUS" ]]; then
    sleep 30
    continue
  fi
  finished=$(awk -F '\t' 'NR > 1 && ($4 == "completed" || $4 == "skipped_existing" || $4 == "failed") {n++} END {print n+0}' "$INITIAL_STATUS")
  failed=$(awk -F '\t' 'NR > 1 && $4 == "failed" {n++} END {print n+0}' "$INITIAL_STATUS")
  if [[ "$finished" -ge 16 ]]; then
    break
  fi
  if [[ "$INITIAL_PID" -gt 0 ]] && ! kill -0 "$INITIAL_PID" 2>/dev/null; then
    echo "Initial runner exited before 16 terminal rows: finished=$finished" >&2
    exit 2
  fi
  sleep 30
done
if [[ "$failed" -gt 0 ]]; then
  echo "Initial runner contains $failed failed runs; refusing BOP correction" >&2
  exit 3
fi

"$PYTHON" "$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py" \
  --allocations "$ALLOCATION_JSON" \
  --run-root "$RUN_ROOT" \
  --run-tag fixedseed20260806 \
  --output "$CORRECTED_JSON"

while IFS=$'\t' read -r budget strategy scale target_bop weight_ratios act_ratios; do
  run_name="ptq_pg_w4a4_out8_${budget}_${strategy}_bopfix2_fixedseed20260806_cal16_semeval_e10"
  run_dir="$RUN_ROOT/$run_name"
  log_path="$LOG_DIR/${budget}_${strategy}.log"
  if [[ -s "$run_dir/val_metrics.json" && \
        -s "$run_dir/quant_summary.json" && \
        -s "$run_dir/outlier_runtime_stats.json" && \
        -s "$run_dir/activation_calibration_stats.json" ]]; then
    status=skipped_existing
  elif env \
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
      PTQ_ALLOCATION_STRATEGY="${strategy}_bopfix2_fixedseed20260806" \
      PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" \
      PTQ_RUN_SEED=20260806 \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    status=completed
  else
    status=failed
    failures=$((failures + 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$budget" "$strategy" "$scale" "$target_bop" "$run_name" "$status" "$log_path" \
    >> "$STATUS_TSV"
done < <("$PYTHON" - "$CORRECTED_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    activations = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(
        item["budget"], item["strategy"], f'{item["bop_correction_scale"]:.12g}',
        f'{item["target_measured_uniform_bop_overhead"]:.12g}', weights, activations,
        sep="\t",
    )
PY
)

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
