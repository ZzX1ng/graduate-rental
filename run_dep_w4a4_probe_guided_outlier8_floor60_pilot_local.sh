#!/usr/bin/env bash
# Pilot DEP allocations at W0.055%/A0.275% with a configurable uniform floor.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
SHAPE_FLOOR="${SHAPE_FLOOR:-0.60}"
FLOOR_TAG="${FLOOR_TAG:-60}"
BASE_ALLOCATION_JSON="$ANALYSIS_DIR/dep_w4a4_probe_guided_outlier8_w0p055_a0p275_floor${FLOOR_TAG}_all.json"
ALLOCATION_JSON="$ANALYSIS_DIR/dep_w4a4_probe_guided_outlier8_w0p055_a0p275_floor${FLOOR_TAG}_pilot.json"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/dep"
SEED="${SEED:-20260806}"
STRATEGIES="${STRATEGIES:-uniform probing inverse early late random_s29}"
RUN_SUFFIX="fixedseed${SEED}_cal16_dep_e3_retry1"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_probe_guided_out8_floor${FLOOR_TAG}_pilot_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tstrategy\tnominal_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task dep \
  --low-weight-ratio 0.00055 \
  --low-activation-ratio 0.00275 \
  --boundary-weight-ratio 0.00055 \
  --boundary-activation-ratio 0.00275 \
  --shape-floor "$SHAPE_FLOOR" \
  --output "$BASE_ALLOCATION_JSON"

"$PYTHON" - "$BASE_ALLOCATION_JSON" "$ALLOCATION_JSON" "$STRATEGIES" <<'PY'
import json
import sys

source_path, output_path, strategy_text = sys.argv[1:]
strategies = strategy_text.split()
source = json.load(open(source_path, encoding="utf-8"))
source["allocations"] = [
    item
    for item in source["allocations"]
    if item["budget"] == "low" and item["strategy"] in strategies
]
found = {item["strategy"] for item in source["allocations"]}
missing = sorted(set(strategies) - found)
if missing:
    raise SystemExit(f"Missing low strategies: {missing}")
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(source, writer, indent=2)
    writer.write("\n")
PY

while IFS=$'\t' read -r strategy nominal_bop weight_ratios act_ratios; do
  run_name="ptq_pgfloor${FLOOR_TAG}_w4a4_out8_w0p055_a0p275_${strategy}_${RUN_SUFFIX}"
  run_dir="$RUN_ROOT/$run_name"
  log_path="$LOG_DIR/${strategy}.log"
  echo "[$(date --iso-8601=seconds)] START strategy=$strategy run=$run_name"
  if [[ -s "$run_dir/val_metrics.json" && \
        -s "$run_dir/quant_summary.json" && \
        -s "$run_dir/outlier_runtime_stats.json" && \
        -s "$run_dir/activation_calibration_stats.json" ]]; then
    status=skipped_existing
    skipped=$((skipped + 1))
  elif env \
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
      PTQ_ALLOCATION_STRATEGY="${strategy}_floor${FLOOR_TAG}" \
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
    "$SEED" "$strategy" "$nominal_bop" "$run_name" "$status" \
    "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} strategy=$strategy"
done < <("$PYTHON" - "$ALLOCATION_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    activations = ",".join(
        f"{value:.12g}" for value in item["activation_layer_ratios"]
    )
    print(
        item["strategy"], f'{item["nominal_bop_overhead"]:.12g}',
        weights, activations, sep="\t"
    )
PY
)

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
