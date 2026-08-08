#!/usr/bin/env bash
# Run DEP floor50 low/boundary six-strategy comparisons at one fixed seed.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
ALLOCATION_JSON="$ANALYSIS_DIR/dep_w4a4_floor50_low_boundary_six_strategy_allocations.json"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/dep"
SEED="${SEED:-20260806}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_floor50_low_boundary_six_strategy_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
STRATEGIES="uniform probing inverse early late random_s29"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tbudget\tlabel\tstrategy\tnominal_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task dep \
  --low-weight-ratio 0.00055 \
  --low-activation-ratio 0.00275 \
  --boundary-weight-ratio 0.00070 \
  --boundary-activation-ratio 0.00350 \
  --shape-floor 0.50 \
  --output "$ALLOCATION_JSON"

while IFS=$'\t' read -r budget strategy nominal_bop weight_ratios act_ratios; do
  if [[ "$budget" == "low" ]]; then
    label=w0p055_a0p275
  else
    label=w0p070_a0p350
  fi

  if [[ "$strategy" == "uniform" ]]; then
    run_name="ptq_oa_w4a4_dualout8_boundaryrefine_${label}_fixedseed${SEED}_cal16_ms256_dep_e3_retry1"
  else
    run_name="ptq_pgfloor50_w4a4_out8_${label}_${strategy}_fixedseed${SEED}_cal16_dep_e3_retry1"
  fi
  run_dir="$RUN_ROOT/$run_name"
  log_path="$LOG_DIR/${budget}_${strategy}.log"

  echo "[$(date --iso-8601=seconds)] START budget=$budget strategy=$strategy run=$run_name"
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
      PTQ_ALLOCATION_STRATEGY="${strategy}_floor50" \
      PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    status=completed
    completed=$((completed + 1))
  else
    status=failed
    failures=$((failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SEED" "$budget" "$label" "$strategy" "$nominal_bop" "$run_name" \
    "$status" "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} budget=$budget strategy=$strategy"
done < <("$PYTHON" - "$ALLOCATION_JSON" "$STRATEGIES" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
strategies = sys.argv[2].split()
for budget in ("low", "boundary"):
    for strategy in strategies:
        item = next(
            item
            for item in payload["allocations"]
            if item["budget"] == budget and item["strategy"] == strategy
        )
        weights = ",".join(
            f"{value:.12g}" for value in item["weight_layer_ratios"]
        )
        activations = ",".join(
            f"{value:.12g}" for value in item["activation_layer_ratios"]
        )
        print(
            budget,
            strategy,
            f'{item["nominal_bop_overhead"]:.12g}',
            weights,
            activations,
            sep="\t",
        )
PY
)

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
