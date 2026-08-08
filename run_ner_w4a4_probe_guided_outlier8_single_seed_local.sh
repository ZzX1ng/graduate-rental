#!/usr/bin/env bash
# Run NER low/boundary pilots and one measured-BOP correction pass.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
CORRECTOR="$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py"
SUMMARIZER="$PROJECT_ROOT/tools/summarize_probe_guided_single_seed_budgets.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
BASE_ALLOCATION_JSON="$ANALYSIS_DIR/ner_w4a4_probe_guided_outlier8_allocations.json"
ALLOCATION_JSON="$ANALYSIS_DIR/ner_w4a4_probe_guided_outlier8_single_seed_allocations.json"
CORRECTED_JSON="$ANALYSIS_DIR/ner_w4a4_probe_guided_outlier8_bopfix2_fixedseed20260806_allocations.json"
SUMMARY_PREFIX="$ANALYSIS_DIR/ner_w4a4_probe_guided_outlier8_low_boundary_fixedseed20260806"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/ner"
SEED="${SEED:-20260806}"
STRATEGIES="uniform probing inverse early late random_s29"
RUN_SUFFIX="cal16_ner_e3"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/ner_w4a4_probe_guided_out8_single_seed_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0
pilot_failures=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tphase\tbudget\tstrategy\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task ner \
  --low-weight-ratio 0.0003125 \
  --low-activation-ratio 0.0015625 \
  --boundary-weight-ratio 0.0004375 \
  --boundary-activation-ratio 0.0021875 \
  --output "$BASE_ALLOCATION_JSON"

"$PYTHON" - "$BASE_ALLOCATION_JSON" "$ALLOCATION_JSON" "$STRATEGIES" <<'PY'
import json
import sys

source_path, output_path, strategy_text = sys.argv[1:]
strategies = strategy_text.split()
source = json.load(open(source_path, encoding="utf-8"))
source["allocations"] = [
    item for item in source["allocations"] if item["strategy"] in strategies
]
for budget in ("low", "boundary"):
    found = {
        item["strategy"]
        for item in source["allocations"]
        if item["budget"] == budget
    }
    missing = sorted(set(strategies) - found)
    if missing:
        raise SystemExit(f"Missing {budget} strategies: {missing}")
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(source, writer, indent=2)
    writer.write("\n")
PY

while IFS=$'\t' read -r budget strategy nominal_bop weight_ratios act_ratios; do
  run_name="ptq_pg_w4a4_out8_${budget}_${strategy}_fixedseed${SEED}_${RUN_SUFFIX}"
  run_dir="$RUN_ROOT/$run_name"
  log_path="$LOG_DIR/pilot_${budget}_${strategy}.log"
  echo "[$(date --iso-8601=seconds)] START phase=pilot budget=$budget strategy=$strategy"
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
      PTQ_RUN_SEED="$SEED" \
      PTQ_ALLOCATION_STRATEGY="$strategy" \
      PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    status=completed
  else
    status=failed
    failures=$((failures + 1))
    pilot_failures=$((pilot_failures + 1))
  fi
  printf '%s\tpilot\t%s\t%s\t\t%s\t%s\t%s\t%s\n' \
    "$SEED" "$budget" "$strategy" "$nominal_bop" "$run_name" "$status" \
    "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} phase=pilot budget=$budget strategy=$strategy"
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
        item["budget"], item["strategy"],
        f'{item["nominal_bop_overhead"]:.12g}', weights, activations, sep="\t"
    )
PY
)

if [[ "$pilot_failures" -gt 0 ]]; then
  echo "[$(date --iso-8601=seconds)] STOP pilot_failures=$pilot_failures status=$STATUS_TSV"
  exit "$failures"
fi

"$PYTHON" "$CORRECTOR" \
  --allocations "$ALLOCATION_JSON" \
  --run-root "$RUN_ROOT" \
  --run-tag "fixedseed${SEED}" \
  --run-suffix "$RUN_SUFFIX" \
  --output "$CORRECTED_JSON"

while IFS=$'\t' read -r budget strategy scale target_bop weight_ratios act_ratios; do
  run_name="ptq_pg_w4a4_out8_${budget}_${strategy}_bopfix2_fixedseed${SEED}_${RUN_SUFFIX}"
  run_dir="$RUN_ROOT/$run_name"
  log_path="$LOG_DIR/bopfix2_${budget}_${strategy}.log"
  echo "[$(date --iso-8601=seconds)] START phase=bopfix2 budget=$budget strategy=$strategy scale=$scale"
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
      PTQ_RUN_SEED="$SEED" \
      PTQ_ALLOCATION_STRATEGY="${strategy}_bopfix2_fixedseed${SEED}" \
      PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    status=completed
  else
    status=failed
    failures=$((failures + 1))
  fi
  printf '%s\tbopfix2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SEED" "$budget" "$strategy" "$scale" "$target_bop" "$run_name" \
    "$status" "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} phase=bopfix2 budget=$budget strategy=$strategy"
done < <("$PYTHON" - "$CORRECTED_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    activations = ",".join(
        f"{value:.12g}" for value in item["activation_layer_ratios"]
    )
    print(
        item["budget"], item["strategy"], f'{item["bop_correction_scale"]:.12g}',
        f'{item["target_measured_uniform_bop_overhead"]:.12g}',
        weights, activations, sep="\t"
    )
PY
)

if [[ "$failures" -eq 0 ]]; then
  "$PYTHON" "$SUMMARIZER" \
    --run-root "$RUN_ROOT" \
    --task ner \
    --seed "$SEED" \
    --budgets low boundary \
    --strategies $STRATEGIES \
    --run-suffix "$RUN_SUFFIX" \
    --output-prefix "$SUMMARY_PREFIX"
fi

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
