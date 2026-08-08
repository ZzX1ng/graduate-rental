#!/usr/bin/env bash
# Replicate the SemEval low-budget probing comparison across calibration seeds.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_fixedseed_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
CORRECTOR="$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py"
SUMMARIZER="$PROJECT_ROOT/tools/summarize_probe_guided_multiseed.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
BASE_ALLOCATION_JSON="$ANALYSIS_DIR/semeval_w4a4_probe_guided_outlier8_allocations.json"
FILTERED_ALLOCATION_JSON="$ANALYSIS_DIR/semeval_w4a4_probe_guided_outlier8_low_multiseed_allocations.json"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
SEEDS="${SEEDS:-20260807 20260808 20260809}"
SUMMARY_SEEDS="${SUMMARY_SEEDS:-20260806 $SEEDS}"
STRATEGIES="${STRATEGIES:-uniform probing inverse early random_s29}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_w4a4_probe_guided_out8_low_multiseed_$RUN_ID"
MASTER_STATUS="$LOG_DIR/status.tsv"
SUMMARY_PREFIX="$ANALYSIS_DIR/semeval_w4a4_probe_guided_outlier8_low_multiseed_20260806_09"
failures=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tphase\tstrategy\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$MASTER_STATUS"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task semeval \
  --output "$BASE_ALLOCATION_JSON"

"$PYTHON" - "$BASE_ALLOCATION_JSON" "$FILTERED_ALLOCATION_JSON" "$STRATEGIES" <<'PY'
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
    raise SystemExit(f"Missing allocation strategies: {missing}")
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(source, writer, indent=2)
    writer.write("\n")
PY

for seed in $SEEDS; do
  seed_log_dir="$LOG_DIR/seed_$seed"
  seed_status="$seed_log_dir/status.tsv"
  corrected_json="$ANALYSIS_DIR/semeval_w4a4_probe_guided_outlier8_low_bopfix2_fixedseed${seed}_allocations.json"
  seed_failures=0
  mkdir -p "$seed_log_dir"
  printf 'seed\tphase\tstrategy\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$seed_status"

  while IFS=$'\t' read -r strategy nominal_bop weight_ratios act_ratios; do
    run_name="ptq_pg_w4a4_out8_low_${strategy}_fixedseed${seed}_cal16_semeval_e10"
    run_dir="$RUN_ROOT/$run_name"
    log_path="$seed_log_dir/pilot_${strategy}.log"
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
        PTQ_RUN_SEED="$seed" \
        PTQ_ALLOCATION_STRATEGY="$strategy" \
        PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" \
        RUN_NAME="$run_name" \
        bash "$WORKER" >"$log_path" 2>&1; then
      status=completed
    else
      status=failed
      failures=$((failures + 1))
      seed_failures=$((seed_failures + 1))
    fi
    printf '%s\tpilot\t%s\t\t%s\t%s\t%s\t%s\n' \
      "$seed" "$strategy" "$nominal_bop" "$run_name" "$status" "$log_path" \
      | tee -a "$seed_status" >> "$MASTER_STATUS"
  done < <("$PYTHON" - "$FILTERED_ALLOCATION_JSON" <<'PY'
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

  if [[ "$seed_failures" -gt 0 ]]; then
    echo "[$(date --iso-8601=seconds)] seed=$seed pilot failures=$seed_failures; skipping BOP fix"
    continue
  fi

  "$PYTHON" "$CORRECTOR" \
    --allocations "$FILTERED_ALLOCATION_JSON" \
    --run-root "$RUN_ROOT" \
    --run-tag "fixedseed${seed}" \
    --output "$corrected_json"

  while IFS=$'\t' read -r strategy scale target_bop weight_ratios act_ratios; do
    run_name="ptq_pg_w4a4_out8_low_${strategy}_bopfix2_fixedseed${seed}_cal16_semeval_e10"
    run_dir="$RUN_ROOT/$run_name"
    log_path="$seed_log_dir/bopfix2_${strategy}.log"
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
        PTQ_RUN_SEED="$seed" \
        PTQ_ALLOCATION_STRATEGY="${strategy}_bopfix2_fixedseed${seed}" \
        PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" \
        RUN_NAME="$run_name" \
        bash "$WORKER" >"$log_path" 2>&1; then
      status=completed
    else
      status=failed
      failures=$((failures + 1))
    fi
    printf '%s\tbopfix2\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$strategy" "$scale" "$target_bop" "$run_name" "$status" "$log_path" \
      | tee -a "$seed_status" >> "$MASTER_STATUS"
  done < <("$PYTHON" - "$corrected_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    activations = ",".join(
        f"{value:.12g}" for value in item["activation_layer_ratios"]
    )
    print(
        item["strategy"], f'{item["bop_correction_scale"]:.12g}',
        f'{item["target_measured_uniform_bop_overhead"]:.12g}',
        weights, activations, sep="\t"
    )
PY
)
done

if [[ "$failures" -eq 0 ]]; then
  "$PYTHON" "$SUMMARIZER" \
    --run-root "$RUN_ROOT" \
    --seeds $SUMMARY_SEEDS \
    --strategies $STRATEGIES \
    --output-prefix "$SUMMARY_PREFIX"
fi

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$MASTER_STATUS"
exit "$failures"
