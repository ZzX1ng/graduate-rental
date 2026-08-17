#!/usr/bin/env bash
# Complete the SemEval boundary-budget six-strategy comparison for seeds 20260807-09.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_fixedseed_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
CORRECTOR="$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
SEEDS="${SEEDS:-20260807 20260808 20260809}"
STRATEGIES="uniform probing inverse early late random_s29"
RUN_SUFFIX="cal16_semeval_e10"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_boundary_multiseed_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
ALLOCATION_JSON="$ANALYSIS_DIR/semeval_w4a4_boundary_multiseed_allocations.json"
failures=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tphase\tstrategy\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task semeval \
  --output "$ALLOCATION_JSON"

"$PYTHON" - "$ALLOCATION_JSON" "$STRATEGIES" <<'PY'
import json
import sys

path, strategy_text = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
strategies = strategy_text.split()
payload["allocations"] = [
    item for item in payload["allocations"]
    if item["budget"] == "boundary" and item["strategy"] in strategies
]
with open(path, "w", encoding="utf-8") as writer:
    json.dump(payload, writer, indent=2)
    writer.write("\n")
PY

for seed in $SEEDS; do
  seed_failures=0
  while IFS=$'\t' read -r strategy nominal_bop weight_ratios act_ratios; do
    run_name="ptq_pg_w4a4_out8_boundary_${strategy}_fixedseed${seed}_${RUN_SUFFIX}"
    run_dir="$RUN_ROOT/$run_name"
    log_path="$LOG_DIR/pilot_${seed}_${strategy}.log"
    if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" && \
          -s "$run_dir/outlier_runtime_stats.json" && \
          -s "$run_dir/activation_calibration_stats.json" ]]; then
      status=skipped_existing
    elif env \
        PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 \
        PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
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
      >> "$STATUS_TSV"
  done < <("$PYTHON" - "$ALLOCATION_JSON" <<'PY'
import json
import sys

for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(item["strategy"], f'{item["nominal_bop_overhead"]:.12g}', weights, acts, sep="\t")
PY
)

  if [[ "$seed_failures" -gt 0 ]]; then
    continue
  fi

  corrected_json="$ANALYSIS_DIR/semeval_w4a4_boundary_bopfix2_fixedseed${seed}_allocations.json"
  "$PYTHON" "$CORRECTOR" \
    --allocations "$ALLOCATION_JSON" \
    --run-root "$RUN_ROOT" \
    --run-tag "fixedseed${seed}" \
    --run-suffix "$RUN_SUFFIX" \
    --output "$corrected_json"

  while IFS=$'\t' read -r strategy scale target_bop weight_ratios act_ratios; do
    run_name="ptq_pg_w4a4_out8_boundary_${strategy}_bopfix2_fixedseed${seed}_${RUN_SUFFIX}"
    run_dir="$RUN_ROOT/$run_name"
    log_path="$LOG_DIR/bopfix2_${seed}_${strategy}.log"
    if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" && \
          -s "$run_dir/outlier_runtime_stats.json" && \
          -s "$run_dir/activation_calibration_stats.json" ]]; then
      status=skipped_existing
    elif env \
        PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 \
        PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
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
      >> "$STATUS_TSV"
  done < <("$PYTHON" - "$corrected_json" <<'PY'
import json
import sys

for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(item["strategy"], f'{item["bop_correction_scale"]:.12g}',
          f'{item["target_measured_uniform_bop_overhead"]:.12g}', weights, acts, sep="\t")
PY
)
done

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
