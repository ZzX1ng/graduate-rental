#!/usr/bin/env bash
# Add the late control to the completed SemEval low-budget calibration seeds.

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
FILTERED_ALLOCATION_JSON="$ANALYSIS_DIR/semeval_w4a4_probe_guided_outlier8_low_late_multiseed_allocations.json"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
SEEDS="${SEEDS:-20260807 20260808 20260809}"
SUMMARY_SEEDS="${SUMMARY_SEEDS:-20260806 $SEEDS}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_w4a4_probe_guided_out8_late_multiseed_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
SUMMARY_PREFIX="$ANALYSIS_DIR/semeval_w4a4_probe_guided_outlier8_low_multiseed_20260806_09"
failures=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tphase\tstrategy\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task semeval \
  --output "$BASE_ALLOCATION_JSON"

"$PYTHON" - "$BASE_ALLOCATION_JSON" "$FILTERED_ALLOCATION_JSON" <<'PY'
import json
import sys

source_path, output_path = sys.argv[1:]
source = json.load(open(source_path, encoding="utf-8"))
source["allocations"] = [
    item
    for item in source["allocations"]
    if item["budget"] == "low" and item["strategy"] in {"uniform", "late"}
]
if {item["strategy"] for item in source["allocations"]} != {"uniform", "late"}:
    raise SystemExit("Expected low-budget uniform and late allocations")
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(source, writer, indent=2)
    writer.write("\n")
PY

while IFS=$'\t' read -r nominal_bop weight_ratios act_ratios; do
  for seed in $SEEDS; do
    uniform_run="$RUN_ROOT/ptq_pg_w4a4_out8_low_uniform_fixedseed${seed}_cal16_semeval_e10"
    corrected_json="$ANALYSIS_DIR/semeval_w4a4_probe_guided_outlier8_low_late_bopfix2_fixedseed${seed}_allocations.json"
    pilot_name="ptq_pg_w4a4_out8_low_late_fixedseed${seed}_cal16_semeval_e10"
    pilot_dir="$RUN_ROOT/$pilot_name"
    pilot_log="$LOG_DIR/fixedseed${seed}_pilot_late.log"

    if [[ ! -s "$uniform_run/quant_summary.json" ]]; then
      echo "Missing uniform anchor for seed $seed: $uniform_run" >&2
      failures=$((failures + 1))
      continue
    fi

    if [[ -s "$pilot_dir/val_metrics.json" && \
          -s "$pilot_dir/quant_summary.json" && \
          -s "$pilot_dir/outlier_runtime_stats.json" && \
          -s "$pilot_dir/activation_calibration_stats.json" ]]; then
      pilot_status=skipped_existing
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
        PTQ_ALLOCATION_STRATEGY=late \
        PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" \
        RUN_NAME="$pilot_name" \
        bash "$WORKER" >"$pilot_log" 2>&1; then
      pilot_status=completed
    else
      pilot_status=failed
      failures=$((failures + 1))
    fi
    printf '%s\tpilot\tlate\t\t%s\t%s\t%s\t%s\n' \
      "$seed" "$nominal_bop" "$pilot_name" "$pilot_status" "$pilot_log" \
      >> "$STATUS_TSV"
    if [[ "$pilot_status" == failed ]]; then
      continue
    fi

    "$PYTHON" "$CORRECTOR" \
      --allocations "$FILTERED_ALLOCATION_JSON" \
      --run-root "$RUN_ROOT" \
      --run-tag "fixedseed${seed}" \
      --output "$corrected_json"

    while IFS=$'\t' read -r scale target_bop corrected_weights corrected_acts; do
      fixed_name="ptq_pg_w4a4_out8_low_late_bopfix2_fixedseed${seed}_cal16_semeval_e10"
      fixed_dir="$RUN_ROOT/$fixed_name"
      fixed_log="$LOG_DIR/fixedseed${seed}_bopfix2_late.log"
      if [[ -s "$fixed_dir/val_metrics.json" && \
            -s "$fixed_dir/quant_summary.json" && \
            -s "$fixed_dir/outlier_runtime_stats.json" && \
            -s "$fixed_dir/activation_calibration_stats.json" ]]; then
        fixed_status=skipped_existing
      elif env \
          PTQ_WEIGHT_BITS=4 \
          PTQ_ACT_BITS=4 \
          PTQ_WEIGHT_OUTLIER_RATIO=0 \
          PTQ_ACT_OUTLIER_RATIO=0 \
          PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$corrected_weights" \
          PTQ_ACT_OUTLIER_LAYER_RATIOS="$corrected_acts" \
          PTQ_OUTLIER_BITS=8 \
          PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
          PTQ_ACT_CALIBRATION_BATCHES=16 \
          PTQ_ACT_CALIBRATION_MAX_VALUES=65536 \
          PTQ_RUN_SEED="$seed" \
          PTQ_ALLOCATION_STRATEGY="late_bopfix2_fixedseed${seed}" \
          PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" \
          RUN_NAME="$fixed_name" \
          bash "$WORKER" >"$fixed_log" 2>&1; then
        fixed_status=completed
      else
        fixed_status=failed
        failures=$((failures + 1))
      fi
      printf '%s\tbopfix2\tlate\t%s\t%s\t%s\t%s\t%s\n' \
        "$seed" "$scale" "$target_bop" "$fixed_name" "$fixed_status" "$fixed_log" \
        >> "$STATUS_TSV"
    done < <("$PYTHON" - "$corrected_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if len(payload["allocations"]) != 1 or payload["allocations"][0]["strategy"] != "late":
    raise SystemExit("Expected one corrected late allocation")
item = payload["allocations"][0]
weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
activations = ",".join(
    f"{value:.12g}" for value in item["activation_layer_ratios"]
)
print(
    f'{item["bop_correction_scale"]:.12g}',
    f'{item["target_measured_uniform_bop_overhead"]:.12g}',
    weights, activations, sep="\t"
)
PY
)
  done
done < <("$PYTHON" - "$FILTERED_ALLOCATION_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
item = next(item for item in payload["allocations"] if item["strategy"] == "late")
weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
activations = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
print(f'{item["nominal_bop_overhead"]:.12g}', weights, activations, sep="\t")
PY
)

if [[ "$failures" -eq 0 ]]; then
  "$PYTHON" "$SUMMARIZER" \
    --run-root "$RUN_ROOT" \
    --seeds $SUMMARY_SEEDS \
    --strategies uniform probing inverse early late random_s29 \
    --output-prefix "$SUMMARY_PREFIX"
fi

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
