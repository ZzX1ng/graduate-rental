#!/usr/bin/env bash
# Run DEP floor50 uniform/probing pairs and probing BOP fixes for three seeds.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
CORRECTOR="$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
BASE_JSON="$ANALYSIS_DIR/dep_w4a4_floor50_low_boundary_paired_multiseed_base.json"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/dep"
SEEDS="${SEEDS:-20260807 20260808 20260809}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_floor50_low_boundary_paired_multiseed_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tphase\tbudget\tlabel\tstrategy\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task dep \
  --low-weight-ratio 0.00055 \
  --low-activation-ratio 0.00275 \
  --boundary-weight-ratio 0.00070 \
  --boundary-activation-ratio 0.00350 \
  --shape-floor 0.50 \
  --output "$BASE_JSON"

run_ptq() {
  local seed="$1"
  local phase="$2"
  local budget="$3"
  local label="$4"
  local strategy="$5"
  local scale="$6"
  local target_bop="$7"
  local weight_ratios="$8"
  local act_ratios="$9"
  local run_name="${10}"
  local run_dir="$RUN_ROOT/$run_name"
  local log_path="$LOG_DIR/${seed}_${phase}_${budget}_${strategy}.log"
  local status

  echo "[$(date --iso-8601=seconds)] START seed=$seed phase=$phase budget=$budget strategy=$strategy"
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
      PTQ_RUN_SEED="$seed" \
      PTQ_ALLOCATION_STRATEGY="${strategy}_floor50_${phase}_fixedseed${seed}" \
      PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    status=completed
    completed=$((completed + 1))
  else
    status=failed
    failures=$((failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$seed" "$phase" "$budget" "$label" "$strategy" "$scale" \
    "$target_bop" "$run_name" "$status" "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} seed=$seed phase=$phase budget=$budget strategy=$strategy"
  [[ "$status" != "failed" ]]
}

for seed in $SEEDS; do
  seed_failures_before="$failures"
  pilot_json="$ANALYSIS_DIR/dep_w4a4_floor50_low_boundary_pilot_fixedseed${seed}_manifest.json"
  corrected_json="$ANALYSIS_DIR/dep_w4a4_floor50_low_boundary_bopfix2_fixedseed${seed}_allocations.json"

  "$PYTHON" - "$BASE_JSON" "$pilot_json" "$seed" <<'PY'
import json
import sys

source_path, output_path, seed = sys.argv[1:]
payload = json.load(open(source_path, encoding="utf-8"))
selected = []
for item in payload["allocations"]:
    if item["strategy"] not in {"uniform", "probing"}:
        continue
    label = "w0p055_a0p275" if item["budget"] == "low" else "w0p070_a0p350"
    source_run = (
        f"ptq_pgfloor50_w4a4_out8_{label}_{item['strategy']}_"
        f"fixedseed{seed}_cal16_dep_e3_retry1"
    )
    selected.append({**item, "source_run": source_run})
payload["allocations"] = selected
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(payload, writer, indent=2)
    writer.write("\n")
PY

  while IFS=$'\t' read -r budget strategy nominal_bop weight_ratios act_ratios; do
    if [[ "$budget" == "low" ]]; then
      label=w0p055_a0p275
    else
      label=w0p070_a0p350
    fi
    run_name="ptq_pgfloor50_w4a4_out8_${label}_${strategy}_fixedseed${seed}_cal16_dep_e3_retry1"
    run_ptq \
      "$seed" pilot "$budget" "$label" "$strategy" "" "$nominal_bop" \
      "$weight_ratios" "$act_ratios" "$run_name" || true
  done < <("$PYTHON" - "$pilot_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for budget in ("low", "boundary"):
    for strategy in ("uniform", "probing"):
        item = next(
            item
            for item in payload["allocations"]
            if item["budget"] == budget and item["strategy"] == strategy
        )
        print(
            budget,
            strategy,
            f'{item["nominal_bop_overhead"]:.12g}',
            ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"]),
            ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"]),
            sep="\t",
        )
PY
  )

  if [[ "$failures" -gt "$seed_failures_before" ]]; then
    echo "[$(date --iso-8601=seconds)] SKIP_BOPFIX seed=$seed due_to_pilot_failure"
    continue
  fi

  "$PYTHON" "$CORRECTOR" \
    --allocations "$pilot_json" \
    --run-root "$RUN_ROOT" \
    --output "$corrected_json"

  while IFS=$'\t' read -r budget strategy scale target_bop weight_ratios act_ratios; do
    if [[ "$budget" == "low" ]]; then
      label=w0p055_a0p275
    else
      label=w0p070_a0p350
    fi
    run_name="ptq_pgfloor50_w4a4_out8_${label}_${strategy}_bopfix2_fixedseed${seed}_cal16_dep_e3_retry1"
    run_ptq \
      "$seed" bopfix2 "$budget" "$label" "$strategy" "$scale" "$target_bop" \
      "$weight_ratios" "$act_ratios" "$run_name" || true
  done < <("$PYTHON" - "$corrected_json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload["allocations"]:
    print(
        item["budget"],
        item["strategy"],
        f'{item["bop_correction_scale"]:.12g}',
        f'{item["target_measured_uniform_bop_overhead"]:.12g}',
        ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"]),
        ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"]),
        sep="\t",
    )
PY
  )
done

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
