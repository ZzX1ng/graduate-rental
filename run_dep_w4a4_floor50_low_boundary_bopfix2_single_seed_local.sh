#!/usr/bin/env bash
# Match DEP floor50 strategy overheads to measured uniform BOP at one fixed seed.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
CORRECTOR="$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
BASE_JSON="$ANALYSIS_DIR/dep_w4a4_floor50_low_boundary_all_allocations.json"
PILOT_JSON="$ANALYSIS_DIR/dep_w4a4_floor50_low_boundary_pilot_manifest.json"
CORRECTED_JSON="$ANALYSIS_DIR/dep_w4a4_floor50_low_boundary_bopfix2_fixedseed20260806_allocations.json"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/dep"
SEED="${SEED:-20260806}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/dep_w4a4_floor50_low_boundary_bopfix2_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
STRATEGIES="uniform probing inverse early late random_s29"
failures=0
completed=0
skipped=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tbudget\tlabel\tstrategy\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

"$PYTHON" "$GENERATOR" \
  --probing-csv "$PROBING_CSV" \
  --task dep \
  --low-weight-ratio 0.00055 \
  --low-activation-ratio 0.00275 \
  --boundary-weight-ratio 0.00070 \
  --boundary-activation-ratio 0.00350 \
  --shape-floor 0.50 \
  --output "$BASE_JSON"

"$PYTHON" - "$BASE_JSON" "$PILOT_JSON" "$SEED" "$STRATEGIES" <<'PY'
import json
import sys

source_path, output_path, seed, strategy_text = sys.argv[1:]
strategies = strategy_text.split()
payload = json.load(open(source_path, encoding="utf-8"))
selected = []
for item in payload["allocations"]:
    if item["strategy"] not in strategies:
        continue
    label = "w0p055_a0p275" if item["budget"] == "low" else "w0p070_a0p350"
    if item["strategy"] == "uniform":
        source_run = (
            "ptq_oa_w4a4_dualout8_boundaryrefine_"
            f"{label}_fixedseed{seed}_cal16_ms256_dep_e3_retry1"
        )
    else:
        source_run = (
            f"ptq_pgfloor50_w4a4_out8_{label}_{item['strategy']}_"
            f"fixedseed{seed}_cal16_dep_e3_retry1"
        )
    selected.append({**item, "source_run": source_run})

for budget in ("low", "boundary"):
    found = {item["strategy"] for item in selected if item["budget"] == budget}
    missing = sorted(set(strategies) - found)
    if missing:
        raise SystemExit(f"Missing {budget} strategies: {missing}")

payload["allocations"] = selected
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(payload, writer, indent=2)
    writer.write("\n")
PY

"$PYTHON" "$CORRECTOR" \
  --allocations "$PILOT_JSON" \
  --run-root "$RUN_ROOT" \
  --output "$CORRECTED_JSON"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  "$PYTHON" - "$CORRECTED_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print("uniform_targets", payload["uniform_measured_bop_targets"])
for item in payload["allocations"]:
    print(
        item["budget"],
        item["strategy"],
        f'scale={item["bop_correction_scale"]:.8f}',
        f'source={item["source_measured_bop_overhead"]:.8f}',
        f'target={item["target_measured_uniform_bop_overhead"]:.8f}',
    )
PY
  exit 0
fi

while IFS=$'\t' read -r budget strategy scale target_bop weight_ratios act_ratios; do
  if [[ "$budget" == "low" ]]; then
    label=w0p055_a0p275
  else
    label=w0p070_a0p350
  fi
  run_name="ptq_pgfloor50_w4a4_out8_${label}_${strategy}_bopfix2_fixedseed${SEED}_cal16_dep_e3_retry1"
  run_dir="$RUN_ROOT/$run_name"
  log_path="$LOG_DIR/${budget}_${strategy}.log"

  echo "[$(date --iso-8601=seconds)] START budget=$budget strategy=$strategy scale=$scale target=$target_bop"
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
      PTQ_ALLOCATION_STRATEGY="${strategy}_floor50_bopfix2_fixedseed${SEED}" \
      PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    status=completed
    completed=$((completed + 1))
  else
    status=failed
    failures=$((failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SEED" "$budget" "$label" "$strategy" "$scale" "$target_bop" \
    "$run_name" "$status" "$log_path" >> "$STATUS_TSV"
  echo "[$(date --iso-8601=seconds)] ${status^^} budget=$budget strategy=$strategy"
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
        item["budget"],
        item["strategy"],
        f'{item["bop_correction_scale"]:.12g}',
        f'{item["target_measured_uniform_bop_overhead"]:.12g}',
        weights,
        activations,
        sep="\t",
    )
PY
)

echo "[$(date --iso-8601=seconds)] FINISH completed=$completed skipped=$skipped failures=$failures status=$STATUS_TSV"
exit "$failures"
