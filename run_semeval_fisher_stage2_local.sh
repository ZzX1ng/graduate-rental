#!/usr/bin/env bash
# Run Fisher and probing+Fisher allocations at low/boundary across four seeds.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_fixedseed_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_fisher_guided_outlier_allocations.py"
CORRECTOR="$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py"
ANALYSIS="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
ALLOCATIONS="$ANALYSIS/semeval_fisher_guided_w4a4_outlier8_allocations.json"
SEEDS="${SEEDS:-20260806 20260807 20260808 20260809}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_fisher_stage2_$(date +%Y%m%d_%H%M%S)"
failures=0

mkdir -p "$LOG_DIR" "$ANALYSIS"
printf 'seed\tbudget\tstrategy\tphase\tscale\ttarget_bop\trun_name\tstatus\tlog\n' \
  > "$LOG_DIR/status.tsv"

for seed in $SEEDS; do
  seed_allocations="$ANALYSIS/semeval_fisher_guided_w4a4_outlier8_fixedseed${seed}_allocations.json"
  "$PYTHON" "$GENERATOR" \
    --fisher-stats "$ANALYSIS/semeval_fisher_w4a4_seed${seed}_cal16.json" \
    --probing-csv "$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv" \
    --task semeval --beta 0.5 --output "$seed_allocations"
  seed_failures=0
  while IFS=$'\t' read -r budget strategy nominal_bop weights acts; do
    run_name="ptq_fisher_w4a4_out8_${budget}_${strategy}_fixedseed${seed}_cal16_semeval_e10"
    run_dir="$RUN_ROOT/$run_name"
    log="$LOG_DIR/pilot_${seed}_${budget}_${strategy}.log"
    if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" && \
          -s "$run_dir/outlier_runtime_stats.json" ]]; then
      status=skipped_existing
    elif env \
        PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 \
        PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
        PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$weights" \
        PTQ_ACT_OUTLIER_LAYER_RATIOS="$acts" PTQ_OUTLIER_BITS=8 \
        PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 PTQ_ACT_CALIBRATION_BATCHES=16 \
        PTQ_ACT_CALIBRATION_MAX_VALUES=65536 PTQ_RUN_SEED="$seed" \
        PTQ_ALLOCATION_STRATEGY="$strategy" \
        PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" RUN_NAME="$run_name" \
        bash "$WORKER" >"$log" 2>&1; then
      status=completed
    else
      status=failed; failures=$((failures + 1)); seed_failures=$((seed_failures + 1))
    fi
    printf '%s\t%s\t%s\tpilot\t\t%s\t%s\t%s\t%s\n' \
      "$seed" "$budget" "$strategy" "$nominal_bop" "$run_name" "$status" "$log" \
      >> "$LOG_DIR/status.tsv"
  done < <("$PYTHON" - "$seed_allocations" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(item["budget"], item["strategy"], f'{item["nominal_bop_overhead"]:.12g}', weights, acts, sep="\t")
PY
)

  if [[ "$seed_failures" -gt 0 ]]; then
    continue
  fi
  manifest="$ANALYSIS/semeval_fisher_stage2_fixedseed${seed}_pilot_manifest.json"
  corrected="$ANALYSIS/semeval_fisher_stage2_fixedseed${seed}_bopfix2_allocations.json"
  "$PYTHON" - "$seed_allocations" "$manifest" "$seed" <<'PY'
import json
import sys
source_path, output_path, seed = sys.argv[1:]
payload = json.load(open(source_path, encoding="utf-8"))
items = []
for item in payload["allocations"]:
    run = f"ptq_fisher_w4a4_out8_{item['budget']}_{item['strategy']}_fixedseed{seed}_cal16_semeval_e10"
    items.append({**item, "source_run": run})
    uniform = {**item, "strategy": "uniform"}
    uniform["source_run"] = f"ptq_pg_w4a4_out8_{item['budget']}_uniform_fixedseed{seed}_cal16_semeval_e10"
    items.append(uniform)
payload["allocations"] = items
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(payload, writer, indent=2)
    writer.write("\n")
PY
  "$PYTHON" "$CORRECTOR" \
    --allocations "$manifest" --run-root "$RUN_ROOT" --output "$corrected"

  while IFS=$'\t' read -r budget strategy scale target_bop weights acts; do
    run_name="ptq_fisher_w4a4_out8_${budget}_${strategy}_bopfix2_fixedseed${seed}_cal16_semeval_e10"
    run_dir="$RUN_ROOT/$run_name"
    log="$LOG_DIR/bopfix2_${seed}_${budget}_${strategy}.log"
    if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" && \
          -s "$run_dir/outlier_runtime_stats.json" ]]; then
      status=skipped_existing
    elif env \
        PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 \
        PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
        PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$weights" \
        PTQ_ACT_OUTLIER_LAYER_RATIOS="$acts" PTQ_OUTLIER_BITS=8 \
        PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 PTQ_ACT_CALIBRATION_BATCHES=16 \
        PTQ_ACT_CALIBRATION_MAX_VALUES=65536 PTQ_RUN_SEED="$seed" \
        PTQ_ALLOCATION_STRATEGY="${strategy}_bopfix2" \
        PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" RUN_NAME="$run_name" \
        bash "$WORKER" >"$log" 2>&1; then
      status=completed
    else
      status=failed; failures=$((failures + 1))
    fi
    printf '%s\t%s\t%s\tbopfix2\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$budget" "$strategy" "$scale" "$target_bop" "$run_name" "$status" "$log" \
      >> "$LOG_DIR/status.tsv"
  done < <("$PYTHON" - "$corrected" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(item["budget"], item["strategy"], f'{item["bop_correction_scale"]:.12g}',
          f'{item["target_measured_uniform_bop_overhead"]:.12g}', weights, acts, sep="\t")
PY
)
done

if [[ "$failures" -eq 0 ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/summarize_semeval_fisher_stage2.py" \
    --run-root "$RUN_ROOT" --output-prefix "$ANALYSIS/semeval_fisher_stage2_multiseed_20260806_09"
fi

echo "FINISH failures=$failures status=$LOG_DIR/status.tsv"
exit "$failures"
