#!/usr/bin/env bash
# Run NER/DEP Fisher and probing+Fisher pilots for one calibration seed.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_fisher_guided_outlier_allocations.py"
CORRECTOR="$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py"
FLOOR_CORRECTOR="$PROJECT_ROOT/tools/make_floor_preserving_bop_corrections.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
SEED="${SEED:-20260806}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/ner_dep_fisher_single_seed_${RUN_ID}"
STATUS="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR" "$ANALYSIS"
printf 'task\tseed\tbudget\tstrategy\tphase\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS"

for task in ner dep; do
  if [[ "$task" == ner ]]; then
    base_run=formal_e3_ms256_tner
    suffix=ner_e3
    low_w=0.0003125; low_a=0.0015625
    boundary_w=0.0004375; boundary_a=0.0021875
    floor=0
  else
    base_run=formal_e3_ms256_retry1
    suffix=dep_e3_retry1
    low_w=0.00055; low_a=0.00275
    boundary_w=0.00070; boundary_a=0.00350
    floor=0.50
  fi
  run_root="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/$task"
  fisher_stats="$ANALYSIS/${task}_fisher_w4a4_seed${SEED}_cal16.json"
  allocations="$ANALYSIS/${task}_fisher_guided_w4a4_outlier8_fixedseed${SEED}_allocations.json"

  if [[ ! -s "$fisher_stats" ]]; then
    TASK="$task" BASE_RUN="$base_run" FISHER_SEED="$SEED" RUN_SUFFIX="$suffix" \
      bash "$PROJECT_ROOT/run_task_fisher_stats_local.sh" \
      > "$LOG_DIR/${task}_fisher_stats.log" 2>&1 || {
        echo "Fisher collection failed for $task" >&2
        exit 1
      }
  fi

  "$PYTHON" "$GENERATOR" \
    --fisher-stats "$fisher_stats" --probing-csv "$PROBING_CSV" \
    --task "$task" --beta 0.5 \
    --low-weight-ratio "$low_w" --low-activation-ratio "$low_a" \
    --boundary-weight-ratio "$boundary_w" \
    --boundary-activation-ratio "$boundary_a" \
    --shape-floor "$floor" --output "$allocations"

  task_pilot_failures=0
  while IFS=$'\t' read -r budget strategy nominal_bop weights acts; do
    run_name="ptq_fisher_w4a4_out8_${budget}_${strategy}_fixedseed${SEED}_cal16_${suffix}"
    run_dir="$run_root/$run_name"
    log="$LOG_DIR/${task}_${budget}_${strategy}_pilot.log"
    if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" && \
          -s "$run_dir/outlier_runtime_stats.json" ]]; then
      status=skipped_existing
    elif env TASK="$task" BASE_RUN="$base_run" \
        PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 PTQ_OUTLIER_BITS=8 \
        PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
        PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$weights" \
        PTQ_ACT_OUTLIER_LAYER_RATIOS="$acts" \
        PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 PTQ_ACT_CALIBRATION_BATCHES=16 \
        PTQ_ACT_CALIBRATION_MAX_VALUES=65536 PTQ_RUN_SEED="$SEED" \
        PTQ_ALLOCATION_STRATEGY="$strategy" \
        PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" RUN_NAME="$run_name" \
        bash "$WORKER" > "$log" 2>&1; then
      status=completed
    else
      status=failed; failures=$((failures + 1)); task_pilot_failures=$((task_pilot_failures + 1))
    fi
    printf '%s\t%s\t%s\t%s\tpilot\t\t%s\t%s\t%s\t%s\n' \
      "$task" "$SEED" "$budget" "$strategy" "$nominal_bop" \
      "$run_name" "$status" "$log" >> "$STATUS"
  done < <("$PYTHON" - "$allocations" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(item["budget"], item["strategy"], f'{item["nominal_bop_overhead"]:.12g}', weights, acts, sep="\t")
PY
)
  if [[ "$task_pilot_failures" -gt 0 ]]; then
    continue
  fi

  manifest="$ANALYSIS/${task}_fisher_single_seed${SEED}_pilot_manifest.json"
  corrected="$ANALYSIS/${task}_fisher_single_seed${SEED}_bopfix2_allocations.json"
  "$PYTHON" - "$allocations" "$manifest" "$task" "$SEED" "$suffix" <<'PY'
import json
import sys
source_path, output_path, task, seed, suffix = sys.argv[1:]
payload = json.load(open(source_path, encoding="utf-8"))
selected = []
for item in payload["allocations"]:
    pilot = f"ptq_fisher_w4a4_out8_{item['budget']}_{item['strategy']}_fixedseed{seed}_cal16_{suffix}"
    selected.append({**item, "source_run": pilot})
    uniform = {**item, "strategy": "uniform"}
    if task == "ner":
        uniform["source_run"] = f"ptq_pg_w4a4_out8_{item['budget']}_uniform_fixedseed{seed}_cal16_ner_e3"
    else:
        label = "w0p055_a0p275" if item["budget"] == "low" else "w0p070_a0p350"
        uniform["source_run"] = (
            f"ptq_oa_w4a4_dualout8_boundaryrefine_{label}_fixedseed{seed}_"
            "cal16_ms256_dep_e3_retry1"
        )
    selected.append(uniform)
payload["allocations"] = selected
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(payload, writer, indent=2)
    writer.write("\n")
PY
  if [[ "$task" == dep ]]; then
    "$PYTHON" "$FLOOR_CORRECTOR" --allocations "$manifest" \
      --run-root "$run_root" --floor-fraction 0.50 --output "$corrected"
  else
    "$PYTHON" "$CORRECTOR" --allocations "$manifest" \
      --run-root "$run_root" --output "$corrected"
  fi

  while IFS=$'\t' read -r budget strategy scale target_bop weights acts; do
    run_name="ptq_fisher_w4a4_out8_${budget}_${strategy}_bopfix2_fixedseed${SEED}_cal16_${suffix}"
    run_dir="$run_root/$run_name"
    log="$LOG_DIR/${task}_${budget}_${strategy}_bopfix2.log"
    if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" && \
          -s "$run_dir/outlier_runtime_stats.json" ]]; then
      status=skipped_existing
    elif env TASK="$task" BASE_RUN="$base_run" \
        PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 PTQ_OUTLIER_BITS=8 \
        PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
        PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$weights" \
        PTQ_ACT_OUTLIER_LAYER_RATIOS="$acts" \
        PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 PTQ_ACT_CALIBRATION_BATCHES=16 \
        PTQ_ACT_CALIBRATION_MAX_VALUES=65536 PTQ_RUN_SEED="$SEED" \
        PTQ_ALLOCATION_STRATEGY="${strategy}_bopfix2" \
        PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" RUN_NAME="$run_name" \
        bash "$WORKER" > "$log" 2>&1; then
      status=completed
    else
      status=failed; failures=$((failures + 1))
    fi
    printf '%s\t%s\t%s\t%s\tbopfix2\t%s\t%s\t%s\t%s\t%s\n' \
      "$task" "$SEED" "$budget" "$strategy" "$scale" "$target_bop" \
      "$run_name" "$status" "$log" >> "$STATUS"
  done < <("$PYTHON" - "$corrected" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    scale = item.get("floor_preserving_excess_scale", item.get("bop_correction_scale"))
    print(item["budget"], item["strategy"], f'{scale:.12g}',
          f'{item["target_measured_uniform_bop_overhead"]:.12g}', weights, acts, sep="\t")
PY
)
done

echo "FINISH failures=$failures status=$STATUS"
exit "$failures"
