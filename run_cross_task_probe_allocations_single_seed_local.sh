#!/usr/bin/env bash
# Evaluate every ordered cross-task probing allocation at target low/boundary budgets.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
CORRECTOR="$PROJECT_ROOT/tools/make_probe_guided_bop_corrections.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
SEED="${SEED:-20260806}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/cross_task_probe_single_seed_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\ttarget\tsource\tbudget\tphase\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

for target in semeval ner dep; do
  case "$target" in
    semeval)
      sources="ner dep"
      worker="$PROJECT_ROOT/run_semeval_outlier_ptq_one_fixedseed_local.sh"
      run_root="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
      run_suffix="cal16_semeval_e10"
      low_w=0.001; low_a=0.005; boundary_w=0.0025; boundary_a=0.01; floor=0
      task_env=()
      ;;
    ner)
      sources="semeval dep"
      worker="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
      run_root="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/ner"
      run_suffix="cal16_ner_e3"
      low_w=0.0003125; low_a=0.0015625; boundary_w=0.0004375; boundary_a=0.0021875; floor=0
      task_env=(TASK=ner BASE_RUN=formal_e3_ms256_tner)
      ;;
    dep)
      sources="semeval ner"
      worker="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
      run_root="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/dep"
      run_suffix="cal16_dep_e3_retry1"
      low_w=0.00055; low_a=0.00275; boundary_w=0.00070; boundary_a=0.00350; floor=0.50
      task_env=(TASK=dep BASE_RUN=formal_e3_ms256_retry1)
      ;;
  esac

  for source in $sources; do
    allocation_json="$ANALYSIS_DIR/cross_probe_target_${target}_source_${source}_allocations.json"
    "$PYTHON" "$GENERATOR" \
      --probing-csv "$PROBING_CSV" \
      --task "$target" \
      --probing-source-task "$source" \
      --low-weight-ratio "$low_w" \
      --low-activation-ratio "$low_a" \
      --boundary-weight-ratio "$boundary_w" \
      --boundary-activation-ratio "$boundary_a" \
      --shape-floor "$floor" \
      --output "$allocation_json"

    while IFS=$'\t' read -r budget nominal_bop weight_ratios act_ratios; do
      run_name="ptq_xprobe_w4a4_out8_${budget}_src${source}_fixedseed${SEED}_${run_suffix}"
      run_dir="$run_root/$run_name"
      log_path="$LOG_DIR/${target}_src${source}_${budget}_pilot.log"
      if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" && \
            -s "$run_dir/outlier_runtime_stats.json" && \
            -s "$run_dir/activation_calibration_stats.json" ]]; then
        status=skipped_existing
      elif env "${task_env[@]}" \
          PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 \
          PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
          PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$weight_ratios" \
          PTQ_ACT_OUTLIER_LAYER_RATIOS="$act_ratios" \
          PTQ_OUTLIER_BITS=8 \
          PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
          PTQ_ACT_CALIBRATION_BATCHES=16 \
          PTQ_ACT_CALIBRATION_MAX_VALUES=65536 \
          PTQ_RUN_SEED="$SEED" \
          PTQ_ALLOCATION_STRATEGY="cross_probe_source_${source}" \
          PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" \
          RUN_NAME="$run_name" \
          bash "$worker" >"$log_path" 2>&1; then
        status=completed
      else
        status=failed
        failures=$((failures + 1))
      fi
      printf '%s\t%s\t%s\t%s\tpilot\t\t%s\t%s\t%s\t%s\n' \
        "$SEED" "$target" "$source" "$budget" "$nominal_bop" \
        "$run_name" "$status" "$log_path" >> "$STATUS_TSV"
    done < <("$PYTHON" - "$allocation_json" <<'PY'
import json
import sys

for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    if item["strategy"] != "probing":
        continue
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(item["budget"], f'{item["nominal_bop_overhead"]:.12g}', weights, acts, sep="\t")
PY
)

    correction_input="$ANALYSIS_DIR/cross_probe_target_${target}_source_${source}_fixedseed${SEED}_pilot_manifest.json"
    corrected_json="$ANALYSIS_DIR/cross_probe_target_${target}_source_${source}_bopfix2_fixedseed${SEED}_allocations.json"
    "$PYTHON" - "$allocation_json" "$correction_input" "$target" "$source" "$SEED" "$run_suffix" <<'PY'
import json
import sys

source_path, output_path, target, probe_source, seed, suffix = sys.argv[1:]
payload = json.load(open(source_path, encoding="utf-8"))
selected = []
for item in payload["allocations"]:
    if item["strategy"] not in ("uniform", "probing"):
        continue
    if item["strategy"] == "probing":
        run = f"ptq_xprobe_w4a4_out8_{item['budget']}_src{probe_source}_fixedseed{seed}_{suffix}"
    elif target == "dep":
        label = "w0p055_a0p275" if item["budget"] == "low" else "w0p070_a0p350"
        run = f"ptq_oa_w4a4_dualout8_boundaryrefine_{label}_fixedseed{seed}_cal16_ms256_dep_e3_retry1"
    else:
        run = f"ptq_pg_w4a4_out8_{item['budget']}_uniform_fixedseed{seed}_{suffix}"
    selected.append({**item, "source_run": run})
payload["allocations"] = selected
with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(payload, writer, indent=2)
    writer.write("\n")
PY

    if ! "$PYTHON" "$CORRECTOR" \
        --allocations "$correction_input" \
        --run-root "$run_root" \
        --output "$corrected_json"; then
      failures=$((failures + 1))
      continue
    fi

    while IFS=$'\t' read -r budget scale target_bop weight_ratios act_ratios; do
      run_name="ptq_xprobe_w4a4_out8_${budget}_src${source}_bopfix2_fixedseed${SEED}_${run_suffix}"
      run_dir="$run_root/$run_name"
      log_path="$LOG_DIR/${target}_src${source}_${budget}_bopfix2.log"
      if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" && \
            -s "$run_dir/outlier_runtime_stats.json" && \
            -s "$run_dir/activation_calibration_stats.json" ]]; then
        status=skipped_existing
      elif env "${task_env[@]}" \
          PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 \
          PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
          PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$weight_ratios" \
          PTQ_ACT_OUTLIER_LAYER_RATIOS="$act_ratios" \
          PTQ_OUTLIER_BITS=8 \
          PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
          PTQ_ACT_CALIBRATION_BATCHES=16 \
          PTQ_ACT_CALIBRATION_MAX_VALUES=65536 \
          PTQ_RUN_SEED="$SEED" \
          PTQ_ALLOCATION_STRATEGY="cross_probe_source_${source}_bopfix2" \
          PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" \
          RUN_NAME="$run_name" \
          bash "$worker" >"$log_path" 2>&1; then
        status=completed
      else
        status=failed
        failures=$((failures + 1))
      fi
      printf '%s\t%s\t%s\t%s\tbopfix2\t%s\t%s\t%s\t%s\t%s\n' \
        "$SEED" "$target" "$source" "$budget" "$scale" "$target_bop" \
        "$run_name" "$status" "$log_path" >> "$STATUS_TSV"
    done < <("$PYTHON" - "$corrected_json" <<'PY'
import json
import sys

for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(item["budget"], f'{item["bop_correction_scale"]:.12g}',
          f'{item["target_measured_uniform_bop_overhead"]:.12g}', weights, acts, sep="\t")
PY
)
  done
done

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
