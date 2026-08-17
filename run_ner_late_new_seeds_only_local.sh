#!/usr/bin/env bash
# Recheck NER late allocation on new calibration seeds without rerunning controls.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
GENERATOR="$PROJECT_ROOT/tools/make_probe_guided_outlier_allocations.py"
PROBING_CSV="$PROJECT_ROOT/base_exp/exp_edge/plots/main_tasks_layer_probing_major.csv"
ANALYSIS="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/ner"
SEEDS="${SEEDS:-20260810 20260811 20260812 20260813}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/ner_late_new_seeds_only_$RUN_ID"
STATUS="$LOG_DIR/status.tsv"
BASE_ALLOC="$ANALYSIS/ner_w4a4_probe_guided_outlier8_allocations.json"
LATE_ALLOC="$ANALYSIS/ner_late_new_seeds_only_allocations.json"
TARGETS="$ANALYSIS/ner_late_new_seeds_only_fixed_targets.json"
failures=0

mkdir -p "$LOG_DIR" "$ANALYSIS"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'seed\tbudget\tphase\tscale\ttarget_bop\trun_name\tstatus\tlog\n' > "$STATUS"

"$PYTHON" "$GENERATOR" --probing-csv "$PROBING_CSV" --task ner \
  --low-weight-ratio 0.0003125 --low-activation-ratio 0.0015625 \
  --boundary-weight-ratio 0.0004375 --boundary-activation-ratio 0.0021875 \
  --output "$BASE_ALLOC"

"$PYTHON" - "$BASE_ALLOC" "$LATE_ALLOC" "$TARGETS" "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

source_path, late_path, target_path, run_root = sys.argv[1:]
payload = json.load(open(source_path, encoding="utf-8"))
payload["allocations"] = [
    item for item in payload["allocations"] if item["strategy"] == "late"
]
Path(late_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

targets = {}
root = Path(run_root)
for budget in ("low", "boundary"):
    values = []
    for seed in (20260806, 20260807, 20260808, 20260809):
        run = root / f"ptq_pg_w4a4_out8_{budget}_uniform_fixedseed{seed}_cal16_ner_e3"
        summary = json.loads((run / "quant_summary.json").read_text(encoding="utf-8"))
        values.append(summary["runtime_totals"]["normalized_bop_overhead"])
    targets[budget] = {"mean": sum(values) / len(values), "source_values": values}
Path(target_path).write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
PY

while IFS=$'\t' read -r budget nominal_bop weights acts; do
  target_bop=$("$PYTHON" - "$TARGETS" "$budget" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]["mean"])
PY
)
  for seed in $SEEDS; do
    pilot_name="ptq_pg_w4a4_out8_${budget}_late_newseed${seed}_cal16_ner_e3"
    pilot_dir="$RUN_ROOT/$pilot_name"
    pilot_log="$LOG_DIR/seed${seed}_${budget}_pilot.log"
    if [[ -s "$pilot_dir/quant_summary.json" && -s "$pilot_dir/val_metrics.json" && \
          -s "$pilot_dir/outlier_runtime_stats.json" ]]; then
      pilot_status=skipped_existing
    elif env TASK=ner BASE_RUN=formal_e3_ms256_tner \
        PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 PTQ_OUTLIER_BITS=8 \
        PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
        PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$weights" \
        PTQ_ACT_OUTLIER_LAYER_RATIOS="$acts" \
        PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 PTQ_ACT_CALIBRATION_BATCHES=16 \
        PTQ_ACT_CALIBRATION_MAX_VALUES=65536 PTQ_RUN_SEED="$seed" \
        PTQ_ALLOCATION_STRATEGY=late_newseed \
        PTQ_NOMINAL_BOP_OVERHEAD="$nominal_bop" RUN_NAME="$pilot_name" \
        bash "$WORKER" > "$pilot_log" 2>&1; then
      pilot_status=completed
    else
      pilot_status=failed
      failures=$((failures + 1))
    fi
    printf '%s\t%s\tpilot\t\t%s\t%s\t%s\t%s\n' \
      "$seed" "$budget" "$target_bop" "$pilot_name" "$pilot_status" "$pilot_log" >> "$STATUS"
    [[ "$pilot_status" == failed ]] && continue

    correction=$("$PYTHON" - "$pilot_dir/quant_summary.json" "$target_bop" "$weights" "$acts" <<'PY'
import json
import math
import sys

path, target, weights, acts = sys.argv[1:]
target = float(target)
totals = json.load(open(path, encoding="utf-8"))["runtime_totals"]
total = totals["low_bit_macs"]
linear = (totals["weight_outlier_macs"] + totals["activation_outlier_macs"]) / total
quadratic = totals["dual_outlier_macs"] / total
scale = target / linear if quadratic == 0 else (
    -linear + math.sqrt(linear * linear + 4 * quadratic * target)
) / (2 * quadratic)
scaled_w = ",".join(f"{float(value) * scale:.12g}" for value in weights.split(","))
scaled_a = ",".join(f"{float(value) * scale:.12g}" for value in acts.split(","))
print(scale, scaled_w, scaled_a, sep="\t")
PY
)
    IFS=$'\t' read -r scale corrected_w corrected_a <<< "$correction"
    fixed_name="ptq_pg_w4a4_out8_${budget}_late_bopfix2_newseed${seed}_cal16_ner_e3"
    fixed_dir="$RUN_ROOT/$fixed_name"
    fixed_log="$LOG_DIR/seed${seed}_${budget}_bopfix2.log"
    if [[ -s "$fixed_dir/quant_summary.json" && -s "$fixed_dir/val_metrics.json" && \
          -s "$fixed_dir/outlier_runtime_stats.json" ]]; then
      fixed_status=skipped_existing
    elif env TASK=ner BASE_RUN=formal_e3_ms256_tner \
        PTQ_WEIGHT_BITS=4 PTQ_ACT_BITS=4 PTQ_OUTLIER_BITS=8 \
        PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
        PTQ_WEIGHT_OUTLIER_LAYER_RATIOS="$corrected_w" \
        PTQ_ACT_OUTLIER_LAYER_RATIOS="$corrected_a" \
        PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 PTQ_ACT_CALIBRATION_BATCHES=16 \
        PTQ_ACT_CALIBRATION_MAX_VALUES=65536 PTQ_RUN_SEED="$seed" \
        PTQ_ALLOCATION_STRATEGY=late_bopfix2_newseed \
        PTQ_NOMINAL_BOP_OVERHEAD="$target_bop" RUN_NAME="$fixed_name" \
        bash "$WORKER" > "$fixed_log" 2>&1; then
      fixed_status=completed
    else
      fixed_status=failed
      failures=$((failures + 1))
    fi
    printf '%s\t%s\tbopfix2\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$budget" "$scale" "$target_bop" "$fixed_name" "$fixed_status" "$fixed_log" >> "$STATUS"
  done
done < <("$PYTHON" - "$LATE_ALLOC" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["allocations"]:
    weights = ",".join(f"{value:.12g}" for value in item["weight_layer_ratios"])
    acts = ",".join(f"{value:.12g}" for value in item["activation_layer_ratios"])
    print(item["budget"], item["nominal_bop_overhead"], weights, acts, sep="\t")
PY
)

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS"
exit "$failures"
