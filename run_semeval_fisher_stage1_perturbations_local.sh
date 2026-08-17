#!/usr/bin/env bash
# Evaluate representative single-layer W4 and A4 perturbations.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_fixedseed_local.sh"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/semeval"
ANALYSIS="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_fisher_stage1_$(date +%Y%m%d_%H%M%S)"
failures=0
LAYERS="${LAYERS:-0 3 7 11 15 19 23}"
SUMMARIZE="${SUMMARIZE:-1}"

mkdir -p "$LOG_DIR"
printf 'layer\tside\trun_name\tstatus\tlog\n' > "$LOG_DIR/status.tsv"

control_name=ptq_fisher_stage1_control_w32a32_seed20260806_semeval_e10
control_dir="$RUN_ROOT/$control_name"
control_log="$LOG_DIR/control_w32a32.log"
if [[ -s "$control_dir/val_metrics.json" ]]; then
  control_status=skipped_existing
elif env \
    PTQ_WEIGHT_BITS=32 PTQ_ACT_BITS=32 \
    PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
    PTQ_OUTLIER_BITS=32 PTQ_ACT_CALIBRATION_BATCHES=0 \
    PTQ_RUN_SEED=20260806 RUN_NAME="$control_name" \
    bash "$WORKER" >"$control_log" 2>&1; then
  control_status=completed
else
  control_status=failed; failures=$((failures + 1))
fi
printf '%s\t%s\t%s\t%s\t%s\n' all control "$control_name" "$control_status" \
  "$control_log" >> "$LOG_DIR/status.tsv"

for layer in $LAYERS; do
  for side in weight activation; do
    run_name="ptq_fisher_stage1_${side}_layer$(printf '%02d' "$layer")_seed20260806_semeval_e10"
    run_dir="$RUN_ROOT/$run_name"
    log="$LOG_DIR/${side}_layer$(printf '%02d' "$layer").log"
    if [[ -s "$run_dir/val_metrics.json" ]]; then
      status=skipped_existing
    else
      if [[ "$side" == weight ]]; then
        weight_bits=4; act_bits=32
      else
        weight_bits=32; act_bits=4
      fi
      if env \
          PTQ_WEIGHT_BITS="$weight_bits" PTQ_ACT_BITS="$act_bits" \
          PTQ_WEIGHT_OUTLIER_RATIO=0 PTQ_ACT_OUTLIER_RATIO=0 \
          PTQ_LAYER_INDICES="$layer" PTQ_OUTLIER_BITS=32 \
          PTQ_ACT_CALIBRATION_BATCHES=0 PTQ_RUN_SEED=20260806 \
          RUN_NAME="$run_name" bash "$WORKER" >"$log" 2>&1; then
        status=completed
      else
        status=failed; failures=$((failures + 1))
      fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$layer" "$side" "$run_name" "$status" "$log" \
      >> "$LOG_DIR/status.tsv"
  done
done

if [[ "$failures" -eq 0 && "$SUMMARIZE" == "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/summarize_semeval_fisher_stage1.py" \
    --fisher-stats "$ANALYSIS/semeval_fisher_w4a4_seed20260806_cal16.json" \
    --run-root "$RUN_ROOT" \
    --output-prefix "$ANALYSIS/semeval_fisher_stage1_representative_layers"
fi
echo "FINISH failures=$failures status=$LOG_DIR/status.tsv"
exit "$failures"
