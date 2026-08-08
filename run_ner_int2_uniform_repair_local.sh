#!/usr/bin/env bash
# Retry NER INT2 uniform controls without activation-outlier calibration.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
WORKER="$PROJECT_ROOT/run_ner_outlier_ptq_one_local.sh"
BASE_RUN="${BASE_RUN:-formal_e3_ms256_tner}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/ner_int2_uniform_repair_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
RUN_ROOT="$PROJECT_ROOT/base_exp/exp_edge/runs/bert-large-uncased/ner"
failures=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'weight_bits\tactivation_bits\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

if [[ -n "$WAIT_FOR_PID" ]]; then
  echo "[$(date --iso-8601=seconds)] WAIT pid=$WAIT_FOR_PID"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 20
  done
fi

run_one() {
  local weight_bits="$1"
  local activation_bits="$2"
  local run_name="$3"
  local run_dir="$RUN_ROOT/$run_name"
  local log_path="$LOG_DIR/${run_name}.log"

  if [[ -s "$run_dir/val_metrics.json" && -s "$run_dir/quant_summary.json" ]]; then
    printf '%s\t%s\t%s\tskipped_existing\t%s\n' \
      "$weight_bits" "$activation_bits" "$run_name" "$log_path" >> "$STATUS_TSV"
    return
  fi

  echo "[$(date --iso-8601=seconds)] START $run_name"
  if env \
      BASE_RUN="$BASE_RUN" \
      PTQ_WEIGHT_BITS="$weight_bits" \
      PTQ_ACT_BITS="$activation_bits" \
      PTQ_WEIGHT_OUTLIER_RATIO=0 \
      PTQ_ACT_OUTLIER_RATIO=0 \
      PTQ_OUTLIER_BITS=8 \
      PTQ_ACT_CALIBRATION_BATCHES=0 \
      PTQ_RUN_SEED=20260806 \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    printf '%s\t%s\t%s\tcompleted\t%s\n' \
      "$weight_bits" "$activation_bits" "$run_name" "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] DONE  $run_name"
  else
    printf '%s\t%s\t%s\tfailed\t%s\n' \
      "$weight_bits" "$activation_bits" "$run_name" "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] FAIL  $run_name"
    failures=$((failures + 1))
  fi
}

run_one 2 4 ptq_oa_w2a4_uniform_fixedseed20260806_ms256_ner_e3
run_one 4 2 ptq_oa_w4a2_uniform_fixedseed20260806_ms256_ner_e3

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
