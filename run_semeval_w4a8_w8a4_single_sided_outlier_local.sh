#!/usr/bin/env bash
# Run isolated SemEval 4-bit weight and activation outlier16 sweeps sequentially.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
WORKER="$PROJECT_ROOT/run_semeval_outlier_ptq_one_local.sh"
BASE_RUN="${BASE_RUN:-formal_e10_ms256}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/semeval_w4a8_w8a4_single_$RUN_ID"
STATUS_TSV="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR"
printf 'kind\tweight_bits\tactivation_bits\tratio\tlabel\trun_name\tstatus\tlog\n' > "$STATUS_TSV"

run_one() {
  local kind="$1"
  local weight_bits="$2"
  local activation_bits="$3"
  local ratio="$4"
  local label="$5"
  local weight_ratio="$6"
  local activation_ratio="$7"
  local run_name="$8"
  local log_path="$LOG_DIR/${kind}_${label}.log"

  echo "[$(date --iso-8601=seconds)] START $kind W${weight_bits}A${activation_bits} ratio=$ratio run=$run_name"
  if env \
      BASE_RUN="$BASE_RUN" \
      PTQ_WEIGHT_BITS="$weight_bits" \
      PTQ_ACT_BITS="$activation_bits" \
      PTQ_WEIGHT_OUTLIER_RATIO="$weight_ratio" \
      PTQ_ACT_OUTLIER_RATIO="$activation_ratio" \
      PTQ_OUTLIER_BITS=16 \
      PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192 \
      RUN_NAME="$run_name" \
      bash "$WORKER" >"$log_path" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tcompleted\t%s\n' \
      "$kind" "$weight_bits" "$activation_bits" "$ratio" "$label" \
      "$run_name" "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] DONE  $run_name"
  else
    printf '%s\t%s\t%s\t%s\t%s\t%s\tfailed\t%s\n' \
      "$kind" "$weight_bits" "$activation_bits" "$ratio" "$label" \
      "$run_name" "$log_path" >> "$STATUS_TSV"
    echo "[$(date --iso-8601=seconds)] FAIL  $run_name"
    failures=$((failures + 1))
  fi
}

weight_configs=(
  "0 r0"
  "0.001 r0p1"
  "0.0025 r0p25"
  "0.005 r0p5"
  "0.01 r1"
  "0.02 r2"
  "0.04 r4"
)

activation_configs=(
  "0 r0"
  "0.005 r0p5"
  "0.01 r1"
  "0.02 r2"
  "0.04 r4"
  "0.06 r6"
)

for cfg in "${weight_configs[@]}"; do
  read -r ratio label <<<"$cfg"
  run_one \
    weight 4 8 "$ratio" "$label" "$ratio" 0 \
    "ptq_oa_w4a8_wout16_${label}_ms256_semeval_e10"
done

for cfg in "${activation_configs[@]}"; do
  read -r ratio label <<<"$cfg"
  run_one \
    activation 8 4 "$ratio" "$label" 0 "$ratio" \
    "ptq_oa_w8a4_aout16_${label}_ms256_semeval_e10"
done

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS_TSV"
exit "$failures"
