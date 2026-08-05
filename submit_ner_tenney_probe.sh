#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK=ner
EPOCHS=3
MAX_SEQ_LENGTH=256

job_ids=()

submit_gpu() {
  local job_name="$1"
  shift
  local output
  output="$(sbatch --parsable --exclude=node6 --job-name="$job_name" "$@")"
  echo "$job_name -> $output"
  job_ids+=("$output")
}

submit_gpu tenney_ner_mix \
  --export=ALL,TASKS="$TASK",MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH",EPOCHS="$EPOCHS",NO_IMPROVEMENTS_FOR_N_EVALS=0,LEARNING_RATE=1e-4,TENNEY_PROBE_MODE=scalar_mix \
  "$SCRIPT_DIR/slurm_tenney_probe.sbatch"

for layer in $(seq 0 24); do
  layer_padded="$(printf "%02d" "$layer")"
  submit_gpu "tenney_ner_c${layer_padded}" \
    --export=ALL,TASKS="$TASK",MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH",EPOCHS="$EPOCHS",NO_IMPROVEMENTS_FOR_N_EVALS=0,LEARNING_RATE=1e-4,TENNEY_PROBE_MODE=cumulative,TENNEY_CUMULATIVE_MAX_LAYER="$layer" \
    "$SCRIPT_DIR/slurm_tenney_probe.sbatch"
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
summary_id="$(sbatch --parsable --dependency=afterok:"$dependency" --job-name=tenney_ner_sum \
  --export=ALL,TASK="$TASK",MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH",EPOCHS="$EPOCHS" \
  "$SCRIPT_DIR/slurm_tenney_probe_summary.sbatch")"

echo "tenney_ner_sum -> $summary_id"
echo "Submitted ${#job_ids[@]} GPU jobs and 1 dependent summary job."
