#!/bin/bash
# Submit NER and DEP stage-1 uniform PTQ sweeps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

configs=(
  "8 8"
  "6 8"
  "8 6"
  "6 6"
  "4 8"
  "8 4"
  "4 4"
)

submit_task() {
  local task="$1"
  local base_run="$2"
  local suffix="$3"
  local job_prefix="$4"

  for cfg in "${configs[@]}"; do
    read -r wbits abits <<<"$cfg"
    run_name="ptq_uniform_w${wbits}a${abits}_ms256_${suffix}"
    job_name="ptq_${job_prefix}_w${wbits}a${abits}"
    sbatch --exclude=node6 \
      --job-name="$job_name" \
      --export=ALL,TASK="$task",BASE_RUN="$base_run",PTQ_WEIGHT_BITS="$wbits",PTQ_ACT_BITS="$abits",RUN_NAME="$run_name" \
      "$SCRIPT_DIR/slurm_uniform_ptq_eval.sbatch"
  done
}

submit_task "ner" "formal_e3_ms256_tner" "ner_e3" "n"
submit_task "dep" "formal_e3_ms256_retry1" "dep_e3_retry1" "d"
