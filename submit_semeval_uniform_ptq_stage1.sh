#!/bin/bash
# Submit SemEval stage-1 uniform PTQ sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUN="${BASE_RUN:-formal_e10_ms256}"

configs=(
  "8 8"
  "6 8"
  "8 6"
  "6 6"
  "4 8"
  "8 4"
  "4 4"
)

for cfg in "${configs[@]}"; do
  read -r wbits abits <<<"$cfg"
  run_name="ptq_uniform_w${wbits}a${abits}_ms256_semeval_e10"
  job_name="ptq_s_w${wbits}a${abits}"
  sbatch --exclude=node6 \
    --job-name="$job_name" \
    --export=ALL,BASE_RUN="$BASE_RUN",PTQ_WEIGHT_BITS="$wbits",PTQ_ACT_BITS="$abits",RUN_NAME="$run_name" \
    "$SCRIPT_DIR/slurm_semeval_uniform_ptq_eval.sbatch"
done
