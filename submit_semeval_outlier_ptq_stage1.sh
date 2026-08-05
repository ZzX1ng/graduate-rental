#!/bin/bash
# Submit SemEval stage-1 single-sided outlier-aware PTQ sweeps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUN="${BASE_RUN:-formal_e10_ms256}"
SBATCH_SCRIPT="$SCRIPT_DIR/slurm_semeval_outlier_ptq_stage1.sbatch"

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
  "0.03 r3"
  "0.04 r4"
  "0.06 r6"
)

for cfg in "${weight_configs[@]}"; do
  read -r ratio label <<<"$cfg"
  run_name="ptq_oa_w6a8_wout16_${label}_ms256_semeval_e10"
  sbatch --exclude=node6 \
    --job-name="oa_s_w_${label}" \
    --export=ALL,BASE_RUN="$BASE_RUN",PTQ_WEIGHT_BITS=6,PTQ_ACT_BITS=8,PTQ_WEIGHT_OUTLIER_RATIO="$ratio",PTQ_ACT_OUTLIER_RATIO=0,PTQ_OUTLIER_BITS=16,RUN_NAME="$run_name" \
    "$SBATCH_SCRIPT"
done

for cfg in "${activation_configs[@]}"; do
  read -r ratio label <<<"$cfg"
  run_name="ptq_oa_w8a6_aout16_${label}_ms256_semeval_e10"
  sbatch --exclude=node6 \
    --job-name="oa_s_a_${label}" \
    --export=ALL,BASE_RUN="$BASE_RUN",PTQ_WEIGHT_BITS=8,PTQ_ACT_BITS=6,PTQ_WEIGHT_OUTLIER_RATIO=0,PTQ_ACT_OUTLIER_RATIO="$ratio",PTQ_OUTLIER_BITS=16,PTQ_ACT_QUANTILE_SAMPLE_SIZE=8192,RUN_NAME="$run_name" \
    "$SBATCH_SCRIPT"
done
