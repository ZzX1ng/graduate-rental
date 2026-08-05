#!/bin/bash
# Summarize completed Tenney-style probing runs.

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=tenney_sum
#SBATCH --output=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/cluster/home/zhangzx/my_project
TASK="${TASK:-semeval}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
EPOCHS="${EPOCHS:-10}"
RUN_SUFFIX="e${EPOCHS}_noearly"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/base_exp/exp_edge/analysis/tenney_probe/${TASK}_e${EPOCHS}_lr1e4_noearly"
DEFAULT_FULL_RUN="tenney_scalar_mix_ms${MAX_SEQ_LENGTH}_${TASK}_lr1e4_${RUN_SUFFIX}"
DEFAULT_CUMULATIVE_PREFIX="tenney_cumulative_l"
DEFAULT_CUMULATIVE_SUFFIX="_ms${MAX_SEQ_LENGTH}_${TASK}_lr1e4_${RUN_SUFFIX}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
fi
if [[ -z "${FULL_RUN:-}" ]]; then
  FULL_RUN="$DEFAULT_FULL_RUN"
fi
if [[ -z "${CUMULATIVE_PATTERN:-}" ]]; then
  CUMULATIVE_PATTERN="${DEFAULT_CUMULATIVE_PREFIX}"'{layer:02d}'"${DEFAULT_CUMULATIVE_SUFFIX}"
fi

module load anaconda3/2024.02
source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pytorch-test

export PYTHONPATH="$PROJECT_ROOT/base_exp/jiant:${PYTHONPATH:-}"

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TASK=$TASK"
echo "FULL_RUN=$FULL_RUN"
echo "CUMULATIVE_PATTERN=$CUMULATIVE_PATTERN"

cd "$PROJECT_ROOT"

python tools/summarize_tenney_probe.py \
  --task "$TASK" \
  --full-run "$FULL_RUN" \
  --cumulative-pattern "$CUMULATIVE_PATTERN" \
  --output-dir "$OUTPUT_DIR"
