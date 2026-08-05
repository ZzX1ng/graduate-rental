#!/bin/bash
# Collect outlier statistics for selected BERT-large baselines.
#
# Submit:
#   sbatch --exclude=node6 slurm_collect_outlier_stats.sbatch
#
# Optional overrides:
#   MAX_VAL_SAMPLES=512 BATCH_SIZE=2 OUTPUT_DIR=... sbatch --exclude=node6 slurm_collect_outlier_stats.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --job-name=outlier_stats
#SBATCH --output=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/cluster/home/zhangzx/my_project
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/base_exp/exp_edge/analysis/outlier_stats/ner_dep_semeval_baselines}"

module load anaconda3/2024.02
module load cuda/12.3

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pytorch-test

export PYTHONPATH="$PROJECT_ROOT/base_exp/jiant:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MAX_VAL_SAMPLES=$MAX_VAL_SAMPLES BATCH_SIZE=$BATCH_SIZE OUTPUT_DIR=$OUTPUT_DIR"

cd "$PROJECT_ROOT"

python tools/collect_outlier_stats.py \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-val-samples "$MAX_VAL_SAMPLES" \
  --batch-size "$BATCH_SIZE"
