#!/bin/bash
# Fixed-layer edge probing for BERT-large.
#
# Required export variables:
#   TASKS=dep|semeval|ner|spr2|pos|nonterminal|srl|coref
#   BERT_LAYER_INDEX=0..24  # 0 = embedding output, 24 = final BERT-large layer
#
# Optional:
#   MAX_SEQ_LENGTH=256|512
#   EPOCHS=3
#   FREEZE_ENCODER=1
#   NO_IMPROVEMENTS_FOR_N_EVALS=10
#   LEARNING_RATE=1e-5
#   RUN_NAME=edge_lXX_msYYY_<task>
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --output=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

: "${TASKS:?Set TASKS to one task name}"
: "${BERT_LAYER_INDEX:?Set BERT_LAYER_INDEX to 0..24}"

PROJECT_ROOT=/cluster/home/zhangzx/my_project
EXP_DIR=$PROJECT_ROOT/base_exp/exp_edge
JIANT_DIR=$PROJECT_ROOT/base_exp/jiant

if [[ "$TASKS" =~ [[:space:]] ]]; then
  echo "ERROR: slurm_edge_layer_probe.sbatch expects a single task in TASKS, got: $TASKS" >&2
  exit 2
fi

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
EPOCHS="${EPOCHS:-3}"
FREEZE_ENCODER="${FREEZE_ENCODER:-1}"
NO_IMPROVEMENTS_FOR_N_EVALS="${NO_IMPROVEMENTS_FOR_N_EVALS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

layer_padded="$(printf "%02d" "$BERT_LAYER_INDEX")"
RUN_NAME="${RUN_NAME:-edge_l${layer_padded}_ms${MAX_SEQ_LENGTH}_${TASKS}}"

mkdir -p "$EXP_DIR/slurm_logs"

module load anaconda3/2024.02
module load cuda/12.3

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pytorch-test

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "TASKS=$TASKS"
echo "RUN_NAME=$RUN_NAME"
echo "BERT_LAYER_INDEX=$BERT_LAYER_INDEX FREEZE_ENCODER=$FREEZE_ENCODER"
echo "MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH EPOCHS=$EPOCHS NO_IMPROVEMENTS_FOR_N_EVALS=$NO_IMPROVEMENTS_FOR_N_EVALS LEARNING_RATE=$LEARNING_RATE"

export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EXP_DIR="$EXP_DIR"
export MODEL=bert-large-uncased
export MAX_SEQ_LENGTH
export EPOCHS
export TRAIN_BATCH_SIZE=1
export EVAL_BATCH_MULTIPLIER=2
export GRADIENT_ACCUMULATION_STEPS=16
export LEARNING_RATE
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME
export BERT_LAYER_INDEX
export FREEZE_ENCODER
export NO_IMPROVEMENTS_FOR_N_EVALS

cd "$PROJECT_ROOT"

# Cache/export must already exist. This job only prepares configs, trains the probe, and summarizes.
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" prepare
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" train
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" summarize
