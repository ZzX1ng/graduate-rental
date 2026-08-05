#!/bin/bash
# OntoNotes LDC edge tasks (pos / nonterminal / srl / coref), MAX_SEQ_LENGTH=512.
# Requires existing ms512 cache (e.g. job 17206). Does not re-run cache/export.
#
# Submit example (or use submit_onto_ms512_formal_e3_e5_e10.sh):
#   sbatch --job-name=bertlarge_pos_ms512_e3 \
#     --export=ALL,TASKS=pos,EPOCHS=3,RUN_NAME=formal_e3_ms512_pos,NO_IMPROVEMENTS_FOR_N_EVALS=0 \
#     slurm_onto_ms512_formal_train.sbatch
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --output=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

: "${TASKS:?Set TASKS (pos|nonterminal|srl|coref)}"
: "${EPOCHS:?Set EPOCHS (e.g. 3 5 10)}"
: "${RUN_NAME:?Set RUN_NAME (e.g. formal_e3_ms512_pos)}"
NO_IMPROVEMENTS_FOR_N_EVALS="${NO_IMPROVEMENTS_FOR_N_EVALS:-0}"

PROJECT_ROOT=/cluster/home/zhangzx/my_project
EXP_DIR=$PROJECT_ROOT/base_exp/exp_edge
JIANT_DIR=$PROJECT_ROOT/base_exp/jiant

mkdir -p "$EXP_DIR/slurm_logs"

module load anaconda3/2024.02
module load cuda/12.3

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pytorch-test

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "TASKS=$TASKS EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME NO_IMPROVEMENTS_FOR_N_EVALS=$NO_IMPROVEMENTS_FOR_N_EVALS"

export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EXP_DIR="$EXP_DIR"
export MODEL=bert-large-uncased
export MAX_SEQ_LENGTH=512
export TRAIN_BATCH_SIZE=1
export EVAL_BATCH_MULTIPLIER=2
export GRADIENT_ACCUMULATION_STEPS=16
export LEARNING_RATE=1e-5
export CUDA_VISIBLE_DEVICES=0
export NO_IMPROVEMENTS_FOR_N_EVALS

cd "$PROJECT_ROOT"

bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" prepare
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" train
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" summarize
