#!/bin/bash
# Tenney-style scalar-mix / cumulative layer probing for a single edge task.
#
# Required/typical submit examples:
#   sbatch --exclude=node6 --export=ALL,TASKS=semeval,TENNEY_PROBE_MODE=scalar_mix slurm_tenney_probe.sbatch
#   sbatch --exclude=node6 --export=ALL,TASKS=semeval,TENNEY_PROBE_MODE=cumulative,TENNEY_CUMULATIVE_MAX_LAYER=16 slurm_tenney_probe.sbatch
#
# SemEval fixed-layer reference curves use e10 no-early, so this template defaults to
# EPOCHS=10 and NO_IMPROVEMENTS_FOR_N_EVALS=0. Other tasks should pass EPOCHS
# explicitly to match their fixed-layer probing reference curves.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --output=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/cluster/home/zhangzx/my_project
EXP_DIR=$PROJECT_ROOT/base_exp/exp_edge
JIANT_DIR=$PROJECT_ROOT/base_exp/jiant

TASKS="${TASKS:-semeval}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
EPOCHS="${EPOCHS:-10}"
FREEZE_ENCODER="${FREEZE_ENCODER:-1}"
NO_IMPROVEMENTS_FOR_N_EVALS="${NO_IMPROVEMENTS_FOR_N_EVALS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TENNEY_PROBE_MODE="${TENNEY_PROBE_MODE:?Set TENNEY_PROBE_MODE=scalar_mix|cumulative}"
TENNEY_CUMULATIVE_MAX_LAYER="${TENNEY_CUMULATIVE_MAX_LAYER:-}"

if [[ "$TASKS" != "semeval" && "$TASKS" != "ner" && "$TASKS" != "dep" ]]; then
  echo "ERROR: supported Tenney-style tasks are semeval, ner, dep; got: $TASKS" >&2
  exit 2
fi

if [[ "$TASKS" == *" "* ]]; then
  echo "ERROR: slurm_tenney_probe.sbatch expects exactly one task, got: $TASKS" >&2
  exit 2
fi

if [[ "$TENNEY_PROBE_MODE" == "scalar_mix" ]]; then
  if [[ -n "$TENNEY_CUMULATIVE_MAX_LAYER" ]]; then
    echo "ERROR: scalar_mix should not set TENNEY_CUMULATIVE_MAX_LAYER" >&2
    exit 2
  fi
  run_suffix="e${EPOCHS}"
  if [[ "$NO_IMPROVEMENTS_FOR_N_EVALS" == "0" ]]; then
    run_suffix="${run_suffix}_noearly"
  fi
  RUN_NAME="${RUN_NAME:-tenney_scalar_mix_ms${MAX_SEQ_LENGTH}_${TASKS}_lr1e4_${run_suffix}}"
elif [[ "$TENNEY_PROBE_MODE" == "cumulative" ]]; then
  : "${TENNEY_CUMULATIVE_MAX_LAYER:?Set TENNEY_CUMULATIVE_MAX_LAYER=0..24}"
  layer_padded="$(printf "%02d" "$TENNEY_CUMULATIVE_MAX_LAYER")"
  run_suffix="e${EPOCHS}"
  if [[ "$NO_IMPROVEMENTS_FOR_N_EVALS" == "0" ]]; then
    run_suffix="${run_suffix}_noearly"
  fi
  RUN_NAME="${RUN_NAME:-tenney_cumulative_l${layer_padded}_ms${MAX_SEQ_LENGTH}_${TASKS}_lr1e4_${run_suffix}}"
else
  echo "ERROR: TENNEY_PROBE_MODE must be scalar_mix or cumulative, got: $TENNEY_PROBE_MODE" >&2
  exit 2
fi

mkdir -p "$EXP_DIR/slurm_logs"

module load anaconda3/2024.02
module load cuda/12.3

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pytorch-test

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TASKS=$TASKS RUN_NAME=$RUN_NAME"
echo "TENNEY_PROBE_MODE=$TENNEY_PROBE_MODE TENNEY_CUMULATIVE_MAX_LAYER=$TENNEY_CUMULATIVE_MAX_LAYER"
echo "MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH EPOCHS=$EPOCHS FREEZE_ENCODER=$FREEZE_ENCODER"
echo "NO_IMPROVEMENTS_FOR_N_EVALS=$NO_IMPROVEMENTS_FOR_N_EVALS LEARNING_RATE=$LEARNING_RATE"

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
export RUN_NAME
export FREEZE_ENCODER
export NO_IMPROVEMENTS_FOR_N_EVALS
export TENNEY_PROBE_MODE
export TENNEY_CUMULATIVE_MAX_LAYER

cd "$PROJECT_ROOT"

bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" prepare
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" train
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" summarize
