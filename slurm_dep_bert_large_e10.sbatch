#!/bin/bash
#SBATCH --job-name=bertlarge_dep_e10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/cluster/home/zhangzx/my_project
EXP_DIR=$PROJECT_ROOT/base_exp/exp_edge
JIANT_DIR=$PROJECT_ROOT/base_exp/jiant

mkdir -p "$EXP_DIR/slurm_logs"

module load anaconda3/2024.02
module load cuda/12.3

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pytorch-test

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
python - <<'PY'
import sys
print(sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch import failed", repr(exc))
PY

export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EXP_DIR="$EXP_DIR"
export MODEL=bert-large-uncased
export TASKS=dep
export MAX_SEQ_LENGTH=256
export EPOCHS=10
export TRAIN_BATCH_SIZE=1
export EVAL_BATCH_MULTIPLIER=2
export GRADIENT_ACCUMULATION_STEPS=16
export LEARNING_RATE=1e-5
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=formal_e10_ms256

cd "$PROJECT_ROOT"

bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" prepare
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" train
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" summarize
