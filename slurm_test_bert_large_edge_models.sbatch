#!/bin/bash
#SBATCH --job-name=edge_test_real_gold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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

export PROJECT_ROOT
export EXP_DIR
export JIANT_DIR
export MODEL=bert-large-uncased
export PYTHON=python
export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT"

# Test metrics with **real** gold: tools/evaluate_edge_test.py rebuilds test_labels from each
# task's test.jsonl (test feature cache still has placeholder labels from jiant).
# Writes per-task TSVs: $EXP_DIR/runs/$MODEL/test_<task>.tsv
#
# Default: every run under runs/$MODEL/*/*/ that has best_model.p (dep/semeval/ner; spr2 when present).
# Optional env SPEC_FILTER=dep|semeval|ner|spr2 to only that task, e.g.:
#   sbatch --export=ALL,SPEC_FILTER=dep slurm_test_bert_large_edge_models.sbatch
# Optional positional args: task:run_name specs (passed through to test_bert_large_edge_models.sh).
export SPEC_FILTER="${SPEC_FILTER:-}"

bash "$PROJECT_ROOT/test_bert_large_edge_models.sh" "$@"
