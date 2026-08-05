#!/bin/bash
#SBATCH --job-name=cache_onto_edge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/cluster/home/zhangzx/my_project
EXP_DIR=$PROJECT_ROOT/base_exp/exp_edge
JIANT_DIR=$PROJECT_ROOT/base_exp/jiant

mkdir -p "$EXP_DIR/slurm_logs"

module load anaconda3/2024.02

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pytorch-test

export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EXP_DIR="$EXP_DIR"
export MODEL=bert-large-uncased
export TASKS="pos nonterminal srl coref"
export INCLUDE_TEST=1
# OntoNotes 四任务单独使用 512；dep/semeval/ner/spr2 等仍由各训练脚本默认 256，勿混用本目录 cache。
export MAX_SEQ_LENGTH=512

cd "$PROJECT_ROOT"

echo "Job: $SLURM_JOB_ID node $(hostname)"
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" prepare
bash "$PROJECT_ROOT/run_bert_large_edge_tasks.sh" cache
echo "cache finished OK"
