#!/bin/bash
# SemEval uniform PTQ validation-only job.
#
# Submit example:
#   sbatch --exclude=node6 \
#     --export=ALL,PTQ_WEIGHT_BITS=8,PTQ_ACT_BITS=8,RUN_NAME=ptq_uniform_w8a8_ms256_semeval_e10 \
#     slurm_semeval_uniform_ptq_eval.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/cluster/home/zhangzx/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

: "${PTQ_WEIGHT_BITS:?Set PTQ_WEIGHT_BITS}"
: "${PTQ_ACT_BITS:?Set PTQ_ACT_BITS}"
: "${RUN_NAME:?Set RUN_NAME}"

PROJECT_ROOT=/cluster/home/zhangzx/my_project
EXP_DIR=$PROJECT_ROOT/base_exp/exp_edge
JIANT_DIR=$PROJECT_ROOT/base_exp/jiant
MODEL=bert-large-uncased
TASK=semeval
BASE_RUN="${BASE_RUN:-formal_e10_ms256}"
BASE_CKPT="${BASE_CKPT:-$EXP_DIR/runs/$MODEL/$TASK/$BASE_RUN/best_model.p}"
OUTPUT_DIR="$EXP_DIR/runs/$MODEL/$TASK/$RUN_NAME"
RUNCONFIG_PATH="$EXP_DIR/runconfigs/$MODEL/$TASK/${RUN_NAME}.json"

mkdir -p "$EXP_DIR/slurm_logs" "$OUTPUT_DIR" "$(dirname "$RUNCONFIG_PATH")"

module load anaconda3/2024.02
module load cuda/12.3

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pytorch-test

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TASK=$TASK BASE_RUN=$BASE_RUN BASE_CKPT=$BASE_CKPT"
echo "RUN_NAME=$RUN_NAME OUTPUT_DIR=$OUTPUT_DIR"
echo "PTQ_WEIGHT_BITS=$PTQ_WEIGHT_BITS PTQ_ACT_BITS=$PTQ_ACT_BITS"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "ERROR: missing baseline checkpoint: $BASE_CKPT" >&2
  exit 2
fi

export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PTQ_WEIGHT_BITS
export PTQ_ACT_BITS

cd "$PROJECT_ROOT"

python "$JIANT_DIR/jiant/proj/main/scripts/configurator.py" \
  SingleTaskConfigurator \
  "$RUNCONFIG_PATH" \
  --task_name "$TASK" \
  --task_config_base_path "$EXP_DIR/tasks/configs" \
  --task_cache_base_path "$EXP_DIR/cache/$MODEL" \
  --train_batch_size 1 \
  --eval_batch_multiplier 2 \
  --do_val

python "$JIANT_DIR/jiant/proj/main/runscript.py" \
  run \
  --hf_pretrained_model_name_or_path "$MODEL" \
  --model_config_path "$EXP_DIR/models/$MODEL/model/config.json" \
  --model_path "$BASE_CKPT" \
  --jiant_task_container_config_path "$RUNCONFIG_PATH" \
  --model_load_mode all \
  --do_val \
  --force_overwrite \
  --output_dir "$OUTPUT_DIR"

python - "$OUTPUT_DIR" "$TASK" "$BASE_RUN" "$BASE_CKPT" "$PTQ_WEIGHT_BITS" "$PTQ_ACT_BITS" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
task = sys.argv[2]
base_run = sys.argv[3]
base_ckpt = sys.argv[4]
weight_bits = int(sys.argv[5])
act_bits = int(sys.argv[6])
metrics_path = output_dir / "val_metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
summary = {
    "mode": "uniform_fake_quant_ptq",
    "task": task,
    "base_run": base_run,
    "base_checkpoint": base_ckpt,
    "weight_bits": weight_bits,
    "activation_bits": act_bits,
    "linear_modules": "bert_encoder_qkv_attn_out_ffn_in_ffn_out",
    "module_count": 144,
    "weight_granularity": "per_tensor_symmetric",
    "activation_granularity": "dynamic_per_tensor_symmetric",
    "outlier_aware": False,
    "val_metrics": metrics,
}
(output_dir / "quant_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
