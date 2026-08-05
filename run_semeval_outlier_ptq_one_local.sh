#!/usr/bin/env bash
# Run one SemEval stage-1 outlier-aware PTQ validation on the rental server.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/root/autodl-tmp/master-gra/my_project/base_exp/exp_edge/slurm_logs/%x-%j.out
#SBATCH --error=/root/autodl-tmp/master-gra/my_project/base_exp/exp_edge/slurm_logs/%x-%j.err

set -euo pipefail

: "${PTQ_WEIGHT_BITS:?Set PTQ_WEIGHT_BITS}"
: "${PTQ_ACT_BITS:?Set PTQ_ACT_BITS}"
: "${PTQ_WEIGHT_OUTLIER_RATIO:=0}"
: "${PTQ_ACT_OUTLIER_RATIO:=0}"
: "${PTQ_OUTLIER_BITS:=16}"
: "${PTQ_ACT_QUANTILE_SAMPLE_SIZE:=8192}"
: "${RUN_NAME:?Set RUN_NAME}"

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
EXP_DIR=$PROJECT_ROOT/base_exp/exp_edge
JIANT_DIR=$PROJECT_ROOT/base_exp/jiant
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
MODEL=bert-large-uncased
TASK=semeval
BASE_RUN="${BASE_RUN:-formal_e10_ms256}"
BASE_CKPT="${BASE_CKPT:-$EXP_DIR/runs/$MODEL/$TASK/$BASE_RUN/best_model.p}"
OUTPUT_DIR="$EXP_DIR/runs/$MODEL/$TASK/$RUN_NAME"
RUNCONFIG_PATH="$EXP_DIR/runconfigs/$MODEL/$TASK/${RUN_NAME}.json"
PTQ_OUTLIER_STATS_PATH="$OUTPUT_DIR/outlier_runtime_stats.json"

mkdir -p "$EXP_DIR/slurm_logs" "$OUTPUT_DIR" "$(dirname "$RUNCONFIG_PATH")"

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TASK=$TASK BASE_RUN=$BASE_RUN BASE_CKPT=$BASE_CKPT"
echo "RUN_NAME=$RUN_NAME OUTPUT_DIR=$OUTPUT_DIR"
echo "PTQ_WEIGHT_BITS=$PTQ_WEIGHT_BITS PTQ_ACT_BITS=$PTQ_ACT_BITS"
echo "PTQ_WEIGHT_OUTLIER_RATIO=$PTQ_WEIGHT_OUTLIER_RATIO"
echo "PTQ_ACT_OUTLIER_RATIO=$PTQ_ACT_OUTLIER_RATIO"
echo "PTQ_OUTLIER_BITS=$PTQ_OUTLIER_BITS"
echo "PTQ_ACT_QUANTILE_SAMPLE_SIZE=$PTQ_ACT_QUANTILE_SAMPLE_SIZE"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "ERROR: missing baseline checkpoint: $BASE_CKPT" >&2
  exit 2
fi

export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/master-gra/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PTQ_WEIGHT_BITS
export PTQ_ACT_BITS
export PTQ_WEIGHT_OUTLIER_RATIO
export PTQ_ACT_OUTLIER_RATIO
export PTQ_OUTLIER_BITS
export PTQ_ACT_QUANTILE_SAMPLE_SIZE
export PTQ_OUTLIER_STATS_PATH

cd "$PROJECT_ROOT"

"$PYTHON" "$JIANT_DIR/jiant/proj/main/scripts/configurator.py" \
  SingleTaskConfigurator \
  "$RUNCONFIG_PATH" \
  --task_name "$TASK" \
  --task_config_base_path "$EXP_DIR/tasks/configs" \
  --task_cache_base_path "$EXP_DIR/cache/$MODEL" \
  --train_batch_size 1 \
  --eval_batch_multiplier 2 \
  --do_val

"$PYTHON" "$JIANT_DIR/jiant/proj/main/runscript.py" \
  run \
  --hf_pretrained_model_name_or_path "$MODEL" \
  --model_config_path "$EXP_DIR/models/$MODEL/model/config.json" \
  --model_path "$BASE_CKPT" \
  --jiant_task_container_config_path "$RUNCONFIG_PATH" \
  --model_load_mode all \
  --do_val \
  --force_overwrite \
  --output_dir "$OUTPUT_DIR"

"$PYTHON" - "$OUTPUT_DIR" "$TASK" "$BASE_RUN" "$BASE_CKPT" \
  "$PTQ_WEIGHT_BITS" "$PTQ_ACT_BITS" "$PTQ_WEIGHT_OUTLIER_RATIO" \
  "$PTQ_ACT_OUTLIER_RATIO" "$PTQ_OUTLIER_BITS" \
  "$PTQ_ACT_QUANTILE_SAMPLE_SIZE" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
task = sys.argv[2]
base_run = sys.argv[3]
base_ckpt = sys.argv[4]
weight_bits = int(sys.argv[5])
act_bits = int(sys.argv[6])
weight_ratio = float(sys.argv[7])
act_ratio = float(sys.argv[8])
outlier_bits = int(sys.argv[9])
sample_size = int(sys.argv[10])
metrics_path = output_dir / "val_metrics.json"
runtime_path = output_dir / "outlier_runtime_stats.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None
if weight_ratio > 0 and act_ratio > 0:
    mode = "dual_sided_outlier_aware_fake_quant_ptq"
    stage = "semeval_stage2_dual_sided_ratio_sweep"
elif weight_ratio > 0 or act_ratio > 0:
    mode = "single_sided_outlier_aware_fake_quant_ptq"
    stage = "semeval_stage1_single_sided_ratio_sweep"
else:
    mode = "uniform_fake_quant_ptq"
    stage = "semeval_uniform_control"
summary = {
    "mode": mode,
    "stage": stage,
    "task": task,
    "base_run": base_run,
    "base_checkpoint": base_ckpt,
    "weight_bits": weight_bits,
    "activation_bits": act_bits,
    "outlier_bits": outlier_bits,
    "weight_target_outlier_ratio": weight_ratio,
    "activation_target_outlier_ratio": act_ratio,
    "linear_modules": "bert_encoder_qkv_attn_out_ffn_in_ffn_out",
    "module_count": 144,
    "weight_outlier_selection": "static_exact_top_abs_per_linear",
    "activation_outlier_selection": "dynamic_sampled_quantile_per_forward",
    "activation_quantile_sample_size": sample_size,
    "cost_model": "logical_replacement_bit_operations",
    "runtime_stats_path": str(runtime_path) if runtime is not None else None,
    "runtime_totals": runtime.get("totals") if runtime is not None else None,
    "val_metrics": metrics,
}
(output_dir / "quant_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
