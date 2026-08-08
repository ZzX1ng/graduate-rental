#!/usr/bin/env bash
# Run one outlier-aware PTQ validation on the rental server.

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
: "${PTQ_ACT_CALIBRATION_BATCHES:=16}"
: "${PTQ_ACT_CALIBRATION_MAX_VALUES:=65536}"
: "${PTQ_WEIGHT_OUTLIER_LAYER_RATIOS:=}"
: "${PTQ_ACT_OUTLIER_LAYER_RATIOS:=}"
: "${PTQ_ALLOCATION_STRATEGY:=uniform}"
: "${PTQ_NOMINAL_BOP_OVERHEAD:=}"
: "${PTQ_RUN_SEED:=20260806}"
: "${RUN_NAME:?Set RUN_NAME}"

if [[ -z "$PTQ_ACT_OUTLIER_LAYER_RATIOS" && \
      "$PTQ_ACT_OUTLIER_RATIO" =~ ^0+([.]0+)?$ ]]; then
  PTQ_ACT_CALIBRATION_BATCHES=0
fi

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
EXP_DIR=$PROJECT_ROOT/base_exp/exp_edge
JIANT_DIR=$PROJECT_ROOT/base_exp/jiant
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
MODEL=bert-large-uncased
TASK="${TASK:-ner}"
BASE_RUN="${BASE_RUN:-formal_e3_ms256_tner}"
BASE_CKPT="${BASE_CKPT:-$EXP_DIR/runs/$MODEL/$TASK/$BASE_RUN/best_model.p}"
OUTPUT_DIR="$EXP_DIR/runs/$MODEL/$TASK/$RUN_NAME"
RUNCONFIG_PATH="$EXP_DIR/runconfigs/$MODEL/$TASK/${RUN_NAME}.json"
PTQ_OUTLIER_STATS_PATH="$OUTPUT_DIR/outlier_runtime_stats.json"
PTQ_ACT_CALIBRATION_STATS_PATH="$OUTPUT_DIR/activation_calibration_stats.json"

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
echo "PTQ_ACT_CALIBRATION_BATCHES=$PTQ_ACT_CALIBRATION_BATCHES"
echo "PTQ_ACT_CALIBRATION_MAX_VALUES=$PTQ_ACT_CALIBRATION_MAX_VALUES"
echo "PTQ_WEIGHT_OUTLIER_LAYER_RATIOS=$PTQ_WEIGHT_OUTLIER_LAYER_RATIOS"
echo "PTQ_ACT_OUTLIER_LAYER_RATIOS=$PTQ_ACT_OUTLIER_LAYER_RATIOS"
echo "PTQ_ALLOCATION_STRATEGY=$PTQ_ALLOCATION_STRATEGY"
echo "PTQ_NOMINAL_BOP_OVERHEAD=$PTQ_NOMINAL_BOP_OVERHEAD"
echo "PTQ_RUN_SEED=$PTQ_RUN_SEED"

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
export PTQ_ACT_CALIBRATION_BATCHES
export PTQ_ACT_CALIBRATION_MAX_VALUES
export PTQ_WEIGHT_OUTLIER_LAYER_RATIOS
export PTQ_ACT_OUTLIER_LAYER_RATIOS
export PTQ_ALLOCATION_STRATEGY
export PTQ_NOMINAL_BOP_OVERHEAD
export PTQ_RUN_SEED
export PTQ_OUTLIER_STATS_PATH
export PTQ_ACT_CALIBRATION_STATS_PATH

cd "$PROJECT_ROOT"

"$PYTHON" "$JIANT_DIR/jiant/proj/main/scripts/configurator.py" \
  SingleTaskConfigurator \
  "$RUNCONFIG_PATH" \
  --task_name "$TASK" \
  --task_config_base_path "$EXP_DIR/tasks/configs" \
  --task_cache_base_path "$EXP_DIR/cache/$MODEL" \
  --train_batch_size 1 \
  --eval_batch_multiplier 2 \
  --do_train \
  --max_steps 0 \
  --do_val

"$PYTHON" "$JIANT_DIR/jiant/proj/main/runscript.py" \
  run \
  --hf_pretrained_model_name_or_path "$MODEL" \
  --model_config_path "$EXP_DIR/models/$MODEL/model/config.json" \
  --model_path "$BASE_CKPT" \
  --jiant_task_container_config_path "$RUNCONFIG_PATH" \
  --model_load_mode all \
  --seed "$PTQ_RUN_SEED" \
  --do_val \
  --force_overwrite \
  --output_dir "$OUTPUT_DIR"

"$PYTHON" - "$OUTPUT_DIR" "$TASK" "$BASE_RUN" "$BASE_CKPT" \
  "$PTQ_WEIGHT_BITS" "$PTQ_ACT_BITS" "$PTQ_WEIGHT_OUTLIER_RATIO" \
  "$PTQ_ACT_OUTLIER_RATIO" "$PTQ_OUTLIER_BITS" \
  "$PTQ_ACT_QUANTILE_SAMPLE_SIZE" <<'PY'
import json
import os
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
weight_layer_spec = os.environ.get("PTQ_WEIGHT_OUTLIER_LAYER_RATIOS", "")
act_layer_spec = os.environ.get("PTQ_ACT_OUTLIER_LAYER_RATIOS", "")
weight_layer_ratios = (
    [float(value) for value in weight_layer_spec.split(",")]
    if weight_layer_spec
    else [weight_ratio] * 24
)
act_layer_ratios = (
    [float(value) for value in act_layer_spec.split(",")]
    if act_layer_spec
    else [act_ratio] * 24
)
metrics_path = output_dir / "val_metrics.json"
runtime_path = output_dir / "outlier_runtime_stats.json"
calibration_path = output_dir / "activation_calibration_stats.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None
if any(value > 0 for value in weight_layer_ratios) and any(
    value > 0 for value in act_layer_ratios
):
    mode = "dual_sided_outlier_aware_fake_quant_ptq"
    stage = f"{task}_probe_guided_layerwise_allocation"
elif any(value > 0 for value in weight_layer_ratios + act_layer_ratios):
    mode = "single_sided_outlier_aware_fake_quant_ptq"
    stage = f"{task}_stage1_single_sided_ratio_sweep"
else:
    mode = "uniform_fake_quant_ptq"
    stage = f"{task}_uniform_control"
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
    "weight_target_layer_ratios": weight_layer_ratios,
    "activation_target_layer_ratios": act_layer_ratios,
    "allocation_strategy": os.environ.get("PTQ_ALLOCATION_STRATEGY", "uniform"),
    "run_seed": int(os.environ.get("PTQ_RUN_SEED", "20260806")),
    "nominal_bop_overhead": (
        float(os.environ["PTQ_NOMINAL_BOP_OVERHEAD"])
        if os.environ.get("PTQ_NOMINAL_BOP_OVERHEAD")
        else None
    ),
    "linear_modules": "bert_encoder_qkv_attn_out_ffn_in_ffn_out",
    "module_count": 144,
    "weight_outlier_selection": "static_exact_top_abs_per_linear",
    "activation_outlier_selection": "fixed_train_calibrated_quantile_per_module",
    "activation_quantile_sample_size": sample_size,
    "activation_calibration_batches": int(
        os.environ.get("PTQ_ACT_CALIBRATION_BATCHES", "0")
    ),
    "activation_calibration_max_values_per_module": int(
        os.environ.get("PTQ_ACT_CALIBRATION_MAX_VALUES", "0")
    ),
    "activation_calibration_stats_path": (
        str(calibration_path) if calibration_path.exists() else None
    ),
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
