#!/usr/bin/env bash
# Collect empirical Fisher sensitivity without changing model parameters.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
PYTHON="${PYTHON:-/home/zhangzx/master-gra/conda-envs/graduate/bin/python}"
EXP_DIR="$PROJECT_ROOT/base_exp/exp_edge"
JIANT_DIR="$PROJECT_ROOT/base_exp/jiant"
FISHER_SEED="${FISHER_SEED:-20260806}"
RUN_NAME="${RUN_NAME:-fisher_stats_seed${FISHER_SEED}_cal16_semeval_e10}"
OUTPUT_DIR="$EXP_DIR/runs/bert-large-uncased/semeval/$RUN_NAME"
RUNCONFIG="$EXP_DIR/runconfigs/bert-large-uncased/semeval/${RUN_NAME}.json"
STATS_PATH="${FISHER_STATS_PATH:-$EXP_DIR/analysis/ptq/semeval_fisher_w4a4_seed${FISHER_SEED}_cal16.json}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$RUNCONFIG")" "$(dirname "$STATS_PATH")"
export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export HF_HOME=/root/autodl-tmp/master-gra/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FISHER_STATS_PATH="$STATS_PATH"
export FISHER_CALIBRATION_BATCHES="${FISHER_CALIBRATION_BATCHES:-16}"
export FISHER_WEIGHT_BITS=4 FISHER_ACT_BITS=4

"$PYTHON" "$JIANT_DIR/jiant/proj/main/scripts/configurator.py" SingleTaskConfigurator \
  "$RUNCONFIG" \
  --task_name semeval \
  --task_config_base_path "$EXP_DIR/tasks/configs" \
  --task_cache_base_path "$EXP_DIR/cache/bert-large-uncased" \
  --train_batch_size 1 --eval_batch_multiplier 2 \
  --do_train --max_steps 0

"$PYTHON" "$JIANT_DIR/jiant/proj/main/runscript.py" run \
  --hf_pretrained_model_name_or_path bert-large-uncased \
  --model_config_path "$EXP_DIR/models/bert-large-uncased/model/config.json" \
  --model_path "$EXP_DIR/runs/bert-large-uncased/semeval/formal_e10_ms256/best_model.p" \
  --jiant_task_container_config_path "$RUNCONFIG" \
  --model_load_mode all --seed "$FISHER_SEED" --force_overwrite \
  --output_dir "$OUTPUT_DIR"
