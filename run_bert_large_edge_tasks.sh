#!/usr/bin/env bash
set -euo pipefail

# Run BERT-large on the eight edge probing tasks supported by jiant 2.x.
#
# Expected data layout:
#   $EXP_DIR/tasks/data/<task>/train.jsonl
#   $EXP_DIR/tasks/data/<task>/val.jsonl
#   $EXP_DIR/tasks/data/<task>/test.jsonl   # optional
#
# Usage:
#   bash run_bert_large_edge_tasks.sh prepare
#   bash run_bert_large_edge_tasks.sh export
#   bash run_bert_large_edge_tasks.sh cache
#   bash run_bert_large_edge_tasks.sh train
#   bash run_bert_large_edge_tasks.sh summarize
#   bash run_bert_large_edge_tasks.sh all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
JIANT_DIR="${JIANT_DIR:-$PROJECT_ROOT/base_exp/jiant}"
EXP_DIR="${EXP_DIR:-$PROJECT_ROOT/base_exp/exp_edge}"
MODEL="${MODEL:-bert-large-uncased}"
PYTHON="${PYTHON:-python3}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_MULTIPLIER="${EVAL_BATCH_MULTIPLIER:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_NAME="${RUN_NAME:-}"
TENNEY_PROBE_MODE="${TENNEY_PROBE_MODE:-}"
TENNEY_CUMULATIVE_MAX_LAYER="${TENNEY_CUMULATIVE_MAX_LAYER:-}"
# 0 = disable early stopping (run full max_steps). Default 10 matches historical smoke/formal jobs.
NO_IMPROVEMENTS_FOR_N_EVALS="${NO_IMPROVEMENTS_FOR_N_EVALS:-10}"

TASKS="${TASKS:-pos nonterminal dep ner srl coref spr1 semeval}"
STAGE="${1:-all}"

export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"
export TENNEY_PROBE_MODE
export TENNEY_CUMULATIVE_MAX_LAYER

cd "$JIANT_DIR"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_jiant() {
  "$PYTHON" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("jiant") else 1)
PY
}

prepare_configs() {
  mkdir -p "$EXP_DIR/tasks/configs"
  "$PYTHON" - "$EXP_DIR" "$TASKS" <<'PY'
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
tasks = sys.argv[2].split()
data_root = exp_dir / "tasks" / "data"
config_root = exp_dir / "tasks" / "configs"

missing = []
for task in tasks:
    task_dir = data_root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    train_path = task_dir / "train.jsonl"
    val_path = task_dir / "val.jsonl"
    test_path = task_dir / "test.jsonl"

    if not train_path.exists():
        missing.append(str(train_path))
    if not val_path.exists():
        missing.append(str(val_path))

    paths = {
        "train": str(train_path),
        "val": str(val_path),
    }
    if test_path.exists():
        paths["test"] = str(test_path)

    config = {
        "task": task,
        "paths": paths,
        "name": task,
    }
    config_path = config_root / f"{task}_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {config_path}")

if missing:
    print("\nMissing required data files:")
    for path in missing:
        print(f"  {path}")
    print("\nPut edge probing JSONL files at the paths above, then rerun:")
    print("  bash run_bert_large_edge_tasks.sh prepare")
    raise SystemExit(2)
PY
}

export_model() {
  "$PYTHON" jiant/proj/main/export_model.py \
    --hf_pretrained_model_name_or_path "$MODEL" \
    --output_base_path "$EXP_DIR/models/$MODEL"
}

cache_tasks() {
  for task in $TASKS; do
    phases="train,val"
    if [[ "${INCLUDE_TEST:-0}" == "1" && -f "$EXP_DIR/tasks/data/$task/test.jsonl" ]]; then
      phases="train,val,test"
    fi

    "$PYTHON" jiant/proj/main/tokenize_and_cache.py \
      --task_config_path "$EXP_DIR/tasks/configs/${task}_config.json" \
      --hf_pretrained_model_name_or_path "$MODEL" \
      --output_dir "$EXP_DIR/cache/$MODEL/$task" \
      --phases "$phases" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      --smart_truncate \
      --do_iter
  done
}

train_tasks() {
  for task in $TASKS; do
    if [[ -n "$RUN_NAME" ]]; then
      output_dir="$EXP_DIR/runs/$MODEL/$task/$RUN_NAME"
      runconfig_path="$EXP_DIR/runconfigs/$MODEL/$task/${RUN_NAME}.json"
    else
      output_dir="$EXP_DIR/runs/$MODEL/$task"
      runconfig_path="$EXP_DIR/runconfigs/$MODEL/${task}.json"
    fi
    mkdir -p "$(dirname "$runconfig_path")"

    "$PYTHON" jiant/proj/main/scripts/configurator.py \
      SingleTaskConfigurator \
      "$runconfig_path" \
      --task_name "$task" \
      --task_config_base_path "$EXP_DIR/tasks/configs" \
      --task_cache_base_path "$EXP_DIR/cache/$MODEL" \
      --epochs "$EPOCHS" \
      --train_batch_size "$TRAIN_BATCH_SIZE" \
      --eval_batch_multiplier "$EVAL_BATCH_MULTIPLIER" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --do_train --do_val

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" jiant/proj/main/runscript.py \
      run \
      --ZZsrc "$EXP_DIR/models/$MODEL/config.json" \
      --jiant_task_container_config_path "$runconfig_path" \
      --model_load_mode from_transformers \
      --learning_rate "$LEARNING_RATE" \
      --eval_every_steps 1000 \
      --no_improvements_for_n_evals "$NO_IMPROVEMENTS_FOR_N_EVALS" \
      --do_train --do_val \
      --do_save --force_overwrite \
      --output_dir "$output_dir"
  done
}

summarize_results() {
  "$PYTHON" - "$EXP_DIR" "$MODEL" "$TASKS" "$RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
model = sys.argv[2]
tasks = sys.argv[3].split()
run_name = sys.argv[4]
rows = []

for task in tasks:
    if run_name:
        path = exp_dir / "runs" / model / task / run_name / "val_metrics.json"
    else:
        path = exp_dir / "runs" / model / task / "val_metrics.json"
    if not path.exists():
        rows.append((task, "MISSING", "", ""))
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    task_metrics = data.get(task, {}).get("metrics", {})
    major = task_metrics.get("major", data.get("aggregated", ""))
    minor = task_metrics.get("minor", {})
    rows.append((task, major, minor.get("f1_micro", ""), minor.get("acc", "")))

if run_name:
    summary_path = exp_dir / "runs" / model / f"summary.{run_name}.tsv"
else:
    summary_path = exp_dir / "runs" / model / "summary.tsv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    f.write("task\tmajor\tf1_micro\tacc\n")
    for row in rows:
        f.write("\t".join(str(x) for x in row) + "\n")

print(f"wrote {summary_path}")
print("task\tmajor\tf1_micro\tacc")
for row in rows:
    print("\t".join(str(x) for x in row))
PY
}

case "$STAGE" in
  prepare)
    prepare_configs
    ;;
  export)
    require_jiant || die "Cannot import jiant. Activate the environment and install requirements first."
    export_model
    ;;
  cache)
    require_jiant || die "Cannot import jiant. Activate the environment and install requirements first."
    prepare_configs
    cache_tasks
    ;;
  train)
    require_jiant || die "Cannot import jiant. Activate the environment and install requirements first."
    train_tasks
    ;;
  summarize)
    summarize_results
    ;;
  all)
    require_jiant || die "Cannot import jiant. Activate the environment and install requirements first."
    prepare_configs
    export_model
    cache_tasks
    train_tasks
    summarize_results
    ;;
  *)
    die "Unknown stage '$STAGE'. Use prepare, export, cache, train, summarize, or all."
    ;;
esac
