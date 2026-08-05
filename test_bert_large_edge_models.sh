#!/usr/bin/env bash
set -euo pipefail

# Evaluate trained BERT-large edge-probing runs on labeled test caches.
#
# Default behavior: evaluate every run directory under
#   $EXP_DIR/runs/$MODEL/<task>/<RUN_NAME>
# that contains best_model.p.
#
# Optional usage:
#   bash test_bert_large_edge_models.sh dep:formal_e10_ms256 semeval:formal_e10_ms256

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
JIANT_DIR="${JIANT_DIR:-$PROJECT_ROOT/base_exp/jiant}"
EXP_DIR="${EXP_DIR:-$PROJECT_ROOT/base_exp/exp_edge}"
MODEL="${MODEL:-bert-large-uncased}"
PYTHON="${PYTHON:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WRITE_PREDS="${WRITE_PREDS:-0}"
NO_CUDA="${NO_CUDA:-0}"
# Optional: restrict to one task, e.g. SPEC_FILTER=dep  -> only specs starting with "dep:"
SPEC_FILTER="${SPEC_FILTER:-}"
# Per-task TSV root (default: one file per task, no cross-task overwrite)
TEST_SUMMARY_DIR="${TEST_SUMMARY_DIR:-$EXP_DIR/runs/$MODEL}"
# If set, all rows go to this single file instead of test_<task>.tsv
SUMMARY_PATH="${SUMMARY_PATH:-}"

export PYTHONPATH="$JIANT_DIR:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

discover_specs() {
  "$PYTHON" - "$EXP_DIR" "$MODEL" <<'PY'
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
model = sys.argv[2]
run_root = exp_dir / "runs" / model
for task_dir in sorted(run_root.glob("*")):
    if not task_dir.is_dir():
        continue
    for run_dir in sorted(task_dir.glob("*")):
        if (run_dir / "best_model.p").exists():
            print(f"{task_dir.name}:{run_dir.name}")
PY
}

if [[ "$#" -gt 0 ]]; then
  mapfile -t SPECS < <(printf '%s\n' "$@")
else
  mapfile -t SPECS < <(discover_specs)
fi

if [[ -n "$SPEC_FILTER" ]]; then
  mapfile -t SPECS < <(printf '%s\n' "${SPECS[@]}" | awk -F: -v p="$SPEC_FILTER" '$1==p {print}')
fi

if [[ "${#SPECS[@]}" -eq 0 ]]; then
  if [[ -n "$SPEC_FILTER" ]]; then
    echo "No specs left after SPEC_FILTER='$SPEC_FILTER' (no matching task:run or no best_model.p)." >&2
  else
    echo "No trained runs with best_model.p found." >&2
  fi
  exit 2
fi

mkdir -p "$TEST_SUMMARY_DIR"

for spec in "${SPECS[@]}"; do
  task="${spec%%:*}"
  run_name="${spec#*:}"
  if [[ -z "$task" || -z "$run_name" || "$task" == "$run_name" ]]; then
    echo "Invalid spec '$spec'. Expected task:run_name" >&2
    exit 2
  fi

  run_dir="$EXP_DIR/runs/$MODEL/$task/$run_name"
  metrics_path="$run_dir/test_eval/test_metrics.json"

  echo "=== Evaluating $task / $run_name ==="
  if [[ -n "$SUMMARY_PATH" ]]; then
    summary_tsv="$SUMMARY_PATH"
  else
    summary_tsv="$TEST_SUMMARY_DIR/test_${task}.tsv"
  fi

  cmd=(
    "$PYTHON" "$PROJECT_ROOT/tools/evaluate_edge_test.py"
    --task "$task"
    --run-name "$run_name"
    --exp-dir "$EXP_DIR"
    --model "$MODEL"
    --summary-tsv "$summary_tsv"
  )
  if [[ "$WRITE_PREDS" == "1" ]]; then
    cmd+=(--write-preds)
  fi
  if [[ "$NO_CUDA" == "1" ]]; then
    cmd+=(--no-cuda)
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
done

if [[ -n "$SUMMARY_PATH" ]]; then
  echo "Wrote $SUMMARY_PATH"
else
  echo "Wrote per-task TSVs under $TEST_SUMMARY_DIR (test_<task>.tsv)"
fi
