#!/usr/bin/env bash
# Submit fixed-layer edge probing scans.
#
# Defaults:
#   tasks: dep semeval ner spr2 pos nonterminal srl coref
#   layers: 0 4 8 12 16 20 24
#   epochs: 3
#   early stopping: NO_IMPROVEMENTS_FOR_N_EVALS=10
#
# Override examples:
#   TASK_LIST="dep semeval" bash submit_edge_layer_probe_scan.sh
#   TASK_LIST=pos LAYERS="0 12 24" EPOCHS=1 bash submit_edge_layer_probe_scan.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="$SCRIPT_DIR/slurm_edge_layer_probe.sbatch"

TASK_LIST="${TASK_LIST:-dep semeval ner spr2 pos nonterminal srl coref}"
LAYERS="${LAYERS:-0 4 8 12 16 20 24}"
EPOCHS="${EPOCHS:-3}"
NO_IMPROVEMENTS_FOR_N_EVALS="${NO_IMPROVEMENTS_FOR_N_EVALS:-10}"

max_seq_length_for_task() {
  case "$1" in
    pos|nonterminal|srl|coref) echo 512 ;;
    dep|semeval|ner|spr2) echo 256 ;;
    *)
      echo "ERROR: unknown task '$1'" >&2
      return 2
      ;;
  esac
}

short_name_for_task() {
  case "$1" in
    nonterminal) echo "nt" ;;
    semeval) echo "sem" ;;
    *) echo "$1" ;;
  esac
}

for task in $TASK_LIST; do
  max_seq_length="$(max_seq_length_for_task "$task")"
  short="$(short_name_for_task "$task")"
  for layer in $LAYERS; do
    layer_padded="$(printf "%02d" "$layer")"
    run_name="edge_l${layer_padded}_ms${max_seq_length}_${task}"
    sbatch \
      --job-name="edge_${short}_l${layer_padded}" \
      --export=ALL,TASKS="${task}",MAX_SEQ_LENGTH="${max_seq_length}",EPOCHS="${EPOCHS}",RUN_NAME="${run_name}",BERT_LAYER_INDEX="${layer}",FREEZE_ENCODER=1,NO_IMPROVEMENTS_FOR_N_EVALS="${NO_IMPROVEMENTS_FOR_N_EVALS}" \
      "$SBATCH_FILE"
  done
done
