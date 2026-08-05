#!/usr/bin/env bash
# Submit full fixed-layer edge probing scans (layers 0..24).
#
# Use this only after the small scan (0/4/8/12/16/20/24) finishes and the
# metrics look reasonable. Defaults to all currently trainable edge tasks:
#   dep semeval ner spr2 pos nonterminal srl coref
#
# Examples:
#   bash submit_edge_layer_probe_full_scan.sh
#   TASK_LIST="dep semeval" bash submit_edge_layer_probe_full_scan.sh
#   TASK_LIST=pos EPOCHS=1 NO_IMPROVEMENTS_FOR_N_EVALS=10 bash submit_edge_layer_probe_full_scan.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK_LIST="${TASK_LIST:-dep semeval ner spr2 pos nonterminal srl coref}" \
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24}" \
bash "$SCRIPT_DIR/submit_edge_layer_probe_scan.sh"
