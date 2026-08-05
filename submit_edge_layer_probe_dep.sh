#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_LIST=dep bash "$SCRIPT_DIR/submit_edge_layer_probe_scan.sh"
