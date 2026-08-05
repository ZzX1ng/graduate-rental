#!/usr/bin/env bash
# Edge probing test eval for **dep** only. Appends to runs/$MODEL/test_dep.tsv (see test_bert_large_edge_models.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_FILTER=dep
exec bash "$SCRIPT_DIR/test_bert_large_edge_models.sh" "$@"
