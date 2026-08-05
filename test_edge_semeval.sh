#!/usr/bin/env bash
# Edge probing test eval for **semeval** only. Appends to runs/$MODEL/test_semeval.tsv.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_FILTER=semeval
exec bash "$SCRIPT_DIR/test_bert_large_edge_models.sh" "$@"
