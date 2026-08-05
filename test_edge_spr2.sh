#!/usr/bin/env bash
# Edge probing test eval for **spr2** only. Appends to runs/$MODEL/test_spr2.tsv.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_FILTER=spr2
exec bash "$SCRIPT_DIR/test_bert_large_edge_models.sh" "$@"
