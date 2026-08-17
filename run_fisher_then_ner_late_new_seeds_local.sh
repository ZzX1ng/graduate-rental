#!/usr/bin/env bash
# Run remaining Fisher seeds, then the NER late-only robustness check.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/base_exp/exp_edge/local_logs/fisher_then_ner_late_new_seeds_$RUN_ID"
STATUS="$LOG_DIR/status.tsv"
failures=0

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/runner.pid"
printf 'stage\tstatus\tlog\n' > "$STATUS"

fisher_log="$LOG_DIR/fisher_remaining.launcher.log"
if env RUN_ID="${RUN_ID}_fisher" bash "$PROJECT_ROOT/run_ner_dep_fisher_remaining_seeds_local.sh" > "$fisher_log" 2>&1; then
  status=completed
else
  status=failed
  failures=$((failures + 1))
fi
printf 'fisher_remaining\t%s\t%s\n' "$status" "$fisher_log" >> "$STATUS"

late_log="$LOG_DIR/ner_late_new_seeds.launcher.log"
if env RUN_ID="${RUN_ID}_late" bash "$PROJECT_ROOT/run_ner_late_new_seeds_only_local.sh" > "$late_log" 2>&1; then
  status=completed
else
  status=failed
  failures=$((failures + 1))
fi
printf 'ner_late_new_seeds\t%s\t%s\n' "$status" "$late_log" >> "$STATUS"

echo "[$(date --iso-8601=seconds)] FINISH failures=$failures status=$STATUS"
exit "$failures"
