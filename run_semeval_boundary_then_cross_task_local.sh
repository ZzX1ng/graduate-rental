#!/usr/bin/env bash
# Run the missing SemEval boundary seeds before the single-seed cross-task study.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
LOG_ROOT="$PROJECT_ROOT/base_exp/exp_edge/local_logs"
STATUS_FILE="$LOG_ROOT/semeval_boundary_then_cross_task.status"

mkdir -p "$LOG_ROOT"
printf 'running_semeval_boundary\t%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
bash "$PROJECT_ROOT/run_semeval_boundary_multiseed_local.sh"

printf 'running_cross_task\t%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
bash "$PROJECT_ROOT/run_cross_task_probe_allocations_single_seed_local.sh"

printf 'completed\t%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
