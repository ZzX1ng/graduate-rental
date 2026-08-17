#!/usr/bin/env bash
# Run Fisher collection, representative perturbations, then stage-2 allocation tests.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/master-gra/my_project}"
LOG_ROOT="$PROJECT_ROOT/base_exp/exp_edge/local_logs"
STATUS="$LOG_ROOT/semeval_fisher_pipeline.status"

mkdir -p "$LOG_ROOT"
for seed in 20260806 20260807 20260808 20260809; do
  stats="$PROJECT_ROOT/base_exp/exp_edge/analysis/ptq/semeval_fisher_w4a4_seed${seed}_cal16.json"
  if [[ ! -s "$stats" ]]; then
    printf 'running_fisher_stats_seed%s\t%s\n' "$seed" "$(date --iso-8601=seconds)" > "$STATUS"
    FISHER_SEED="$seed" bash "$PROJECT_ROOT/run_semeval_fisher_stats_local.sh"
  fi
done
printf 'running_stage1_perturbations\t%s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$PROJECT_ROOT/run_semeval_fisher_stage1_perturbations_local.sh"
printf 'running_stage2\t%s\n' "$(date --iso-8601=seconds)" > "$STATUS"
bash "$PROJECT_ROOT/run_semeval_fisher_stage2_local.sh"
printf 'completed\t%s\n' "$(date --iso-8601=seconds)" > "$STATUS"
