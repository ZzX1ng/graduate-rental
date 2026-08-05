#!/usr/bin/env bash
# Submit 12 Slurm jobs: pos / nonterminal / srl / coref × epochs 3 / 5 / 10.
# No early stopping (NO_IMPROVEMENTS_FOR_N_EVALS=0), Slurm time UNLIMITED (see .sbatch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="$SCRIPT_DIR/slurm_onto_ms512_formal_train.sbatch"

job_name_for_task() {
  local t="$1"
  case "$t" in
    nonterminal) echo "nt" ;;
    *) echo "$t" ;;
  esac
}

for task in pos nonterminal srl coref; do
  short="$(job_name_for_task "$task")"
  for ep in 3 5 10; do
    rn="formal_e${ep}_ms512_${task}"
    sbatch \
      --job-name="bertlarge_${short}_ms512_e${ep}" \
      --export=ALL,TASKS="${task}",EPOCHS="${ep}",RUN_NAME="${rn}",NO_IMPROVEMENTS_FOR_N_EVALS=0 \
      "$SBATCH_FILE"
  done
done
