#!/usr/bin/env bash
set -euo pipefail

cd /cluster/home/zhangzx/my_project

submit_one() {
  local job_name="$1"
  local run_name="$2"
  local config="$3"

  export GELU_APPROX_LAYER_SEGMENTS="$config"
  export GELU_APPROX_MIN=-4.0
  export GELU_APPROX_MAX=4.0
  unset GELU_APPROX_SEGMENTS

  sbatch --exclude=node6 \
    --job-name="$job_name" \
    --export=ALL,TASKS=dep,EPOCHS=3,RUN_NAME="$run_name",NO_IMPROVEMENTS_FOR_N_EVALS=0 \
    slurm_edge_ms256_formal_train.sbatch
}

submit_one \
  "dep_avg108_aw15_16_e3" \
  "gelu_pwl1_l00_14_pwl2_l15_16_pwl1_l17_23_e3_ms256_dep_avg108_aware15_16" \
  "0-14:1;15-16:2;17-23:1"

submit_one \
  "dep_avg108_early2_e3" \
  "gelu_pwl2_l00_01_pwl1_l02_23_e3_ms256_dep_avg108_early2" \
  "0-1:2;2-23:1"

submit_one \
  "dep_avg108_late2_e3" \
  "gelu_pwl1_l00_21_pwl2_l22_23_e3_ms256_dep_avg108_late2" \
  "0-21:1;22-23:2"
