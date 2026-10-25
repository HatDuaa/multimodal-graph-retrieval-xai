#!/usr/bin/env bash
# Control for the micro-batch change: the main_r3 recipe (phrase triple channel, all defaults) with micro-batch 32
# instead of 128, seed 0, two epochs; compare with main_r3_seed0 epochs 1-2. Effective batch stays 256.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/improve/microbatch-control
mkdir -p $OUT/logs
$PY src/train_graph_reranker.py --seed 0 --run-name main_mb32_seed0 --runs-root improve/microbatch-control \
    --micro-batch-size 32 --epochs 2 --resume >> $OUT/logs/main_mb32_seed0.stdout.log 2>&1
echo "exit $?" >> $OUT/logs/main_mb32_seed0.stdout.log
