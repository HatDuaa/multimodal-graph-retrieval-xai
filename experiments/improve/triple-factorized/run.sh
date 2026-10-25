#!/usr/bin/env bash
# Step 2, experiment A: factorized triple matching sim = c(s) * c(p) * c(o), c = (1 + cos) / 2.
# Same recipe as main_r3 (all train_graph_reranker defaults); only the flags below differ. Seeds 0, 1, 2.
# A failed seed is resumed from its checkpoint_last.pt (identical continuation, see tests/test_train_resume.py).
# The allocator override avoids NVML, broken on this machine until the driver update of 2026-10-15 is rebooted;
# it changes memory layout only, not the computation. Without expandable segments the cache fragments, so the
# micro-batch is 32 (effective batch still 256, gradients accumulated; see ../microbatch-control for the check).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
EXP=triple-factorized
FLAGS="--triple-match factorized"
OUT=experiments/improve/$EXP
MICRO="--micro-batch-size 32"
mkdir -p $OUT/logs
for seed in 0 1 2; do
  run=factorized_seed$seed
  [ -f $OUT/$run/metrics.json ] && continue
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') start $run attempt $attempt" >> $OUT/logs/timeline.log
    if $PY src/train_graph_reranker.py --seed $seed --run-name $run --runs-root improve/$EXP $MICRO $FLAGS --resume \
        >> $OUT/logs/$run.stdout.log 2>&1; then
      echo "$(date '+%F %T') done $run" >> $OUT/logs/timeline.log; break
    fi
    echo "$(date '+%F %T') failed $run attempt $attempt" >> $OUT/logs/timeline.log
  done
done
RUNS="$OUT/factorized_seed0 $OUT/factorized_seed1 $OUT/factorized_seed2"
echo "$(date '+%F %T') start diagnostics" >> $OUT/logs/timeline.log
$PY scripts/improve/triple_perturbation.py --runs $RUNS --out $OUT/triple_perturbation.json > $OUT/logs/triple_perturbation.log 2>&1
$PY scripts/checks/part_similarity.py --runs $RUNS --step0 phrase factorized --out $OUT/part_similarity.json > $OUT/logs/part_similarity.log 2>&1
echo "$(date '+%F %T') all done" >> $OUT/logs/timeline.log
