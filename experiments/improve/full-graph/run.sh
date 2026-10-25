#!/usr/bin/env bash
# Experiment 3 (one combined variant, Lộc's plan of 22/10): full image graphs on top of 1a (--fusion raw), data
# version merge_v1 (factual_v1 + duplicate boxes merged, IoU >= 0.7, same type). All changes at once:
# merged duplicates + six box-geometry numbers on every edge + 5-nearest completion edges and bridges until each
# image graph is connected ([none] predicate, not triples) + one global node per image and per sentence.
# Same recipe as raw_factual_r3 otherwise: same --fusion-scale constants (train statistics of step 0, factual_v1),
# micro-batch 32, allocator override (../README.md), 10 epochs, seeds 0-2 in order, --resume; one GPU job at a time.
# Diagnostics follow in diagnostics.sh (run if present when training ends).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/merge_v1.yaml
OUT=experiments/improve/full-graph
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
SCALE=$($PY -c "import json; print(*json.load(open('experiments/improve/fusion/train_channel_stats.json'))['fusion_scale'])")
for seed in 0 1 2; do
  run=fullgraph_factual_r3_seed$seed
  [ -f $OUT/$run/metrics.json ] && continue
  for attempt in 1 2 3; do
    log "start $run attempt $attempt"
    if $PY src/train_graph_reranker.py --config $CFG --seed $seed --run-name $run --runs-root improve/full-graph \
        --micro-batch-size 32 --resume --fusion raw --fusion-scale $SCALE \
        --edge-geometry --knn-edges 5 --global-node >> $OUT/logs/$run.stdout.log 2>&1; then
      log "done $run"; break
    fi
    log "failed $run attempt $attempt"
  done
done
[ -f $OUT/diagnostics.sh ] && { log "start diagnostics"; bash $OUT/diagnostics.sh; log "done diagnostics"; }
log "all done"
