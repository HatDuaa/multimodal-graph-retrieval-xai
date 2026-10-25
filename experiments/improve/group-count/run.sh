#!/usr/bin/env bash
# Experiment 4: group nodes and counts on top of the full image graph (fullgraph_factual_r3), data version group_v1.
# Same as fullgraph_factual_r3 (--fusion raw with the same --fusion-scale, edge geometry, 5-nearest completion edges and
# bridges, global node, micro-batch 32, allocator override, 10 epochs, seeds 0-2 in order, --resume) except:
# image graphs with group nodes (majority labels) and node counts, caption count classes (configs/group_v1.yaml), and
# the learned count table g in the object channel (--count-table). Diagnostics follow in diagnostics.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/group_v1.yaml
OUT=experiments/improve/group-count
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
SCALE=$($PY -c "import json; print(*json.load(open('experiments/improve/fusion/train_channel_stats.json'))['fusion_scale'])")
for seed in 0 1 2; do
  run=groupcount_factual_r3_seed$seed
  [ -f $OUT/$run/metrics.json ] && continue
  for attempt in 1 2 3; do
    log "start $run attempt $attempt"
    if $PY src/train_graph_reranker.py --config $CFG --seed $seed --run-name $run --runs-root improve/group-count \
        --micro-batch-size 32 --resume --fusion raw --fusion-scale $SCALE \
        --edge-geometry --knn-edges 5 --global-node --count-table >> $OUT/logs/$run.stdout.log 2>&1; then
      log "done $run"; break
    fi
    log "failed $run attempt $attempt"
  done
done
[ -f $OUT/diagnostics.sh ] && { log "start diagnostics"; bash $OUT/diagnostics.sh; log "done diagnostics"; }
log "all done"
