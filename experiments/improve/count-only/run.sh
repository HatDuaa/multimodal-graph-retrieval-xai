#!/usr/bin/env bash
# Experiment 5: counts and the count table g on top of fullgraph_factual_r3, without grouping (data version count_v1:
# merge_v1 graphs unchanged, each node counting the same-type instances of its image; caption count classes as in
# experiment 4). Same recipe as fullgraph_factual_r3 (--fusion raw with the same --fusion-scale, edge geometry,
# 5-nearest completion edges and bridges, global node, micro-batch 32, allocator override, 10 epochs, seeds 0-2,
# --resume) plus --count-table. The disk guard stops the chain below 20 GB free (logged, nothing deleted).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/count_v1.yaml
OUT=experiments/improve/count-only
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
disk_ok() {
  local free; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 20 ]; then log "stop: $free GB free < 20 GB before $1 (nothing deleted)"; return 1; fi
  log "disk $free GB free before $1"
}
SCALE=$($PY -c "import json; print(*json.load(open('experiments/improve/fusion/train_channel_stats.json'))['fusion_scale'])")
for seed in 0 1 2; do
  run=countonly_factual_r3_seed$seed
  [ -f $OUT/$run/metrics.json ] && continue
  disk_ok $run || exit 1
  for attempt in 1 2 3; do
    log "start $run attempt $attempt"
    if $PY src/train_graph_reranker.py --config $CFG --seed $seed --run-name $run --runs-root improve/count-only \
        --micro-batch-size 32 --resume --fusion raw --fusion-scale $SCALE \
        --edge-geometry --knn-edges 5 --global-node --count-table >> $OUT/logs/$run.stdout.log 2>&1; then
      log "done $run"; break
    fi
    log "failed $run attempt $attempt"
  done
done
disk_ok diagnostics || exit 1
[ -f $OUT/diagnostics.sh ] && { log "start diagnostics"; bash $OUT/diagnostics.sh; log "done diagnostics"; }
log "all done"
