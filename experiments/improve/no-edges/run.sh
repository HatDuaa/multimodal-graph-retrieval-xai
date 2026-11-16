#!/usr/bin/env bash
# Ablation: countonly_factual_r3 retrained without any edge (countonly_noedges_r3). Same recipe as
# experiments/improve/count-only/run.sh (count_v1, --fusion raw with the same --fusion-scale, edge geometry, 5-nearest
# completion edges and bridges, global node, count table, micro-batch 32, allocator override, 10 epochs, seeds 0-2,
# --resume) plus --no-use-edges, as level 1's no_edges_r3: the GAT gets no edge at all (relations, [none] completion
# edges and global-node edges) and the triple channel is off. Val only. The disk guard stops the chain below 20 GB free
# (logged, nothing deleted).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/count_v1.yaml
OUT=experiments/improve/no-edges
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
disk_ok() {
  local free; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 20 ]; then log "stop: $free GB free < 20 GB before $1 (nothing deleted)"; return 1; fi
  log "disk $free GB free before $1"
}
SCALE=$($PY -c "import json; print(*json.load(open('experiments/improve/fusion/train_channel_stats.json'))['fusion_scale'])")
for seed in 0 1 2; do
  run=countonly_noedges_r3_seed$seed
  [ -f $OUT/$run/metrics.json ] && continue
  disk_ok $run || exit 1
  for attempt in 1 2 3; do
    log "start $run attempt $attempt"
    if $PY src/train_graph_reranker.py --config $CFG --seed $seed --run-name $run --runs-root improve/no-edges \
        --micro-batch-size 32 --resume --fusion raw --fusion-scale $SCALE \
        --edge-geometry --knn-edges 5 --global-node --count-table --no-use-edges >> $OUT/logs/$run.stdout.log 2>&1; then
      log "done $run"; break
    fi
    log "failed $run attempt $attempt"
  done
done
log "all done"
