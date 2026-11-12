#!/usr/bin/env bash
# Scene-ranker options on top of countonly_factual_r3 (docs/scene-ranker.html), not run yet. Same recipe as
# count-only/run.sh (--fusion raw with the same --fusion-scale, edge geometry, 5-nearest completion edges and bridges,
# global node, count table, micro-batch 32, 10 epochs, seeds 0-2, --resume) on data version concept_v1, plus:
#   --part-match sinkhorn       transport-plan matching with a learned dustbin in both channels;
#   --graph-loss-weight 0.3     cross-entropy of the graph score alone, added to the fused loss;
#   --concept-gat               relational GAT over the concept graph, added to every linked node's input;
#   --scene-terms               attributes, attributes_soft, salience and soft_objects with learned weights.
# Build the data version once first: python scripts/build_concept_data.py --config configs/concept_v1.yaml
# The disk guard stops the chain below 20 GB free (logged, nothing deleted).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/concept_v1.yaml
OUT=experiments/improve/scene-ranker
mkdir -p $OUT/logs
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
disk_ok() {
  local free; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 20 ]; then log "stop: $free GB free < 20 GB before $1 (nothing deleted)"; return 1; fi
  log "disk $free GB free before $1"
}
[ -f data/graphs/concept_v1/MANIFEST.json ] || { echo "missing concept_v1: run scripts/build_concept_data.py" >&2; exit 1; }
SCALE=$($PY -c "import json; print(*json.load(open('experiments/improve/fusion/train_channel_stats.json'))['fusion_scale'])")
for seed in 0 1 2; do
  run=scene_countonly_r3_seed$seed
  [ -f $OUT/$run/metrics.json ] && continue
  disk_ok $run || exit 1
  for attempt in 1 2 3; do
    log "start $run attempt $attempt"
    if $PY src/train_graph_reranker.py --config $CFG --seed $seed --run-name $run --runs-root improve/scene-ranker \
        --micro-batch-size 32 --resume --fusion raw --fusion-scale $SCALE \
        --edge-geometry --knn-edges 5 --global-node --count-table \
        --part-match sinkhorn --graph-loss-weight 0.3 --concept-gat --scene-terms >> $OUT/logs/$run.stdout.log 2>&1; then
      log "done $run"; break
    fi
    log "failed $run attempt $attempt"
  done
done
log "all done"
