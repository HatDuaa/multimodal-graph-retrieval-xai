#!/usr/bin/env bash
# Step 1d with FACTUAL-T5 large caption graphs (data version factual_v1), after run_parse.sh, one GPU job at a time:
# package -> checks (probe on the packaged data, step-0 model on the training data path) -> train the main_r3 recipe
# for seeds 0-2 -> val diagnostics. Same recipe as main_r3 except the caption parser; micro-batch 32 and the allocator
# override as every improvement run (../README.md). Test is never evaluated.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/factual_v1.yaml
OUT=experiments/improve/factual-retrain
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
[ -f data/processed/factual_v1/MANIFEST.json ] || { log "start package"; $PY scripts/improve/package_factual.py > $OUT/logs/package.log 2>&1 || { log "failed package"; exit 1; }; log "done package"; }
log "start checks"
$PY scripts/improve/step0_eval.py --config $CFG --out $OUT/check_step0.json > $OUT/logs/check_step0.log 2>&1
CUDA_VISIBLE_DEVICES= $PY scripts/improve/parser_probe.py --parser factual-large --queries data/processed/factual_v1/query_graphs_val.json \
    --table data/features/factual_v1/vg_coco_graph_parts --out-name ../factual-retrain/check_probe.json > $OUT/logs/check_probe.log 2>&1
log "done checks"
for seed in 0 1 2; do
  run=main_factual_r3_seed$seed
  [ -f $OUT/$run/metrics.json ] && continue
  for attempt in 1 2 3; do
    log "start $run attempt $attempt"
    if $PY src/train_graph_reranker.py --config $CFG --seed $seed --run-name $run --runs-root improve/factual-retrain \
        --micro-batch-size 32 --resume >> $OUT/logs/$run.stdout.log 2>&1; then
      log "done $run"; break
    fi
    log "failed $run attempt $attempt"
  done
done
RUNS="$OUT/main_factual_r3_seed0 $OUT/main_factual_r3_seed1 $OUT/main_factual_r3_seed2"
log "start diagnostics"
$PY scripts/improve/triple_perturbation.py --config $CFG --runs $RUNS --out $OUT/triple_perturbation.json > $OUT/logs/triple_perturbation.log 2>&1
$PY scripts/checks/part_similarity.py --config $CFG --runs $RUNS --step0 phrase --out $OUT/part_similarity.json > $OUT/logs/part_similarity.log 2>&1
$PY scripts/improve/channel_correlation.py --config $CFG --group main_factual_r3 $RUNS --out $OUT/channel_correlation.json > $OUT/logs/channel_correlation.log 2>&1
log "all done"
