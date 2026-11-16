#!/usr/bin/env bash
# Experiment 1 (fusion normalisation) on data version factual_v1, against main_factual_r3; one GPU job at a time.
# 0. val scores and encoded parts of main_factual_r3 for the CPU graph-matching probes (experiment 2);
# 1a. --fusion raw, started equivalent to the z-score fusion (--fusion-scale = median per-query train std, step 0);
# 1b. --zscore-floor = 5th percentile of the per-query train std per channel (step 0).
# Both constants come from train_channel_stats.json (scripts/improve/channel_stats.py), never from val.
# Same recipe as main_factual_r3 otherwise: micro-batch 32, allocator override (../README.md), seeds 0-2, --resume.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/factual_v1.yaml
OUT=experiments/improve/fusion
BASE=experiments/improve/factual-retrain
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
stat() { $PY -c "import json; print(*json.load(open('$OUT/train_channel_stats.json'))['$1'])"; }
[ -f data/processed/graph_matching/main_factual_r3_seed2.npz ] || {
  log "start dump"; $PY scripts/improve/dump_val_scores.py --config $CFG --runs $BASE/main_factual_r3_seed{0,1,2} \
      > $OUT/logs/dump_val_scores.log 2>&1 && log "done dump" || log "failed dump"; }
train() {  # variant prefix flags...
  local variant=$1 prefix=$2; shift 2
  mkdir -p $OUT/$variant
  for seed in 0 1 2; do
    local run=${prefix}_seed$seed
    [ -f $OUT/$variant/$run/metrics.json ] && continue
    for attempt in 1 2 3; do
      log "start $variant/$run attempt $attempt"
      if $PY src/train_graph_reranker.py --config $CFG --seed $seed --run-name $run --runs-root improve/fusion/$variant \
          --micro-batch-size 32 --resume "$@" >> $OUT/logs/$run.stdout.log 2>&1; then
        log "done $variant/$run"; break
      fi
      log "failed $variant/$run attempt $attempt"
    done
  done
  local runs="$OUT/$variant/${prefix}_seed0 $OUT/$variant/${prefix}_seed1 $OUT/$variant/${prefix}_seed2"
  log "start diagnostics $variant"
  $PY scripts/improve/triple_perturbation.py --config $CFG --runs $runs --out $OUT/$variant/triple_perturbation.json > $OUT/logs/triple_perturbation_$variant.log 2>&1
  $PY scripts/checks/part_similarity.py --config $CFG --runs $runs --step0 phrase --out $OUT/$variant/part_similarity.json > $OUT/logs/part_similarity_$variant.log 2>&1
  $PY scripts/improve/channel_correlation.py --config $CFG --group $prefix $runs --out $OUT/$variant/channel_correlation.json > $OUT/logs/channel_correlation_$variant.log 2>&1
  log "done diagnostics $variant"
}
train raw raw_factual_r3 --fusion raw --fusion-scale $(stat fusion_scale)
train floor floor_factual_r3 --zscore-floor $(stat zscore_floor)
log "all done"
