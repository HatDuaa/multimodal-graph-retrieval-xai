#!/usr/bin/env bash
# Diagnostics of groupcount_factual_r3 against fullgraph_factual_r3 (val only, no training), run by run.sh after the
# three seeds; one GPU job at a time; a step whose output exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/group_v1.yaml
OUT=experiments/improve/group-count
BASE=experiments/improve/full-graph
RUNS="$OUT/groupcount_factual_r3_seed0 $OUT/groupcount_factual_r3_seed1 $OUT/groupcount_factual_r3_seed2"
BASE_RUNS="$BASE/fullgraph_factual_r3_seed0 $BASE/fullgraph_factual_r3_seed1 $BASE/fullgraph_factual_r3_seed2"
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
step() {  # name output command...
  local name=$1 output=$2; shift 2
  [ -f "$output" ] && return
  log "start $name"
  if "$@" > $OUT/logs/$name.log 2>&1; then log "done $name"; else log "failed $name"; fi
}
step count_subset $OUT/count_subset.json \
  $PY scripts/improve/count_subset_eval.py --group groupcount_factual_r3 $RUNS --group fullgraph_factual_r3 $BASE_RUNS \
  --out $OUT/count_subset.json
step triple_perturbation $OUT/triple_perturbation.json \
  $PY scripts/improve/triple_perturbation.py --config $CFG --runs $RUNS --out $OUT/triple_perturbation.json
step part_similarity $OUT/part_similarity.json \
  $PY scripts/checks/part_similarity.py --config $CFG --runs $RUNS --step0 phrase --out $OUT/part_similarity.json
step channel_correlation $OUT/channel_correlation.json \
  $PY scripts/improve/channel_correlation.py --config $CFG --group groupcount_factual_r3 $RUNS --out $OUT/channel_correlation.json
step gradient_probe $OUT/gradient_probe.json \
  $PY scripts/improve/gradient_probe.py --group groupcount_factual_r3 $RUNS --out $OUT/gradient_probe.json
step summarize $OUT/summary.json \
  $PY scripts/improve/summarize_improve.py --exp $OUT --prefix groupcount_factual_r3 --label groupcount_factual_r3 \
  --baseline fullgraph_factual_r3=$BASE --baseline raw_factual_r3=experiments/improve/fusion/raw \
  --reference fullgraph_factual_r3=$BASE/triple_perturbation.json \
  --spread-reference fullgraph_factual_r3=$BASE/part_similarity.json
