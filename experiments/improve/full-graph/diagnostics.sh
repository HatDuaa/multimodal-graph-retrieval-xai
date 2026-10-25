#!/usr/bin/env bash
# Diagnostics of fullgraph_factual_r3 (val only, no training), run by run.sh after the three seeds; one GPU job at a
# time. A step whose output exists is skipped, so the script can be rerun. The baseline raw_factual_r3 (1a) already
# has its perturbation, spread and correlation files in experiments/improve/fusion/raw; only the gradient probe
# measures it again, with the same fixed train batches as the new runs.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
CFG=configs/merge_v1.yaml
OUT=experiments/improve/full-graph
BASE=experiments/improve/fusion/raw
RUNS="$OUT/fullgraph_factual_r3_seed0 $OUT/fullgraph_factual_r3_seed1 $OUT/fullgraph_factual_r3_seed2"
log() { echo "$(date '+%F %T') $*" >> $OUT/logs/timeline.log; }
step() {  # name output command...
  local name=$1 output=$2; shift 2
  [ -f "$output" ] && return
  log "start $name"
  if "$@" > $OUT/logs/$name.log 2>&1; then log "done $name"; else log "failed $name"; fi
}
step triple_perturbation $OUT/triple_perturbation.json \
  $PY scripts/improve/triple_perturbation.py --config $CFG --runs $RUNS --out $OUT/triple_perturbation.json
step part_similarity $OUT/part_similarity.json \
  $PY scripts/checks/part_similarity.py --config $CFG --runs $RUNS --step0 phrase --out $OUT/part_similarity.json
step channel_correlation $OUT/channel_correlation.json \
  $PY scripts/improve/channel_correlation.py --config $CFG --group fullgraph_factual_r3 $RUNS --out $OUT/channel_correlation.json
step gradient_probe $OUT/gradient_probe.json \
  $PY scripts/improve/gradient_probe.py --group raw_factual_r3 $BASE/raw_factual_r3_seed0 $BASE/raw_factual_r3_seed1 \
  $BASE/raw_factual_r3_seed2 --group fullgraph_factual_r3 $RUNS --out $OUT/gradient_probe.json
step gat_attention $OUT/attention.json $PY scripts/improve/gat_attention.py --runs $RUNS
step summarize $OUT/summary.json \
  $PY scripts/improve/summarize_improve.py --exp $OUT --prefix fullgraph_factual_r3 --label fullgraph_factual_r3 \
  --baseline raw_factual_r3=$BASE --reference raw_factual_r3=$BASE/triple_perturbation.json \
  --spread-reference raw_factual_r3=$BASE/part_similarity.json
