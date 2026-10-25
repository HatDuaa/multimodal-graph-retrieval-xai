#!/usr/bin/env bash
# Step 1: FACTUAL-large on the train and test captions (val was parsed in ../parser-compare), same settings as val
# (beam 5, batch 64). Resumable: finished chunks are skipped. Test is parsed only to have the data ready.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/improve/factual-retrain
for split in train test; do
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') start parse $split attempt $attempt" >> $OUT/logs/timeline.log
    if $PY scripts/improve/parse_captions.py --parser factual-large --split $split --chunk-size 8192 \
        >> $OUT/logs/parse_${split}_factual-large.log 2>&1; then
      echo "$(date '+%F %T') done parse $split" >> $OUT/logs/timeline.log; break
    fi
    echo "$(date '+%F %T') failed parse $split attempt $attempt" >> $OUT/logs/timeline.log
  done
done
