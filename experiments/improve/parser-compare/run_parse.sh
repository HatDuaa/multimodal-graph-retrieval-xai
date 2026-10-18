#!/usr/bin/env bash
# Part 1a: parse the 10,633 val captions with each parser (one GPU job at a time). sng runs on CPU.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/improve/parser-compare
for parser in sng factual-base factual-large qwen2.5-3b; do
  echo "$(date '+%F %T') start $parser" >> $OUT/logs/timeline.log
  if $PY scripts/improve/parse_captions.py --parser $parser --split val > $OUT/logs/parse_val_$parser.log 2>&1; then
    echo "$(date '+%F %T') done $parser" >> $OUT/logs/timeline.log
  else
    echo "$(date '+%F %T') failed $parser" >> $OUT/logs/timeline.log
  fi
done
echo "$(date '+%F %T') all done" >> $OUT/logs/timeline.log
