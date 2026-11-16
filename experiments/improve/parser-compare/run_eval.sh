#!/usr/bin/env bash
# Parts 1b-1c and 2c after run_parse.sh: CPU probes, intrinsic metrics, then one GPU job at a time.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/improve/parser-compare
for parser in sng factual-base factual-large qwen2.5-3b; do
  [ -f $OUT/probe_$parser.json ] || CUDA_VISIBLE_DEVICES= $PY scripts/improve/parser_probe.py --parser $parser > $OUT/logs/probe_$parser.log 2>&1
done &
$PY scripts/improve/parser_intrinsic.py > $OUT/logs/intrinsic.log 2>&1
$PY scripts/improve/parser_rerank_eval.py --parsers sng factual-base factual-large qwen2.5-3b > $OUT/logs/rerank_main_r3.log 2>&1
$PY scripts/improve/build_boxes.py --split val > experiments/improve/geometry-channel/logs/build_boxes.log 2>&1
$PY scripts/improve/geometry_channel.py > experiments/improve/geometry-channel/logs/geometry_channel.log 2>&1
wait
echo "$(date '+%F %T') eval done" >> $OUT/logs/timeline.log
