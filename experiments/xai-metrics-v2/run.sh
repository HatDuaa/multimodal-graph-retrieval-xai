#!/usr/bin/env bash
# Level-3 explanation metrics, version 2 (scripts/xai_metrics.py), on val: countonly_factual_r3 and main_r3, seeds 0-2,
# checkpoint_best; no training, no test. The explainer comparison runs on 1,000 fixed val captions (GNNExplainer on
# 200 of them). Pass models as arguments to run only some, e.g. `bash run.sh main_r3`; later runs add to metrics.json.
# The last step writes summary.json and logs/tables.md (the appendix of REPORT.md) with scripts/summarize_xai_metrics.py.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/xai-metrics-v2
MODELS=${*:-countonly_factual_r3 main_r3}
mkdir -p $OUT/logs
for model in $MODELS; do
  merge=$([ -f $OUT/metrics.json ] && echo --merge)
  { df -h / | tail -1 | awk '{print "free", $4}'; date; echo "model $model"; } >> $OUT/logs/run.log
  $PY scripts/xai_metrics.py --models $model $merge 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" >> $OUT/logs/run.log
  echo "exit ${PIPESTATUS[0]}" >> $OUT/logs/run.log
done
date >> $OUT/logs/run.log
$PY scripts/summarize_xai_metrics.py > $OUT/logs/tables.md
