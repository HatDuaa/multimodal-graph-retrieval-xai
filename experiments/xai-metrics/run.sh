#!/usr/bin/env bash
# Level-3 explanation metrics (fidelity, validity, sparsity, explanation subgraph) on val for countonly_factual_r3 and
# main_r3, seeds 0-2, checkpoint_best; no training, no test. One GPU job; scripts/xai_metrics.py stops before a run when
# less than 10 GB is free and writes nothing but metrics.json / cases.json (scores stay in memory).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/xai-metrics
mkdir -p $OUT/logs
{ df -h / | tail -1 | awk '{print "free", $4}'; date; } > $OUT/logs/run.log
$PY scripts/xai_metrics.py 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" >> $OUT/logs/run.log
echo "exit ${PIPESTATUS[0]}" >> $OUT/logs/run.log
date >> $OUT/logs/run.log
$PY scripts/summarize_xai_metrics.py > $OUT/logs/tables.md
