#!/usr/bin/env bash
# Level-3 explanations against known answers (scripts/xai_ground_truth.py): planted parts (C) and Visual Genome region
# graphs (A) on countonly_factual_r3 and main_r3, seeds 0-2, checkpoint_best, val only; and a synthetic task (B) with
# a model of the same architecture trained here (3 seeds). 100 explained cases per run; every one of the 16
# explainers of scripts/xai_metrics.py (version 3) is scored by GEA, precision@g, AUROC and first position.
# Steps that already have their output are skipped (regions_val.json, the synthetic checkpoints).
# The last step writes summary.json and logs/tables.md (the appendix of REPORT.md) with
# scripts/summarize_xai_ground_truth.py.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/xai-ground-truth
mkdir -p $OUT/logs
run() {
  { date; echo "step $*"; } >> $OUT/logs/run.log
  $PY scripts/xai_ground_truth.py "$@" 2>&1 | grep --line-buffered -v -E "FutureWarning|warnings.warn" >> $OUT/logs/run.log
  echo "exit ${PIPESTATUS[0]}" >> $OUT/logs/run.log
}
[ -f data/processed/xai_regions/regions_val.json ] || run vg-regions prepare
[ -f $OUT/synthetic/models/seed2.pt ] || run synthetic train
run planted
run vg-regions
run synthetic
date >> $OUT/logs/run.log
$PY scripts/summarize_xai_ground_truth.py > $OUT/logs/tables.md
