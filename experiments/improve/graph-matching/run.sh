#!/usr/bin/env bash
# Experiment 2 (training-free whole-graph matching), CPU only, after fusion/run.sh wrote the main_factual_r3 dumps
# (scripts/improve/dump_val_scores.py). nice + 12 workers so that a training job on the GPU is not slowed down.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/improve/graph-matching
CUDA_VISIBLE_DEVICES= nice -n 10 $PY scripts/improve/graph_matching_probe.py --workers 12 > $OUT/logs/graph_matching_probe.log 2>&1
CUDA_VISIBLE_DEVICES= nice -n 10 $PY scripts/improve/graph_matching_eval.py > $OUT/logs/graph_matching_eval.log 2>&1
CUDA_VISIBLE_DEVICES= nice -n 10 $PY scripts/improve/graph_matching_examples.py > $OUT/logs/graph_matching_examples.log 2>&1
