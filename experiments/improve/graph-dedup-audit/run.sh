#!/usr/bin/env bash
# Image-graph cleaning audit (merge duplicates, group by role): CPU only, no training, nice + few processes.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/improve/graph-dedup-audit
for split in val train; do nice -n 10 $PY scripts/improve/build_boxes.py --split $split --synsets; done > $OUT/logs/build_objects.log 2>&1
nice -n 10 $PY scripts/improve/graph_dedup_audit.py --workers 6 > $OUT/logs/structure.log 2>&1
nice -n 10 $PY scripts/improve/graph_dedup_val.py > $OUT/logs/val_checks.log 2>&1
nice -n 10 $PY scripts/improve/graph_dedup_draw.py > $OUT/logs/draw.log 2>&1
