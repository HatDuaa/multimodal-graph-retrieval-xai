#!/usr/bin/env bash
# Step 1 diagnosis (val only, no training). The allocator override avoids NVML, which fails on this machine while the
# NVIDIA driver update of 2026-10-15 waits for a reboot; it changes memory layout only, not the computation.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
OUT=experiments/improve/triple-perturbation
runs() { for s in 0 1 2; do echo "experiments/level1/gat/$1_seed$s"; done; }
$PY scripts/improve/triple_perturbation.py --runs $(runs main_r3) --out $OUT/main_r3.json > $OUT/main_r3.log 2>&1
$PY scripts/improve/triple_perturbation.py --runs $(runs no_triple_r3) --out $OUT/no_triple_r3.json > $OUT/no_triple_r3.log 2>&1
$PY scripts/improve/triple_perturbation.py --runs $(runs no_gat_r3) --out $OUT/no_gat_r3.json > $OUT/no_gat_r3.log 2>&1
$PY scripts/improve/triple_perturbation.py --step0 --out $OUT/step0_frozen_clip.json > $OUT/step0_frozen_clip.log 2>&1
