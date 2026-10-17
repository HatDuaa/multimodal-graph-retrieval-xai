#!/usr/bin/env bash
# Per-query correlation between channel scores on val (no training), for the level-1 runs and both experiments.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-~/venvs/mgrx/bin/python}
L=experiments/level1/gat; I=experiments/improve
$PY scripts/improve/channel_correlation.py \
    --group main_r3 $L/main_r3_seed0 $L/main_r3_seed1 $L/main_r3_seed2 \
    --group no_gat_r3 $L/no_gat_r3_seed0 $L/no_gat_r3_seed1 $L/no_gat_r3_seed2 \
    --group factorized $I/triple-factorized/factorized_seed0 $I/triple-factorized/factorized_seed1 $I/triple-factorized/factorized_seed2 \
    --group graph_channel $I/graph-channel/graph_channel_seed0 $I/graph-channel/graph_channel_seed1 $I/graph-channel/graph_channel_seed2 \
    --out $I/channel-correlation/channel_correlation.json > $I/channel-correlation/channel_correlation.log 2>&1
