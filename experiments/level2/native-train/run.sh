#!/usr/bin/env bash
# Level 2: train NativE (zjukg/NativE c72291c) on the RTX 4090, one run at a time, then export the rmask embeddings.
#
# Runs, in order: rmask seeds 0, 1, 2 (MKG-W-rmask, step b) and repro seed 42 (original MKG-W, step a).
# Hyper-parameters are the authors' scripts/run_mkgw.sh, exactly as notebooks/mkgw_molab.py builds them; only the
# dataset and the seed change between runs.
#
# Usage (from anywhere, inside tmux so it survives a disconnect):
#   tmux new -d -s native "bash experiments/level2/native-train/run.sh"
# Optional: PY=<python> (default ~/venvs/mgrx/bin/python), RUNS="rmask:MKG-W-rmask:0 ..." to run a subset.
#
# A run whose checkpoint already exists is skipped (rerun after a crash continues where it stopped).
# Before each run the driver checks that at least MIN_FREE_GB are free and stops otherwise; it never deletes anything.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-$HOME/venvs/mgrx/bin/python}"
EPOCH=1000
MIN_FREE_GB=20
RUNS="${RUNS:-rmask:MKG-W-rmask:0 rmask:MKG-W-rmask:1 rmask:MKG-W-rmask:2 repro:MKG-W:42}"
OUT="$REPO/experiments/level2/native-train"
DRIVER="$OUT/driver.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

cd "$REPO"
NATIVE="$("$PY" -c "from pathlib import Path; from src.utils.config import load_config; \
print(Path(load_config()['mkgw']['native_repo']).expanduser())")"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER"; }

say "driver start: runs=[$RUNS] epoch=$EPOCH native=$NATIVE ($(git -C "$NATIVE" rev-parse --short HEAD))"
say "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
say "python: $("$PY" -c 'import torch, sys; print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)')"

for spec in $RUNS; do
    IFS=: read -r tag dataset seed <<< "$spec"
    name="${tag}_seed${seed}_e${EPOCH}"
    ckpt="$NATIVE/checkpoint/$name.ckpt"
    log="$OUT/$name.log"
    if [[ -f "$ckpt" ]]; then
        say "skip $name: checkpoint exists"
    else
        free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
        if (( free_gb < MIN_FREE_GB )); then
            say "STOP before $name: ${free_gb} GB free < ${MIN_FREE_GB} GB"
            exit 2
        fi
        cmd="$PY -u run_adv_wgan_gp.py -dataset=$dataset -batch_size=1024 -margin=4 -epoch=$EPOCH -dim=250 \
-save=./checkpoint/$name.ckpt -neg_num=128 -mu=0.0001 -learning_rate=1e-4 -lrg=1e-4 -seed=$seed"
        say "start $name (${free_gb} GB free)"
        echo "\$ $cmd" > "$log"
        start=$(date +%s)
        (cd "$NATIVE" && $cmd >> "$log" 2>&1)
        status=$?
        wall=$(( $(date +%s) - start ))
        say "end $name: exit $status, wall ${wall} s ($(awk -v s="$wall" 'BEGIN{printf "%.2f", s/3600}') h)"
        if (( status != 0 )); then
            say "STOP: $name failed, see $log"
            exit 1
        fi
        say "$name result: $(grep -a 'averaged(filter)' "$log" | tail -1 | tr -s ' \t' ' ')"
    fi
    if [[ "$tag" == rmask ]]; then
        if "$PY" scripts/export_native_embeddings.py --checkpoint "$ckpt" --what ent --suffix "_seed$seed" >> "$DRIVER" 2>&1; then
            say "exported $name -> data/features/mkgw_native_ent_seed$seed.npy"
        else
            say "export of $name failed (training kept), see $DRIVER"
        fi
    fi
done
say "driver done"
