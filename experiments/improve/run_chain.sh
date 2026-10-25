#!/usr/bin/env bash
# Micro-batch control, then steps 2 and 3, one after the other: one GPU job at a time.
cd "$(dirname "$0")"
./microbatch-control/run.sh
./triple-factorized/run.sh
./graph-channel/run.sh
