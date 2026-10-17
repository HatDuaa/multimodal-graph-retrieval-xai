#!/usr/bin/env bash
# Steps 2 and 3 one after the other: one GPU job at a time.
cd "$(dirname "$0")"
./triple-factorized/run.sh
./graph-channel/run.sh
