#!/usr/bin/env bash
# Run a list of graph reranker trainings one after another, surviving crashes and reboots.
#
# Usage: bash scripts/run_level1_chain.sh configs/level1_runs.txt
# Each non-comment line of the list: <run_name> <seed> [extra train flags...]
# - a run whose metrics.json exists is skipped;
# - a run folder that already has artifacts is continued with --resume;
# - a run that exits non-zero is resumed, at most 3 times, then the chain moves on.
# Progress goes to ~/mgrx-night/STATUS, each run's output to ~/mgrx-night/<run_name>.log.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LIST="$(realpath "${1:?run list file required}")"
PY="${PYTHON:-$HOME/venvs/mgrx/bin/python}"
NIGHT="$HOME/mgrx-night"
RUNS="$REPO/experiments/level1/gat"
MAX_RETRIES=3
mkdir -p "$NIGHT"
status() { printf "%s %s\n" "$(date "+%F %T %z")" "$*" >> "$NIGHT/STATUS"; }

exec 9>"$NIGHT/chain.lock"
if ! flock -n 9; then
    echo "another chain driver is already running" >&2
    exit 1
fi
cd "$REPO"
status "chain driver started: $LIST"
while read -r name seed flags; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue
    dir="$RUNS/$name"
    if [[ -f "$dir/metrics.json" ]]; then
        status "$name skipped: metrics.json exists"
        continue
    fi
    attempt=0
    while :; do
        resume=()
        [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]] && resume=(--resume)
        status "$name started seed=$seed flags=${flags:-none} ${resume[*]:-fresh} attempt=$attempt"
        # shellcheck disable=SC2086
        "$PY" -X faulthandler src/train_graph_reranker.py --run-name "$name" --seed "$seed" $flags "${resume[@]}" \
            >> "$NIGHT/$name.log" 2>&1 < /dev/null
        code=$?
        if [[ $code -eq 0 && -f "$dir/metrics.json" ]]; then
            status "$name done exit=0"
            break
        fi
        status "$name failed exit=$code"
        attempt=$((attempt + 1))
        if [[ $attempt -gt $MAX_RETRIES ]]; then
            status "$name gave up after $MAX_RETRIES resumes"
            break
        fi
    done
done < "$LIST"
status "chain driver finished: $LIST"
