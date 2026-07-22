#!/bin/bash
# sc2.2 -- run all 6 methods x 5 nu (30 jobs total) with a concurrency cap.
# Single uncontended job uses ~500MB GPU memory (measured) out of 24GB, so
# memory is not the bottleneck at this concurrency; CONCURRENCY=15 was picked
# to comfortably beat "at least 10 parallel" while leaving headroom on both
# GPU memory and the 24 CPU cores.
set -u
cd "$(dirname "$0")"
PY=/home/df.gomezb/.env/bin/python
CONCURRENCY=30

METHODS="td_strong td_weak_projection td_agnostic vanilla_strong vanilla_weak_projection vanilla_agnostic"
NUS="0.05 0.02 0.01 0.005 0.001"

mkdir -p logs

jobs_list=()
for nu in $NUS; do
  for m in $METHODS; do
    jobs_list+=("$m $nu")
  done
done

echo "Total jobs: ${#jobs_list[@]}  (concurrency=$CONCURRENCY)"

run_one() {
  local method="$1" nu="$2"
  local tag="${nu/./p}"
  local name="${method}_nu${tag}"
  if [ -f "results/${name}.DONE" ]; then
    echo "[skip] $name already DONE"
    return
  fi
  echo "[start] $name"
  "$PY" -u train_one.py --method "$method" --nu "$nu" > "logs/${name}.log" 2>&1
  echo "[end]   $name (exit $?)"
}

running=0
for job in "${jobs_list[@]}"; do
  set -- $job
  run_one "$1" "$2" &
  running=$((running + 1))
  if [ "$running" -ge "$CONCURRENCY" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "=== all sc2.2 jobs finished ==="
