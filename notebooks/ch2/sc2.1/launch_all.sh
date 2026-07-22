#!/bin/sh
# sc2.1 -- launch the 6 training scripts + the auto-compare waiter, each in
# its own tmux window inside a detached session named "sc21". Survives
# terminal/SSH/session disconnects; re-attach any time with:
#   tmux attach -t sc21
# List windows without attaching:
#   tmux list-windows -t sc21
# Kill everything:
#   tmux kill-session -t sc21
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
PY=/home/df.gomezb/.env/bin/python
SESSION=sc21

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists -- kill it first if you want a clean restart:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

rm -f results/*.DONE

tmux new-session  -d -s "$SESSION" -n td_str   "cd '$DIR' && $PY -u train_td_strong.py 2>&1 | tee logs/td_strong.log; exec sh"
tmux new-window   -t "$SESSION"    -n td_wp    "cd '$DIR' && $PY -u train_td_weak_projection.py 2>&1 | tee logs/td_weak_projection.log; exec sh"
tmux new-window   -t "$SESSION"    -n td_agn   "cd '$DIR' && $PY -u train_td_agnostic.py 2>&1 | tee logs/td_agnostic.log; exec sh"
tmux new-window   -t "$SESSION"    -n van_str  "cd '$DIR' && $PY -u train_vanilla_strong.py 2>&1 | tee logs/vanilla_strong.log; exec sh"
tmux new-window   -t "$SESSION"    -n van_wp   "cd '$DIR' && $PY -u train_vanilla_weak_projection.py 2>&1 | tee logs/vanilla_weak_projection.log; exec sh"
tmux new-window   -t "$SESSION"    -n van_agn  "cd '$DIR' && $PY -u train_vanilla_agnostic.py 2>&1 | tee logs/vanilla_agnostic.log; exec sh"
tmux new-window   -t "$SESSION"    -n compare  "cd '$DIR' && sh wait_and_compare.sh; exec sh"

echo "launched tmux session '$SESSION' with 7 windows: td_str, td_wp, td_agn, van_str, van_wp, van_agn, compare"
echo "attach:      tmux attach -t $SESSION"
echo "list windows: tmux list-windows -t $SESSION"
echo "tail a log:  tail -f $DIR/logs/td_weak_projection.log"
