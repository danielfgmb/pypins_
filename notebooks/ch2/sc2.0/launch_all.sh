#!/bin/sh
# sc2.0 -- launch the 4 training scripts + the auto-compare waiter, each in
# its own tmux window inside a detached session named "sc20". Runs alongside
# (not instead of) the sc2.1 session -- different tmux session, different
# GPU-sharing but same GPU. Survives terminal/SSH/session disconnects;
# re-attach any time with:
#   tmux attach -t sc20
# List windows without attaching:
#   tmux list-windows -t sc20
# Kill everything:
#   tmux kill-session -t sc20
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
PY=/home/df.gomezb/.env/bin/python
SESSION=sc20

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists -- kill it first if you want a clean restart:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

rm -f results/*.DONE

tmux new-session  -d -s "$SESSION" -n td_mlp   "cd '$DIR' && $PY -u train_td_mlp.py 2>&1 | tee logs/td_mlp.log; exec sh"
tmux new-window   -t "$SESSION"    -n td_kan   "cd '$DIR' && $PY -u train_td_kan.py 2>&1 | tee logs/td_kan.log; exec sh"
tmux new-window   -t "$SESSION"    -n van_mlp  "cd '$DIR' && $PY -u train_vanilla_mlp.py 2>&1 | tee logs/vanilla_mlp.log; exec sh"
tmux new-window   -t "$SESSION"    -n van_kan  "cd '$DIR' && $PY -u train_vanilla_kan.py 2>&1 | tee logs/vanilla_kan.log; exec sh"
tmux new-window   -t "$SESSION"    -n compare  "cd '$DIR' && sh wait_and_compare.sh; exec sh"

echo "launched tmux session '$SESSION' with 5 windows: td_mlp, td_kan, van_mlp, van_kan, compare"
echo "attach:      tmux attach -t $SESSION"
echo "list windows: tmux list-windows -t $SESSION"
echo "tail a log:  tail -f $DIR/logs/td_mlp.log"
