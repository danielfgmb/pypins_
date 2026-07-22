#!/bin/sh
# sc2.0 -- polls for all 4 DONE markers, then runs compare_results.py,
# plot_solutions.py, and plot_loss_curves.py.
# Run inside its own tmux window by launch_all.sh.
set -e
cd "$(dirname "$0")"
PY=/home/df.gomezb/.env/bin/python

echo "waiting for all 4 runs to finish..."
while :; do
    if [ -f results/td_mlp.DONE ] && \
       [ -f results/td_kan.DONE ] && \
       [ -f results/vanilla_mlp.DONE ] && \
       [ -f results/vanilla_kan.DONE ]; then
        break
    fi
    sleep 15
done

echo "all 4 runs finished -- running comparison + plotting scripts"
"$PY" compare_results.py  2>&1 | tee logs/compare.log
"$PY" plot_solutions.py   2>&1 | tee logs/plot_solutions.log
"$PY" plot_loss_curves.py 2>&1 | tee logs/plot_loss_curves.log
echo "comparison + plots done -- see results/ and figs/"
