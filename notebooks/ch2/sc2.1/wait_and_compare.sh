#!/bin/sh
# sc2.1 -- polls for all 6 DONE markers, then runs compare_results.py,
# plot_solutions.py, plot_extra.py, and plot_loss_curves.py.
# Run inside its own tmux window by launch_all.sh.
set -e
cd "$(dirname "$0")"
PY=/home/df.gomezb/.env/bin/python

echo "waiting for all 6 runs to finish..."
while :; do
    if [ -f results/td_strong.DONE ] && \
       [ -f results/td_weak_projection.DONE ] && \
       [ -f results/td_agnostic.DONE ] && \
       [ -f results/vanilla_strong.DONE ] && \
       [ -f results/vanilla_weak_projection.DONE ] && \
       [ -f results/vanilla_agnostic.DONE ]; then
        break
    fi
    sleep 15
done

echo "all 6 runs finished -- running comparison + plotting scripts"
"$PY" compare_results.py    2>&1 | tee logs/compare.log
"$PY" plot_solutions.py     2>&1 | tee logs/plot_solutions.log
"$PY" plot_extra.py         2>&1 | tee logs/plot_extra.log
"$PY" plot_loss_curves.py   2>&1 | tee logs/plot_loss_curves.log
echo "comparison + plots done -- see results/ and figs/"
