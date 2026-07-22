#!/usr/bin/env python
"""
sc2.3 -- epoch-vs-training-loss and epoch-vs-real-error curves, all 3 forms
overlaid, one column per k. Uses results/{form}_k{k}_loss_history.csv
(written by train_one.py every ERROR_EVERY epochs). This is the actual
requested comparison: does a form's raw training loss going down track its
real error (vs the exact solution) going down, and how do the three forms'
convergence trajectories compare epoch-for-epoch (not just final value)?
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import K_LIST, FORMS, name_for, RESULTS_DIR, FIGS_DIR

STYLE = {'strong': ('tab:red', '-'), 'weak_projection': ('tab:blue', '-'), 'agnostic': ('tab:green', '-')}

fig, axes = plt.subplots(2, len(K_LIST), figsize=(4.0 * len(K_LIST), 7.5), sharex='col')

for col, k in enumerate(K_LIST):
    ax_loss = axes[0, col]
    ax_err  = axes[1, col]
    for form in FORMS:
        color, ls = STYLE[form]
        df = pd.read_csv(f'{RESULTS_DIR}/{name_for(form, k)}_loss_history.csv')
        ax_loss.semilogy(df['epoch'], df['pde_loss'], ls, color=color, lw=1.6, label=form)
        ax_err.semilogy(df['epoch'], df['rel_L2'], ls, color=color, lw=1.6, label=form)
    ax_loss.set_title(f'k = {k}')
    ax_loss.grid(True, which='both', alpha=0.3)
    ax_err.grid(True, which='both', alpha=0.3)
    ax_err.set_xlabel('epoch')
    if col == 0:
        ax_loss.set_ylabel('training PDE loss (raw)')
        ax_err.set_ylabel('real relative L2 error')
        ax_loss.legend(fontsize=8)

fig.suptitle('sc2.3 -- training loss vs real error, by epoch (top: loss, bottom: real error)', fontsize=14)
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/sc23_loss_curves.png', dpi=140, bbox_inches='tight')
fig.savefig(f'{FIGS_DIR}/sc23_loss_curves.pdf', bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc23_loss_curves.png')
