#!/usr/bin/env python
"""
sc2.0 -- epoch vs. training-loss and epoch vs. real-error curves, from each
method's results/<name>_loss_history.csv. MLP and KAN use very different
epoch budgets (KAN is much slower per epoch) so they get separate x-axes --
this is two independent 1x2 panels (MLP row, KAN row), not one shared plot.
Writes figs/sc20_loss_curves.png (+ .pdf). Run manually: python plot_loss_curves.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (
    MLP_IC_EPOCHS, MLP_TD_EPOCHS, KAN_IC_EPOCHS, KAN_TD_EPOCHS, RESULTS_DIR, FIGS_DIR,
)

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size'       : 12,
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 12,
    'legend.fontsize' : 9,
})

TD_STAGE_ORDER = ['step 0 / IC', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5']

def _td_stage_offsets(ic_epochs, td_epochs):
    budget = {'step 0 / IC': ic_epochs, **{f'step {k}': td_epochs for k in range(1, 6)}}
    offsets, cum = {}, 0
    for lab in TD_STAGE_ORDER:
        offsets[lab] = cum
        cum += budget[lab]
    return offsets

def load_td_global(name, ic_epochs, td_epochs):
    df = pd.read_csv(f'{RESULTS_DIR}/{name}_loss_history.csv')
    offsets = _td_stage_offsets(ic_epochs, td_epochs)
    df = df[df['label'].isin(offsets)].copy()
    df['global_epoch'] = df['label'].map(offsets) + df['epoch']
    return df.sort_values('global_epoch')

def load_van(name):
    df = pd.read_csv(f'{RESULTS_DIR}/{name}_loss_history.csv')
    df = df.rename(columns={'epoch': 'global_epoch'})
    return df.sort_values('global_epoch')


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ── MLP (top row) ────────────────────────────────────────────────────────
ax_loss, ax_err = axes[0]
td_mlp = load_td_global('td_mlp', MLP_IC_EPOCHS, MLP_TD_EPOCHS)
van_mlp = load_van('vanilla_mlp')
for lab, df, color in [('TD-PINNs', td_mlp, 'tab:blue'), ('Vanilla', van_mlp, 'tab:orange')]:
    ax_loss.semilogy(df['global_epoch'], df['loss'].abs().clip(lower=1e-12), '-', lw=1.3, color=color, label=lab)
    if 'error' in df:
        ax_err.semilogy(df['global_epoch'], df['error'], '-', lw=1.3, color=color, label=lab)
for lab in TD_STAGE_ORDER[1:]:
    off = _td_stage_offsets(MLP_IC_EPOCHS, MLP_TD_EPOCHS)[lab]
    for ax in (ax_loss, ax_err):
        ax.axvline(off, color='gray', ls=':', lw=0.8, alpha=0.6)
ax_loss.set_title('MLP -- training loss'); ax_loss.set_ylabel('loss')
ax_err.set_title('MLP -- real error vs analytical'); ax_err.set_ylabel('relative L2 error')
for ax in (ax_loss, ax_err):
    ax.set_xlabel('global epoch (dotted = TD step boundaries)')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)

# ── KAN (bottom row) ─────────────────────────────────────────────────────
ax_loss, ax_err = axes[1]
td_kan = load_td_global('td_kan', KAN_IC_EPOCHS, KAN_TD_EPOCHS)
van_kan = load_van('vanilla_kan')
for lab, df, color in [('TD-PINNs', td_kan, 'tab:cyan'), ('Vanilla', van_kan, 'tab:red')]:
    ax_loss.semilogy(df['global_epoch'], df['loss'].abs().clip(lower=1e-12), '-', lw=1.3, color=color, label=lab)
    if 'error' in df:
        ax_err.semilogy(df['global_epoch'], df['error'], '-', lw=1.3, color=color, label=lab)
for lab in TD_STAGE_ORDER[1:]:
    off = _td_stage_offsets(KAN_IC_EPOCHS, KAN_TD_EPOCHS)[lab]
    for ax in (ax_loss, ax_err):
        ax.axvline(off, color='gray', ls=':', lw=0.8, alpha=0.6)
ax_loss.set_title('KAN -- training loss'); ax_loss.set_ylabel('loss')
ax_err.set_title('KAN -- real error vs analytical'); ax_err.set_ylabel('relative L2 error')
for ax in (ax_loss, ax_err):
    ax.set_xlabel('global epoch (dotted = TD step boundaries)')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)

fig.suptitle('Burgers sawtooth (sc2.0) -- loss and real error vs. epoch, MLP vs KAN', fontsize=15)
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/sc20_loss_curves.png', dpi=140, bbox_inches='tight')
fig.savefig(f'{FIGS_DIR}/sc20_loss_curves.pdf', bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc20_loss_curves.png')
print('saved ->', f'{FIGS_DIR}/sc20_loss_curves.pdf')
plt.close(fig)
