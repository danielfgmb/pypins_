#!/usr/bin/env python
"""
sc2.1 -- epoch vs. training-loss and epoch vs. real-error curves, from each
method's results/<name>_loss_history.csv (written by PINTrainer's error_fn /
loss_history tracking, sampled every ERROR_EVERY=1000 epochs).
Writes figs/sc21_loss_curves.png (+ .pdf). Run manually: python plot_loss_curves.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import TD_EPOCHS, IC_EPOCHS, VAN_EPOCHS, RESULTS_DIR, FIGS_DIR

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size'       : 12,
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 12,
    'legend.fontsize' : 9,
})

TD_METHODS = {
    'strong':          ('td_strong',          'tab:green'),
    'weak_projection': ('td_weak_projection',  'tab:blue'),
    'agnostic':        ('td_agnostic',         'tab:cyan'),
}
VAN_METHODS = {
    'strong':          ('vanilla_strong',          'tab:purple'),
    'weak_projection': ('vanilla_weak_projection',  'tab:orange'),
    'agnostic':        ('vanilla_agnostic',         'tab:red'),
}

# TD-PINNs stages run in this fixed order, tolerance-based early stop is
# disabled so each always uses its full budget: IC_EPOCHS for step 0, then
# TD_EPOCHS for every PDE step (steps 1..5) -- so a global, continuous epoch
# axis is a cumulative sum of those per-stage budgets, not a uniform stride.
TD_STAGE_ORDER = ['step 0 / IC', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5']
TD_STAGE_BUDGET = {'step 0 / IC': IC_EPOCHS,
                    **{f'step {k}': TD_EPOCHS for k in range(1, 6)}}

def _td_stage_offsets():
    offsets, cum = {}, 0
    for lab in TD_STAGE_ORDER:
        offsets[lab] = cum
        cum += TD_STAGE_BUDGET[lab]
    return offsets, cum  # cum ends up == IC_EPOCHS + 5*TD_EPOCHS == VAN_EPOCHS

TD_STAGE_OFFSET, TD_TOTAL_EPOCHS = _td_stage_offsets()

def load_td_global(name):
    df = pd.read_csv(f'{RESULTS_DIR}/{name}_loss_history.csv')
    df = df[df['label'].isin(TD_STAGE_OFFSET)].copy()
    df['global_epoch'] = df['label'].map(TD_STAGE_OFFSET) + df['epoch']
    return df.sort_values('global_epoch')

def load_van(name):
    df = pd.read_csv(f'{RESULTS_DIR}/{name}_loss_history.csv')
    df = df.rename(columns={'epoch': 'global_epoch'})
    return df.sort_values('global_epoch')


fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex='row')

# ── TD-PINNs (top row) ──────────────────────────────────────────────────
ax_loss, ax_err = axes[0]
for label, (name, color) in TD_METHODS.items():
    df = load_td_global(name)
    ax_loss.semilogy(df['global_epoch'], df['loss'].clip(lower=1e-12), '-', lw=1.3, color=color, label=label)
    if 'error' in df:
        ax_err.semilogy(df['global_epoch'], df['error'], '-', lw=1.3, color=color, label=label)
for lab in TD_STAGE_ORDER[1:]:
    for ax in (ax_loss, ax_err):
        ax.axvline(TD_STAGE_OFFSET[lab], color='gray', ls=':', lw=0.8, alpha=0.6)
ax_loss.set_title('TD-PINNs -- training loss'); ax_loss.set_ylabel('loss')
ax_err.set_title('TD-PINNs -- real error vs analytical'); ax_err.set_ylabel('relative L2 error')
for ax in (ax_loss, ax_err):
    ax.set_xlabel('global epoch (dotted lines = step boundaries)')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)

# ── Vanilla (bottom row) ─────────────────────────────────────────────────
ax_loss, ax_err = axes[1]
for label, (name, color) in VAN_METHODS.items():
    df = load_van(name)
    ax_loss.semilogy(df['global_epoch'], df['loss'].abs().clip(lower=1e-12), '-', lw=1.3, color=color, label=label)
    if 'error' in df:
        ax_err.semilogy(df['global_epoch'], df['error'], '-', lw=1.3, color=color, label=label)
ax_loss.set_title('Vanilla -- |training loss| (log scale)'); ax_loss.set_ylabel('|loss|')
ax_err.set_title('Vanilla -- real error vs analytical'); ax_err.set_ylabel('relative L2 error')
for ax in (ax_loss, ax_err):
    ax.set_xlabel('epoch')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)

fig.suptitle('Burgers sawtooth (sc2.1) -- loss and real error vs. epoch, all forms', fontsize=15)
plt.tight_layout()
fig.savefig(f'{FIGS_DIR}/sc21_loss_curves.png', dpi=140, bbox_inches='tight')
fig.savefig(f'{FIGS_DIR}/sc21_loss_curves.pdf', bbox_inches='tight')
print('saved ->', f'{FIGS_DIR}/sc21_loss_curves.png')
print('saved ->', f'{FIGS_DIR}/sc21_loss_curves.pdf')
plt.close(fig)
