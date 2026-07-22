#!/usr/bin/env python
"""
sc2.3 -- final training-loss magnitude vs the real relative-L2 error against
the exact solution, side by side. Motivation: agnostic's raw PDE loss values
look huge (e.g. ~6.5e4 at k=16) next to strong's (e.g. ~2.2e6 at k=16, also
huge) or weak_projection's (~6.3 at k=16) -- these are NOT comparable across
forms. strong's loss is a plain per-point MSE of (-u''-f); weak_projection's
is a mean of K squared MC-projected residuals; agnostic's is a mean of K
squared quadrature-projected residuals with a different implicit
normalisation (no domain_volume correction, unlike vanilla_agnostic_loss in
sc2.1/sc2.2). A big number in the training log does not by itself mean the
fit is bad -- what matters is the real error against the known exact
solution (rel_L2 in results/*_summary.json), which is what compare_results.py
already plots. This script just makes the disconnect explicit.
"""
import re, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import K_LIST, FORMS, name_for, RESULTS_DIR, LOGS_DIR, FIGS_DIR

STYLE = {'strong': ('tab:red', 'o'), 'weak_projection': ('tab:blue', 's'), 'agnostic': ('tab:green', '^')}

_EPOCH_RE = re.compile(r'epoch\s+(\d+)\s*\|\s*pde\s+([\d.eE+-]+)\s*\|\s*bc\s+([\d.eE+-]+)')


def final_pde_loss(form, k):
    """Grab the last 'epoch ... | pde ... | bc ...' line from whichever log
    exists for this (form, k) -- prefers the most recent log suffix.
    NOTE: must not glob '{form}_k{k}*' directly -- that also matches k=16's
    log when k=1 (prefix collision), so require an exact boundary after k{k}."""
    candidates = sorted(
        glob.glob(f'{LOGS_DIR}/{form}_k{k}.log') + glob.glob(f'{LOGS_DIR}/{form}_k{k}_*.log'),
        key=os.path.getmtime,
    )
    for path in reversed(candidates):
        with open(path) as f:
            lines = [l for l in f if _EPOCH_RE.search(l)]
        if lines:
            m = _EPOCH_RE.search(lines[-1])
            return float(m.group(2))
    return np.nan


def main():
    rows = []
    for form in FORMS:
        for k in K_LIST:
            s = pd.read_json(f'{RESULTS_DIR}/{name_for(form, k)}_summary.json', typ='series')
            rows.append({'form': form, 'k': k, 'rel_L2': s['rel_L2'],
                         'final_pde_loss': final_pde_loss(form, k)})
    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/loss_vs_error.csv', index=False)
    pd.set_option('display.float_format', '{:.4e}'.format)
    print(df.pivot(index='k', columns='form', values='final_pde_loss').to_string())
    print()
    print(df.pivot(index='k', columns='form', values='rel_L2').to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for form in FORMS:
        color, marker = STYLE[form]
        sub = df[df.form == form].sort_values('k')
        ax.semilogy(sub['k'], sub['final_pde_loss'], '-', marker=marker, ms=7, color=color, label=form)
    ax.set_xlabel('k'); ax.set_ylabel('final training PDE loss (raw, log scale)')
    ax.set_title('Raw training loss -- NOT comparable across forms')
    ax.grid(True, which='both', alpha=0.3); ax.legend()

    ax = axes[1]
    for form in FORMS:
        color, marker = STYLE[form]
        sub = df[df.form == form].sort_values('k')
        ax.semilogy(sub['k'], sub['rel_L2'], '-', marker=marker, ms=7, color=color, label=form)
    ax.set_xlabel('k'); ax.set_ylabel('relative L2 error vs exact solution')
    ax.set_title('Real error -- what actually matters')
    ax.grid(True, which='both', alpha=0.3); ax.legend()

    fig.suptitle('sc2.3 -- training loss magnitude is not the real error (different scales per form)', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc23_loss_vs_error.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc23_loss_vs_error.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc23_loss_vs_error.png')


if __name__ == '__main__':
    main()
