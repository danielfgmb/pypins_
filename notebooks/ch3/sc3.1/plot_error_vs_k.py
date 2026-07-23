#!/usr/bin/env python
"""
sc3.1 -- rel_L2 vs k, this work (direct Legendre-Galerkin solve) overlaid
against ch2/sc2.3's PINN-trained strong / weak_projection / agnostic
numbers on the identical benchmark. Same STYLE color convention as
sc2.3/plot_loss_vs_error.py (form -> color) so the mapping is consistent
across chapters; PGDr-PIKAN gets its own distinct color+marker.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common import K_LIST, RESULTS_DIR, FIGS_DIR

SC23_RESULTS = '/home/df.gomezb/Tesis/notebooks/ch2/sc2.3/results'

STYLE = {
    'strong': ('tab:red', 'o'),
    'weak_projection': ('tab:blue', 's'),
    'agnostic': ('tab:green', '^'),
    'pgdr_pikan': ('tab:purple', 'D'),
}
LABELS = {
    'strong': 'sc2.3 strong (PINN, trained)',
    'weak_projection': 'sc2.3 weak_projection (PINN, trained)',
    'agnostic': 'sc2.3 agnostic (PINN, trained)',
    'pgdr_pikan': 'sc3.1 PGDr-PIKAN (direct solve, no training)',
}


def main():
    rows = []
    for form in ('strong', 'weak_projection', 'agnostic'):
        for k in K_LIST:
            with open(f'{SC23_RESULTS}/{form}_k{k}_summary.json') as f:
                s = json.load(f)
            rows.append({'method': form, 'k': k, 'rel_L2': s['rel_L2']})

    df31 = pd.read_csv(f'{RESULTS_DIR}/summary_all.csv')
    for _, r in df31.iterrows():
        rows.append({'method': 'pgdr_pikan', 'k': int(r['k']), 'rel_L2': r['rel_L2']})

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/comparison_vs_sc23.csv', index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ('strong', 'weak_projection', 'agnostic', 'pgdr_pikan'):
        color, marker = STYLE[method]
        sub = df[df.method == method].sort_values('k')
        lw = 2.5 if method == 'pgdr_pikan' else 1.5
        ax.loglog(sub['k'], sub['rel_L2'], '-', marker=marker, ms=8, lw=lw,
                   color=color, label=LABELS[method])

    ax.set_xlabel('k (target frequency)')
    ax.set_ylabel('relative L2 error vs exact solution')
    ax.set_xticks(K_LIST)
    ax.set_xticklabels([str(k) for k in K_LIST])
    ax.set_title('1D Poisson benchmark: direct Legendre-Galerkin solve (sc3.1)\nvs PINN training (sc2.3), same k sweep')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc31_error_vs_k.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc31_error_vs_k.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc31_error_vs_k.png')


if __name__ == '__main__':
    main()
