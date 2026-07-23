#!/usr/bin/env python
"""
sc3.3 -- rel_L2 vs n_cells for the manufactured advection-diffusion
problem. All points sit near float64 round-off (in fact rising slightly
with cell/interface count, from accumulated round-off across more
interface terms -- not a real accuracy loss) -- the point of the plot is
that adding interior interfaces doesn't leak accuracy, confirming
SS2.5's flux implementation (upwind + SIPG) is correct.
"""
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import RESULTS_DIR, FIGS_DIR


def main():
    df = pd.read_csv(f'{RESULTS_DIR}/summary_all.csv')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(df['n_cells'], df['rel_L2'], '-o', color='tab:purple', ms=8, label='rel_L2 vs exact solution')
    ax.axhline(2.22e-16, color='k', ls=':', lw=1, alpha=0.6, label='float64 eps (2.22e-16)')
    ax.set_xlabel('n_cells (interior interfaces = n_cells - 1)')
    ax.set_ylabel('relative L2 error vs exact solution')
    ax.set_title('sc3.3 -- multi-cell interface flux (upwind + SIPG) validation\nflat near machine precision -> no accuracy leaking at interfaces')
    ax.set_xticks(df['n_cells'])
    ax.set_xticklabels(df['n_cells'])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc33_convergence.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc33_convergence.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc33_convergence.png')


if __name__ == '__main__':
    main()
