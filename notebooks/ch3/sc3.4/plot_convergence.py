#!/usr/bin/env python
"""
sc3.4 -- error vs refinement round (log scale) alongside cell count. Point
of the plot: rel_L2 drops several orders of magnitude over a handful of
rounds while n_cells grows only modestly (4 -> 16) -- the indicator is
doing its job of adding cells only where needed, not uniformly refining
everywhere.
"""
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import RESULTS_DIR, FIGS_DIR


def main():
    df = pd.read_csv(f'{RESULTS_DIR}/summary_all.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.semilogy(df['round'], df['rel_L2'], '-o', color='tab:purple', ms=8)
    ax1.set_xlabel('refinement round')
    ax1.set_ylabel('relative L2 error vs exact solution')
    ax1.set_title('Error drops fast...')
    ax1.grid(True, which='both', alpha=0.3)

    ax2.plot(df['round'], df['n_cells'], '-s', color='tab:gray', ms=7)
    ax2.set_xlabel('refinement round')
    ax2.set_ylabel('n_cells')
    ax2.set_title('...while cell count grows only\nwhere the indicator says to')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('sc3.4 -- adaptive h-refinement', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc34_convergence.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc34_convergence.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc34_convergence.png')


if __name__ == '__main__':
    main()
