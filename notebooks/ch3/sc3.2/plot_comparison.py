#!/usr/bin/env python
"""
sc3.2 -- mean_L2 bar comparison, this work's single-cell nu=0.02 solve
(identical benchmark to ch2/sc2.4) against sc2.4's 4 trained-PINN methods.
Unlike sc3.1 (where the direct solve won by many orders of magnitude), this
one is expected -- and shown -- to LOSE to every trained method at this
nu: a single global polynomial cell cannot represent a shock this sharp
(see plot_solutions.py), which is exactly the case for chapter 3 phase 2's
adaptive h-refinement, not a flaw in the linear-algebra machinery itself
(sc3.2's own nu=0.2 case, and sc3.1's Poisson case, both already show that
machinery reaching spectral accuracy when the target is representable).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import RESULTS_DIR, FIGS_DIR

SC24_RESULTS = '/home/df.gomezb/Tesis/notebooks/ch2/sc2.4/results'
SC24_METHODS = ['strong', 'weak_projection', 'agnostic', 'agnostic_weak_time']
SC24_COLORS = {'strong': 'tab:red', 'weak_projection': 'tab:blue',
               'agnostic': 'tab:green', 'agnostic_weak_time': 'tab:olive'}


def main():
    labels, values, colors = [], [], []
    for method in SC24_METHODS:
        with open(f'{SC24_RESULTS}/{method}_summary.json') as f:
            s = json.load(f)
        labels.append(f'sc2.4\n{method}\n(trained)')
        values.append(s['mean_L2'])
        colors.append(SC24_COLORS[method])

    with open(f'{RESULTS_DIR}/sc32_nu0p02_summary.json') as f:
        s32 = json.load(f)
    labels.append('sc3.2\nsingle-cell\n(direct solve)')
    values.append(s32['mean_L2'])
    colors.append('tab:purple')

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale('log')
    ax.set_ylabel('mean relative L2 error vs exact solution')
    ax.set_title('Burgers traveling shock, nu=0.02: single-cell direct solve\nvs PINN training (identical benchmark, ch2/sc2.4)')
    ax.grid(True, which='both', axis='y', alpha=0.3)
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:.2e}', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc32_comparison_vs_sc24.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc32_comparison_vs_sc24.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc32_comparison_vs_sc24.png')


if __name__ == '__main__':
    main()
