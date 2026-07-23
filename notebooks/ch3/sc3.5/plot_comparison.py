#!/usr/bin/env python
"""
sc3.5 -- mean_L2 bar comparison, extending sc3.2's chart: ch2/sc2.4's 4
trained-PINN methods, sc3.2's single-cell direct solve, and this work's
graded multi-cell direct solve, all on the identical nu=0.02 benchmark.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import RESULTS_DIR, FIGS_DIR

SC24_RESULTS = '/home/df.gomezb/Tesis/notebooks/ch2/sc2.4/results'
SC32_RESULTS = '/home/df.gomezb/Tesis/notebooks/ch3/sc3.2/results'
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

    with open(f'{SC32_RESULTS}/sc32_nu0p02_summary.json') as f:
        s32 = json.load(f)
    labels.append('sc3.2\nsingle-cell\n(direct solve)')
    values.append(s32['mean_L2'])
    colors.append('tab:gray')

    with open(f'{RESULTS_DIR}/sc35_nu0p02_summary.json') as f:
        s35 = json.load(f)
    labels.append('sc3.5\ngraded multi-cell\n(direct solve)')
    values.append(s35['mean_L2'])
    colors.append('tab:purple')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale('log')
    ax.set_ylabel('mean relative L2 error vs exact solution')
    ax.set_title('Burgers traveling shock, nu=0.02: finishing validation-ladder step 5\nsingle-cell (sc3.2) vs graded multi-cell (sc3.5) vs PINN training (sc2.4)')
    ax.grid(True, which='both', axis='y', alpha=0.3)
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:.2e}', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc35_comparison.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc35_comparison.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc35_comparison.png')


if __name__ == '__main__':
    main()
