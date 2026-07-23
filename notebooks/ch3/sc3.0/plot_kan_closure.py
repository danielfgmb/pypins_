#!/usr/bin/env python
"""
sc3.0 -- plot the KAN-closure sweep from kan_closure_test.py: residual
between a composed Legendre-KAN's raw output and its exact Legendre
re-projection, vs p^L (effective polynomial degree), one line per depth
(1/2/3 layers). The point of the plot is what does NOT happen: if closure
were only approximate, residual would grow with p^L; instead every point
sits within a couple of decades of float64 round-off (~1e-16), flat
regardless of degree or depth.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
FIGS_DIR = os.path.join(HERE, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

STYLE = {
    '1-layer (Bubnov-Galerkin case, phase 1)': ('tab:gray', 'o', '1-layer (p^L = p)'),
    '2-layer': ('tab:blue', 's', '2-layer (p^L = p^2)'),
    '3-layer': ('tab:purple', '^', '3-layer (p^L = p^3)'),
}


def main():
    with open(f'{RESULTS_DIR}/kan_closure_report.json') as f:
        report = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (color, marker, legend_label) in STYLE.items():
        cases = sorted((c for c in report['cases'] if c['label'] == label),
                        key=lambda c: c['p_pow_L'])
        xs = [c['p_pow_L'] for c in cases]
        ys = [c['residual'] for c in cases]
        ax.loglog(xs, ys, '-', marker=marker, ms=8, color=color, label=legend_label)

    ax.axhline(2.22e-16, color='k', ls=':', lw=1, alpha=0.6, label='float64 eps (2.22e-16)')
    ax.set_xlabel('p^L  (effective composed polynomial degree)')
    ax.set_ylabel('max |raw KAN output - Legendre re-projection|')
    ax.set_title('sc3.0 -- KAN-Legendre closure residual vs degree and depth\n(flat -> exact closure, not "small enough to not matter")')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc30_kan_closure.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc30_kan_closure.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc30_kan_closure.png')


if __name__ == '__main__':
    main()
