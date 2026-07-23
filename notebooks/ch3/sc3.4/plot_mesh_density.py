#!/usr/bin/env python
"""
sc3.4 -- the final adaptively-refined mesh's cell edges, overlaid on the
exact solution profile it was built to resolve. Point of the plot: cells
concentrate tightly around the narrow Gaussian bump (x0=0.5) and stay coarse
everywhere else -- exactly the payoff h-refinement is supposed to deliver
(contrast with sc3.2, where a single global cell had no such option).
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from common import DOMAIN, FIGS_DIR, RESULTS_DIR, exact_solution


def main():
    with open(f'{RESULTS_DIR}/final_mesh_edges.csv') as f:
        edges = [float(row['edge']) for row in csv.DictReader(f)]

    x = torch.linspace(*DOMAIN, 2000, dtype=torch.float64)
    u = exact_solution(x)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(x, u, 'k-', lw=2, label='exact solution (narrow Gaussian)')
    for e in edges:
        ax1.axvline(e, color='tab:purple', lw=0.8, alpha=0.7)
    ax1.set_ylabel('u(x)')
    ax1.set_title(f'sc3.4 -- final mesh ({len(edges)-1} cells) vs. the feature it resolves')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.eventplot([edges], colors='tab:purple', linelengths=0.8)
    ax2.set_yticks([])
    ax2.set_xlabel('x')
    ax2.set_ylabel('cell\nedges')
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc34_mesh_density.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc34_mesh_density.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc34_mesh_density.png')


if __name__ == '__main__':
    main()
