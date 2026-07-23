#!/usr/bin/env python
"""
sc3.2 -- solution profiles at t=1.0 (final time, shock furthest from its
t=0 position): single-cell PGDr-PIKAN solution vs the exact traveling
shock, one panel per nu. Point of the plot: nu=0.2 tracks the exact profile
tightly; nu=0.02 visibly cannot -- a single global polynomial cell shows
Gibbs-type ringing/smearing around a front this sharp, which is exactly the
representability limit discussed in notebooks/ch3/README.md (motivating
adaptive h-refinement, chapter 3 phase 2).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from common import DOMAIN, T_END, NU_CASES, FIGS_DIR, exact_solution
from pinns_framework.pgdr import solve_burgers_single_cell


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, (nu, cfg) in zip(axes, sorted(NU_CASES.items())):
        N = cfg['N']
        out = solve_burgers_single_cell(
            exact_fn=lambda t, x, nu=nu: exact_solution(t, x, nu),
            domain=DOMAIN, nu=nu, u_left=2.0, u_right=0.0,
            N=N, n_steps=100, T_end=T_END, snapshot_times=[T_END],
        )
        c = out['snapshots'][T_END]['c']

        from pinns_framework.pgdr import legendre_basis
        lo, hi = DOMAIN
        x_eval = torch.linspace(lo, hi, 2000, dtype=torch.float64)
        xi_eval = 2.0 * (x_eval - lo) / (hi - lo) - 1.0
        u_pred = (legendre_basis(xi_eval, N) * c).sum(dim=-1)
        u_exact = exact_solution(torch.tensor(T_END, dtype=torch.float64), x_eval, nu)

        ax.plot(x_eval, u_exact, 'k-', lw=2, label='exact')
        ax.plot(x_eval, u_pred, '--', lw=1.8, color='tab:purple',
                label=f'PGDr-PIKAN single cell (N={N})')
        ax.set_xlim(-2, 3)  # zoom near the shock (at x=s*T_END=1 here)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t=1.0)')
        ax.set_title(cfg['label'])
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle('sc3.2 -- single-cell Burgers solve vs exact traveling shock, at t=1.0', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc32_solutions.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc32_solutions.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc32_solutions.png')


if __name__ == '__main__':
    main()
