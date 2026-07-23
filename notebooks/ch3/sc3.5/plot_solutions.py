#!/usr/bin/env python
"""
sc3.5 -- solution profile at t=1.0 vs exact, the direct counterpart to
sc3.2/figs/sc32_solutions.png's nu=0.02 (left) panel: same benchmark, same
zoom window, but on the graded multi-cell mesh instead of one global cell.
Point of the plot: no Gibbs ringing this time. Also marks the fine-band
cell edges to show where the extra resolution actually went.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from common import X0, XF, T_END, RESULTS_DIR, FIGS_DIR, exact_solution, graded_mesh_edges
from pinns_framework.pgdr import Mesh1D, solve_burgers_step_newton, legendre_basis, project_to_legendre
from common import N, N_STEPS, NU, U_L, U_R, PENALTY_COEFF


def main():
    mesh = Mesh1D(graded_mesh_edges())
    c_list = [project_to_legendre(lambda x: exact_solution(torch.tensor(0.0, dtype=torch.float64), x),
                                   mesh.cell(e), N) for e in range(mesh.n_cells)]
    c = torch.cat(c_list)
    dt = T_END / N_STEPS
    for _ in range(N_STEPS):
        c, _ = solve_burgers_step_newton(c, mesh, N, NU, dt, U_L, U_R, penalty_coeff=PENALTY_COEFF)

    x_eval = torch.linspace(X0, XF, 2000, dtype=torch.float64)
    u_pred = torch.empty_like(x_eval)
    for e in range(mesh.n_cells):
        lo_e, hi_e = mesh.cell(e)
        mask = (x_eval >= lo_e) & (x_eval <= hi_e)
        xi_e = 2.0 * (x_eval[mask] - lo_e) / (hi_e - lo_e) - 1.0
        u_pred[mask] = (legendre_basis(xi_e, N) * c[e * N:(e + 1) * N]).sum(dim=-1)
    u_exact = exact_solution(torch.tensor(T_END, dtype=torch.float64), x_eval)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_eval, u_exact, 'k-', lw=2, label='exact')
    ax.plot(x_eval, u_pred, '--', lw=1.8, color='tab:purple', label=f'PGDr-PIKAN graded mesh ({mesh.n_cells} cells)')
    for e_edge in mesh.edges:
        if -2 <= e_edge <= 3:
            ax.axvline(e_edge, color='tab:purple', lw=0.5, alpha=0.4)
    ax.set_xlim(-2, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t=1.0)')
    ax.set_title('sc3.5 -- graded multi-cell Burgers solve vs exact traveling shock, at t=1.0\n(nu=0.02, same benchmark as sc3.2 -- compare against its ringing)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{FIGS_DIR}/sc35_solutions.png', dpi=140, bbox_inches='tight')
    fig.savefig(f'{FIGS_DIR}/sc35_solutions.pdf', bbox_inches='tight')
    print('saved ->', f'{FIGS_DIR}/sc35_solutions.png')


if __name__ == '__main__':
    main()
