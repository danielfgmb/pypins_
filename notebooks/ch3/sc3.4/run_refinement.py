#!/usr/bin/env python
"""
sc3.4 -- runs the adaptive h-refinement loop (design doc SS4) on common.py's
narrow-Gaussian advection-diffusion benchmark: solve -> indicate -> mark ->
bisect+transfer -> resolve, for N_ROUNDS rounds starting from a uniform
N_CELLS_INITIAL mesh.

Writes results/round_{i}_summary.json (n_cells, rel_L2, Linf, marked cells,
per-cell etas) and results/summary_all.csv, plus results/final_mesh_edges.csv
for the mesh-density figure.
"""
import json

import pandas as pd
import torch

from common import (
    DOMAIN, A_ADV, NU, N, N_CELLS_INITIAL, N_ROUNDS, THETA, ENRICH,
    G_LEFT, G_RIGHT, RESULTS_DIR, exact_solution, rhs,
)
from pinns_framework.pgdr import (
    Mesh1D, assemble_advection_diffusion, compute_indicators, dorfler_mark, refine_mesh, legendre_basis,
)


def eval_error(mesh, c, n_eval=3000):
    lo, hi = DOMAIN
    x_eval = torch.linspace(lo, hi, n_eval, dtype=torch.float64)
    u_pred = torch.empty_like(x_eval)
    for e in range(mesh.n_cells):
        lo_e, hi_e = mesh.cell(e)
        mask = (x_eval >= lo_e) & (x_eval <= hi_e)
        xi_e = 2.0 * (x_eval[mask] - lo_e) / (hi_e - lo_e) - 1.0
        u_pred[mask] = (legendre_basis(xi_e, N) * c[e]).sum(dim=-1)
    u_exact = exact_solution(x_eval)
    diff = u_pred - u_exact
    l2 = float(torch.sqrt((diff ** 2).mean()))
    denom = float(torch.sqrt((u_exact ** 2).mean())) + 1e-12
    return l2 / denom, float(diff.abs().max())


def main():
    mesh = Mesh1D.uniform(*DOMAIN, N_CELLS_INITIAL)
    rows = []

    for round_i in range(N_ROUNDS):
        A, b = assemble_advection_diffusion(mesh, N, A_ADV, NU, rhs, G_LEFT, G_RIGHT)
        c = torch.linalg.solve(A, b).view(mesh.n_cells, N)

        rel_l2, linf = eval_error(mesh, c)
        etas = compute_indicators(mesh, c, N, A_ADV, NU, rhs, G_LEFT, G_RIGHT, enrich=ENRICH)
        marked = dorfler_mark(etas, theta=THETA)

        print(f'round {round_i}: n_cells={mesh.n_cells:3d}  total_dof={mesh.n_cells*N:4d}  '
              f'rel_L2={rel_l2:.3e}  Linf={linf:.3e}  marked={marked}')

        rows.append({
            'round': round_i, 'n_cells': mesh.n_cells, 'total_dof': mesh.n_cells * N,
            'rel_L2': rel_l2, 'Linf': linf, 'n_marked': len(marked),
        })
        with open(f'{RESULTS_DIR}/round_{round_i}_summary.json', 'w') as f:
            json.dump({
                'round': round_i, 'n_cells': mesh.n_cells, 'rel_L2': rel_l2, 'Linf': linf,
                'edges': mesh.edges, 'etas': etas.tolist(), 'marked': marked,
            }, f, indent=2)

        if round_i < N_ROUNDS - 1:
            mesh, c = refine_mesh(mesh, c, N, marked)

    pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}/summary_all.csv', index=False)
    with open(f'{RESULTS_DIR}/final_mesh_edges.csv', 'w') as f:
        f.write('edge\n' + '\n'.join(str(e) for e in mesh.edges))
    print(f'\nfinal mesh: {mesh.n_cells} cells')
    print(f'wrote {RESULTS_DIR}/summary_all.csv, {RESULTS_DIR}/final_mesh_edges.csv')


if __name__ == '__main__':
    main()
