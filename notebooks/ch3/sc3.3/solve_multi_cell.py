#!/usr/bin/env python
"""
sc3.3 -- sweep n_cells over the manufactured advection-diffusion problem
(common.py) at fixed per-cell degree N, via
pinns_framework.pgdr.solve_advection_diffusion_multi_cell. n_cells=1 must
reproduce sc3.1-style spectral accuracy (sanity check the interface
machinery is a strict generalization); n_cells>1 exercises SS2.5's
interface flux for the first time in this chapter.

Writes results/sc33_ncells{n}_summary.json and results/summary_all.csv.
"""
import json
import time

import pandas as pd

from common import DOMAIN, A_ADV, NU, N, N_CELLS_LIST, G_LEFT, G_RIGHT, RESULTS_DIR, exact_solution, rhs, name_for
from pinns_framework.pgdr import Mesh1D, solve_advection_diffusion_multi_cell


def main():
    rows = []
    for n_cells in N_CELLS_LIST:
        NAME = name_for(n_cells)
        mesh = Mesh1D.uniform(*DOMAIN, n_cells)

        t0 = time.perf_counter()
        out = solve_advection_diffusion_multi_cell(
            mesh, N=N, a=A_ADV, nu=NU, f_fn=rhs, exact_fn=exact_solution,
            g_left=G_LEFT, g_right=G_RIGHT,
        )
        wall = time.perf_counter() - t0

        total_dof = n_cells * N
        summary = {
            'n_cells': n_cells, 'N': N, 'total_dof': total_dof, 'wall_s': wall,
            'rel_L2': out['rel_L2'], 'Linf': out['Linf'],
        }
        rows.append(summary)
        with open(f'{RESULTS_DIR}/{NAME}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'n_cells={n_cells:3d}  N={N}  total_dof={total_dof:4d}  '
              f'wall {wall*1000:6.1f}ms  rel_L2={out["rel_L2"]:.3e}  Linf={out["Linf"]:.3e}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/summary_all.csv', index=False)
    print(f'\nwrote {RESULTS_DIR}/summary_all.csv')


if __name__ == '__main__':
    main()
