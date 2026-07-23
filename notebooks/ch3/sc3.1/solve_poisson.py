#!/usr/bin/env python
"""
sc3.1 -- validation-ladder step 3 (design doc SS7): "linear fixed-mesh demo
... demonstrates the principal contribution's essential mechanism with no
training loop involved." Single cell spanning the whole domain, single-layer
Legendre-KAN trial space (== Bubnov-Galerkin, see common.py's caveat), one
torch.linalg.solve per k -- no optimiser, no epochs.

Also closes the loop on "one KAN layer": for each k, the solved Legendre
coefficients are copied into a pinns_framework.pgdr.LegendreEdge's theta and
re-evaluated, checked bit-close to the manual polynomial reconstruction --
i.e. this literally IS a (single-layer) KAN's output, not just "KAN-flavored"
linear algebra.

Writes results/sc31_k{k}_summary.json and results/summary_all.csv.
"""
import json
import time

import pandas as pd
import torch

from common import DOMAIN, K_LIST, N_FOR_K, N_QUAD, RESULTS_DIR, exact_solution, rhs, name_for
from pinns_framework.pgdr import solve_poisson_single_cell, LegendreEdge, legendre_basis


def main():
    rows = []
    for k in K_LIST:
        N = N_FOR_K[k]
        NAME = name_for(k)
        t0 = time.perf_counter()
        out = solve_poisson_single_cell(
            f_fn=lambda x, k=k: rhs(k, x),
            exact_fn=lambda x, k=k: exact_solution(k, x),
            domain=DOMAIN, N=N, n_quad=N_QUAD,
        )
        wall = time.perf_counter() - t0

        # "one KAN layer" check: theta = c reproduces the same function.
        edge = LegendreEdge(degree=N, domain=DOMAIN)
        with torch.no_grad():
            edge.theta.copy_(out['c'])
            x_chk = torch.linspace(*DOMAIN, 500, dtype=torch.float64).unsqueeze(-1)
            u_edge = edge(x_chk).squeeze(-1)
            xi_chk = 2.0 * (x_chk.squeeze(-1) - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) - 1.0
            u_manual = (legendre_basis(xi_chk, N) * out['c']).sum(dim=-1)
        kan_edge_vs_manual = float((u_edge - u_manual).abs().max())

        summary = {
            'k': k, 'N': N, 'n_quad': N_QUAD, 'wall_s': wall,
            'rel_L2': out['rel_L2'], 'Linf': out['Linf'],
            'quad_vs_closed_form_rel': out['quad_vs_closed_form'],
            'kan_edge_vs_manual': kan_edge_vs_manual,
        }
        rows.append(summary)
        with open(f'{RESULTS_DIR}/{NAME}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'k={k:2d} N={N:3d} | wall {wall*1000:6.2f}ms | rel_L2 {out["rel_L2"]:.3e} '
              f'| Linf {out["Linf"]:.3e} | quad_vs_closed_form(rel) {out["quad_vs_closed_form"]:.3e} '
              f'| kan_edge_vs_manual {kan_edge_vs_manual:.3e}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/summary_all.csv', index=False)
    print(f'\nwrote {RESULTS_DIR}/summary_all.csv')


if __name__ == '__main__':
    main()
