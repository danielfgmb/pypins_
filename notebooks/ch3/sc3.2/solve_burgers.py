#!/usr/bin/env python
"""
sc3.2 -- single-cell viscous Burgers (traveling shock, ch2/sc2.4's
benchmark), solved via pinns_framework.pgdr's Crank-Nicolson + Newton
time-marching instead of PINN training. Still one cell spanning the whole
domain, still one KAN layer (Bubnov-Galerkin) -- the point of this script is
to exercise the nonlinear flux tensor (ref_tensors.deriv_triple_matrix) and
the CN+Newton machinery for the first time in a real solve, and to be
honest about where a single global cell succeeds (mild nu) vs where it
doesn't (sc2.4's actual nu=0.02 -- motivates chapter 3 phase 2's adaptive
h-refinement).

Writes results/sc32_nu{nu}_summary.json (per-t breakdown + mean/final_L2)
and results/summary_all.csv.
"""
import json
import time

import pandas as pd
import torch

from common import (
    DOMAIN, T_END, U_L, U_R, N_STEPS, SNAPSHOT_TIMES, NU_CASES,
    RESULTS_DIR, exact_solution, check_boundary_is_constant, name_for,
)
from pinns_framework.pgdr import solve_burgers_single_cell


def main():
    rows = []
    for nu, cfg in NU_CASES.items():
        N = cfg['N']
        NAME = name_for(nu)
        bc_dev = check_boundary_is_constant(nu)
        print(f'[{NAME}] {cfg["label"]}  N={N}  '
              f'(boundary-constancy check: max|deviation|={bc_dev:.3e})')

        t0 = time.perf_counter()
        out = solve_burgers_single_cell(
            exact_fn=lambda t, x, nu=nu: exact_solution(t, x, nu),
            domain=DOMAIN, nu=nu, u_left=U_L, u_right=U_R,
            N=N, n_steps=N_STEPS, T_end=T_END, snapshot_times=SNAPSHOT_TIMES,
        )
        wall = time.perf_counter() - t0

        per_t = {t: {'rel_L2': s['rel_L2'], 'Linf': s['Linf']}
                 for t, s in sorted(out['snapshots'].items())}
        rel_l2s = [s['rel_L2'] for _, s in sorted(out['snapshots'].items())]
        final_l2 = rel_l2s[-1]
        mean_l2 = sum(rel_l2s) / len(rel_l2s)

        summary = {
            'nu': nu, 'N': N, 'n_steps': N_STEPS, 'wall_s': wall,
            'boundary_constancy_max_dev': bc_dev,
            'mean_L2': mean_l2, 'final_L2': final_l2,
            'per_t': per_t,
            'newton_iters_min': min(out['newton_iters']),
            'newton_iters_max': max(out['newton_iters']),
        }
        with open(f'{RESULTS_DIR}/{NAME}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'  wall {wall:6.1f}s | mean_L2 {mean_l2:.3e} | final_L2 {final_l2:.3e} '
              f'| newton iters {min(out["newton_iters"])}-{max(out["newton_iters"])}')
        for t, s in per_t.items():
            print(f'    t={t:.2f}  rel_L2={s["rel_L2"]:.3e}  Linf={s["Linf"]:.3e}')

        rows.append({'nu': nu, 'N': N, 'wall_s': wall, 'mean_L2': mean_l2, 'final_L2': final_l2})

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/summary_all.csv', index=False)
    print(f'\nwrote {RESULTS_DIR}/summary_all.csv')


if __name__ == '__main__':
    main()
