#!/usr/bin/env python
"""
sc3.5 -- finishing validation-ladder step 5: retries sc3.2's failing
nu=0.02 Burgers benchmark on a genuine multi-cell, graded mesh
(common.py's graded_mesh_edges), Crank-Nicolson + Newton time-marching
(pinns_framework.pgdr.mesh.solve_burgers_step_newton).

First reports the calibration check (pure-projection error of a single
fine-band cell against the exact profile, no dynamics) that justified
common.py's FINE_H/N choice, then runs the actual time march and reports
rel_L2/Linf at the same 5 snapshot times as sc3.2, for direct comparison.
"""
import json
import time

import pandas as pd
import torch

from common import (
    X0, XF, T_END, U_L, U_R, NU, N, N_STEPS, PENALTY_COEFF, FINE_H, SNAPSHOT_TIMES,
    RESULTS_DIR, exact_solution, graded_mesh_edges,
)
from pinns_framework.pgdr import Mesh1D, solve_burgers_step_newton, legendre_basis, project_to_legendre


def calibration_check():
    """Pure-projection error of ONE fine-band-width cell centered on the
    shock, against the exact t=0 profile -- no dynamics, just: is FINE_H/N
    even capable of representing this feature well?"""
    lo, hi = -FINE_H / 2, FINE_H / 2
    c = project_to_legendre(lambda x: exact_solution(torch.tensor(0.0, dtype=torch.float64), x), (lo, hi), N)
    x_eval = torch.linspace(lo, hi, 1000, dtype=torch.float64)
    xi = 2.0 * (x_eval - lo) / (hi - lo) - 1.0
    u_pred = (legendre_basis(xi, N) * c).sum(dim=-1)
    u_exact = exact_solution(torch.tensor(0.0, dtype=torch.float64), x_eval)
    diff = u_pred - u_exact
    rel = float(torch.sqrt((diff ** 2).mean()) / torch.sqrt((u_exact ** 2).mean()))
    return rel


def eval_error(mesh, c, t, n_eval=4000):
    x_eval = torch.linspace(X0, XF, n_eval, dtype=torch.float64)
    u_pred = torch.empty_like(x_eval)
    for e in range(mesh.n_cells):
        lo_e, hi_e = mesh.cell(e)
        mask = (x_eval >= lo_e) & (x_eval <= hi_e)
        xi_e = 2.0 * (x_eval[mask] - lo_e) / (hi_e - lo_e) - 1.0
        u_pred[mask] = (legendre_basis(xi_e, N) * c[e * N:(e + 1) * N]).sum(dim=-1)
    u_exact = exact_solution(torch.tensor(t, dtype=torch.float64), x_eval)
    diff = u_pred - u_exact
    l2 = float(torch.sqrt((diff ** 2).mean()))
    denom = float(torch.sqrt((u_exact ** 2).mean())) + 1e-12
    return l2 / denom, float(diff.abs().max())


def main():
    cal = calibration_check()
    print(f'calibration: single {FINE_H}-wide cell at N={N}, pure-projection '
          f'rel_L2 vs exact profile = {cal:.3e} (no dynamics)')

    mesh = Mesh1D(graded_mesh_edges())
    print(f'graded mesh: {mesh.n_cells} cells, total_dof={mesh.n_cells * N}')

    c_list = [project_to_legendre(lambda x: exact_solution(torch.tensor(0.0, dtype=torch.float64), x),
                                   mesh.cell(e), N) for e in range(mesh.n_cells)]
    c = torch.cat(c_list)

    dt = T_END / N_STEPS
    per_t = {}
    rel0, linf0 = eval_error(mesh, c, 0.0)
    per_t[0.0] = {'rel_L2': rel0, 'Linf': linf0}

    t0 = time.perf_counter()
    newton_iters = []
    for step in range(N_STEPS):
        c, it = solve_burgers_step_newton(c, mesh, N, NU, dt, U_L, U_R, penalty_coeff=PENALTY_COEFF)
        newton_iters.append(it)
        t_now = (step + 1) * dt
        for st in SNAPSHOT_TIMES:
            if abs(t_now - st) < 1e-9:
                rel, linf = eval_error(mesh, c, st)
                per_t[st] = {'rel_L2': rel, 'Linf': linf}
    wall = time.perf_counter() - t0

    rel_l2s = [per_t[t]['rel_L2'] for t in SNAPSHOT_TIMES]
    mean_l2 = sum(rel_l2s) / len(rel_l2s)
    final_l2 = per_t[SNAPSHOT_TIMES[-1]]['rel_L2']

    print(f'wall {wall:.1f}s | newton iters {min(newton_iters)}-{max(newton_iters)} | '
          f'mean_L2 {mean_l2:.3e} | final_L2 {final_l2:.3e}')
    for t in SNAPSHOT_TIMES:
        print(f'  t={t:.2f}  rel_L2={per_t[t]["rel_L2"]:.3e}  Linf={per_t[t]["Linf"]:.3e}')

    summary = {
        'nu': NU, 'n_cells': mesh.n_cells, 'N': N, 'total_dof': mesh.n_cells * N,
        'n_steps': N_STEPS, 'wall_s': wall, 'calibration_rel_L2': cal,
        'mean_L2': mean_l2, 'final_L2': final_l2, 'per_t': per_t,
        'newton_iters_min': min(newton_iters), 'newton_iters_max': max(newton_iters),
        'mesh_edges': mesh.edges,
    }
    with open(f'{RESULTS_DIR}/sc35_nu0p02_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([{'nu': NU, 'n_cells': mesh.n_cells, 'total_dof': mesh.n_cells * N,
                    'wall_s': wall, 'mean_L2': mean_l2, 'final_L2': final_l2}]).to_csv(
        f'{RESULTS_DIR}/summary_all.csv', index=False)
    print(f'\nwrote {RESULTS_DIR}/sc35_nu0p02_summary.json, {RESULTS_DIR}/summary_all.csv')


if __name__ == '__main__':
    main()
