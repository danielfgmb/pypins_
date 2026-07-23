"""
Shared setup for sc3.5: finishing validation-ladder step 5 (design doc SS7)
-- the same viscous-Burgers traveling-shock benchmark as sc3.2/ch2/sc2.4
(u_L=2, u_R=0, nu=0.02, domain=[-5,5], T_END=1), now solved on a genuine
multi-cell mesh with Crank-Nicolson + Newton (pgdr.mesh.solve_burgers_step_newton),
instead of sc3.2's single global cell.

sc3.2 showed a single cell cannot resolve this benchmark's actual shock
width (0.4% of the domain) -- Gibbs ringing, mean_L2~5.6e-2. This
experiment asks: does concentrating resolution where the shock actually
travels fix that?

Mesh: STATIC and graded, built from what's already known about the exact
solution's trajectory (shock travels rigidly from x=0 at t=0 to x=1 at
t=1) -- NOT full dynamic AMR (that would need refine.py's indicator
generalized to a moving nonlinear target and the mesh re-adapted mid-march,
flagged as future work). Fine, uniform-width cells (h=0.2) tile a band
[-0.5, 1.5] comfortably covering the shock's whole path with margin;
coarse cells cover the rest of [-5,5], where the solution is essentially
constant. N=15 modes per cell (calibrated: a single 0.2-wide cell at N=15
achieves ~1.1e-4 pure-projection error against the exact profile -- see
solve_burgers_mesh.py's printed calibration check), used uniformly across
every cell (this phase doesn't attempt per-cell/p-adaptive degree).
"""
import os
import sys

ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
FIGS_DIR = os.path.join(HERE, 'figs')
for _d in (RESULTS_DIR, FIGS_DIR):
    os.makedirs(_d, exist_ok=True)

X0, XF = -5.0, 5.0
T_END = 1.0
U_L, U_R = 2.0, 0.0
S_SHOCK = (U_L + U_R) / 2.0
NU = 0.02

N = 15
N_STEPS = 40
PENALTY_COEFF = 10.0
FINE_BAND = (-0.5, 1.5)
FINE_H = 0.2
SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def exact_solution(t, x):
    xi = x - S_SHOCK * t
    return (U_L + U_R) / 2.0 - (U_L - U_R) / 2.0 * torch.tanh((U_L - U_R) * xi / (4.0 * NU))


def graded_mesh_edges():
    """coarse | fine (h=0.2, tiling FINE_BAND) | coarse."""
    lo, hi = FINE_BAND
    n_fine = round((hi - lo) / FINE_H)
    left_coarse = np.linspace(X0, lo, 4).tolist()   # 3 coarse cells
    fine = np.linspace(lo, hi, n_fine + 1).tolist()
    right_coarse = np.linspace(hi, XF, 4).tolist()  # 3 coarse cells
    return left_coarse[:-1] + fine[:-1] + right_coarse
