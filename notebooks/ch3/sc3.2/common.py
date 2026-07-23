"""
Shared setup for sc3.2: the same viscous-Burgers traveling-shock benchmark
as ch2/sc2.4 --

    u(x,t) = (u_L+u_R)/2 - (u_L-u_R)/2 * tanh[(u_L-u_R)(x - s*t)/(4*nu)]
    s = (u_L+u_R)/2   (Rankine-Hugoniot shock speed)

u_L=2, u_R=0 => s=1, domain x in [-5,5], T_END=1 -- but solved here via
PGDr-PIKAN's single-cell machinery (pinns_framework.pgdr) instead of PINN
training, exactly as sc3.1 did for sc2.3's Poisson benchmark: Crank-Nicolson
+ Newton time-marching (pgdr.solve_burgers_single_cell), one KAN layer
(Bubnov-Galerkin), single cell spanning the whole domain -- no interior
interfaces, so design doc SS2.5's flagged open risk never comes up.

The domain edges x=-5, x=5 are far enough from the shock's path (x=s*t for
t in [0,1], i.e. x=0..1) that the true boundary values are u_L, u_R to
float64 round-off for the ENTIRE simulated time window (checked below) --
that is what licenses treating the BC as a time-CONSTANT Dirichlet
condition (pgdr.assembly.constant_dirichlet_lift), rather than needing a
time-varying lift.

Two viscosities, deliberately contrasting (see notebooks/ch3/README.md for
why this matters for a SINGLE global cell with no h-refinement yet):
  - NU_HARD = 0.02  -- identical to sc2.4, shock width ~4*nu/(u_L-u_R)=0.04,
    i.e. 0.4% of the domain width. A single global polynomial cell is
    expected to struggle here (Gibbs-type ringing near a genuinely sharp,
    if smooth, front) -- this is precisely the motivation for adaptive
    h-refinement (chapter 3 phase 2).
  - NU_MILD = 0.2   -- 10x milder, shock width ~0.4 (4% of the domain), well
    within a single cell's reach at a moderate mode count.
"""
import os
import sys

ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
FIGS_DIR = os.path.join(HERE, 'figs')
for _d in (RESULTS_DIR, FIGS_DIR):
    os.makedirs(_d, exist_ok=True)

DOMAIN = (-5.0, 5.0)
T_END = 1.0
U_L, U_R = 2.0, 0.0
S_SHOCK = (U_L + U_R) / 2.0
N_STEPS = 100
SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]

# nu -> (N legendre modes, human label)
NU_CASES = {
    0.02: {'N': 60, 'label': 'nu=0.02 (identical to sc2.4 -- sharp shock)'},
    0.2: {'N': 40, 'label': 'nu=0.2 (10x milder -- well-resolved)'},
}


def exact_solution(t, x, nu):
    xi = x - S_SHOCK * t
    return (U_L + U_R) / 2.0 - (U_L - U_R) / 2.0 * torch.tanh((U_L - U_R) * xi / (4.0 * nu))


def check_boundary_is_constant(nu):
    """Verifies this experiment's core assumption: the exact solution at
    both domain edges is u_L / u_R to float round-off for every t we
    simulate -- returns the max deviation observed."""
    dev = 0.0
    for t in (0.0, T_END):
        for x, target in ((DOMAIN[0], U_L), (DOMAIN[1], U_R)):
            val = float(exact_solution(torch.tensor(t, dtype=torch.float64),
                                        torch.tensor(x, dtype=torch.float64), nu))
            dev = max(dev, abs(val - target))
    return dev


def name_for(nu):
    return f"sc32_nu{str(nu).replace('.', 'p')}"
