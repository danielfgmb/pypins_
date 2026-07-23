"""
Shared setup for sc3.4: adaptive h-refinement (design doc SS4, phase 2b),
validation-ladder step 4 ("refinement demo on linear problem").

Same linear advection-diffusion operator as sc3.3 --

    d/dx(a*u) - nu*u'' = f(x),  x in [0,1],  u(0)=g_left, u(1)=g_right

-- but with a manufactured solution that is DELIBERATELY hard for a coarse
uniform mesh: a narrow Gaussian bump,

    u_exact(x) = exp(-(x-x0)^2 / (2*sigma^2))

with sigma=0.015 (1.5% of the domain width), centered at x0=0.5. u(0) and
u(1) are the (extremely small but nonzero, computed exactly) tail values of
this Gaussian, used as the Dirichlet data so u_exact solves the BVP exactly
regardless of a/nu.

Starting from 4 uniform cells (N=10 modes each) this is badly under-resolved
(rel_L2 ~ 6e-2) -- analogous in spirit to why sc3.2's Burgers shock defeated
a single global cell, but here on the LINEAR problem so the refinement loop
itself (indicate -> mark -> bisect -> transfer -> resolve,
pinns_framework.pgdr's compute_indicators / dorfler_mark / refine_mesh) can
be validated without any nonlinear-solve complexity.
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

DOMAIN = (0.0, 1.0)
A_ADV = 1.0
NU = 0.05
N = 10
N_CELLS_INITIAL = 4
N_ROUNDS = 8
THETA = 0.5      # Doerfler marking fraction
ENRICH = 4       # extra test modes used by the error indicator
X0 = 0.5
SIGMA = 0.015


def exact_solution(x):
    return torch.exp(-(x - X0) ** 2 / (2 * SIGMA ** 2))


def rhs(x):
    u = torch.exp(-(x - X0) ** 2 / (2 * SIGMA ** 2))
    u_x = -(x - X0) / SIGMA ** 2 * u
    u_xx = (-1.0 / SIGMA ** 2 + (x - X0) ** 2 / SIGMA ** 4) * u
    return A_ADV * u_x - NU * u_xx


G_LEFT = float(exact_solution(torch.tensor(DOMAIN[0], dtype=torch.float64)))
G_RIGHT = float(exact_solution(torch.tensor(DOMAIN[1], dtype=torch.float64)))
