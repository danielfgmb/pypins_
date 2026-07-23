"""
Shared setup for sc3.3: validation-ladder step "linear multi-cell interface
flux" (design doc SS2.5), phase 2a. Manufactured steady linear
advection-diffusion problem --

    d/dx(a*u) - nu*u'' = f(x),  x in [0,1],  u(0)=u(1)=0
    u_exact(x) = sin(2*pi*x)   =>  f derived analytically below

solved via pinns_framework.pgdr.solve_advection_diffusion_multi_cell: a
genuine multi-cell mesh (pgdr.Mesh1D), upwind advective + SIPG viscous
interface flux (pgdr.mesh.assemble_advection_diffusion) -- the first solve
in this chapter that actually needs SS2.5's "sole unverified analytic
component". Still one global torch.linalg.solve, no training loop, same
spirit as sc3.1/sc3.2.

u_exact vanishes at both x=0 and x=1, so it's compatible with a plain
n_cells=1 sanity check (must reproduce sc3.1-style spectral accuracy --
the interface machinery generalizes the single-cell case, it doesn't
replace it) before adding interior interfaces.
"""
import math
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
N = 30
N_CELLS_LIST = [1, 2, 4, 8, 16]
G_LEFT, G_RIGHT = 0.0, 0.0


def exact_solution(x):
    return torch.sin(2 * math.pi * x)


def rhs(x):
    u_x = 2 * math.pi * torch.cos(2 * math.pi * x)
    u_xx = -(2 * math.pi) ** 2 * torch.sin(2 * math.pi * x)
    return A_ADV * u_x - NU * u_xx


def name_for(n_cells):
    return f'sc33_ncells{n_cells}'
