"""
Shared setup for sc3.1: the same 1D Poisson benchmark as ch2/sc2.3 --

    -u''(x) = f(x),  x in [-1, 1],   u(-1) = u(1) = 0
    u_exact(x) = sin(k*pi*x)   =>   f(x) = k^2 * pi^2 * sin(k*pi*x)

for k in {1, 2, 4, 8, 16} -- but solved here via PGDr-PIKAN's single-cell
Legendre-Galerkin machinery (pinns_framework.pgdr) instead of PINN training:
one direct linear solve, no optimiser, no epochs (design doc SS7,
validation-ladder step 3).

Deliberately the SAME benchmark as sc2.3 so the two results tables sit next
to each other: sc2.3 asked "does a hand-derived weak form let a PINN *train*
better at high k"; sc3.1 asks "does a cell-local Legendre trial/test space
let you skip training altogether and solve the same benchmark by linear
algebra, to spectral accuracy". See notebooks/ch2/sc2.3/common.py for the
literature context (Kharazmi, Zhang & Karniadakis, VPINN, arXiv:1912.00873).

Caveat carried over from the design doc (SS3.3) and stated honestly in
notebooks/ch3/README.md: a single KAN layer collapses to Bubnov-Galerkin
(trial == test), so this does NOT yet demonstrate the trial != test
Petrov-Galerkin argument that depth is supposed to provide -- that needs a
deeper KAN solved nonlinearly in theta, which is chapter-3 phase 2+.
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

DOMAIN = (-1.0, 1.0)
K_LIST = [1, 2, 4, 8, 16]

# Legendre-Galerkin mode count per k -- needs to grow with k to resolve
# sin(k*pi*x) to spectral accuracy (unlike sc2.3's PINN runs, which used a
# fixed N_TEST=100 for every k because Monte Carlo noise, not mode count,
# was the binding constraint there).
N_FOR_K = {1: 40, 2: 40, 4: 40, 8: 44, 16: 68}
N_QUAD = 400


def exact_solution(k, x):
    return torch.sin(k * np.pi * x)


def rhs(k, x):
    return (k * np.pi) ** 2 * torch.sin(k * np.pi * x)


def name_for(k):
    return f'sc31_k{k}'
