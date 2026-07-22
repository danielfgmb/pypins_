"""
Shared setup for sc2.3: 1D Poisson with a high-frequency manufactured
solution, the canonical benchmark from Kharazmi, Zhang & Karniadakis,
"Variational Physics-Informed Neural Networks For Solving Partial
Differential Equations" (VPINN, arXiv:1912.00873) and its hp-VPINN
follow-up (arXiv:2003.05385). Picked as a *simple, cheap* literature-grounded
case where strong-form PINN is expected to fail relative to the weak
(Petrov-Galerkin) form -- unlike sc2.1/sc2.2's Burgers shock, which turned
out to be dominated by discretization/collocation noise rather than the
strong-vs-weak distinction itself.

    -u''(x) = f(x),  x in [-1, 1],   u(-1) = u(1) = 0
    u_exact(x) = sin(k*pi*x)   =>   f(x) = k^2 * pi^2 * sin(k*pi*x)

Why strong form is expected to fail as k grows: the strong-form residual
needs u_xx pointwise via autodiff through a *fixed-size* MLP -- fitting a
high-frequency, unit-amplitude target forces large curvature into a small
number of parameters, and the second derivative amplifies whatever
representation error remains (same spectral-bias mechanism as Wang, Teng &
Perdikaris, "Understanding and mitigating gradient pathologies in PINNs").

IMPORTANT: 'weak_projection' (PINLossFunction's generic form) does NOT
actually avoid this -- residual_fn still computes -u_xx-f pointwise via
autodiff; weak_projection only changes how that residual is *aggregated*
into a scalar loss (projected against several test functions instead of
averaged into one MSE number). It still differentiates u twice. The only
form that actually needs just u_x is 'agnostic' (poisson_agnostic_loss
below): the weak form is derived by hand via integration by parts,
int(-u'')v dx = int(u' v') dx - [u' v]_boundary, and v_k is chosen to vanish
at the boundary so that bracket drops -- so it only ever differentiates u
once. That is the actual literature claim (VPINN's reduced regularity
requirement), and 'weak_projection' vs 'agnostic' here isolates it: same
Legendre test-function machinery, only difference is whether u_xx is ever
computed.

Static problem (no time-stepping) -- PINTrainer's TD-PINNs/Vanilla schemes
don't apply, so train_one.py bypasses it entirely: just PINArchitecture +
PINLossFunction + a bare Adam loop. Small net, modest epochs, k in
{1, 2, 4, 8, 16}: seconds per run, not another long sweep.
"""
import sys, os
ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from pinns_framework import PINSpatialDimension

HERE            = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(HERE, 'results')
CHECKPOINTS_DIR = os.path.join(HERE, 'checkpoints')
LOGS_DIR        = os.path.join(HERE, 'logs')
FIGS_DIR        = os.path.join(HERE, 'figs')
for _d in (RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGS_DIR):
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X0, XF = -1.0, 1.0
K_LIST = [1, 2, 4, 8, 16]
FORMS  = ['strong', 'weak_projection', 'agnostic']
N_TEST_AGNOSTIC = 100   # was 20 -- too few modes to resolve k up to 16

LAYERS   = (1, 20, 20, 1)
N_COLLOC = 10000  # was 200 -> 1000 -> 10000 -- 1000 was still noisy/coarse
                  # for weak_projection's Monte Carlo estimate and agnostic's
                  # trapezoid quadrature against 100 Legendre modes
N_EVAL   = 500
N_TEST   = 100    # was 20 -- too few modes to resolve k up to 16
EPOCHS   = 100000   # back up to 100k -- 10k epochs wasn't enough to converge
                    # 100 simultaneous test-function constraints (agnostic's
                    # k=16 result collapsed to the trivial u=0 solution)
LR       = 2e-3
BC_WEIGHT = 10.0
ERROR_EVERY = 2000   # epoch interval for real (vs exact) error tracking during training

X_DIM = PINSpatialDimension('x', domain=(X0, XF), n_colloc=N_COLLOC)


def exact_solution(k, x):
    return np.sin(k * np.pi * np.asarray(x, dtype=float))


def residual(u, x, k):
    """-u'' - f, f = k^2 pi^2 sin(k pi x) so u=sin(k pi x) solves -u''=f."""
    u_x  = torch.autograd.grad(u.sum(),   x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    f = (k * np.pi) ** 2 * torch.sin(k * np.pi * x)
    return -u_xx - f


# ── agnostic (form='agnostic') hand-derived weak form -- only needs u_x ─────
# Same vanishing-Legendre-test / trapezoid-quadrature machinery as
# sc2.1/sc2.2's td_agnostic_loss, adapted to this static 1D problem.

def _legendre_bank(xi, n_max):
    Ps = [torch.ones_like(xi), xi]
    for n in range(1, n_max + 1):
        Ps.append(((2 * n + 1) * xi * Ps[n] - n * Ps[n - 1]) / (n + 1))
    return Ps

def _vanishing_tests(xi, n_test):
    """v_k = P_{k+1} - P_{k-1} (vanishes at xi=+-1), dv_k/dxi = (2k+1) P_k."""
    Ps = _legendre_bank(xi, n_test)
    V  = torch.cat([Ps[k + 1] - Ps[k - 1] for k in range(1, n_test + 1)], dim=1)
    dV = torch.cat([(2 * k + 1) * Ps[k]   for k in range(1, n_test + 1)], dim=1)
    k = torch.arange(1, n_test + 1, dtype=xi.dtype, device=xi.device)
    norm = torch.sqrt(2.0 / (2 * k + 3) + 2.0 / (2 * k - 1))
    return V / norm, dV / norm

def _trapz_weights(x_sorted):
    dx = x_sorted[1:] - x_sorted[:-1]
    w = torch.zeros_like(x_sorted)
    w[:-1] += dx / 2
    w[1:]  += dx / 2
    return w

def _spatial_weak_setup(x, n_test=N_TEST_AGNOSTIC):
    idx = torch.argsort(x.view(-1))
    xs  = x.view(-1, 1)[idx]
    L   = XF - X0
    xi  = (2.0 * (xs - X0) / L - 1.0).detach()
    V, dV_dxi = _vanishing_tests(xi, n_test)
    dV_dx = dV_dxi * (2.0 / L)
    w = _trapz_weights(xs)
    return idx, V.detach(), dV_dx.detach(), w.detach()

def _project(w, f, G):
    return (w * f * G).sum(dim=0)

def poisson_agnostic_loss(u, x, k):
    """Weak form of -u''=f, derived by hand via integration by parts:
        int(-u'')v dx = int(u' v') dx - [u' v]_boundary = int(u' v') dx
    (the boundary bracket drops because v_k(+-1)=0 by construction), so:
        R_k = int(u' v_k') dx - int(f v_k) dx
    Only ever needs u_x (never u_xx) -- unlike 'strong' and 'weak_projection'
    (see module docstring). Real trapezoid quadrature, no Monte Carlo."""
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    idx, V, dVdx, w = _spatial_weak_setup(x)
    uxs = u_x[idx]
    xs  = x[idx]
    f_vals = (k * np.pi) ** 2 * torch.sin(k * np.pi * xs)
    R = _project(w, uxs, dVdx) - _project(w, f_vals, V)
    return (R ** 2).mean()


def name_for(form, k):
    return f'{form}_k{k}'
