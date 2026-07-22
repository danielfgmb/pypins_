"""
sc2.4 -- viscous Burgers traveling shock, a genuinely time-dependent
discontinuous-as-nu->0 benchmark (unlike sc2.3's static Poisson case) chosen
specifically to exercise the Vanilla-agnostic time-weakened vs
not-time-weakened distinction, which only makes sense when there's a real
time dimension.

Classical exact solution (Rankine-Hugoniot shock speed, viscous profile) --
see e.g. Whitham, "Linear and Nonlinear Waves", or the Evans-function
literature on Burgers viscous shocks:

    u(x,t) = (u_L+u_R)/2 - (u_L-u_R)/2 * tanh[(u_L-u_R)(x - s*t)/(4*nu)]
    s = (u_L+u_R)/2   (shock speed, Rankine-Hugoniot for F(u)=u^2/2)

u_L=2, u_R=0 => s=1: the shock translates rigidly from x=0 at t=0 to x=1 at
t=1, exact for ALL t (no need to wait for a smooth IC to steepen into a
shock, unlike sc2.2's Gaussian-IC case) -- cheap to train, sharp immediately
for small nu. |u_xx| at the shock ~ O(1/nu^2), so strong-form's pointwise
second derivative is expected to blow up as nu -> 0, same mechanism as
sc2.3's high-k Poisson case but now for a real evolving PDE.

4 methods, VANILLA scheme only (single space-time network) -- the point is
to compare weak_projection vs agnostic (space-weak/time-strong) vs
agnostic_weak_time (space+time-weak) against strong, isolating the "does
weakening TIME help too" question that a static problem (sc2.3) or
TD-PINNs (time already discretized via Crank-Nicolson, not weakened) can't
address.
"""
import sys, os
ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn

from pinns_framework import PINSpatialDimension

HERE            = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(HERE, 'results')
CHECKPOINTS_DIR = os.path.join(HERE, 'checkpoints')
LOGS_DIR        = os.path.join(HERE, 'logs')
FIGS_DIR        = os.path.join(HERE, 'figs')
for _d in (RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGS_DIR):
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── physics ──────────────────────────────────────────────────────────────
X0, XF = -5.0, 5.0
T_END  = 1.0
U_L, U_R = 2.0, 0.0
S_SHOCK  = (U_L + U_R) / 2.0
NU = 0.02   # small -> sharp (near-discontinuous) shock

def exact_solution(t, x):
    x = np.asarray(x, dtype=float)
    xi = x - S_SHOCK * t
    return (U_L + U_R) / 2.0 - (U_L - U_R) / 2.0 * np.tanh((U_L - U_R) * xi / (4.0 * NU))

# ── architecture / training config ──────────────────────────────────────
PARAMS_VAN = {'layers': (2, 32, 32, 32, 1), 'activation': 'tanh'}
N_COLLOC = 300     # cut from 1000 -- GPU is heavily contended by sc2.2/sc2.3
N_EVAL   = 1000
N_TEST   = 25          # spatial test functions (weak_projection AND agnostic)
N_TEST_T = 8            # temporal test functions (agnostic_weak_time only)
EPOCHS   = 2000         # quick, per request -- not another long sweep. GPU is
                        # heavily contended by sc2.2/sc2.3 right now (~1-1.6s/
                        # epoch measured for the agnostic forms even after
                        # trimming N_COLLOC/time-nodes), so this is a
                        # compromise between "quick" and what's actually
                        # feasible under current load, not a from-scratch
                        # ideal choice.
LR, MAX_LR, MIN_LR = 1e-2, 1e-2, 1e-6
_UNREACHABLE_TOL = -1e18
_norm = (15**2 + 5**2 + 1**2) ** 0.5
IC_WEIGHT  = 15.0 / _norm
BC_WEIGHT  = 5.0  / _norm
PDE_WEIGHT = 1.0  / _norm
ERROR_EVERY = 1000

DOMAIN_VOLUME    = XF - X0
SPACETIME_VOLUME = DOMAIN_VOLUME * T_END

# time nodes for the two Vanilla-agnostic variants' tensor-product collocation
# (cut from 13/25 -- each node costs a sequential Python-loop iteration with
# several small GPU kernel launches, which is brutal under heavy contention)
AGNOSTIC_TIME_NODES      = 6    # space-weak/time-strong
AGNOSTIC_WEAK_TIME_NODES = 11   # space+time-weak (needs more to resolve N_TEST_T modes)

X_DIM = PINSpatialDimension('x', domain=(X0, XF), n_colloc=N_COLLOC)

x_ic_np = np.linspace(X0, XF, 400)
u_ic_np = exact_solution(0.0, x_ic_np)
X_IC = torch.tensor(x_ic_np, dtype=torch.float32).unsqueeze(1)
U_IC = torch.tensor(u_ic_np, dtype=torch.float32).unsqueeze(1)

X_BC = torch.tensor([[X0], [XF]], dtype=torch.float32)
# BC values are technically time-varying (exact_solution(t, X0/XF)) but the
# shock stays >4 units from either edge for all t in [0,1] at this nu, so
# u(X0,t)~=U_L and u(XF,t)~=U_R to within numerical precision -- fixed BC.
U_BC = torch.tensor([[U_L], [U_R]], dtype=torch.float32)

def exact_fn(t, x_pts):
    u = exact_solution(t, x_pts.detach().cpu().numpy().ravel())
    return torch.tensor(u, dtype=torch.float32, device=x_pts.device).unsqueeze(1)

_ERR_XG = None
def _err_grid(device):
    global _ERR_XG
    if _ERR_XG is None or _ERR_XG.device != device:
        _ERR_XG = torch.linspace(X0, XF, N_EVAL).unsqueeze(1).to(device)
    return _ERR_XG

def real_error_vanilla(net, _t_unused, times=(0.0, 0.25, 0.5, 0.75, 1.0)):
    xg = _err_grid(next(net.parameters()).device)
    was_training = net.training
    net.eval()
    errs = []
    with torch.no_grad():
        for t in times:
            tcol = torch.full((xg.shape[0], 1), float(t), device=xg.device, dtype=xg.dtype)
            up = net(torch.cat([tcol, xg], dim=1))
            ue = exact_fn(t, xg)
            errs.append(torch.sqrt(((up - ue) ** 2).mean()) / (torch.sqrt((ue ** 2).mean()) + 1e-12))
    if was_training:
        net.train()
    return float(torch.stack(errs).mean())

# ── strong residual (also reused for weak_projection) ───────────────────
def vanilla_residual(u, t, x, nu):
    u_t  = torch.autograd.grad(u.sum(),   t, create_graph=True)[0]
    u_x  = torch.autograd.grad(u.sum(),   x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx

# ── agnostic weak-form machinery (same as sc2.1/sc2.2) ───────────────────
def _legendre_bank(xi, n_max):
    Ps = [torch.ones_like(xi), xi]
    for n in range(1, n_max + 1):
        Ps.append(((2 * n + 1) * xi * Ps[n] - n * Ps[n - 1]) / (n + 1))
    return Ps

def _vanishing_tests(xi, n_test):
    Ps = _legendre_bank(xi, n_test)
    V  = torch.cat([Ps[k + 1] - Ps[k - 1] for k in range(1, n_test + 1)], dim=1)
    dV = torch.cat([(2 * k + 1) * Ps[k]   for k in range(1, n_test + 1)], dim=1)
    k = torch.arange(1, n_test + 1, dtype=xi.dtype, device=xi.device)
    norm = torch.sqrt(2.0 / (2 * k + 3) + 2.0 / (2 * k - 1))
    return V / norm, dV / norm

def _legendre_plain(tau, n_max):
    Ps = [torch.ones_like(tau)]
    if n_max > 1:
        Ps.append(tau.clone())
    for k in range(1, n_max - 1):
        Ps.append(((2 * k + 1) * tau * Ps[-1] - k * Ps[-2]) / (k + 1))
    Psi = torch.cat(Ps[:n_max], dim=1)
    m = torch.arange(0, n_max, dtype=tau.dtype, device=tau.device)
    norm = torch.sqrt(2.0 / (2 * m + 1))
    return Psi / norm

def _trapz_weights(x_sorted):
    dx = x_sorted[1:] - x_sorted[:-1]
    w = torch.zeros_like(x_sorted)
    w[:-1] += dx / 2
    w[1:]  += dx / 2
    return w

def _spatial_weak_setup(x, n_test=N_TEST):
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

def vanilla_agnostic_loss(u, t, x, nu):
    """Space-weak-only (no temporal test functions -- u_t exact via autograd,
    tested only against v_k(x)), Monte Carlo + two-sample trick (see
    sc2.1/NOTA_perdida_negativa.md). No tensor-product time_colloc_nodes
    plumbing needed -- ordinary random (t,x) collocation, same as
    strong/weak_projection. Loss CAN go negative near convergence; expected,
    not a bug -- judge real error instead."""
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    L  = XF - X0
    xi = (2.0 * (x - X0) / L - 1.0).detach()
    V, dV_dxi = _vanishing_tests(xi, N_TEST)
    dV_dx = dV_dxi * (2.0 / L)

    F_u = 0.5 * u ** 2
    f = u_t * V - F_u * dV_dx + nu * u_x * dV_dx   # (N, Kx)

    N = f.shape[0]
    half = N // 2
    R_A = f[:half].mean(dim=0) * SPACETIME_VOLUME
    R_B = f[half:2 * half].mean(dim=0) * SPACETIME_VOLUME
    return (R_A * R_B).mean()

def vanilla_agnostic_weak_time_loss(u, t, x, nu):
    """Space+time weak, product test basis Phi=v_k(x)*psi_m(t), Monte Carlo
    + two-sample trick. Same notes as vanilla_agnostic_loss above."""
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    L   = XF - X0
    xi  = (2.0 * (x - X0) / L - 1.0).detach()
    tau = (2.0 * t / T_END - 1.0).detach()

    V, dV_dxi = _vanishing_tests(xi, N_TEST)
    dV_dx = dV_dxi * (2.0 / L)
    Psi = _legendre_plain(tau, N_TEST_T)

    Phi     = (V.unsqueeze(2)     * Psi.unsqueeze(1)).reshape(x.shape[0], -1).detach()
    dPhi_dx = (dV_dx.unsqueeze(2) * Psi.unsqueeze(1)).reshape(x.shape[0], -1).detach()

    F_u = 0.5 * u ** 2
    f = u_t * Phi - F_u * dPhi_dx + nu * u_x * dPhi_dx

    N = f.shape[0]
    half = N // 2
    R_A = f[:half].mean(dim=0) * SPACETIME_VOLUME
    R_B = f[half:2 * half].mean(dim=0) * SPACETIME_VOLUME
    return (R_A * R_B).mean()

def name_for(method):
    return method
