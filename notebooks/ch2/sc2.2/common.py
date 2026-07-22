"""
Shared setup for the sc2.2 experiment: viscous Burgers with a Gaussian IC,
Hopf-Lax entropy solution as the nu-independent ground truth, sweeping nu
down toward the near-inviscid/shock regime (nu = 0.05, 0.02, 0.01, 0.005,
0.001) for all 6 sc2.1 methods (strong / weak_projection / agnostic x
TD-PINNs / Vanilla). Goal: show how strong vs weak (variational) formulations
behave near the shock as nu -> 0 -- the weak form should need less pointwise
regularity of u (only first derivatives, via integration by parts) and
degrade more gracefully than the strong form's pointwise second-derivative
residual.

Physics/IC/exact-solution mirror notebooks/.tests/mlp_burgers_strong_vs_weak
copy 2.ipynb (Hopf-Lax construction), but that notebook trains inviscid
(nu=0) TD-PINNs weak-form only; here nu > 0 is a runtime parameter and all 6
sc2.1 method combinations are exercised via train_one.py, plus a 7th method
(vanilla_agnostic_weak_time) unique to the agnostic redo. Domain [0, 2*pi]
matches sc2.1's toothsaw problem so the generic weak-form/agnostic machinery
(_vanishing_tests, _legendre_plain, _spatial_weak_setup, _project) carries
over unchanged -- only the IC, exact solution, and time horizon differ.

Architecture, collocation count, test-function counts, learning rates,
per-step/IC epoch budgets, and error-tracking interval are all kept
identical to sc2.1 (per explicit request) except: NT (time steps) is capped
at 4 (T_END=0.6, dt=0.15 -- coarser than the reference notebook's dt=0.05,
chosen to keep the total epoch budget (and therefore wall-clock cost) of
this 6-methods x 5-nu sweep tractable), so Vanilla's "fair" epoch budget is
IC_EPOCHS + NT*TD_EPOCHS = 10000 + 4*100000 = 410,000.

Collocation: sc2.1 used collocation_last_residual for TD (falls back to
uniform for Vanilla, crashes for TD-agnostic -- see sc2.1/common.py's
IMPORTANT ASYMMETRY notes). To keep all 6 methods x 5 nu directly comparable
here without that asymmetry, every script in sc2.2 uses plain
collocation_uniform.
"""
import sys, os

ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from scipy.special import erf as sp_erf

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from pinns_framework import (
    PINSpatialDimension, collocation_uniform,
)

HERE            = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(HERE, 'results')
CHECKPOINTS_DIR = os.path.join(HERE, 'checkpoints')
LOGS_DIR        = os.path.join(HERE, 'logs')
FIGS_DIR        = os.path.join(HERE, 'figs')
for _d in (RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGS_DIR):
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── problem: Gaussian IC, Hopf-Lax entropy solution (nu-independent) ───────
X0, XF  = 0.0, 2 * np.pi
A_G     = 2.0
SIGMA_G = 0.5
X0_G    = np.pi
T_END   = 0.6            # T_G in the reference notebook
NT      = 4               # capped at 4 time steps (coarser Crank-Nicolson, cheaper sweep)
DT      = T_END / NT      # 0.15
TIMES   = [k * DT for k in range(NT + 1)]

T_C = SIGMA_G * np.sqrt(np.e) / A_G   # shock formation time (~0.303); T_END is post-shock

NU_LIST = [0.05, 0.02, 0.01, 0.005, 0.001]

_sqrt2   = np.sqrt(2.0)
_erf_ref = sp_erf(-X0_G / (SIGMA_G * _sqrt2))

def _F(xi, A=A_G, x0=X0_G, sigma=SIGMA_G):
    return A * sigma * np.sqrt(np.pi / 2) * (
        sp_erf((xi - x0) / (sigma * _sqrt2)) - _erf_ref
    )

def exact_solution(t, x_arr, A=A_G, x0=X0_G, sigma=SIGMA_G, n_xi=40000):
    """Hopf-Lax entropy solution for inviscid Burgers with a Gaussian IC.
    Used as the nu-independent reference for all nu in NU_LIST (vanishing
    viscosity limit): as nu -> 0 the viscous solution should converge to
    this pointwise a.e., but not uniformly near the shock."""
    x_arr = np.asarray(x_arr, dtype=float)
    if t == 0:
        return A * np.exp(-(x_arr - x0) ** 2 / (2 * sigma ** 2))
    xi_lo = min(X0, x0 - 10 * sigma) - 1.0
    xi_hi = max(XF, x0 + A * t + 5 * sigma) + 1.0
    xi    = np.linspace(xi_lo, xi_hi, n_xi)
    F_xi  = _F(xi)
    h     = F_xi[None, :] + (x_arr[:, None] - xi[None, :]) ** 2 / (2 * t)
    return (x_arr - xi[h.argmin(axis=1)]) / t

def viscous_solution(t, x_arr, nu, A=A_G, x0=X0_G, sigma=SIGMA_G, n_xi=40000):
    """Exact Cole-Hopf solution of viscous Burgers (whole-line, Gaussian IC),
    for a specific nu > 0 -- the 'soft' (Laplace-integral) version of
    exact_solution's hard argmin: as nu -> 0 this -> exact_solution (Laplace's
    method). Domain [0, 2pi] is >6 sigma from x0=pi, so edge/BC truncation
    effects are negligible and this matches the true whole-line solution to
    high precision. Uses the same h(xi) = F(xi) + (x-xi)^2/(2t) as
    exact_solution; only the weighting (softmax at temperature 2*nu vs hard
    argmin) differs.

        u_nu(x,t) = [ int (x-xi)/t * exp(-h(xi)/(2 nu)) dxi ]
                    / [ int exp(-h(xi)/(2 nu)) dxi ]
    """
    x_arr = np.asarray(x_arr, dtype=float)
    if t == 0:
        return A * np.exp(-(x_arr - x0) ** 2 / (2 * sigma ** 2))
    xi_lo = min(X0, x0 - 10 * sigma) - 1.0
    xi_hi = max(XF, x0 + A * t + 5 * sigma) + 1.0
    xi    = np.linspace(xi_lo, xi_hi, n_xi)
    F_xi  = _F(xi)
    h     = F_xi[None, :] + (x_arr[:, None] - xi[None, :]) ** 2 / (2 * t)   # (Nx, Nxi)
    w     = -h / (2 * nu)
    w    -= w.max(axis=1, keepdims=True)          # numerically stable softmax
    ew    = np.exp(w)
    num   = _trapz((x_arr[:, None] - xi[None, :]) / t * ew, xi, axis=1)
    den   = _trapz(ew, xi, axis=1)
    return num / den

# ── architecture / training config (identical to sc2.1 unless noted) ───────
PARAMS_TD  = {'layers': (1, 32, 32, 32, 1), 'activation': 'tanh'}   # u(x)  per step
PARAMS_VAN = {'layers': (2, 32, 32, 32, 1), 'activation': 'tanh'}   # u(t, x)

DOMAIN_VOLUME    = XF - X0
SPACETIME_VOLUME = DOMAIN_VOLUME * T_END

# Number of fixed time nodes for VANILLA agnostic's tensor-product PDE
# collocation (see PINTrainer's time_colloc_nodes). Space-weak/time-strong
# variant (vanilla_agnostic_loss): NT+1 lands on TD-PINNs' own step times.
# Space+time-weak variant (vanilla_agnostic_weak_time_loss): needs its own
# denser node count (25) to resolve N_TEST_T=10 temporal Legendre modes.
VANILLA_AGNOSTIC_TIME_NODES = NT + 1
VANILLA_AGNOSTIC_WEAK_TIME_NODES = 25

LR, MAX_LR, MIN_LR = 1e-2, 1e-2, 1e-6
_norm      = (15**2 + 5**2 + 1**2) ** 0.5
IC_WEIGHT  = 15.0 / _norm
BC_WEIGHT  = 5.0  / _norm
PDE_WEIGHT = 1.0  / _norm
N_COLLOC   = 500
N_EVAL     = 1000
N_TEST     = 10   # weak_projection: Legendre test functions per dim (unchanged)
N_TEST_AGNOSTIC = 20   # agnostic: vanishing Legendre test functions in x (TD and both
                        # Vanilla variants) -- separate from N_TEST, 10 was too few.
N_TEST_T   = 10   # agnostic (Vanilla weak-time variant only): plain Legendre
                   # test functions in t. 5 was too few.

_UNREACHABLE_TOL = -1e18
IC_EPOCHS, TD_EPOCHS = 10000, 100000        # per-step budgets, same as sc2.1
TD_TOL = _UNREACHABLE_TOL
VAN_EPOCHS = IC_EPOCHS + NT * TD_EPOCHS      # 10000 + 4*100000 = 410,000
VAN_TOL    = _UNREACHABLE_TOL

ERROR_EVERY = 1000

COLLOCATION_FN = collocation_uniform  # same for every method here, see module docstring

x_ic_np = np.linspace(X0, XF, 300)
u_ic_np = exact_solution(0, x_ic_np)
X_IC = torch.tensor(x_ic_np, dtype=torch.float32).unsqueeze(1)
U_IC = torch.tensor(u_ic_np, dtype=torch.float32).unsqueeze(1)

X_BC = torch.tensor([[X0], [XF]], dtype=torch.float32)
U_BC = torch.zeros(2, 1)   # Gaussian IC ~ 0 at both ends of [0, 2pi]

X_DIM = PINSpatialDimension('x', domain=(X0, XF), n_colloc=N_COLLOC)

def exact_fn(t, x_pts):
    u = exact_solution(t, x_pts.detach().cpu().numpy().ravel())
    return torch.tensor(u, dtype=torch.float32, device=x_pts.device).unsqueeze(1)

class TimeSlice(nn.Module):
    """Expose a space-time net u(t,x) as a spatial-only net u(x) at fixed t."""
    def __init__(self, net, t):
        super().__init__(); self.net = net; self.t = float(t)
    def forward(self, x):
        tcol = torch.full((x.shape[0], 1), self.t, device=x.device, dtype=x.dtype)
        return self.net(torch.cat([tcol, x], dim=1))

def add_L1(df, slice_nets, times=None, n_eval=N_EVAL):
    times = times if times is not None else TIMES
    xg = torch.linspace(X0, XF, n_eval).unsqueeze(1).to(DEVICE)
    l1 = []
    for t, net in zip(times, slice_nets):
        net.eval()
        with torch.no_grad():
            up = net(xg)
            if up.dim() == 1:
                up = up.unsqueeze(1)
        l1.append(float((up - exact_fn(t, xg)).abs().mean()))
    df['L1'] = l1
    return df

# ── real (ground-truth) error tracking during training ──────────────────────
_ERR_XG = None
def _err_grid(device):
    global _ERR_XG
    if _ERR_XG is None or _ERR_XG.device != device:
        _ERR_XG = torch.linspace(X0, XF, N_EVAL).unsqueeze(1).to(device)
    return _ERR_XG

def real_error_td(net, t):
    xg = _err_grid(next(net.parameters()).device)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        up = net(xg)
        ue = exact_fn(t, xg)
        err = torch.sqrt(((up - ue) ** 2).mean()) / (torch.sqrt((ue ** 2).mean()) + 1e-12)
    if was_training:
        net.train()
    return float(err)

def real_error_vanilla(net, _t_unused):
    xg = _err_grid(next(net.parameters()).device)
    was_training = net.training
    net.eval()
    errs = []
    with torch.no_grad():
        for t in TIMES:
            tcol = torch.full((xg.shape[0], 1), float(t), device=xg.device, dtype=xg.dtype)
            up = net(torch.cat([tcol, xg], dim=1))
            ue = exact_fn(t, xg)
            errs.append(torch.sqrt(((up - ue) ** 2).mean()) / (torch.sqrt((ue ** 2).mean()) + 1e-12))
    if was_training:
        net.train()
    return float(torch.stack(errs).mean())

# ── strong pointwise residuals -- reused as-is for form='weak_projection' ───

def td_residual(u, u_prev, x, dt, nu):
    """TD-PINNs Crank-Nicolson residual (viscous)."""
    def S(w):
        w_x  = torch.autograd.grad(w.sum(),   x, create_graph=True)[0]
        w_xx = torch.autograd.grad(w_x.sum(), x, create_graph=True)[0]
        return w * w_x - nu * w_xx
    return (u - u_prev.detach()) / dt + 0.5 * (S(u) + S(u_prev).detach())

def vanilla_residual(u, t, x, nu):
    """Vanilla strong-form viscous Burgers residual u_t + u u_x - nu u_xx."""
    u_t  = torch.autograd.grad(u.sum(),   t, create_graph=True)[0]
    u_x  = torch.autograd.grad(u.sum(),   x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx

# ── agnostic (form='agnostic') weak-form machinery -- identical to sc2.1 ────
# (domain [0, 2pi] is the same, only T_END/NU differ, both already
# parameterised below or module-level).

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
    """Plain (non-vanishing) P_0 .. P_{n_max-1}, L2([-1,1])-normalised --
    used only by vanilla_agnostic_weak_time_loss's temporal test functions."""
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

def td_agnostic_loss(u, u_prev, x, dt, nu):
    """Weak-form Crank-Nicolson loss for TD-PINNs (form='agnostic'), nu > 0.
    Deterministic trapezoid quadrature over all collocation points -- no
    Monte Carlo volume-constant issue (unlike vanilla_agnostic_loss below)."""
    u_x  = torch.autograd.grad(u.sum(),      x, create_graph=True)[0]
    up_x = torch.autograd.grad(u_prev.sum(), x, create_graph=True)[0]

    idx, V, dVdx, w = _spatial_weak_setup(x)
    us,  uxs  = u[idx],               u_x[idx]
    ups, upxs = u_prev.detach()[idx], up_x.detach()[idx]

    R  = _project(w, (us - ups) / dt, V)
    R += 0.5 * (-_project(w, 0.5 * us**2,  dVdx) + nu * _project(w, uxs,  dVdx))
    R += 0.5 * (-_project(w, 0.5 * ups**2, dVdx) + nu * _project(w, upxs, dVdx))
    return (R ** 2).mean()

def vanilla_agnostic_loss(u, t, x, nu):
    """Space-weak-ONLY loss for Vanilla PINNs (form='agnostic'), nu > 0:
    u_t exact via autograd (NOT weakened in time), tested only against
    spatial vanishing-Legendre tests v_k(x). Monte Carlo + two-sample trick
    -- see sc2.1/NOTA_perdida_negativa.md. No tensor-product
    time_colloc_nodes plumbing needed.

    IMPORTANT: distinct from vanilla_agnostic_weak_time_loss below (which
    additionally tests against temporal basis psi_m(t), product basis
    v_k(x)*psi_m(t)) -- keep them structurally different or they collapse to
    the same method (caught by their outputs matching bit-for-bit on the
    same batch during this session's first attempt at this revert).

    Splits the batch into two independent halves A, B; loss is R_A*R_B
    (unbiased: E[R_A*R_B] = R_k^2, no MC-noise-floor bias), NOT R_hat^2
    (always >= 0 but biased upward by Var(R_hat), never reaching zero even
    at convergence). This CAN and regularly DOES go negative near
    convergence -- expected, not a bug. Only the real error (tracked
    separately via error_fn) should be judged, not this loss's sign/value."""
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    L  = XF - X0
    xi = (2.0 * (x - X0) / L - 1.0).detach()

    V, dV_dxi = _vanishing_tests(xi, N_TEST_AGNOSTIC)
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
    + two-sample trick (see sc2.1/NOTA_perdida_negativa.md) -- same pattern
    as vanilla_agnostic_loss above but WITH temporal test functions psi_m(t).
    No tensor-product time_colloc_nodes plumbing needed."""
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    L   = XF - X0
    xi  = (2.0 * (x - X0) / L - 1.0).detach()
    tau = (2.0 * t / T_END - 1.0).detach()

    V, dV_dxi = _vanishing_tests(xi, N_TEST_AGNOSTIC)
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
