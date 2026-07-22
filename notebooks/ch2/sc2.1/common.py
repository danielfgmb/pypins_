"""
Shared setup for the sc2.1 experiment: Burgers sawtooth, TD-PINNs vs Vanilla,
weak_projection vs agnostic (nu > 0 reduced-order weak form). Imported by
every train_*.py script and by compare_results.py so the physics, config,
and weak-form machinery are defined exactly once. Mirrors
notebooks/ch2/burgers_sawtooth_weak_forms.ipynb -- this is that notebook's
content split into standalone, tmux-runnable scripts because the notebook
kernel was failing on the full-epoch-budget runs.
"""
import sys, os

ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn

from pinns_framework import (
    PINSpatialDimension, collocation_last_residual, collocation_uniform,
)

HERE            = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(HERE, 'results')
CHECKPOINTS_DIR = os.path.join(HERE, 'checkpoints')
LOGS_DIR        = os.path.join(HERE, 'logs')
FIGS_DIR        = os.path.join(HERE, 'figs')
for _d in (RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGS_DIR):
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── problem ──────────────────────────────────────────────────────────────
NU     = 0.07
X0, XF = 0.0, 2 * np.pi

def toothsaw(t, x):
    """Hopf-Cole analytic solution for viscous Burgers on [0, 2pi]."""
    x  = np.asarray(x, dtype=float)
    nu = NU
    c  = 4 * nu * (t + 1)
    a  = (-4*t + x)**2           / c
    b  = (-4*t + x - 2*np.pi)**2 / c
    num = ((-8*t + 2*x)            * np.exp(-a)
         + (-8*t + 2*x - 4*np.pi) * np.exp(-b)) / c
    den = np.exp(-b) + np.exp(-a)
    return 2 * nu * num / den + 4

# ── experiment config (matches burgers_sawtooth_weak_forms.ipynb) ──────────
PARAMS_TD  = {'layers': (1, 32, 32, 32, 1), 'activation': 'tanh'}   # u(x)  per step
PARAMS_VAN = {'layers': (2, 32, 32, 32, 1), 'activation': 'tanh'}   # u(t, x)

DT     = 0.005
T_END  = 0.025
NT     = round(T_END / DT)
TIMES  = [k * DT for k in range(NT + 1)]

# Domain-volume constants for the weak_projection Monte Carlo residual fix --
# see NOTA_analisis_agnostic.md. DOMAIN_VOLUME = |Omega| (just x, for
# TD-PINNs' purely-spatial weak_projection). SPACETIME_VOLUME = |Omega|*T (for
# VANILLA weak_projection, where each collocation point also carries an
# independent random t that a plain Monte Carlo mean silently marginalizes
# out). NOT used by vanilla_agnostic_loss anymore -- that now does a real
# per-time-slice quadrature, no MC volume correction needed.
DOMAIN_VOLUME    = XF - X0
SPACETIME_VOLUME = DOMAIN_VOLUME * T_END

# Number of fixed time nodes for VANILLA agnostic's tensor-product PDE
# collocation (see PINTrainer's time_colloc_nodes) -- NT+1 lands exactly on
# TD-PINNs' own step times (0, dt, 2dt, ..., T_END), so the two schemes'
# agnostic residuals are directly comparable slice-for-slice. Used by the
# space-weak/time-strong variant (vanilla_agnostic_loss).
VANILLA_AGNOSTIC_TIME_NODES = NT + 1
# The space+time-weak variant (vanilla_agnostic_weak_time_loss) tests against
# N_TEST_T temporal Legendre modes and needs its own, denser set of time
# nodes to resolve them by trapezoid quadrature -- NT+1 (=6) is nowhere near
# enough for N_TEST_T=10 modes. Independent of TD-PINNs' step count (this
# variant isn't meant to be slice-comparable to TD, unlike the one above).
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
                        # Vanilla variants) -- separate from N_TEST so bumping this
                        # doesn't force a weak_projection rerun. 10 was judged too few.
N_TEST_T   = 10   # agnostic (Vanilla weak-time variant only): plain Legendre
                   # test functions in t. 5 was judged too few.

# Tolerance-based early stopping is disabled everywhere (unreachable threshold):
# every run trains for its full epoch budget, so the "epoch vs error" curves are
# all sampled over the same range and comparable across methods.
_UNREACHABLE_TOL = -1e18
TD_EPOCHS,  TD_TOL  = 100000, _UNREACHABLE_TOL   # per PDE step (steps 1..5), full budget always
# The IC step (step 0) is a much easier fit (no PDE residual -- just matching
# u(0,x) and the boundary values) and doesn't need the same budget as a PDE
# step; it converged in <20k epochs even under the old tolerance-based
# stopping. Giving it its own, smaller budget instead of reusing TD_EPOCHS
# means TD-PINNs' TOTAL epoch count (IC_EPOCHS + 5*TD_EPOCHS) can be made to
# land on a round number that Vanilla is then given exactly, for a fair
# epoch-for-epoch compute comparison between the two schemes.
IC_EPOCHS = 10000
# 10000 (IC) + 5 * 100000 (PDE steps) = 510000 -- Vanilla gets the same total.
VAN_EPOCHS, VAN_TOL = IC_EPOCHS + 5 * TD_EPOCHS, _UNREACHABLE_TOL

ERROR_EVERY = 1000   # epoch interval for real (vs analytical) error tracking

# ── collocation: residual-driven, not uniform ───────────────────────────────
# collocation_last_residual concentrates points where the PDE residual was highest
# at the END of the previous training step, falling back to uniform on the first
# step (see pinns_framework/collocation.py).
#
# IMPORTANT ASYMMETRY #1 (silent, Vanilla only): this only has an effect for
# TD-PINNs. PINTrainer only populates the residual-history buffers
# (_last_residual / _mean_residual) inside _fit_td; _fit_vanilla never touches
# them, so for the Vanilla-weak_projection script below collocation_last_residual's
# `last_residual is None` branch fires on EVERY call and it silently falls back to
# plain uniform sampling throughout training. This is a framework limitation
# (Vanilla has no per-step residual snapshot to key off of), not a bug in that
# script -- it prints a warning about it at start-up rather than pretending
# adaptive collocation is active there.
COLLOCATION_FN = collocation_last_residual

# IMPORTANT ASYMMETRY #2 (crashes, agnostic only): _fit_td's residual-history
# bookkeeping calls self.loss_fn.residual_fn(...) DIRECTLY (bypassing
# PINLossFunction.__call__) to get a per-point (N,1) residual to rank collocation
# points by magnitude -- see trainer.py's epoch_cb inside _fit_td. That assumes
# residual_fn returns a pointwise (N,1) tensor, true for "strong"/"weak_projection"
# residual functions (td_residual) but NOT for form='agnostic' ones: td_agnostic_loss
# returns the finished SCALAR loss by design (that's the point of agnostic mode).
# Calling it from epoch_cb produces a 0-dim "residual," and collocation_last_residual
# then crashes trying to torch.argsort/slice a 0-dim tensor. So: residual-history
# adaptive collocation is structurally incompatible with agnostic-mode residual
# functions for TD-PINNs (it doesn't just no-op like the Vanilla case above, it
# raises). Use plain uniform sampling for both agnostic scripts instead.
COLLOCATION_FN_AGNOSTIC_SAFE = collocation_uniform

x_ic_np = np.linspace(X0, XF, 201)
u_ic_np = toothsaw(0, x_ic_np)
X_IC = torch.tensor(x_ic_np, dtype=torch.float32).unsqueeze(1)
U_IC = torch.tensor(u_ic_np, dtype=torch.float32).unsqueeze(1)

u_bc_val = float(toothsaw(0, X0))
X_BC = torch.tensor([[X0], [XF]], dtype=torch.float32)
U_BC = torch.full((2, 1), u_bc_val, dtype=torch.float32)

X_DIM = PINSpatialDimension('x', domain=(X0, XF), n_colloc=N_COLLOC)

def exact_fn(t, x_pts):
    u = toothsaw(t, x_pts.detach().cpu().numpy().ravel())
    return torch.tensor(u, dtype=torch.float32, device=x_pts.device).unsqueeze(1)

class TimeSlice(nn.Module):
    """Expose a space-time net u(t,x) as a spatial-only net u(x) at fixed t."""
    def __init__(self, net, t):
        super().__init__(); self.net = net; self.t = float(t)
    def forward(self, x):
        tcol = torch.full((x.shape[0], 1), self.t, device=x.device, dtype=x.dtype)
        return self.net(torch.cat([tcol, x], dim=1))

def add_L1(df, slice_nets, times=None, n_eval=N_EVAL):
    """Add an L1 (mean-abs) error column to a compute_metrics DataFrame."""
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
# Passed as PINTrainer(error_fn=...); evaluated every ERROR_EVERY epochs on a
# fixed spatial grid and logged into trainer.loss_history alongside the
# training loss, independent of what form of loss (strong/weak_projection/
# agnostic) is actually driving the gradient. Relative L2 error vs the
# analytical toothsaw solution.
_ERR_XG = None
def _err_grid(device):
    global _ERR_XG
    if _ERR_XG is None or _ERR_XG.device != device:
        _ERR_XG = torch.linspace(X0, XF, N_EVAL).unsqueeze(1).to(device)
    return _ERR_XG

def real_error_td(net, t):
    """Relative L2 error of a TD-PINNs per-step net (u(x) at fixed t) vs analytical."""
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
    """Relative L2 error of a Vanilla net u(t,x), averaged over all TIMES vs analytical.
    _t_unused is ignored (PINTrainer always calls error_fn(net, None) for VANILLA);
    kept so the signature matches real_error_td and either can be passed to error_fn."""
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
    """TD-PINNs Crank-Nicolson residual."""
    def S(w):
        w_x  = torch.autograd.grad(w.sum(),   x, create_graph=True)[0]
        w_xx = torch.autograd.grad(w_x.sum(), x, create_graph=True)[0]
        return w * w_x - nu * w_xx
    return (u - u_prev.detach()) / dt + 0.5 * (S(u) + S(u_prev).detach())

def vanilla_residual(u, t, x, nu):
    """Vanilla strong-form Burgers residual u_t + u u_x - nu u_xx."""
    u_t  = torch.autograd.grad(u.sum(),   t, create_graph=True)[0]
    u_x  = torch.autograd.grad(u.sum(),   x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx

# ── agnostic (form='agnostic') weak-form machinery -- full nu>0 physics ─────
#
# Vanishing Legendre test functions v_k(xi) = P_{k+1}(xi) - P_{k-1}(xi), which
# satisfy v_k(+-1)=0, so integrating nu*u_xx (and u*u_x = F(u)_x) by parts against
# v_k drops the boundary-flux term entirely -- only first derivatives of u are
# ever computed. dv_k/dxi = (2k+1) P_k comes for free from the Legendre recurrence,
# no autograd needed for it. Both v_k and the plain (non-vanishing) temporal test
# functions psi_m are L2([-1,1])-normalised so low-order modes don't dominate a
# product basis.

def _legendre_bank(xi, n_max):
    """P_0 .. P_{n_max+1} via the three-term recurrence. xi: (N,1) in [-1,1].
    Returns a list of (N,1) tensors, length n_max+2."""
    Ps = [torch.ones_like(xi), xi]
    for n in range(1, n_max + 1):
        Ps.append(((2 * n + 1) * xi * Ps[n] - n * Ps[n - 1]) / (n + 1))
    return Ps

def _vanishing_tests(xi, n_test):
    """v_k = P_{k+1} - P_{k-1}  (vanishes at xi=+-1), dv_k/dxi = (2k+1) P_k,
    for k=1..n_test, L2([-1,1])-normalised (||v_k||=1). Analytic norm:
    ||v_k||^2 = 2/(2k+3) + 2/(2k-1) (P_{k+1} perp P_{k-1}, ||P_n||^2=2/(2n+1)).
    xi: (N,1) -> V, dV_dxi: (N, n_test), both already normalised."""
    Ps = _legendre_bank(xi, n_test)
    V  = torch.cat([Ps[k + 1] - Ps[k - 1] for k in range(1, n_test + 1)], dim=1)
    dV = torch.cat([(2 * k + 1) * Ps[k]   for k in range(1, n_test + 1)], dim=1)
    k = torch.arange(1, n_test + 1, dtype=xi.dtype, device=xi.device)
    norm = torch.sqrt(2.0 / (2 * k + 3) + 2.0 / (2 * k - 1))
    return V / norm, dV / norm

def _legendre_plain(tau, n_max):
    """Plain (non-vanishing) P_0 .. P_{n_max-1}, L2([-1,1])-normalised --
    used only by vanilla_agnostic_weak_time_loss's temporal test functions
    (t=0 carries the IC, so unlike the spatial tests there's no boundary to
    eliminate by construction). tau: (N,1) in [-1,1] -> (N, n_max)."""
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
    """Trapezoid quadrature weights for sorted 1-D nodes. (N,1) -> (N,1)."""
    dx = x_sorted[1:] - x_sorted[:-1]
    w = torch.zeros_like(x_sorted)
    w[:-1] += dx / 2
    w[1:]  += dx / 2
    return w

def _spatial_weak_setup(x, n_test=N_TEST_AGNOSTIC):
    """Sort 1-D collocation points; build vanishing Legendre test functions
    and trapezoid quadrature weights on the physical domain [X0, XF]."""
    idx = torch.argsort(x.view(-1))
    xs  = x.view(-1, 1)[idx]
    L   = XF - X0
    xi  = (2.0 * (xs - X0) / L - 1.0).detach()
    V, dV_dxi = _vanishing_tests(xi, n_test)
    dV_dx = dV_dxi * (2.0 / L)
    w = _trapz_weights(xs)
    return idx, V.detach(), dV_dx.detach(), w.detach()

def _project(w, f, G):
    """Quadrature estimate of integral(f * g_k dx) for every test function k."""
    return (w * f * G).sum(dim=0)

def td_agnostic_loss(u, u_prev, x, dt, nu):
    """Weak-form Crank-Nicolson loss for TD-PINNs (form='agnostic'), nu > 0.
    See notebooks/ch2/burgers_sawtooth_weak_forms.ipynb Section 3 for the derivation."""
    u_x  = torch.autograd.grad(u.sum(),      x, create_graph=True)[0]
    up_x = torch.autograd.grad(u_prev.sum(), x, create_graph=True)[0]

    idx, V, dVdx, w = _spatial_weak_setup(x)
    us,  uxs  = u[idx],               u_x[idx]
    ups, upxs = u_prev.detach()[idx], up_x.detach()[idx]

    R  = _project(w, (us - ups) / dt, V)
    R += 0.5 * (-_project(w, 0.5 * us**2,  dVdx) + nu * _project(w, uxs,  dVdx))
    R += 0.5 * (-_project(w, 0.5 * ups**2, dVdx) + nu * _project(w, upxs, dVdx))
    return (R ** 2).mean()

def _vanishing_tests_batched(xi, n_test):
    """Batched version of _vanishing_tests: xi (n_times, n_col) -> V, dV_dxi
    each (n_times, n_col, n_test). Same recurrence, just carrying an extra
    leading (time-slice) dimension through every op instead of looping over
    it in Python."""
    Ps = [torch.ones_like(xi), xi]
    for n in range(1, n_test + 1):
        Ps.append(((2 * n + 1) * xi * Ps[n] - n * Ps[n - 1]) / (n + 1))
    V  = torch.stack([Ps[k + 1] - Ps[k - 1] for k in range(1, n_test + 1)], dim=-1)
    dV = torch.stack([(2 * k + 1) * Ps[k]   for k in range(1, n_test + 1)], dim=-1)
    k = torch.arange(1, n_test + 1, dtype=xi.dtype, device=xi.device)
    norm = torch.sqrt(2.0 / (2 * k + 3) + 2.0 / (2 * k - 1))
    return V / norm, dV / norm

def _trapz_weights_batched(x_sorted):
    """x_sorted: (n_times, n_col), each row already sorted -> (n_times, n_col)."""
    dx = x_sorted[:, 1:] - x_sorted[:, :-1]
    w = torch.zeros_like(x_sorted)
    w[:, :-1] += dx / 2
    w[:, 1:]  += dx / 2
    return w

def _project_batched(w, f, G):
    """w, f: (n_times, n_col); G: (n_times, n_col, n_test) -> (n_times, n_test)."""
    return (w.unsqueeze(-1) * f.unsqueeze(-1) * G).sum(dim=1)

def _reshape_time_blocks(t, *tensors):
    """PINTrainer's time_colloc_nodes builds the PDE batch as n_times
    contiguous blocks of n_col points each (see trainer.py's _fit_vanilla) --
    NOT shuffled -- so recovering the (n_times, n_col) grid is just a
    .view(), no torch.unique + boolean-mask grouping needed. That Python-loop
    grouping (the original implementation, still used by td_agnostic_loss's
    per-step call, where it's genuinely a single small batch) was the actual
    bottleneck for the Vanilla-agnostic forms: ~21-77x faster batched than
    looped, verified against the loop version's output (matches to float
    precision) before this rewrite. Returns t_unique (n_times,) plus each
    input tensor reshaped to (n_times, n_col)."""
    t_flat = t.view(-1).detach()
    t_unique = torch.unique(t_flat)
    n_times = t_unique.numel()
    n_col = t_flat.shape[0] // n_times
    return (t_unique, n_times) + tuple(z.view(n_times, n_col) for z in tensors)

def vanilla_agnostic_loss(u, t, x, nu):
    """Space-weak-ONLY loss for Vanilla PINNs (form='agnostic'), nu > 0:
    u_t is exact via autograd (no temporal test function -- NOT weakened in
    time), tested only against spatial vanishing-Legendre tests v_k(x).
    Monte Carlo version with the two-sample (U-statistic) trick -- see
    NOTA_perdida_negativa.md. No tensor-product time_colloc_nodes plumbing
    needed: works directly on ordinary random (t,x) collocation, same as
    strong/weak_projection.

    IMPORTANT: distinct from vanilla_agnostic_weak_time_loss below, which
    additionally tests against temporal basis psi_m(t) (product basis
    v_k(x)*psi_m(t)) -- that WAS this function's old formulation (before
    this session split "agnostic" into a with/without-time-weakening pair);
    keep them structurally different or they collapse to the same method
    (a bug caught by their outputs matching bit-for-bit on the same batch).

    The batch is split into two independent halves A, B; R_A and R_B are
    each an MC estimate of the same true weak residual R_k (scaled by
    SPACETIME_VOLUME for the missing 1/|Omega|/T factor -- see
    NOTA_analisis_agnostic.md), and the loss is R_A*R_B (unbiased:
    E[R_A*R_B] = R_k^2, no Var(R_hat) bias term), not R_hat^2 (biased
    upward, never gets below the MC noise floor even at convergence).

    This CAN and regularly DOES go negative near convergence (R_A, R_B are
    independent noisy estimates of ~0, so their product has random sign) --
    that's expected, not a bug; see NOTA_perdida_negativa.md for the full
    derivation. Only the real error (tracked separately via error_fn) is
    what should be judged, not the sign or magnitude of this loss."""
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    L  = XF - X0
    xi = (2.0 * (x - X0) / L - 1.0).detach()

    V, dV_dxi = _vanishing_tests(xi, N_TEST_AGNOSTIC)   # (N, Kx), unit-norm
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
    + two-sample trick (see NOTA_perdida_negativa.md) -- same pattern as
    vanilla_agnostic_loss above but WITH temporal test functions psi_m(t).
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

