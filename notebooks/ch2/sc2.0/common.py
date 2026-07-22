"""
Shared setup for the sc2.0 experiment: Burgers sawtooth, basic stats for the
4 (architecture x scheme) combinations -- {MLP, KAN} x {TD-PINNs, Vanilla} --
all using form='strong' (plain pointwise MSE, no test-function projection).
Runs in parallel with sc2.1 (weak_projection/agnostic study); this experiment
is the architecture-comparison baseline instead. Imported by every train_*.py
script and by compare_results.py.
"""
import sys, os, shutil

ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn

from pinns_framework import PINSpatialDimension, collocation_uniform

HERE            = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(HERE, 'results')
CHECKPOINTS_DIR = os.path.join(HERE, 'checkpoints')
LOGS_DIR        = os.path.join(HERE, 'logs')
FIGS_DIR        = os.path.join(HERE, 'figs')
for _d in (RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGS_DIR):
    os.makedirs(_d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── problem (same as sc2.1) ─────────────────────────────────────────────────
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

# ── architectures ────────────────────────────────────────────────────────
PARAMS_TD_MLP  = {'layers': (1, 32, 32, 32, 1), 'activation': 'tanh'}   # u(x)  per step
PARAMS_VAN_MLP = {'layers': (2, 32, 32, 32, 1), 'activation': 'tanh'}   # u(t, x)

# KAN backbones -- grid_range set to the input domain, grid_eps mostly-adaptive.
# Same hyperparameters as notebooks/ch2/burgers_tdpinns_vs_vanilla_sawtooth_kan.ipynb.
GRID_SIZE, SPLINE_ORDER, GRID_EPS = 20, 3, 0.05
PARAMS_TD_KAN  = {'layers': (1, 16, 16, 1), 'grid_size': GRID_SIZE, 'spline_order': SPLINE_ORDER,
                  'grid_range': (X0, XF), 'grid_eps': GRID_EPS}   # u(x)    over x in [X0, XF]
PARAMS_VAN_KAN = {'layers': (2, 16, 16, 1), 'grid_size': GRID_SIZE, 'spline_order': SPLINE_ORDER,
                  'grid_range': (0.0, XF), 'grid_eps': GRID_EPS}  # u(t, x) over [0, XF] (covers t and x)

def clear_kan_cache():
    """pykan writes a ./model checkpoint dir (relative to cwd) on every KAN
    construction. Each KAN script chdir()s into its own KAN_SCRATCH dir (see
    below) before ever constructing a KAN net, specifically so that running
    train_td_kan.py and train_vanilla_kan.py in parallel (same tmux session)
    can't race on a shared ./model directory -- each process's './model' then
    resolves to a different physical path."""
    shutil.rmtree('./model', ignore_errors=True)

def kan_scratch(name):
    """Create (if needed) and chdir into a private scratch dir for a KAN
    script, so pykan's './model' writes don't collide with another KAN
    script running in parallel. Call once, near the top of main(), AFTER
    importing everything from this module (all paths above are already
    absolute at that point, so the chdir doesn't affect them)."""
    path = os.path.join(HERE, f'_kan_scratch_{name}')
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

DT     = 0.005
T_END  = 0.025
NT     = round(T_END / DT)
TIMES  = [k * DT for k in range(NT + 1)]

LR, MAX_LR, MIN_LR = 1e-2, 1e-2, 1e-6
_norm      = (15**2 + 5**2 + 1**2) ** 0.5
IC_WEIGHT  = 15.0 / _norm
BC_WEIGHT  = 5.0  / _norm
PDE_WEIGHT = 1.0  / _norm
N_COLLOC   = 500
N_EVAL     = 1000

COLLOCATION_FN = collocation_uniform  # kept simple/basic -- same strategy for all 4 combos

# Tolerance-based early stopping disabled (unreachable threshold): every run
# always trains its full epoch budget, same rationale as sc2.1. MLP and KAN
# get the SAME epoch counts -- this is a basic-stats comparison of what each
# architecture achieves in the same number of optimizer steps, not a
# wall-clock-matched comparison. KAN is much slower per epoch (pykan spline
# overhead measured at ~2.5-5s/epoch on this machine), so it simply takes
# longer to finish the same budget -- that's fine, it's the point of the
# comparison, not something to compensate for.
_UNREACHABLE_TOL = -1e18

IC_EPOCHS, TD_EPOCHS = 200, 800           # IC step, per PDE step (x5) -- shared by MLP and KAN
VAN_EPOCHS = IC_EPOCHS + 5 * TD_EPOCHS      # matched total: 4200
TOL = _UNREACHABLE_TOL

# Backward-compatible aliases (train_*.py scripts import the arch-specific
# names) -- same values, since the budget is now shared.
MLP_IC_EPOCHS, MLP_TD_EPOCHS, MLP_VAN_EPOCHS, MLP_TOL = IC_EPOCHS, TD_EPOCHS, VAN_EPOCHS, TOL
KAN_IC_EPOCHS, KAN_TD_EPOCHS, KAN_VAN_EPOCHS, KAN_TOL = IC_EPOCHS, TD_EPOCHS, VAN_EPOCHS, TOL

ERROR_EVERY_MLP = 50
ERROR_EVERY_KAN = 50

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

# ── strong pointwise residuals (same math as sc2.1) ──────────────────────
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

# ── real (ground-truth) error tracking during training ──────────────────────
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
    """Relative L2 error of a Vanilla net u(t,x), averaged over all TIMES vs analytical."""
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
