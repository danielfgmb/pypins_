"""
Higher-order time integration for TD-PINNs.

  burgers_1d_cn_loss   : Crank-Nicolson   O(dt^2) — averages PDE at t^n and t^{n+1}
  burgers_1d_bdf2_loss : BDF2             O(dt^2) — 3-level, only implicit spatial term
  TrainerBurger1D_HO   : drops-in for TrainerBurger1D, adds 'scheme' argument
"""

import torch
import time
import numpy as np
from .model import TD_PINNS
from .model import burgers_1d_loss as _be_loss


# ── Loss functions ─────────────────────────────────────────────────────────

def burgers_1d_cn_loss(model, X, u0, u1, PINN_old):
    """
    Crank-Nicolson residual:

        (u1 - u0)/dt
        + 0.5 * [ u1*du1/dx - nu*d2u1/dx2 ]   <- implicit, t^{n+1}
        + 0.5 * [ u0*du0/dx - nu*d2u0/dx2 ]   <- explicit, t^n  (frozen PINN_old)
    """
    # t^{n+1} — gradients flow through PINN being trained
    du1_dx = torch.autograd.grad(
        u1, X, grad_outputs=torch.ones_like(u1),
        retain_graph=True, create_graph=True)[0]
    du1_dxx = torch.autograd.grad(
        du1_dx, X, grad_outputs=torch.ones_like(du1_dx),
        retain_graph=True, create_graph=True)[0]

    # t^n — from frozen PINN_old, treated as constants
    X_old = X.detach().clone().requires_grad_(True)
    u0_val = PINN_old.forward(X_old)
    # create_graph=True is required so that the second derivative can be computed
    du0_dx = torch.autograd.grad(
        u0_val, X_old, grad_outputs=torch.ones_like(u0_val),
        retain_graph=True, create_graph=True)[0]
    du0_dxx = torch.autograd.grad(
        du0_dx, X_old, grad_outputs=torch.ones_like(du0_dx),
        retain_graph=False, create_graph=False)[0].detach()
    du0_dx = du0_dx.detach()  # detach only after second derivative is done

    rhs_n1 = u1     * du1_dx - model.nu * du1_dxx
    rhs_n0 = u0.detach() * du0_dx - model.nu * du0_dxx
    res = (u1 - u0.detach()) / model.dt + 0.5 * (rhs_n1 + rhs_n0)
    return torch.mean(res**2), res


def burgers_1d_bdf2_loss(model, X, u0, u_prev, u1):
    """
    BDF2 residual (3-level, no old derivatives needed):

        (3*u1 - 4*u0 + u_prev) / (2*dt)   <- time term from Taylor
        + u1 * du1/dx - nu * d2u1/dx2      <- implicit spatial term
    """
    du1_dx = torch.autograd.grad(
        u1, X, grad_outputs=torch.ones_like(u1),
        retain_graph=True, create_graph=True)[0]
    du1_dxx = torch.autograd.grad(
        du1_dx, X, grad_outputs=torch.ones_like(du1_dx),
        retain_graph=True, create_graph=True)[0]

    res = ((3*u1 - 4*u0.detach() + u_prev.detach()) / (2*model.dt)
           + u1 * du1_dx - model.nu * du1_dxx)
    return torch.mean(res**2), res


# ── Device helper (duplicated to keep module self-contained) ───────────────

def _optimize_device():
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    torch.set_float32_matmul_precision('medium')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.backends.mps.is_available():
        print('Using MPS'); return torch.device('mps')
    if torch.cuda.is_available():
        print('Using CUDA'); return torch.device('cuda')
    print('Using CPU'); return torch.device('cpu')


# ── Trainer ────────────────────────────────────────────────────────────────

class TrainerBurger1D_HO:
    """
    Higher-order TD-PINN trainer for 1-D Burgers.

    scheme : 'BE'   backward Euler   O(dt)    [baseline]
             'CN'   Crank-Nicolson   O(dt^2)
             'BDF2' BDF2             O(dt^2)  (falls back to CN at step 1)
    """

    def __init__(self, scheme='CN'):
        assert scheme in ('BE', 'CN', 'BDF2'), f"Unknown scheme: {scheme}"
        self.scheme = scheme
        self.device = _optimize_device()

    # ── Setup (identical to TrainerBurger1D) ──────────────────────────────

    def set_simulation_parameters(self, nu=0.07, t0=0.0, dt=0.1, nt=10, x_bc=None):
        self.nu, self.t0, self.dt, self.nt = nu, t0, dt, nt
        if x_bc is None:
            x_bc = torch.Tensor([0, 2 * np.pi])
        self.x_bc   = x_bc
        self.x_test = torch.linspace(x_bc[0], x_bc[1], 1000)

    def set_training_parameters(self, tol=1e-2, tol_ic=None, max_epochs=80000,
                                 n_prints=10, lr=1e-2, min_lr=1e-8):
        self.tol, self.tol_ic = tol, (tol_ic or tol)
        self.max_epochs       = max_epochs
        self.n_prints         = n_prints
        self.lr, self.min_lr  = lr, min_lr

    def set_point_collocation_distribution(self, func):
        self.func_distribution = func

    def set_initial_condition(self, x_initial, func=None, u_initial=None):
        self.x_initial = x_initial
        if func:
            x_np            = x_initial.cpu().numpy() if torch.is_tensor(x_initial) else x_initial
            self.u_initial  = torch.as_tensor(func(t=self.t0, x=x_np), dtype=torch.float32)
            self.bc_initial = torch.as_tensor(func(t=self.t0, x=self.x_bc.cpu().numpy()), dtype=torch.float32)
        else:
            self.u_initial  = torch.as_tensor(u_initial, dtype=torch.float32)
            self.bc_initial = self.u_initial[[0, -1]]

    def set_boundary_function(self, func):
        self.boundary_func = func

    def set_device(self, name):
        self.device = torch.device(name)

    # ── Training loop ──────────────────────────────────────────────────────

    def train_td_implicit(self):
        self.PINNS_list  = []
        self.PINNS_times = []

        # IC step — always backward Euler (no previous PINN exists)
        PINN_old, pde_res = self._train_iter(
            self.x_initial, self.u_initial,
            initial_condition=True, tol=self.tol_ic,
            x_bc=self.x_bc, bc=self.bc_initial,
            pde_loss_fn=None  # BE default
        )
        self.PINNS_list.append(PINN_old)
        PINN_prev = None   # needed for BDF2 (2 steps back)

        for i in range(1, self.nt):
            t = self.t0 + i * self.dt
            print(f'\n=== [{self.scheme}] Step {i}/{self.nt-1}  t={t:.4f} ===')
            start = time.perf_counter()

            x_i = self.func_distribution(PINN_old, t, pde_res).reshape(-1, 1).to(self.device)
            u_0 = PINN_old.forward(x_i)
            bc  = torch.as_tensor(
                self.boundary_func(t=t, x=self.x_bc.cpu().numpy()), dtype=torch.float32)

            # Pick the loss function for this step
            pde_loss_fn = self._make_pde_loss(i, PINN_old, PINN_prev, x_i)

            PINN_i, pde_res = self._train_iter(
                x_i, u_0, tol=self.tol, previous_PINN=PINN_old,
                x_bc=self.x_bc, bc=bc,
                pde_loss_fn=pde_loss_fn
            )
            elapsed = time.perf_counter() - start
            self.PINNS_list.append(PINN_i)
            self.PINNS_times.append(elapsed)
            print(f'  Done in {elapsed:.2f}s')

            PINN_prev = PINN_old
            PINN_old  = PINN_i

    def _make_pde_loss(self, step_i, PINN_old, PINN_prev, x_i):
        """Return the appropriate pde_loss_fn closure for this step."""
        if self.scheme == 'BE':
            return None  # uses burgers_1d_loss default

        if self.scheme == 'CN':
            _po = PINN_old
            return lambda model, X, u0, u1: burgers_1d_cn_loss(model, X, u0, u1, _po)

        if self.scheme == 'BDF2':
            if step_i < 2 or PINN_prev is None:
                # Not enough history — fall back to CN for the first step
                _po = PINN_old
                return lambda model, X, u0, u1: burgers_1d_cn_loss(model, X, u0, u1, _po)
            _po   = PINN_old
            _pp   = PINN_prev
            _xi   = x_i.detach()
            def _bdf2(model, X, u0, u1, po=_po, pp=_pp, xi=_xi):
                u_prev = pp.forward(xi.to(model.device)).detach()
                return burgers_1d_bdf2_loss(model, X, u0, u_prev, u1)
            return _bdf2

    def _train_iter(self, x_train, u_train, previous_PINN=None,
                    initial_condition=False, tol=1e-3, x_bc=None, bc=None,
                    pde_loss_fn=None):
        def to_t(a):
            a = a.detach() if torch.is_tensor(a) else a
            return torch.as_tensor(a, dtype=torch.float32).reshape(-1, 1).to(self.device)
        x_train = to_t(x_train).requires_grad_(True)
        u_train = to_t(u_train)
        x_bc    = to_t(x_bc).requires_grad_(True)
        bc      = to_t(bc)

        PINN = TD_PINNS(device=self.device, x_dim=1, nu=self.nu, dt=self.dt,
                        pde_loss_fn=pde_loss_fn)
        if previous_PINN:
            PINN.load_state_dict(previous_PINN.state_dict())

        opt   = torch.optim.AdamW(PINN.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', patience=300, factor=0.5, min_lr=self.min_lr)
        print_every = max(1, self.max_epochs // self.n_prints)

        for i in range(self.max_epochs):
            opt.zero_grad()
            if initial_condition:
                loss, l_ic, l_bc, res = PINN.loss_initial(x_train, u_train, x_bc, bc)
            else:
                loss, l_pde, l_bc, res = PINN.loss(x_train, u_train, x_bc, bc)
            loss.backward(retain_graph=True)
            v = loss.item()
            if v <= tol:
                print(f'  Converged epoch={i}  loss={v:.6f}')
                break
            opt.step()
            sched.step(v)
            if i % print_every == 0:
                print(f'  Epoch {i:6d}: loss={v:.6f}  lr={opt.param_groups[0]["lr"]:.2e}')
        return PINN, res
