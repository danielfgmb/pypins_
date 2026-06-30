import torch
import time
import numpy as np
from .model import TD_PINNS


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


# ── 1-D ───────────────────────────────────────────────────────────────────

class TrainerBurger1D:

    def __init__(self):
        self.device = _optimize_device()

    def set_simulation_parameters(self, nu=0.07, t0=0.0, dt=0.1, nt=10, x_bc=None):
        self.nu, self.t0, self.dt, self.nt = nu, t0, dt, nt
        if x_bc is None:
            x_bc = torch.Tensor([0, 2 * np.pi])
        self.x_bc   = x_bc
        self.x_test = torch.linspace(x_bc[0], x_bc[1], 1000)

    def set_training_parameters(self, tol=1e-2, tol_ic=None, max_epochs=80000,
                                 n_prints=10, lr=1e-2, min_lr=1e-8, pde_loss_fn=None):
        self.tol, self.tol_ic = tol, (tol_ic or tol)
        self.max_epochs       = max_epochs
        self.n_prints         = n_prints
        self.lr, self.min_lr  = lr, min_lr
        self.pde_loss_fn      = pde_loss_fn

    def set_point_collocation_distribution(self, func):
        self.func_distribution = func

    def set_initial_condition(self, x_initial, func=None, u_initial=None):
        self.x_initial = x_initial
        if func:
            x_np            = x_initial.cpu().numpy() if torch.is_tensor(x_initial) else x_initial
            self.u_initial  = torch.as_tensor(func(t=self.t0, x=x_np), dtype=torch.float32)
            bc_np           = self.x_bc.cpu().numpy()
            self.bc_initial = torch.as_tensor(func(t=self.t0, x=bc_np), dtype=torch.float32)
        else:
            self.u_initial  = torch.as_tensor(u_initial, dtype=torch.float32)
            self.bc_initial = self.u_initial[[0, -1]]

    def set_boundary_function(self, func):
        self.boundary_func = func

    def set_device(self, name):
        self.device = torch.device(name)

    def train_td_implicit(self):
        self.PINNS_list  = []
        self.PINNS_times = []
        PINN_old, pde_res = self._train_iter(
            self.x_initial, self.u_initial,
            initial_condition=True, tol=self.tol_ic,
            x_bc=self.x_bc, bc=self.bc_initial
        )
        self.PINNS_list.append(PINN_old)
        for i in range(1, self.nt):
            t = self.t0 + i * self.dt
            print(f'\n=== Step {i}/{self.nt-1}  t={t:.4f} ===')
            start   = time.perf_counter()
            x_i     = self.func_distribution(PINN_old, t, pde_res).reshape(-1, 1)
            u_0     = PINN_old.forward(x_i)
            bc      = torch.as_tensor(
                self.boundary_func(t=t, x=self.x_bc.cpu().numpy()), dtype=torch.float32)
            PINN_i, pde_res = self._train_iter(
                x_i, u_0, tol=self.tol, previous_PINN=PINN_old,
                x_bc=self.x_bc, bc=bc
            )
            elapsed = time.perf_counter() - start
            self.PINNS_list.append(PINN_i)
            self.PINNS_times.append(elapsed)
            print(f'  Done in {elapsed:.2f}s')
            PINN_old = PINN_i

    def _train_iter(self, x_train, u_train, previous_PINN=None,
                    initial_condition=False, tol=1e-3, x_bc=None, bc=None):
        def to_t(a):
            a = a.detach() if torch.is_tensor(a) else a
            return torch.as_tensor(a, dtype=torch.float32).reshape(-1, 1).to(self.device)
        x_train = to_t(x_train).requires_grad_(True)
        u_train = to_t(u_train)
        x_bc    = to_t(x_bc).requires_grad_(True)
        bc      = to_t(bc)
        PINN = TD_PINNS(device=self.device, x_dim=1, nu=self.nu, dt=self.dt,
                        pde_loss_fn=getattr(self, 'pde_loss_fn', None))
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


# ── 2-D ───────────────────────────────────────────────────────────────────

class TrainerBurger2D:

    def __init__(self):
        self.device = _optimize_device()

    def set_simulation_parameters(self, nu=0.01, t0=0.0, dt=0.2, nt=6,
                                   x_range=(-1., 1.), y_range=(-1., 1.)):
        self.nu, self.t0, self.dt, self.nt = nu, t0, dt, nt
        self.x_range, self.y_range = x_range, y_range

    def set_training_parameters(self, tol=1e-2, tol_ic=None, max_epochs=30000,
                                 n_prints=10, lr=1e-2, min_lr=1e-8, pde_loss_fn=None):
        self.tol, self.tol_ic = tol, (tol_ic or tol)
        self.max_epochs       = max_epochs
        self.n_prints         = n_prints
        self.lr, self.min_lr  = lr, min_lr
        self.pde_loss_fn      = pde_loss_fn

    def set_point_collocation_distribution(self, func):
        self.func_distribution = func

    def set_device(self, name):
        self.device = torch.device(name)

    def create_2d_grid(self, nx, ny):
        x = torch.linspace(self.x_range[0], self.x_range[1], nx)
        y = torch.linspace(self.y_range[0], self.y_range[1], ny)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([xx.flatten(), yy.flatten()], dim=1).to(self.device)

    def get_boundary_points(self, n=50):
        xmn, xmx = self.x_range
        ymn, ymx = self.y_range
        bot = torch.stack([torch.linspace(xmn, xmx, n), torch.full((n,), ymn)], 1)
        top = torch.stack([torch.linspace(xmn, xmx, n), torch.full((n,), ymx)], 1)
        lft = torch.stack([torch.full((n,), xmn), torch.linspace(ymn, ymx, n)], 1)
        rgt = torch.stack([torch.full((n,), xmx), torch.linspace(ymn, ymx, n)], 1)
        return torch.cat([bot, top, lft, rgt]).to(self.device)

    def set_initial_condition(self, xy_initial, func=None, u_initial=None):
        self.xy_initial = xy_initial.to(self.device)
        if func:
            self.initial_func = func
            x = xy_initial[:, 0].cpu().numpy()
            y = xy_initial[:, 1].cpu().numpy()
            self.u_initial = torch.tensor(
                func(t=self.t0, x=x, y=y), dtype=torch.float32, device=self.device)
        else:
            self.u_initial = u_initial.to(self.device)
            self.initial_func = None

    def set_boundary_function(self, func):
        self.boundary_func = func

    def _get_bc_values(self, t, pts):
        x = pts[:, 0].cpu().numpy()
        y = pts[:, 1].cpu().numpy()
        return torch.tensor(
            self.boundary_func(t=t, x=x, y=y), dtype=torch.float32, device=self.device)

    def train_td_implicit(self):
        self.PINNS_list  = []
        self.PINNS_times = []
        self.bc_points   = self.get_boundary_points()
        bc0 = self._get_bc_values(self.t0, self.bc_points)
        PINN_old, pde_res = self._train_iter(
            self.xy_initial, self.u_initial,
            initial_condition=True, tol=self.tol_ic,
            xy_bc=self.bc_points, bc=bc0
        )
        self.PINNS_list.append(PINN_old)
        for i in range(1, self.nt):
            t = self.t0 + i * self.dt
            print(f'\n=== Step {i}/{self.nt-1}  t={t:.4f} ===')
            start    = time.perf_counter()
            xy_i     = self.func_distribution(PINN_old, t, pde_res)
            u_0      = PINN_old.forward(xy_i)
            bc_t     = self._get_bc_values(t, self.bc_points)
            PINN_i, pde_res = self._train_iter(
                xy_i, u_0, tol=self.tol, previous_PINN=PINN_old,
                xy_bc=self.bc_points, bc=bc_t
            )
            elapsed  = time.perf_counter() - start
            self.PINNS_list.append(PINN_i)
            self.PINNS_times.append(elapsed)
            print(f'  Done in {elapsed:.2f}s')
            PINN_old = PINN_i

    def _train_iter(self, xy_train, u_train, previous_PINN=None,
                    initial_condition=False, tol=1e-3, xy_bc=None, bc=None):
        xy_train = xy_train.reshape(-1, 2).to(self.device).detach().requires_grad_(True)
        u_train  = u_train.reshape(-1, 1).to(self.device).detach()
        xy_bc    = xy_bc.reshape(-1, 2).to(self.device).detach().requires_grad_(True)
        bc       = bc.reshape(-1, 1).to(self.device).detach()
        PINN = TD_PINNS(device=self.device, x_dim=2, nu=self.nu, dt=self.dt,
                        pde_loss_fn=getattr(self, 'pde_loss_fn', None))
        if previous_PINN:
            PINN.load_state_dict(previous_PINN.state_dict())
        opt   = torch.optim.AdamW(PINN.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', patience=300, factor=0.5, min_lr=self.min_lr)
        print_every = max(1, self.max_epochs // self.n_prints)
        for i in range(self.max_epochs):
            opt.zero_grad()
            if initial_condition:
                loss, l_ic, l_bc, res = PINN.loss_initial(xy_train, u_train, xy_bc, bc)
            else:
                loss, l_pde, l_bc, res = PINN.loss(xy_train, u_train, xy_bc, bc)
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
