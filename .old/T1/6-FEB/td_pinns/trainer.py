import torch
from TD_PINNS import TD_PINNS
import time
import numpy as np
from logger import Logger


METHOD = "residual"

logger = Logger()


class TrainerDiffusion1D:

    def __init__(self, ):
        self.device = self.optimize_device()
        self.analytic_solution_func = None

    def set_simulation_parameters(self, nu=5.0, t0=0.01, dt=0.001, nt=10, x_bc=torch.Tensor([-3,3])):
        self.nu = nu
        self.t0 = t0
        self.dt = dt
        self.nt = nt
        self.x_test= torch.linspace(x_bc[0], x_bc[1], 1000)
        self.x_bc = x_bc


class TrainerBurger2D:

    def __init__(self):
        self.device = self.optimize_device()
        self.analytic_solution_func = None

    def set_simulation_parameters(self, nu=0.01, t0=0.0, dt=0.01, nt=10, 
                                   x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)):
        self.nu = nu
        self.t0 = t0
        self.dt = dt
        self.nt = nt
        self.x_range = x_range
        self.y_range = y_range

    def set_training_parameters(self, tol=1e-2, tol_ic=None, max_epochs=30000, 
                                 n_prints_per_training=10, PINNs_trained_per_plot=1, 
                                 lr=1e-2, min_lr=1e-8):
        self.tol = tol
        self.tol_ic = tol_ic if tol_ic else tol
        self.max_epochs = max_epochs
        self.n_prints_per_training = n_prints_per_training
        self.PINNs_trained_per_plot = PINNs_trained_per_plot
        self.lr = lr
        self.min_lr = min_lr

    def set_point_collocation_distribution(self, func_distribution):
        self.func_distribution = func_distribution

    def create_2d_grid(self, nx, ny):
        x = torch.linspace(self.x_range[0], self.x_range[1], nx)
        y = torch.linspace(self.y_range[0], self.y_range[1], ny)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return xy.to(self.device)

    def get_boundary_points(self, n_per_edge=50):
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        
        x_bottom = torch.linspace(x_min, x_max, n_per_edge)
        bottom = torch.stack([x_bottom, torch.full_like(x_bottom, y_min)], dim=1)
        
        x_top = torch.linspace(x_min, x_max, n_per_edge)
        top = torch.stack([x_top, torch.full_like(x_top, y_max)], dim=1)
        
        y_left = torch.linspace(y_min, y_max, n_per_edge)
        left = torch.stack([torch.full_like(y_left, x_min), y_left], dim=1)
        
        y_right = torch.linspace(y_min, y_max, n_per_edge)
        right = torch.stack([torch.full_like(y_right, x_max), y_right], dim=1)
        
        boundary = torch.cat([bottom, top, left, right], dim=0)
        return boundary.to(self.device)

    def set_initial_condition(self, xy_initial, u_initial=None, func=None):
        self.xy_initial = xy_initial.to(self.device)
        
        if func is not None:
            self.initial_func = func
            x = xy_initial[:, 0].cpu().numpy()
            y = xy_initial[:, 1].cpu().numpy()
            self.u_initial = torch.tensor(func(t=self.t0, x=x, y=y), 
                                          dtype=torch.float32, device=self.device)
        else:
            self.u_initial = u_initial.to(self.device)
            self.initial_func = None

    def set_boundary_function(self, func):
        self.boundary_func = func

    def get_boundary_values(self, t, bc_points):
        x = bc_points[:, 0].cpu().numpy()
        y = bc_points[:, 1].cpu().numpy()
        bc_values = self.boundary_func(t=t, x=x, y=y)
        return torch.tensor(bc_values, dtype=torch.float32, device=self.device)

    def set_device(self, device_name):
        self.device = torch.device(device_name)
        return self.device

    def optimize_device(self):
        torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
        torch.set_float32_matmul_precision('medium')

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) GPU acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU acceleration")
        else:
            device = torch.device("cpu")
            print("Using CPU")

        self.device = device
        return device

    def train_td_implicit(self):
        self.PINNS_list = []
        self.PINNS_times = []

        self.bc_points = self.get_boundary_points()
        bc_initial = self.get_boundary_values(self.t0, self.bc_points)

        PINN_initial, pde_residual = self.train_iteration(
            self.xy_initial, self.u_initial, 
            initial_condition=True, tol=self.tol_ic,
            xy_bc=self.bc_points, bc=bc_initial
        )

        self.PINNS_list.append(PINN_initial)
        PINN_old = PINN_initial

        for i in range(1, self.nt):
            t = self.t0 + i * self.dt
            print(f"\n=== Time step {i}/{self.nt-1}, t = {t:.4f} ===")

            start = time.perf_counter()

            xy_i = self.func_distribution(PINN_old, t, pde_residual)
            
            u_0 = PINN_old.forward(xy_i)

            bc_current = self.get_boundary_values(t, self.bc_points)

            PINN_i, pde_residual = self.train_iteration(
                xy_i, u_0, tol=self.tol, previous_PINN=PINN_old,
                xy_bc=self.bc_points, bc=bc_current
            )

            end = time.perf_counter()
            elapsed = end - start

            self.PINNS_list.append(PINN_i)
            self.PINNS_times.append(elapsed)

            print(f"Time step completed in {elapsed:.2f}s")

            PINN_old = PINN_i

    def train_iteration(self, xy_train, u_train, previous_PINN=None, 
                        initial_condition=False, tol=1e-3, xy_bc=None, bc=None):
        xy_train = xy_train.reshape(-1, 2).to(self.device)
        u_train = u_train.reshape(-1, 1).to(self.device)
        xy_bc = xy_bc.reshape(-1, 2).to(self.device)
        bc = bc.reshape(-1, 1).to(self.device)

        xy_train.requires_grad_(True)
        xy_bc.requires_grad_(True)

        if initial_condition:
            u1_train = u_train
        else:
            u0_train = u_train

        PINN = TD_PINNS(nu=self.nu, dt=self.dt, device=self.device, x_dim=2)

        optimizer = torch.optim.AdamW(PINN.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=300, factor=0.5, min_lr=self.min_lr
        )

        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        use_amp = torch.cuda.is_available()

        if previous_PINN:
            PINN.load_state_dict(previous_PINN.state_dict())

        for i in range(self.max_epochs):
            optimizer.zero_grad()

            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    if initial_condition:
                        loss, loss_ic, loss_bc, pde_residual = PINN.loss_initial(
                            xy_train, u1_train, xy_bc, bc
                        )
                    else:
                        loss, loss_pde, loss_bc, pde_residual = PINN.loss(
                            xy_train, u0_train, xy_bc, bc
                        )
                scaler.scale(loss).backward(retain_graph=True)
            else:
                if initial_condition:
                    loss, loss_ic, loss_bc, pde_residual = PINN.loss_initial(
                        xy_train, u1_train, xy_bc, bc
                    )
                else:
                    loss, loss_pde, loss_bc, pde_residual = PINN.loss(
                        xy_train, u0_train, xy_bc, bc
                    )
                loss.backward(retain_graph=True)

            loss_value = loss.item()
            if loss_value <= tol:
                if initial_condition:
                    print(f"Converged at epoch {i}: Loss={loss_value:.7f}, LossIC={loss_ic:.7f}, LossBC={loss_bc:.7f}")
                else:
                    print(f"Converged at epoch {i}: Loss={loss_value:.7f}, LossPDE={loss_pde:.7f}, LossBC={loss_bc:.7f}")
                break

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step(loss_value)

            if i % 500 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            if i % (self.max_epochs // self.n_prints_per_training) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if initial_condition:
                    print(f"Epoch {i}: Loss={loss_value:.7f}, LossIC={loss_ic:.7f}, LossBC={loss_bc:.7f}, LR={current_lr:.2e}")
                else:
                    print(f"Epoch {i}: Loss={loss_value:.7f}, LossPDE={loss_pde:.7f}, LossBC={loss_bc:.7f}, LR={current_lr:.2e}")

        return PINN, pde_residual

    def plot_solution(self, PINN, t, nx=50, ny=50):
        import matplotlib.pyplot as plt
        
        xy_plot = self.create_2d_grid(nx, ny)
        with torch.no_grad():
            u_plot = PINN.forward(xy_plot).cpu().numpy()
        
        x = xy_plot[:, 0].cpu().numpy().reshape(nx, ny)
        y = xy_plot[:, 1].cpu().numpy().reshape(nx, ny)
        u = u_plot.reshape(nx, ny)
        
        plt.figure(figsize=(8, 6))
        plt.contourf(x, y, u, levels=50, cmap='viridis')
        plt.colorbar(label='u')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'u(x, y) at t = {t:.4f}')
        plt.show()

    def plot_solution_3d(self, PINN, t, nx=50, ny=50, elev=30, azim=45):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        xy_plot = self.create_2d_grid(nx, ny)
        with torch.no_grad():
            u_plot = PINN.forward(xy_plot).cpu().numpy()
        
        x = xy_plot[:, 0].cpu().numpy().reshape(nx, ny)
        y = xy_plot[:, 1].cpu().numpy().reshape(nx, ny)
        u = u_plot.reshape(nx, ny)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(x, y, u, cmap='viridis', edgecolor='none', alpha=0.9)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(f'u(x, y) at t = {t:.4f}')
        ax.view_init(elev=elev, azim=azim)
        
        fig.colorbar(surf, shrink=0.5, aspect=10, label='u')
        plt.tight_layout()
        plt.show()

    def plot_evolution_3d(self, nx=50, ny=50, elev=30, azim=45):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        for i, PINN in enumerate(self.PINNS_list):
            t = self.t0 + i * self.dt
            self.plot_solution_3d(PINN, t, nx, ny, elev, azim)

    def plot_evolution_subplots(self, nx=50, ny=50, ncols=3, elev=30, azim=45):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        n_plots = len(self.PINNS_list)
        nrows = (n_plots + ncols - 1) // ncols
        
        fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
        
        all_u = []
        xy_plot = self.create_2d_grid(nx, ny)
        for PINN in self.PINNS_list:
            with torch.no_grad():
                u = PINN.forward(xy_plot).cpu().numpy()
                all_u.append(u)
        
        vmin = min(u.min() for u in all_u)
        vmax = max(u.max() for u in all_u)
        
        x = xy_plot[:, 0].cpu().numpy().reshape(nx, ny)
        y = xy_plot[:, 1].cpu().numpy().reshape(nx, ny)
        
        for i, (PINN, u) in enumerate(zip(self.PINNS_list, all_u)):
            t = self.t0 + i * self.dt
            u_grid = u.reshape(nx, ny)
            
            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
            surf = ax.plot_surface(x, y, u_grid, cmap='viridis', 
                                   vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.9)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title(f't = {t:.4f}')
            ax.set_zlim(vmin, vmax)
            ax.view_init(elev=elev, azim=azim)
        
        plt.suptitle('Burger Equation Evolution', fontsize=14)
        plt.tight_layout()
        plt.show()

    def create_animation(self, nx=50, ny=50, elev=30, azim=45, interval=500, save_path=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation
        
        xy_plot = self.create_2d_grid(nx, ny)
        
        all_u = []
        for PINN in self.PINNS_list:
            with torch.no_grad():
                u = PINN.forward(xy_plot).cpu().numpy().reshape(nx, ny)
                all_u.append(u)
        
        x = xy_plot[:, 0].cpu().numpy().reshape(nx, ny)
        y = xy_plot[:, 1].cpu().numpy().reshape(nx, ny)
        
        vmin = min(u.min() for u in all_u)
        vmax = max(u.max() for u in all_u)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            t = self.t0 + frame * self.dt
            surf = ax.plot_surface(x, y, all_u[frame], cmap='viridis',
                                   vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.9)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title(f'Burger Equation: t = {t:.4f}')
            ax.set_zlim(vmin, vmax)
            ax.view_init(elev=elev, azim=azim)
            return [surf]
        
        anim = FuncAnimation(fig, update, frames=len(self.PINNS_list), 
                            interval=interval, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        return anim
