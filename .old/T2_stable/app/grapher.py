import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _grid_2d(x_range, y_range, nx, ny, device):
    x = torch.linspace(x_range[0], x_range[1], nx)
    y = torch.linspace(y_range[0], y_range[1], ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)


# ── 1-D ───────────────────────────────────────────────────────────────────

def plot_1d(PINN, t, x0, xf, device, analytic_func=None, nx=1000):
    x = torch.linspace(x0, xf, nx).reshape(-1, 1).to(device)
    with torch.no_grad():
        u = PINN.forward(x).cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(x.cpu().numpy(), u, label=f'PINN  t={t:.3f}')
    if analytic_func is not None:
        u_ex = analytic_func(t=t, x=x.cpu().numpy().ravel())
        plt.plot(x.cpu().numpy(), u_ex, '--', label='exact')
    plt.legend(); plt.xlabel('x'); plt.ylabel('u')
    plt.title(f'1-D Burgers  t={t:.4f}'); plt.grid(True); plt.show()


def plot_1d_evolution(pinns_list, t0, dt, x0, xf, device, analytic_func=None, nx=1000):
    for i, P in enumerate(pinns_list):
        plot_1d(P, t0 + i * dt, x0, xf, device, analytic_func, nx)


# ── 2-D ───────────────────────────────────────────────────────────────────

def plot_2d_contour(PINN, t, x_range, y_range, device, nx=50, ny=50):
    xy = _grid_2d(x_range, y_range, nx, ny, device)
    with torch.no_grad():
        u = PINN.forward(xy).cpu().numpy().reshape(nx, ny)
    x = xy[:, 0].cpu().numpy().reshape(nx, ny)
    y = xy[:, 1].cpu().numpy().reshape(nx, ny)
    plt.figure(figsize=(7, 5))
    plt.contourf(x, y, u, levels=50, cmap='viridis')
    plt.colorbar(label='u'); plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'2-D Burgers  t={t:.4f}'); plt.show()



# ── 1-D comparison & metrics ───────────────────────────────────────────────

def plot_comparison_1d(t, x, u_pinn, u_fd, u_exact=None):
    """Overlay PINN, FD, and exact solutions on the same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    if u_exact is not None:
        ax.plot(x, u_exact, 'k-',  lw=2,   label='Exact')
    ax.plot(x, u_fd,   'b--', lw=1.5, label='FD (CFD Python)')
    ax.plot(x, u_pinn, 'r-',  lw=1.5, label='TD-PINN')
    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.set_title(f'Solutions  t = {t:.4f}')
    ax.legend(); ax.grid(True)

    ax = axes[1]
    if u_exact is not None:
        ax.plot(x, np.abs(u_pinn - u_exact), 'r-',  lw=1.5, label='|PINN − exact|')
        ax.plot(x, np.abs(u_fd   - u_exact), 'b--', lw=1.5, label='|FD − exact|')
        ax.set_ylabel('|error|')
    else:
        ax.plot(x, np.abs(u_pinn - u_fd), 'g-', lw=1.5, label='|PINN − FD|')
        ax.set_ylabel('|diff|')
    ax.set_xlabel('x')
    ax.set_title(f'Point-wise error  t = {t:.4f}')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_comparison_evolution(times, x, pinn_sols, fd_sols, exact_func=None, ncols=3):
    """Grid of comparison subplots across all time steps."""
    n     = len(times)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for i, t in enumerate(times):
        ax = axes[i]
        if exact_func is not None:
            ax.plot(x, exact_func(t=t, x=x), 'k-',  lw=2,   label='Exact')
        ax.plot(x, fd_sols[t],   'b--', lw=1.5, label='FD')
        ax.plot(x, pinn_sols[t], 'r-',  lw=1.5, label='PINN')
        ax.set_title(f't = {t:.3f}'); ax.grid(True)
        if i == 0:
            ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.suptitle('1-D Burgers — PINN vs FD vs Exact', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_error_evolution(times, df, metrics=('L2', 'Linf')):
    """
    Semilogy plot of error metrics over time.
    `df` is the DataFrame from `build_comparison_table`.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        ax.semilogy(df.index, df[f'PINN_{m}'], 'r-o', ms=5, label=f'PINN {m}')
        ax.semilogy(df.index, df[f'FD_{m}'],   'b-s', ms=5, label=f'FD {m}')
        ax.set_xlabel('t'); ax.set_ylabel(m + ' error')
        ax.set_title(f'{m} error vs time')
        ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_conservation(df):
    """Plot mass and energy over time for both methods and exact."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(df.index, df['exact_mass'], 'k-',  lw=2,   label='Exact')
    ax.plot(df.index, df['FD_mass'],    'b--', lw=1.5, label='FD')
    ax.plot(df.index, df['PINN_mass'],  'r-',  lw=1.5, label='PINN')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\int u\,dx$')
    ax.set_title('Mass conservation  (should be constant)')
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.plot(df.index, df['exact_energy'], 'k-',  lw=2,   label='Exact')
    ax.plot(df.index, df['FD_energy'],    'b--', lw=1.5, label='FD')
    ax.plot(df.index, df['PINN_energy'],  'r-',  lw=1.5, label='PINN')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\frac{1}{2}\int u^2\,dx$')
    ax.set_title('Energy dissipation  (should decrease)')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_2d_evolution(pinns_list, t0, dt, x_range, y_range, device,
                      nx=40, ny=40, ncols=3):
    n     = len(pinns_list)
    nrows = (n + ncols - 1) // ncols
    xy    = _grid_2d(x_range, y_range, nx, ny, device)
    all_u = []
    for P in pinns_list:
        with torch.no_grad():
            all_u.append(P.forward(xy).cpu().numpy())
    vmin = min(u.min() for u in all_u)
    vmax = max(u.max() for u in all_u)
    x = xy[:, 0].cpu().numpy().reshape(nx, ny)
    y = xy[:, 1].cpu().numpy().reshape(nx, ny)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, u in enumerate(all_u):
        t_i = t0 + i * dt
        axes[i].contourf(x, y, u.reshape(nx, ny), levels=30,
                         cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f't={t_i:.3f}')
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.suptitle('2-D Burgers Evolution')
    plt.tight_layout()
    plt.show()
