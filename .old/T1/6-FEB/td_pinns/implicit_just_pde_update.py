import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import os
import json
from datetime import datetime
import pandas as pd
from TD_PINNS import TD_PINNS
from trainer import TrainerBurger2D


trainer = TrainerBurger2D()

trainer.set_device("cpu")

nu = 0.01

x0, xf = -1.0, 1.0
y0, yf = -1.0, 1.0

trainer.set_simulation_parameters(
    nu=nu, 
    t0=0.0, 
    dt=0.2, 
    nt=10, 
    x_range=(x0, xf), 
    y_range=(y0, yf)
)

trainer.set_training_parameters(
    tol=1e-2, 
    tol_ic=1e-2, 
    max_epochs=30000, 
    n_prints_per_training=10, 
    PINNs_trained_per_plot=1, 
    lr=1e-2, 
    min_lr=1e-8
)


def burger_initial_condition(t=0, x=0, y=0):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.exp(-(x**2 + y**2) / 0.1)


def burger_boundary_condition(t=0, x=0, y=0):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.zeros_like(x)


def sine_initial_condition(t=0, x=0, y=0):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sin(np.pi * x) * np.sin(np.pi * y)


nx = 20
ny = 20
n_total = nx * ny


def uniform_collocation_2d(PINN, t, pde_residual):
    return trainer.create_2d_grid(nx, ny)


def lhs_like_2d(x_range, y_range, n_samples, device=None, dtype=None):
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    cut = torch.linspace(0, 1, n_samples + 1, device=device, dtype=dtype)

    u = torch.rand(n_samples, 2, device=device, dtype=dtype)
    a = cut[:-1]
    b = cut[1:]
    samples = a.unsqueeze(1) + (b - a).unsqueeze(1) * u

    for j in range(2):
        perm = torch.randperm(n_samples, device=device)
        samples[:, j] = samples[perm, j]

    x_min, x_max = x_range
    y_min, y_max = y_range
    
    samples[:, 0] = x_min + samples[:, 0] * (x_max - x_min)
    samples[:, 1] = y_min + samples[:, 1] * (y_max - y_min)
    
    return samples


def lhs_collocation_2d(PINN, t, pde_residual):
    return lhs_like_2d(
        (x0, xf), (y0, yf), n_total, 
        device=trainer.device
    )


def gradient_collocation_2d(PINN, t, pde_residual):
    m = 2  
    n_adaptive = n_total // 2

    xy_test = trainer.create_2d_grid(nx * m, ny * m)
    xy_test.requires_grad_(True)

    grad_u, laplacian = PINN.grad(xy_test)
    
    grad_magnitude = torch.sqrt(grad_u[:, 0]**2 + grad_u[:, 1]**2).detach()

    grad_scaled = grad_magnitude / torch.sum(grad_magnitude)
    sorted_vals, sorted_idx = torch.sort(grad_scaled, descending=True)

    adaptive_points = xy_test[sorted_idx[:n_adaptive]].detach()

    uniform_points = trainer.create_2d_grid(int(np.sqrt(n_total - n_adaptive)), 
                                            int(np.sqrt(n_total - n_adaptive)))

    result = torch.cat([adaptive_points, uniform_points], dim=0)
    
    return result.to(trainer.device)


def residual_collocation_2d(PINN, t, pde_residual):
    if pde_residual is None:
        return uniform_collocation_2d(PINN, t, pde_residual)
    
    n_adaptive = n_total // 2
    
    xy_test = trainer.create_2d_grid(nx, ny)
    
    residual_magnitude = torch.sqrt(pde_residual[:, 0]**2).detach()
    
    residual_scaled = residual_magnitude / (torch.sum(residual_magnitude) + 1e-8)
    sorted_vals, sorted_idx = torch.sort(residual_scaled, descending=True)
    
    n_select = min(n_adaptive, len(sorted_idx))
    adaptive_points = xy_test[sorted_idx[:n_select]].detach()
    
    n_uniform = int(np.sqrt(n_total - n_select))
    uniform_points = trainer.create_2d_grid(n_uniform, n_uniform)
    
    result = torch.cat([adaptive_points, uniform_points], dim=0)
    
    return result.to(trainer.device)


xy_initial = trainer.create_2d_grid(nx, ny)

trainer.set_initial_condition(xy_initial=xy_initial, func=burger_initial_condition)

trainer.set_boundary_function(burger_boundary_condition)

trainer.set_point_collocation_distribution(uniform_collocation_2d)

print("Starting 2D Burger equation training...")
print(f"Domain: x in [{x0}, {xf}], y in [{y0}, {yf}]")
print(f"Parameters: nu = {nu}, dt = {trainer.dt}, nt = {trainer.nt}")
print(f"Grid: {nx} x {ny} = {n_total} collocation points")
print("=" * 50)

trainer.train_td_implicit()


print("\nPlotting 2D contour solutions...")
for i, PINN in enumerate(trainer.PINNS_list):
    t = trainer.t0 + i * trainer.dt
    trainer.plot_solution(PINN, t, nx=50, ny=50)


print("\nPlotting 3D surface solutions...")
for i, PINN in enumerate(trainer.PINNS_list):
    t = trainer.t0 + i * trainer.dt
    trainer.plot_solution_3d(PINN, t, nx=50, ny=50, elev=30, azim=45)


print("\nPlotting evolution subplots...")
trainer.plot_evolution_subplots(nx=50, ny=50, ncols=3, elev=30, azim=45)


print("\nCreating animation...")
trainer.create_animation(nx=50, ny=50, interval=500, save_path='burger_evolution.gif')
