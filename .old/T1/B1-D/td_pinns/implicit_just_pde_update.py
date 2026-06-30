#%% Libraries

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
from trainer import TrainerDiffusion1D

# Trainer Diffusion

trainer = TrainerDiffusion1D()

trainer.set_device("cpu")

nu = 0.07
# x_bc = torch.Tensor([-3.0,3.0]).to(trainer.device)
x_bc = torch.Tensor([0,2*np.pi])
trainer.set_simulation_parameters(nu=nu, t0=0,dt=0.1,nt=7, x_bc=x_bc)

trainer.set_training_parameters(tol=1e-2, tol_ic=1e-2, max_epochs=80000, n_prints_per_training=10, 
                                PINNs_trained_per_plot=1, lr=1e-2,min_lr=1e-8)

# x0 y xf

x0 = 0
xf = 2*np.pi

# Analytic Solution

def heat_kernel(t=0,x=0):
    # return ( 1/np.sqrt(4*np.pi*nu*t) )*np.exp(-x**2/(4*nu*t))
    # print("sol heat_kernel", f"t {t}, x {x}, nu {nu}")
    return (1/(4*np.pi*t)**(1/2))*np.exp(-x**2/(4*t))
def step(t=0, x=0, width=0.05, height=0.5):
    """
    Step function for initial conditio<n.
    Returns 0.5 inside [-width, width], 0 outside.
    Works for scalar or numpy array x.
    """
    x = np.asarray(x)
    u = np.zeros_like(x)
    mask = (x >= -width) & (x <= width)
    u[mask] = height
    return u

def sinusoidal_ic(t=0,x=0):
    return -np.sin(np.pi*x)

def toothsaw_solution(t=0,x=0):
    return -2*nu*(-(-8*t + 2*x)*np.exp(-(-4*t + x)**2/(4*nu*(t + 1)))/(4*nu*(t + 1)) - (-8*t + 2*x - 4*np.pi)*np.exp(-(-4*t + x - 2*np.pi)**2/(4*nu*(t + 1)))/(4*nu*(t + 1)))/(np.exp(-(-4*t + x - 2*np.pi)**2/(4*nu*(t + 1))) + np.exp(-(-4*t + x)**2/(4*nu*(t + 1)))) + 4

# Point Collocation Func





nx = 100 # puntos sin contar BC
nx_uniform = 50


def uniform_collocation(PINN, t, pde_residual):
    """
    x_test = torch.linspace(x0,xf,nx_test).reshape(-1, 1)

    # du_x, du2_xx = PINN.grad(x_test)

    # du_x = du_x.cpu().numpy()

    du2_xx = du2_xx.detach().cpu().numpy()

    print("du2_xx",du2_xx)

    norm_du2_xx = du2_xx / np.sum(du2_xx)

    # 2. Compute the CDF
    cdf = np.cumsum(norm_du2_xx)
    cdf /= cdf[-1]

    # Uniform cumulative positions
    u = np.linspace(0, 1, nx)

    points = np.interp(u, cdf, x_test.detach().numpy())

    #return torch.tensor(points).to(trainer.device)
    """
    nx_test = nx - nx_uniform
    m=3
    x_test = torch.linspace(x0,xf,nx_test*m).reshape(-1, 1)
    du_x, du2_xx = PINN.grad(x_test) # for times

    return torch.linspace(x0,xf,nx).to(trainer.device)


def lhs_like_linspace(start, end, steps, dims=1, device=None, dtype=None):
    """
    Latin Hypercube Sampling usando el mismo estilo de torch.linspace.
    
    start: escalar o tensor (d,)
    end:   escalar o tensor (d,)
    steps: cantidad de muestras
    dims:  si start/end son escalares, número de dimensiones
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    # Asegurar representaciones vectoriales
    start = torch.tensor(start, dtype=dtype, device=device)
    end   = torch.tensor(end, dtype=dtype, device=device)

    if start.ndim == 0:  # caso escalar
        start = start.repeat(dims)
        end   = end.repeat(dims)

    d = start.shape[0]  # número de dimensiones

    # División en estratos
    cut = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)

    # Crear valores aleatorios dentro de cada estrato
    u = torch.rand(steps, d, device=device, dtype=dtype)
    a = cut[:-1]
    b = cut[1:]
    samples = a.unsqueeze(1) + (b - a).unsqueeze(1) * u

    # Permutar estratos por dimensión
    for j in range(d):
        perm = torch.randperm(steps, device=device)
        samples[:, j] = samples[perm, j]

    # Escalar a [start, end]
    return start + samples * (end - start)

def lhs_collocation(PINN, t, pde_residual):
    """
    x_test = torch.linspace(x0,xf,nx_test).reshape(-1, 1)

    # du_x, du2_xx = PINN.grad(x_test)

    # du_x = du_x.cpu().numpy()

    du2_xx = du2_xx.detach().cpu().numpy()

    print("du2_xx",du2_xx)

    norm_du2_xx = du2_xx / np.sum(du2_xx)

    # 2. Compute the CDF
    cdf = np.cumsum(norm_du2_xx)
    cdf /= cdf[-1]

    # Uniform cumulative positions
    u = np.linspace(0, 1, nx)

    points = np.interp(u, cdf, x_test.detach().numpy())

    #return torch.tensor(points).to(trainer.device)
    """
    nx_test = nx - nx_uniform
    m=3
    x_test = torch.linspace(x0,xf,nx_test*m).reshape(-1, 1)
    du_x, du2_xx = PINN.grad(x_test) # for times

    return lhs_like_linspace(x0,xf,nx).to(trainer.device)


def gradient_collocation(PINN, t, pde_residual):

    m = 3

    min_d = 0.01

    nx_test = nx - nx_uniform

    x_test = torch.linspace(x0,xf,nx_test*m).reshape(-1, 1)

    du_x, du2_xx = PINN.grad(x_test)

    du_x = torch.sqrt(du_x[:,0].detach()**2)

    print(f" du shape {du_x}")

    du_x_escaled = du_x / torch.sum(du_x)
    sorted_vals, sorted_idx = torch.sort(du_x_escaled,descending=True)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x_test.detach().numpy(), du_x_escaled.detach().numpy(), 'o-', markersize=2)
    plt.show()
    
    res = torch.tensor([])

    for i in range(nx_test):
        res= torch.hstack((res,x_test[sorted_idx[i]]))

    uniform = torch.linspace(x0,xf,nx - nx_test).to(trainer.device) 
    res = torch.hstack((res,uniform))
    
    res, _ = torch.sort(res)



    return res.detach().to(trainer.device)
    


def curvature_collocation(PINN, t, pde_residual):

    m = 3

    min_d = 0.01

    nx_test = nx - nx_uniform

    x_test = torch.linspace(x0,xf,nx_test*m).reshape(-1, 1)

    du_x, du2_xx = PINN.grad(x_test)

    du2_xx = torch.sqrt(du2_xx[:,0].detach()**2)

    print(f" du shape {du2_xx}")

    du2_xx_escaled = du2_xx / torch.sum(du2_xx)
    sorted_vals, sorted_idx = torch.sort(du2_xx_escaled,descending=True)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x_test.detach().numpy(), du2_xx_escaled.detach().numpy(), 'o-', markersize=2)
    plt.show()
    
    res = torch.tensor([])

    for i in range(nx_test):
        res= torch.hstack((res,x_test[sorted_idx[i]]))

    uniform = torch.linspace(x0,xf,nx//2).to(trainer.device) 
    res = torch.hstack((res,uniform))
    
    res, _ = torch.sort(res)



    return res.detach().to(trainer.device)


def residual_collocation(PINN, t, pde_residual):

    m = 3

    min_d = 0.01

    nx_test = nx - nx_uniform

    x_test = torch.linspace(x0,xf,nx).reshape(-1, 1)

    du_x, du2_xx = PINN.grad(x_test) # for times

    pde_residual_positive = torch.sqrt(pde_residual[:,0]**2)

    sorted_vals, sorted_idx = torch.sort(pde_residual_positive,descending=True)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x_test.detach().numpy(), pde_residual_positive.detach().numpy(), 'o-', markersize=2)
    plt.show()
    
    res = torch.tensor([])

    for i in range(nx_test):
        res= torch.hstack((res,x_test[sorted_idx[i]]))

    uniform = torch.linspace(x0,xf,nx//2).to(trainer.device) 
    res = torch.hstack((res,uniform))
    
    res, _ = torch.sort(res)



    return res.detach().to(trainer.device)




trainer.set_initial_condition(x_initial=torch.linspace(x0,xf,nx), func=toothsaw_solution)

trainer.set_boundary_function(toothsaw_solution)

trainer.set_point_collocation_distribution(curvature_collocation)


trainer.bc_initial


trainer.train_td_implicit()



#%% Get and plot solution

nx_test = 10000

options = [('uniform',),('gradient'),('curvature'),('')]

import matplotlib.pyplot as plt

def plot_solution(self, PINN, t, nx=1000):
        
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
        plt.title(f'u(x) at t = {t:.4f}')
        plt.show()

for option in options:

    trainer = TrainerDiffusion1D()
    trainer.set_device("cpu")

    nu = 0.07
    # x_bc = torch.Tensor([-3.0,3.0]).to(trainer.device)
    x_bc = torch.Tensor([0,2*np.pi])
    trainer.set_simulation_parameters(nu=nu, t0=0,dt=0.01,nt=5, x_bc=x_bc)

    trainer.set_training_parameters(tol=1e-2, tol_ic=1e-2, max_epochs=30000, n_prints_per_training=10, 
                                    PINNs_trained_per_plot=1, lr=1e-2,min_lr=1e-8)

    # x0 y xf

    x0 = 0
    xf = 2*np.pi

    trainer.set_initial_condition(x_initial=torch.linspace(x0,xf,nx), func=toothsaw_solution)

    trainer.set_boundary_function(toothsaw_solution)

    trainer.set_point_collocation_distribution(residual_collocation)


    trainer.bc_initial


    trainer.train_td_implicit()

    PINN_list = trainer.PINNS_list





    






# %%

=

