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

nu = 1
# x_bc = torch.Tensor([-3.0,3.0]).to(trainer.device)
x_bc = torch.Tensor([-0.25,0.25])
trainer.set_simulation_parameters(nu=nu, t0=0.0001,dt=0.0001,nt=3, x_bc=x_bc)

trainer.set_training_parameters(tol=1e-2, tol_ic=1e-2, max_epochs=30000, n_prints_per_training=10, PINNs_trained_per_plot=1, lr=1e-2,min_lr=1e-8)

# x0 y xf

x0 = -0.25
xf = 0.25

# Analytic Solution

def heat_kernel(t=0,x=0):
    # return ( 1/np.sqrt(4*np.pi*nu*t) )*np.exp(-x**2/(4*nu*t))
    print("sol heat_kernel", f"t {t}, x {x}, nu {nu}")
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

# Point Collocation Func





nx = 100 # puntos sin contar BC
nx_test = 400

def collocation_func_der(PINN, t):
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

    return torch.linspace(x0,xf,nx).to(trainer.device)


trainer.set_initial_condition(x_initial=torch.linspace(x0,xf,nx), func=heat_kernel)

trainer.set_boundary_function(heat_kernel)

trainer.set_point_collocation_distribution(collocation_func_der)


trainer.bc_initial


trainer.train_td_implicit_2()



#%% Get and plot solution

nx_test = 10000




    






# %%

