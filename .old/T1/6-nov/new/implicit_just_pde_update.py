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

nu = 0.3
# x_bc = torch.Tensor([-3.0,3.0]).to(trainer.device)
x_bc = np.array([0.0,2.0])
trainer.set_simulation_parameters(nu=nu, t0=0.0001,dt=0.01,nt=10, x_bc=x_bc)

trainer.set_training_parameters(tol=1e-3, tol_ic=1e-3, max_epochs=300000, n_prints_per_training=300, PINNs_trained_per_plot=1, lr=1e-2,min_lr=1e-8)

# x0 y xf

x0 = 0.0
xf = 2.0

# Initial condition function to match working version
def step_initial_condition(t=0, x=0):
    """Step function initial condition exactly like in working version"""
    if isinstance(x, torch.Tensor):
        result = torch.ones_like(x, dtype=torch.float32)
        # For domain [0,2], the step is between 0.25*2=0.5 and 0.5*2=1.0
        mask = (x >= 0.5) & (x <= 1.0) 
        result[mask] = 2.0
        return result
    elif isinstance(x, np.ndarray):
        result = np.ones_like(x, dtype=np.float32)
        mask = (x >= 0.5) & (x <= 1.0)
        result[mask] = 2.0
        return result
    else:
        return 2.0 if 0.5 <= x <= 1.0 else 1.0

# Boundary condition function 
def boundary_condition(t=0, x=0):
    """Boundary conditions - constant values at boundaries"""
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return torch.ones_like(x) if torch.is_tensor(x) else np.ones_like(x)
    else:
        return 1.0

# Point Collocation Func





nx = 40 # puntos sin contar BC
nx_test = 400

def collocation_func_der(PINN, t):

    x_test = torch.linspace(x0,xf,nx_test)

    du_x, du2_xx = PINN.grad(x_test)

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

    return torch.tensor(points).to(trainer.device)

    # return torch.linspace(x0,xf,nx).to(trainer.device)  # This breaks adaptive distribution!


trainer.set_initial_condition(x_initial=torch.linspace(x0,xf,nx), func=step_initial_condition)

trainer.set_boundary_function(boundary_condition)

trainer.set_point_collocation_distribution(collocation_func_der)

#%%

trainer.bc_initial

#%% Train 

trainer.train_td_implicit()


 #%% Get and plot solution

nx_test = 10000




    






# %%

