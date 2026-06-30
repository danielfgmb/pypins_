# -*- coding: utf-8 -*-
#%% 
"""
Created on Sun Nov 20 12:20:38 2022

@author: Felipe
"""
# Solve the PDE du/dt = d2u/dx2
# via GS

import torch
import matplotlib.pyplot as plt
import numpy as np
import time

# Library for Design of Experiments
from pyDOE import lhs # Latin Hypercube Sampling

# Solve Diffusion in 1D

#%% Declare parameters

nu = 0.3        # parametro nu que acompania a la ecuacion diferencial

epochs = 100000 # numero de epocas de entrenamiento por si no se cumple epsilon
n_prints = 10   # numero de impresiones del loss en cada PINN
n_plots = 1200     # numero de graficas para ver evolucion de la solucion
tol = 1e-6      # epsilon o tolerancia permitida
layers = np.array([1, 32, 32, 1]) # arquitectura de PINNs

# create tensors for the domains

nx = 100        # numero de puntos para generar la condicion inicial
nt = 1200       # numero de pasos de tiempo (numero de PINNs a la vez)
tf = 0.1        # tiempo final de simulacion
Nf = 100        # numero de puntos de colocacion para entrenamiento

dt = tf/nt
print(dt)
x = torch.linspace(0, 2, nx+1)
t = torch.linspace(0, tf, nt+1)

print("tamaÃ±o de x: ", x.size(),
      "tamaÃ±o de t: ", t.size())

X, T = torch.meshgrid(x, t)

lb = x[0]   # lower bound o cuando x = 0
ub = x[-1]  # upper bound o cuando x = 2

x1 = X[:, 0][:, None].detach() # vector de x en tiempo t = 0

#%% Design the architecture of the PINN

class FCN(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = torch.nn.Tanh()
        self.loss_function = torch.nn.MSELoss(reduction="mean")
        self.linears = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for i in range(len(layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.)
            torch.nn.init.zeros_(self.linears[i].bias.data)
        
    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def gradiente(self, x):
        g = x.clone()
        g.requires_grad=True
        f = self.forward(g)
        f_x = torch.autograd.grad(f, g,
                                  torch.ones([g.shape[0], 1]),
                                  retain_graph=True,
                                  create_graph=True)[0]
        f_xx = torch.autograd.grad(f_x, g,
                                  torch.ones([g.shape[0], 1]),
                                  retain_graph=True,
                                  create_graph=True)[0]
        return f_xx
    
    def lossBC(self, x_BC, y_BC):
        loss_BC = self.loss_function(self.forward(x_BC), y_BC)
        return loss_BC
    
    def loss(self, x_BC, y_BC):
        return self.lossBC(x_BC, y_BC)

#%% Boundary and Initial Conditions
# Initial condition
x_init = x[:, None]
y_init = torch.ones(x.shape[0], 1)
print("tamaÃ±o de y_init: ", y_init.size())
y_init[int(0.25*nx):(int(0.5*nx)+1), 0] = 2.

# se declara una lista de PINNs donde se ira guardando cada PINN
# y otra lista donde se guardan los tiempos de entrenamiento de estos
lista_PINNs = []
train_time = []
ii = 0

# empieza a correr el tiempo de entrenamiento
TIC = time.time()

for j in range(len(t)):
    print("------------------------------\nIteration # ", j)
    # Boundary condition
    # f(x=0,t) = 0
    bottom_x = X[0, j][None, None]
    bottom_y = torch.ones(bottom_x.shape[0], 1)
    
    # f(x=1,t) = 0
    top_x = X[-1, j][None, None]
    top_y = torch.ones(bottom_x.shape[0], 1)
    
    bottom_x.size(), bottom_y.size(),
    top_x.size(), top_y.size()
    
    # se agrupan las condiciones iniciales y condiciones de borde
    X_train = torch.vstack([x_init, bottom_x, top_x])
    Y_train = torch.vstack([y_init, bottom_y, top_y])
        
    # Create the j-th model for the j-th time step
    
    PINN_j = FCN(layers)
    
    # Train model
    # si el modelo no es el primer PINN, pues hereda los parametros del modelo
    # anterior (Transfered Learning)
    if j != 0:
        PINN_j.load_state_dict(lista_PINNs[j-1].state_dict())
        X_train = torch.vstack([X_train_apres, bottom_x, top_x])
        Y_train = torch.vstack([y_train_apres, bottom_y, top_y])
    
    optimizer = torch.optim.Adam(PINN_j.parameters(), lr=1e-3, amsgrad=False)
    
    # empieza un contador de tiempo para cada PINN_j
    tic = time.time()
    
    for i in range(epochs):
        loss = PINN_j.loss(X_train, Y_train)
        optimizer.zero_grad()
        loss.backward()
        if loss.item() <= tol:
            print("epoch until tolerance: ", i)
            break
        optimizer.step()
        
        if i%(epochs/n_prints) == 0:
            print("training: ", round(loss.item(), 7))
    
    # termina el contador de tiempo del PINN_j  
    toc = time.time()
    elapsed_time = toc-tic
    lista_PINNs.append(PINN_j)
    train_time.append(elapsed_time)
    print("Elapsed time: ", elapsed_time, " seconds")
    
    # Advance in time with EULER METHOD
    # apres es equivalente a la solucion "despues" o avanzada
    
    X_train_apres = lb + (ub - lb)*lhs(1, Nf)
    X_train_apres = X_train_apres.float()
    
    y1 = PINN_j(X_train_apres).detach()
    f_x = PINN_j.gradiente(X_train_apres).detach()
    y_train_apres = y1 + dt*nu*f_x # en esta linea se avanza la solucion
    
    
    #% Plot
    if j%(nt/n_plots) == 0:
        ii += 1
        print (ii)
        y1 = PINN_j(X[:, j][:, None]).detach()
        plt.plot(x1, y1)

TOC = time.time()
TOTAL_TIME = TOC - TIC

plt.xticks(np.linspace(0, 2, 11))
plt.yticks(np.linspace(1, 2, 6))
plt.xlabel("x", fontsize=15)
plt.ylabel("u(x,t)", fontsize=15)
plt.legend(["CondiciÃ³n Inicial", "SoluciÃ³n en t = 0.1"])
plt.title("MÃ©todo GS RÃ©plica 4", fontsize=15)
plt.savefig("D:/Felipe/Documents/Uniandes/Semestre9/Proyecto IMEC/DiffMM.svg")

# termina el temporizador y se estima el tiempo de entrenamiento total
print("Total runtime: ", TOTAL_TIME, " seconds")

# %%
